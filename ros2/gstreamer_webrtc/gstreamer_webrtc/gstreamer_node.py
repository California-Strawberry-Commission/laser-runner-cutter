import asyncio
import gc
import logging
import os
import threading
import time

import gi
import rclpy
import websockets
from gi.repository import GLib, Gst
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from gstreamer_webrtc.gstreamer_to_webrtc import GstreamerToWebRTCBridge

gi.require_version("Gst", "1.0")


class GStreamerRosWebRTCNode(Node):

    def __init__(self):
        super().__init__("gstreamer_ros_webrtc_node")
        self.get_logger().info("GStreamer ROS WebRTC Node initializing...")

        Gst.init(None)

        self.is_jetson_device = os.path.exists("/etc/nv_tegra_release")

        self.pipeline = None
        self.webrtcbin = None
        self.webrtc_handler = None
        self.appsrc = None
        self.frame_count = 0
        self.connection_established = False
        self.pipeline_is_playing = False

        self.glib_main_loop = GLib.MainLoop()
        self._glib_thread = threading.Thread(
            target=self._run_glib_main_loop, daemon=True
        )
        self._glib_thread.start()

        self.topics = {}
        self._server_thread = threading.Thread(
            target=self._run_signaling_server, daemon=True
        )
        self._server_thread.start()
        self.get_logger().info("Signaling server thread started.")

        self.declare_parameter("video_topic", "/camera0/debug_frame")
        self.video_topic = (
            self.get_parameter("video_topic").get_parameter_value().string_value
        )

        self.declare_parameter("image_width", 640)
        self.width = (
            self.get_parameter("image_width").get_parameter_value().integer_value
        )

        self.declare_parameter("image_height", 480)
        self.height = (
            self.get_parameter("image_height").get_parameter_value().integer_value
        )

        self.declare_parameter("image_framerate", 30)
        self.framerate = (
            self.get_parameter("image_framerate").get_parameter_value().integer_value
        )

        self.declare_parameter("gst_input_encoding", "rgb8")
        self.gst_input_encoding = (
            self.get_parameter("gst_input_encoding").get_parameter_value().string_value
        )

        self.declare_parameter(
            "signaling_server_url", "ws://localhost:8080/?topic=/camera0/debug_frame"
        )
        self.signaling_server_url = (
            self.get_parameter("signaling_server_url")
            .get_parameter_value()
            .string_value
        )

        self.declare_parameter("connection_timeout", 3.0)
        self.connection_timeout = (
            self.get_parameter("connection_timeout").get_parameter_value().double_value
        )

        self.subscription = self.create_subscription(
            Image, self.video_topic, self.image_callback, qos_profile_sensor_data
        )
        self.get_logger().info(f"Subscribed to ROS topic: {self.video_topic}")

    # This function destroys the webrtc session and pipeline
    def _teardown_webrtc_session(self):

        self.get_logger().info("Tearing down GStreamer Pipeline and WebRTC Handler")
        self.pipeline_is_playing = False
        self.connection_established = False  # Reset connection state
        self.frame_count = 0  # Reset frame counter

        # Store references to old objects
        old_handler = self.webrtc_handler
        old_pipeline = self.pipeline
        old_webrtcbin = self.webrtcbin

        # Nullify
        self.webrtc_handler = None
        self.pipeline = None
        self.appsrc = None
        self.webrtcbin = None

        # Shut down the old handler and wait for its thread to exit
        if old_handler:
            self.get_logger().info("Shutting down old WebRTC handler...")
            try:
                old_handler.shutdown()

                old_handler.join(timeout=10)  # Time for thread to exit

                # Check if thread is still alive
                if (
                    hasattr(old_handler, "_thread")
                    and old_handler._thread
                    and old_handler._thread.is_alive()
                ):
                    self.get_logger().warning(
                        "WebRTC handler thread still alive after timeout!"
                    )
                    # Force kill the thread's event loop
                    if old_handler.loop:
                        old_handler.loop.call_soon_threadsafe(old_handler.loop.stop)
                else:
                    self.get_logger().info("Old WebRTC handler thread has exited.")
            except Exception as e:
                self.get_logger().error(f"Error during handler shutdown: {e}")

        # Disconnect webrtcbin before cleaning up pipeline
        if old_webrtcbin:
            try:
                old_webrtcbin.disconnect_by_func(self.on_connection_state_changed)
                old_webrtcbin.disconnect_by_func(self.on_ice_state_changed)
            except:
                pass

        # Shut down the old pipeline
        if old_pipeline:
            self.get_logger().info("Setting old pipeline to NULL...")
            try:
                # Send EOS to flush the pipeline
                old_pipeline.send_event(Gst.Event.new_eos())

                time.sleep(0.1)  # Time delay to process EOS

                old_pipeline.set_state(Gst.State.NULL)

                # Wait for state change with timeout
                ret, current, pending = old_pipeline.get_state(5 * Gst.SECOND)

                if ret == Gst.StateChangeReturn.SUCCESS:
                    self.get_logger().info(
                        f"Old pipeline is now in {current.value_nick} state."
                    )
                else:
                    self.get_logger().warning(
                        f"Pipeline state change returned: {ret.value_nick}"
                    )

                # Remove bus watch
                bus = old_pipeline.get_bus()
                if bus:
                    bus.remove_signal_watch()

            except Exception as e:
                self.get_logger().error(f"Error during pipeline cleanup: {e}")

        gc.collect()

        self.get_logger().info("Teardown complete.")
        return False  # for GLib.idle_add

    # Called for first pipeline and all subsequent pipelines
    # (Re)creates pipeline and webrtcbin
    def _recreate_pipeline_and_webrtc_handler(self):

        # Ensure we are working with a clean slate
        self._teardown_webrtc_session()

        time.sleep(0.5)  # Time delay to ensure cleanup is performed

        self.get_logger().info("Creating GStreamer Pipeline and WebRTC Handler")

        # Reset state variables
        self.connection_established = False
        self.frame_count = 0
        if hasattr(self, "start_time_ns"):
            delattr(self, "start_time_ns")  # Reset timestamp baseline

        # Create a new pipeline
        if self.is_jetson_device:
            self.create_jetson_pipeline()
        else:
            self.create_pipeline_manually()

        assert self.pipeline is not None, "Pipeline creation failed!"

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_gst_message)

        # Setup connection monitoring for the new webrtcbin
        if self.webrtcbin:
            self.setup_connection_monitor()

        # Create and start a new WebRTC handler for the new pipeline
        self.webrtc_handler = GstreamerToWebRTCBridge(
            pipeline=self.pipeline,
            logger=self.get_logger(),
            server_url=self.signaling_server_url,
            webrtc_pipeline_element_name="webrtc",
            connection_timeout=self.connection_timeout,
        )
        self.webrtc_handler.start()

        # Set the new pipeline to PLAYING
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            self.get_logger().error("Failed to start pipeline")
            return False

        # Wait for state change
        ret, current_state, pending = self.pipeline.get_state(5 * Gst.SECOND)

        if current_state == Gst.State.PLAYING:
            self.get_logger().info("New pipeline is now in PLAYING state.")
            self.pipeline_is_playing = True
            # Small delay before triggering offer
            GLib.timeout_add(
                200, lambda: self.webrtc_handler.handle_new_client() or False
            )
        else:
            self.get_logger().error(
                f"Failed to set new pipeline to PLAYING. Current state: {current_state.value_nick}"
            )

        return False  # for GLib.idle_add

    # Runs the asyncio event loop for the server
    def _run_signaling_server(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._start_server_main())
            loop.run_forever()
        finally:
            loop.close()

    # Start the websockets server
    async def _start_server_main(self):
        self.get_logger().info("Starting WebRTC signaling server on ws://0.0.0.0:8080")
        server = await websockets.serve(self._signaling_handler, "0.0.0.0", 8080)
        await server.wait_closed()

    # The handler for websocket connection(s)
    async def _signaling_handler(self, websocket):
        path = websocket.request.path
        topic_name = path.split("?topic=")[-1] if "?topic=" in path else None
        if not topic_name:
            await websocket.close()
            return

        self.get_logger().info(f"Peer connected on topic: '{topic_name}'")
        self.topics.setdefault(topic_name, set()).add(websocket)

        if self.webrtc_handler is None:
            self.get_logger().info("First peer connected, creating new WebRTC session.")
            GLib.idle_add(self._recreate_pipeline_and_webrtc_handler)
        else:
            self.get_logger().info("Additional peer connected (likely self), ignoring.")

        try:
            async for message_str in websocket:
                peers = self.topics.get(topic_name, set())
                send_tasks = [
                    peer.send(message_str) for peer in peers if peer is not websocket
                ]
                await asyncio.gather(*send_tasks)
        finally:
            self.get_logger().info(f"Peer disconnected from topic: '{topic_name}'")
            self.topics.get(topic_name, set()).discard(websocket)

            remaining_peers = self.topics.get(topic_name, set())
            num_remaining = len(remaining_peers)
            handler_exists = self.webrtc_handler is not None

            self.get_logger().info(
                f"Teardown Check: Peers remaining = {num_remaining}. "
                f"Handler exists = {handler_exists}."
            )

            if num_remaining <= 1 and handler_exists:
                self.get_logger().info("CONDITION MET. Tearing down WebRTC session.")
                GLib.idle_add(self._teardown_webrtc_session)
            else:
                self.get_logger().info("CONDITION NOT MET. Not tearing down session.")

    # Monitor WebRTC connection state and unblock pad when connected
    def setup_connection_monitor(self):

        self.get_logger().info(f"Setting up connection monitor")
        self.webrtcbin.connect(
            "notify::connection-state", self.on_connection_state_changed
        )
        self.webrtcbin.connect(
            "notify::ice-connection-state", self.on_ice_state_changed
        )

    # Handle connection state changes
    def on_connection_state_changed(self, webrtcbin, pspec):

        state = webrtcbin.get_property("connection-state")
        self.get_logger().info(f"WebRTC connection state: {state}")

        if state == 2 and not self.connection_established:
            self.connection_established = True
            self.get_logger().info("Connection established - ensuring media flow")
            self.ensure_media_flow()

    # Handle ICE connection state changes with detailed logging for debugging
    def on_ice_state_changed(self, webrtcbin, pspec):

        state = webrtcbin.get_property("ice-connection-state")
        state_name = {
            0: "NEW",
            1: "CHECKING",
            2: "CONNECTED",
            3: "COMPLETED",
            4: "FAILED",
            5: "DISCONNECTED",
            6: "CLOSED",
        }.get(state, f"UNKNOWN({state})")

        self.get_logger().info(f"ICE connection state: {state_name}")

        # Force pipeline refresh when ICE is connected or completed
        if (
            state in [2, 3] and not self.connection_established
        ):  # CONNECTED or COMPLETED
            self.connection_established = True
            self.get_logger().info("ICE connection established - ensuring media flow")

            # Trick to ensure decoder can start
            if hasattr(self, "encoder"):
                structure = Gst.Structure.new_empty("GstForceKeyUnit")
                event = Gst.Event.new_custom(Gst.EventType.CUSTOM_DOWNSTREAM, structure)
                self.encoder.send_event(event)
                self.get_logger().info("Forced keyframe generation")

            self.ensure_media_flow()
        elif state == 4:  # FAILED
            self.get_logger().error("ICE connection failed!")
            self.connection_established = False

    # Ensure media is flowing through the pipeline, not needed with tee implementation
    def ensure_media_flow(self):

        self.get_logger().info("Forcing pipeline state refresh to unblock pads")

        success, position = self.pipeline.query_position(Gst.Format.TIME)

        self.pipeline.set_state(Gst.State.PAUSED)
        self.pipeline.get_state(Gst.CLOCK_TIME_NONE)
        self.pipeline.set_state(Gst.State.PLAYING)
        self.pipeline.get_state(Gst.CLOCK_TIME_NONE)

        if success:
            self.pipeline.seek_simple(
                Gst.Format.TIME,
                Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT,
                position,
            )

        self.get_logger().info("Pipeline state refreshed")

    # Jetson pipeline creation
    def create_jetson_pipeline(self):

        self.pipeline = Gst.Pipeline.new("webrtc-pipeline")

        self.webrtcbin = Gst.ElementFactory.make("webrtcbin", "webrtc")
        self.webrtcbin.set_property("bundle-policy", "max-bundle")

        self.appsrc = Gst.ElementFactory.make("appsrc", "video-source")
        appsrc_caps = Gst.Caps.from_string(
            "video/x-raw,format=RGB,width=640,height=480,framerate=30/1"
        )
        self.appsrc.set_property("caps", appsrc_caps)
        self.appsrc.set_property("is-live", True)
        self.appsrc.set_property("format", Gst.Format.TIME)
        self.appsrc.set_property("do-timestamp", False)

        videoconvert = Gst.ElementFactory.make("videoconvert", "convert")

        capsfilter1 = Gst.ElementFactory.make("capsfilter", "capsfilter1")
        capsfilter1_caps = Gst.Caps.from_string(
            "video/x-raw,format=I420,width=640,height=480,framerate=30/1"
        )
        capsfilter1.set_property("caps", capsfilter1_caps)

        nvvidconv = Gst.ElementFactory.make("nvvidconv", "nvvidconv")

        capsfilter2 = Gst.ElementFactory.make("capsfilter", "capsfilter2")
        capsfilter2_caps = Gst.Caps.from_string(
            "video/x-raw(memory:NVMM),format=NV12,width=640,height=480,framerate=30/1"
        )
        capsfilter2.set_property("caps", capsfilter2_caps)

        encoder = Gst.ElementFactory.make("nvv4l2h264enc", "h264-encoder")
        encoder.set_property("preset-level", 1)
        parser = Gst.ElementFactory.make("h264parse", "h264-parser")

        rtph264pay = Gst.ElementFactory.make("rtph264pay", "payloader")
        rtph264pay.set_property("pt", 96)

        self.pipeline.add(self.appsrc)
        self.pipeline.add(videoconvert)
        self.pipeline.add(capsfilter1)
        self.pipeline.add(nvvidconv)
        self.pipeline.add(capsfilter2)
        self.pipeline.add(encoder)
        self.pipeline.add(parser)
        self.pipeline.add(rtph264pay)
        self.pipeline.add(self.webrtcbin)

        if not self.appsrc.link(videoconvert):
            self.get_logger().error("Failed to link appsrc to videoconvert")
        if not videoconvert.link(capsfilter1):
            self.get_logger().error("Failed to link videoconvert to capsfilter1")
        if not capsfilter1.link(nvvidconv):
            self.get_logger().error("Failed to link capsfilter1 to nvvidconv")
        if not nvvidconv.link(capsfilter2):
            self.get_logger().error("Failed to link nvvidconv to capsfilter2")
        if not capsfilter2.link(encoder):
            self.get_logger().error("Failed to link capsfilter2 to encoder")
        if not encoder.link(parser):
            self.get_logger().error("Failed to link encoder to parser")
        if not parser.link(rtph264pay):
            self.get_logger().error("Failed to link parser to rtph264pay")

        rtp_sink_pad = self.webrtcbin.get_request_pad("sink_%u")
        src_pad = rtph264pay.get_static_pad("src")
        src_pad.link(rtp_sink_pad)

    # Non jetson pipeline creation
    # Minimal pipeline appsrc → videoconvert → NV12 → NVENC → H264Parse → RTP → webrtcbin
    def create_pipeline_manually(self):

        self.pipeline = Gst.Pipeline.new("simple-webrtc-pipeline")

        self.appsrc = Gst.ElementFactory.make("appsrc", "mysource")
        self.videoconvert = Gst.ElementFactory.make("videoconvert", "videoconvert")
        self.capsfilter = Gst.ElementFactory.make("capsfilter", "capsfilter")
        self.encoder = Gst.ElementFactory.make("nvh264enc", "encoder")
        self.parser = Gst.ElementFactory.make("h264parse", "parser")
        self.rtppay = Gst.ElementFactory.make("rtph264pay", "pay")
        self.webrtcbin = Gst.ElementFactory.make("webrtcbin", "webrtc")

        if not all(
            [
                self.appsrc,
                self.videoconvert,
                self.capsfilter,
                self.encoder,
                self.parser,
                self.rtppay,
                self.webrtcbin,
            ]
        ):
            raise RuntimeError("Failed to create all elements")

        self.appsrc.set_property("is-live", True)
        self.appsrc.set_property("format", Gst.Format.TIME)
        self.appsrc.set_property("do-timestamp", False)
        self.appsrc.set_property(
            "caps",
            Gst.Caps.from_string(
                f"video/x-raw,format={self._get_gst_format(self.gst_input_encoding)},"
                f"width={self.width},height={self.height},framerate={self.framerate}/1"
            ),
        )

        nv12_caps = Gst.Caps.from_string(
            f"video/x-raw,format=NV12,width={self.width},height={self.height},framerate={self.framerate}/1"
        )
        self.capsfilter.set_property("caps", nv12_caps)

        self.parser.set_property("config-interval", 1)
        self.rtppay.set_property("pt", 96)
        self.rtppay.set_property("mtu", 1200)

        for el in [
            self.appsrc,
            self.videoconvert,
            self.capsfilter,
            self.encoder,
            self.parser,
            self.rtppay,
            self.webrtcbin,
        ]:
            self.pipeline.add(el)

        assert self.appsrc.link(self.videoconvert)
        assert self.videoconvert.link(self.capsfilter)
        assert self.capsfilter.link(self.encoder)
        assert self.encoder.link(self.parser)
        assert self.parser.link(self.rtppay)
        assert self.rtppay.link(self.webrtcbin)

    # Convert ROS image encoding to GStreamer format
    def _get_gst_format(self, encoding):

        format_map = {
            "rgb8": "RGB",
            "bgr8": "BGR",
            "mono8": "GRAY8",
            "yuv422": "YUY2",
            "nv12": "NV12",
            "bgra8": "BGRA",
            "rgba8": "RGBA",
            "16UC1": "GRAY16_LE",
            "mono16": "GRAY16_LE",
        }

        if encoding not in format_map:
            self.get_logger().error(f"Unsupported ROS image encoding: {encoding}")
            raise ValueError(f"Unsupported ROS image encoding: {encoding}")

        return format_map[encoding]

    def _on_gst_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            self.get_logger().info("GStreamer: End-Of-Stream reached.")
            self.glib_main_loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            self.get_logger().error(
                f"GStreamer Error: {err.message} from {message.src.get_name()} ({debug})"
            )
            self.glib_main_loop.quit()
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            self.get_logger().warning(
                f"GStreamer Warning: {err.message} from {message.src.get_name()} ({debug})"
            )
        elif t == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipeline:
                old, new, pending = message.parse_state_changed()
                self.get_logger().debug(
                    f"Pipeline state changed: {old.value_nick} -> {new.value_nick}"
                )
        elif t == Gst.MessageType.STREAM_STATUS:
            status_type, owner = message.parse_stream_status()
            self.get_logger().debug(
                f"Stream status: {status_type.value_nick} for {owner.get_name()}"
            )
        elif t == Gst.MessageType.LATENCY:
            self.get_logger().info("Latency message received, recalculating...")
            self.pipeline.recalculate_latency()
        return True

    def _run_glib_main_loop(self):
        self.get_logger().info("GLib MainLoop running in dedicated thread.")
        try:
            self.glib_main_loop.run()
        except Exception as e:
            self.get_logger().error(f"Error in GLib MainLoop thread: {e}")
        finally:
            self.get_logger().info("GLib MainLoop exited from dedicated thread.")

    # Handle incoming ROS image messages with debugging
    def image_callback(self, msg: Image):

        if not self.pipeline_is_playing or not self.appsrc:
            self.get_logger().debug("Dropping frame, pipeline not ready.")
            return

        # Log first frame and every 30th frame
        if not hasattr(self, "first_frame_logged"):
            self.get_logger().info(
                f"First frame received! Size: {msg.width}x{msg.height}, encoding: {msg.encoding}"
            )
            self.first_frame_logged = True

        if msg.encoding != self.gst_input_encoding:
            self.get_logger().warning(
                f"Unexpected image encoding: {msg.encoding}, expected: {self.gst_input_encoding}"
            )
            return

        if msg.width != self.width or msg.height != self.height:
            self.get_logger().warning(
                f"Image size mismatch: {msg.width}x{msg.height}, expected: {self.width}x{self.height}"
            )
            return

        gst_buffer = Gst.Buffer.new_wrapped(bytes(msg.data))

        if not hasattr(self, "start_time_ns"):
            self.start_time_ns = (
                msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            )

        timestamp_ns = (  # Assigns ROS timestamp to video frame
            msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        ) - self.start_time_ns
        gst_buffer.pts = timestamp_ns
        gst_buffer.dts = timestamp_ns
        gst_buffer.duration = Gst.CLOCK_TIME_NONE

        ret = self.appsrc.emit("push-buffer", gst_buffer)

        if ret != Gst.FlowReturn.OK:
            if ret == Gst.FlowReturn.FLUSHING:
                self.get_logger().debug("Pipeline is flushing")
            elif ret == Gst.FlowReturn.EOS:
                self.get_logger().warning("Pipeline received EOS")
            else:
                self.get_logger().warning(
                    f"Failed to push buffer to appsrc: {ret.value_name}"
                )
        else:
            self.frame_count += 1

            # More debugging logging
            if self.frame_count % 10 == 0:
                success, position = self.pipeline.query_position(Gst.Format.TIME)
                if success:
                    position_s = position / Gst.SECOND

                    # Check if encoder is receiving frames
                    if hasattr(self, "encoder"):
                        encoder_pad = self.encoder.get_static_pad("sink")
                        if encoder_pad:
                            success, current = encoder_pad.query_position(
                                Gst.Format.TIME
                            )
                else:
                    self.get_logger().info(
                        f"Pushed {self.frame_count} frames to pipeline (position query failed)"
                    )

    def destroy_node(self):

        # Shutdown the WebRTC handler (async components)
        if hasattr(self, "webrtc_handler") and self.webrtc_handler:
            self.webrtc_handler.shutdown()
            self.webrtc_handler.join(timeout=5)

        # Cleanup GStreamer pipeline
        if hasattr(self, "pipeline") and self.pipeline:
            self.pipeline.send_event(Gst.Event.new_eos())
            time.sleep(0.5)
            self.pipeline.set_state(Gst.State.NULL)

        # Stop GLib main loop
        if (
            hasattr(self, "glib_main_loop")
            and self.glib_main_loop
            and self.glib_main_loop.is_running()
        ):
            self.glib_main_loop.quit()

        # Wait for GLib thread to finish
        if (
            hasattr(self, "_glib_thread")
            and self._glib_thread
            and self._glib_thread.is_alive()
        ):
            self._glib_thread.join(timeout=2)

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    executor = MultiThreadedExecutor()
    try:
        node = GStreamerRosWebRTCNode()
        executor.add_node(node)
        while rclpy.ok():
            executor.spin_once()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if node:
            node.get_logger().error(f"Caught exception: {e}")
        else:
            logging.error(f"Caught exception before node creation: {e}")
    finally:
        executor.shutdown()
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
