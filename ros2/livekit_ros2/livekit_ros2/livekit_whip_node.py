import os
import threading
from typing import Optional

import cv2
import gi
import numpy as np
import rclpy
import requests
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

gi.require_version("Gst", "1.0")
gi.require_version("GLib", "2.0")
gi.require_version("GstSdp", "1.0")
gi.require_version("GstWebRTC", "1.0")
import asyncio
from urllib.parse import urlparse

from gi.repository import GLib, Gst, GstSdp, GstWebRTC
from livekit import api
from livekit.api import ingress_service as ingress_api
from livekit.protocol.ingress import IngressInput


async def get_or_create_whip_endpoint(
    livekit_server_url: str,
    livekit_api_key: str,
    livekit_api_secret: str,
    room_name: str,
) -> str:
    """
    Gets or creates a WHIP URL for ingress.
    """
    async with api.LiveKitAPI(
        livekit_server_url, livekit_api_key, livekit_api_secret
    ) as lk:
        # Fetch all ingresses for the room
        resp = await lk.ingress.list_ingress(
            ingress_api.ListIngressRequest(room_name=room_name)
        )

        # Create WHIP ingress if none already exists
        whips = [
            i
            for i in resp.items
            if getattr(i, "input_type", None) == IngressInput.WHIP_INPUT
        ]
        info = whips[0] if whips else None
        if info is None:
            create_req = ingress_api.CreateIngressRequest(
                input_type=IngressInput.WHIP_INPUT,
                enable_transcoding=False,  # already transcoded by GStreamer
                name="ingress",
                room_name=room_name,
                participant_identity="publisher",
                participant_name="publisher",
            )
            info = await lk.ingress.create_ingress(create_req)

        endpoint = f"{info.url.rstrip('/')}/{info.stream_key}"
        return endpoint


class LiveKitWhipNode(Node):
    """
    Subscribes to a sensor_msgs/Image topic, GPU-encodes and publishes to LiveKit
    Ingress via WHIP.
    """

    def __init__(self):
        super().__init__("livekit_whip_node")

        # Parameters
        self.declare_parameter("livekit_url", "ws://localhost:7880")
        self.declare_parameter("livekit_api_key", "")
        self.declare_parameter("livekit_api_secret", "")
        self.declare_parameter("fps", 30)
        self.declare_parameter("bitrate_kbps", 4000)
        # TODO: Support multiple topics on demand, and one pipeline per topic
        self.declare_parameter("topic", "/camera0/debug_frame")

        livekit_url = (
            self.get_parameter("livekit_url").get_parameter_value().string_value
        )
        livekit_api_key = (
            self.get_parameter("livekit_api_key").get_parameter_value().string_value
        )
        livekit_api_secret = (
            self.get_parameter("livekit_api_secret").get_parameter_value().string_value
        )
        self._fps = self.get_parameter("fps").get_parameter_value().integer_value
        self._bitrate_kbps = (
            self.get_parameter("bitrate_kbps").get_parameter_value().integer_value
        )
        topic = self.get_parameter("topic").get_parameter_value().string_value

        # Jetson flag
        self._is_jetson = os.path.exists("/etc/nv_tegra_release")

        # Start GStreamer thread
        Gst.init(None)
        self._glib_mainloop = GLib.MainLoop()
        self._glib_thread = threading.Thread(
            target=self._glib_mainloop.run, daemon=True
        )
        self._glib_thread.start()

        self._pipeline: Optional[Gst.Pipeline] = None
        self._appsrc: Optional[Gst.Element] = None
        self._whip_resource_url: Optional[str] = None

        # Get WHIP URL
        if not livekit_url or not livekit_api_key or not livekit_api_secret:
            raise RuntimeError(
                "Parameters 'livekit_url', 'livekit_api_key', and 'livekit_api_secret' are required"
            )
        self._whip_url = asyncio.run(
            get_or_create_whip_endpoint(
                livekit_url, livekit_api_key, livekit_api_secret, topic
            )
        )
        self.get_logger().info(f"WHIP URL: {self._whip_url}")

        # Subscriptions
        self.create_subscription(Image, topic, self._on_image, qos_profile_sensor_data)
        self.get_logger().info(f"Subscribed to: {topic}")

    # region GStreamer pipeline

    def _make_element(self, factory: str, name: str) -> Gst.Element:
        elem = Gst.ElementFactory.make(factory, name)
        if not elem:
            raise RuntimeError(
                f"Failed to create GStreamer element '{factory}'. "
                f"Ensure the plugin providing it is installed."
            )
        return elem

    def _build_pipeline(self, width: int, height: int, format: str):
        # TODO: Implement Jetson pipeline
        self.get_logger().info("Building pipeline...")

        self._pipeline = Gst.Pipeline.new("pipeline")

        # appsrc
        self._appsrc = self._make_element("appsrc", "mysource")
        self._appsrc.set_property(
            "caps",
            Gst.Caps.from_string(
                f"video/x-raw,format={format},width={width},height={height},framerate={self._fps}/1"
            ),
        )
        self._appsrc.set_property("is-live", True)
        self._appsrc.set_property("format", Gst.Format.TIME)
        self._appsrc.set_property("do-timestamp", True)

        # videoconvert + capsfilter to convert to NV12
        videoconvert_nv12 = self._make_element("videoconvert", "videoconvert_nv12")
        capsfilter_nv12 = self._make_element("capsfilter", "capsfilter_nv12")
        capsfilter_nv12.set_property(
            "caps",
            Gst.Caps.from_string(
                f"video/x-raw,format=NV12,width={width},height={height},framerate={self._fps}/1"
            ),
        )

        # Encoder - nvh264enc for hardware encoding
        encoder = self._make_element("nvh264enc", "encoder_h264")
        # Target bitrate
        encoder.set_property("bitrate", self._bitrate_kbps)

        # Parser
        parser = self._make_element("h264parse", "parser_h264")
        # Ensure SPS/PPS is present for new subscribers
        parser.set_property("config-interval", 1)

        # Payloader - rtph264pay converts H264 video into RTP packets
        rtp_payloader = self._make_element("rtph264pay", "payloader")
        # Ensure SPS/PPS is present for new subscribers
        rtp_payloader.set_property("config-interval", 1)
        rtp_payloader.set_property("pt", 96)

        # webrtcbin
        self.webrtcbin = self._make_element("webrtcbin", "webrtcbin")
        self.webrtcbin.set_property("bundle-policy", "max-bundle")

        # Add elements to pipeline and link up
        self._pipeline.add(self._appsrc)
        self._pipeline.add(videoconvert_nv12)
        self._pipeline.add(capsfilter_nv12)
        self._pipeline.add(encoder)
        self._pipeline.add(parser)
        self._pipeline.add(rtp_payloader)
        self._pipeline.add(self.webrtcbin)

        assert self._appsrc.link(videoconvert_nv12)
        assert videoconvert_nv12.link(capsfilter_nv12)
        assert capsfilter_nv12.link(encoder)
        assert encoder.link(parser)
        assert parser.link(rtp_payloader)

        # Link payloader to webrtcbin
        sinkpad = self.webrtcbin.get_request_pad("sink_%u")
        if not sinkpad:
            raise RuntimeError("Failed to get request pad sink_%u from webrtcbin")
        srcpad = rtp_payloader.get_static_pad("src")
        if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
            raise RuntimeError("Failed to link RTP payloader to webrtcbin")

        # Bus
        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        # webrtc signals
        self.webrtcbin.connect("on-negotiation-needed", self._on_negotiation_needed)
        self.webrtcbin.connect("on-ice-candidate", self._on_ice_candidate)
        self.webrtcbin.connect(
            "notify::ice-gathering-state",
            self._on_ice_gathering_state,
        )

        self.get_logger().info("Pipeline created")

        # Start pipeline
        self._pipeline.set_state(Gst.State.PLAYING)
        self.get_logger().info("Pipeline PLAYING")

    # endregion

    # region WebRTC / WHIP signaling

    """
    1. Create the SDP offer (create-offer) and set it locally (set-local-description)
    2. Wait for ICE candidate network paths
    3. HTTP POST the SDP offer to the WHIP URL to register the session with the ingress server
    4. Upon receiving a response from the ingress server, set the answer SDP locally (set-remote-description)
    """

    def _on_negotiation_needed(self, webrtcbin):
        """
        Creates and emits an SDP offer from the webrtcbin element.
        """

        self.get_logger().info("Negotiation needed. Creating offer")
        promise = Gst.Promise.new_with_change_func(
            self._on_offer_created, webrtcbin, None
        )
        webrtcbin.emit("create-offer", None, promise)

    def _on_offer_created(self, promise, webrtcbin, _):
        """
        Sets the SDP offer locally.
        """

        promise.wait()
        reply = promise.get_reply()
        offer = reply.get_value("offer")
        p = Gst.Promise.new()
        webrtcbin.emit("set-local-description", offer, p)
        p.interrupt()  # don't block. We wait for ICE gathering to complete
        self.get_logger().info(
            "Offer created and local description set. Waiting for ICE gathering to complete..."
        )

    def _on_ice_gathering_state(self, webrtcbin, state):
        # TODO: use Trickle ICE for faster start
        state = self.webrtcbin.get_property("ice-gathering-state")
        if int(state) == int(GstWebRTC.WebRTCICEGatheringState.COMPLETE):
            self.get_logger().info(f"ICE gathering complete")
            self._send_offer_via_whip(self.webrtcbin)

    def _send_offer_via_whip(self, webrtcbin):
        """
        HTTP POST the SDP offer to the WHIP URL to register the session with the ingress server.
        Upon receiving a response from the ingress server, set the answer SDP locally.
        """

        self.get_logger().info(f"Sending offer via WHIP...")

        # At this point ICE candidates have been gathered. Grab the local SDP which includes all of
        # the candidates.
        local = webrtcbin.get_property("local-description")
        if local is None:
            self.get_logger().error("Missing local description; cannot POST to WHIP.")
            return
        offer_sdp_text = local.sdp.as_text()

        self.get_logger().info(f"POSTing offer to WHIP endpoint: {self._whip_url}")
        try:
            resp = requests.post(
                self._whip_url,
                data=offer_sdp_text,
                headers={"Content-Type": "application/sdp"},
                timeout=10,
            )
        except Exception as e:
            self.get_logger().error(f"WHIP POST failed: {e}")
            return

        if resp.status_code not in (200, 201):
            self.get_logger().error(
                f"WHIP POST HTTP {resp.status_code}: {resp.text[:200]}"
            )
            return

        self._whip_resource_url = resp.headers.get("Location", None)
        if self._whip_resource_url:
            self.get_logger().info(f"WHIP resource: {self._whip_resource_url}")

        answer_text = resp.text
        ret, answer_sdp = GstSdp.SDPMessage.new_from_text(answer_text)
        if ret != GstSdp.SDPResult.OK:
            self.get_logger().error("Failed to parse WHIP SDP answer.")
            return

        answer = GstWebRTC.WebRTCSessionDescription.new(
            GstWebRTC.WebRTCSDPType.ANSWER, answer_sdp
        )
        promise = Gst.Promise.new()
        webrtcbin.emit("set-remote-description", answer, promise)
        promise.interrupt()
        self.get_logger().info("Remote description set. WebRTC is live.")

    def _on_ice_candidate(self, webrtcbin, mlineindex, candidate):
        self.get_logger().info(
            f"Local ICE candidate: mline={mlineindex} cand={candidate}"
        )

    def _on_connection_state(self, webrtcbin, pspec):
        state = webrtcbin.get_property("connection-state")
        self.get_logger().info(f"WebRTC connection state: {state}")

    def _on_ice_state(self, webrtcbin, pspec):
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

    # endregion

    # region GStreamer bus

    def _on_bus_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            self.get_logger().error(f"GStreamer ERROR: {err} debug:{dbg}")
            self._pipeline.set_state(Gst.State.NULL)
        elif t == Gst.MessageType.WARNING:
            w, dbg = message.parse_warning()
            self.get_logger().warn(f"GStreamer WARN: {w} debug:{dbg}")
        elif t == Gst.MessageType.EOS:
            self.get_logger().info("GStreamer EOS")
            self._pipeline.set_state(Gst.State.NULL)
        elif t == Gst.MessageType.STATE_CHANGED and message.src == self._pipeline:
            old, new, pending = message.parse_state_changed()
            self.get_logger().info(
                f"Pipeline state changed: {old.value_nick} -> {new.value_nick}"
            )

    # endregion

    def _on_image(self, msg: Image):
        if self._pipeline is None:
            enc = (msg.encoding or "").lower()
            format = "BGR"
            if enc in ("rgb8", "rgb"):
                format = "RGB"
            elif enc in ("bgr8", "bgr"):
                format = "BGR"
            elif enc in ("i420", "yuv420", "yuv420p"):
                format = "I420"
            elif enc in ("nv12",):
                format = "NV12"
            else:
                self.get_logger().warn(
                    f"Unknown encoding '{msg.encoding}', assuming BGR"
                )
                format = "BGR"
            self._build_pipeline(msg.width, msg.height, format)

        data = bytes(msg.data)
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)

        if not hasattr(self, "start_time_ns"):
            self.start_time_ns = (
                msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            )

        timestamp_ns = (  # Assigns ROS timestamp to video frame
            msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        ) - self.start_time_ns
        buf.pts = timestamp_ns
        buf.dts = timestamp_ns
        buf.duration = Gst.CLOCK_TIME_NONE

        flow = self._appsrc.emit("push-buffer", buf)
        if flow != Gst.FlowReturn.OK:
            self.get_logger().warn(f"push-buffer -> {flow}")

    def destroy_node(self):
        # Tear down WHIP resource
        try:
            if self._whip_resource_url:
                headers = {}
                parsed_url = urlparse(self._whip_url)
                full_whip_resource_url = (
                    parsed_url.scheme
                    + "://"
                    + parsed_url.netloc
                    + self._whip_resource_url
                )
                requests.delete(full_whip_resource_url, headers=headers, timeout=3)
        except Exception as e:
            self.get_logger().warn(f"WHIP DELETE failed: {e}")

        # GStreamer shutdown
        try:
            if self._appsrc:
                try:
                    self._appsrc.emit("end-of-stream")
                except Exception:
                    pass
            if self._pipeline:
                self._pipeline.set_state(Gst.State.NULL)
        finally:
            try:
                if self._glib_mainloop.is_running():
                    self._glib_mainloop.quit()
            except Exception:
                pass

        super().destroy_node()


def main(argv=None):
    rclpy.init(args=argv)

    node = None
    executor = MultiThreadedExecutor()
    try:
        node = LiveKitWhipNode()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
