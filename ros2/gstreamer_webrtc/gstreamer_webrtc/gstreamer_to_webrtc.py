import asyncio
import json
import logging
import threading
from typing import List, Optional

import gi
import websockets
from websockets.client import WebSocketClientProtocol

gi.require_version("Gst", "1.0")
from gi.repository import Gst

gi.require_version("GstWebRTC", "1.0")
from gi.repository import GstWebRTC

gi.require_version("GstSdp", "1.0")
from gi.repository import GstSdp

gi.require_version("GLib", "2.0")
from gi.repository import GLib


class GstreamerToWebRTCBridge:

    def __init__(
        self,
        pipeline,
        logger,
        server_url="ws://0.0.0.0:8080/?topic=/camera0/debug_frame",
        webrtc_pipeline_element_name="webrtc",
        connection_timeout=3.0,
    ) -> None:
        """
        Initializes the GstreamerToWebRTCBridge.

        Args:
            pipeline (Gst.Pipeline): The GStreamer pipeline object to manage.
            logger (Optional[logging.Logger]): An optional logger instance. If None,
                a default logger is created.
            server_url (str): The URL of the WebSocket signaling server.
            webrtc_pipeline_element_name (str): The name of the 'webrtcbin'
                element within the GStreamer pipeline.
            connection_timeout (float): Timeout in seconds for establishing the
                WebSocket connection.
        """

        self._pipeline = pipeline

        if logger:
            self._logger = logger
        else:
            self._logger = logging.getLogger(__name__)
            self._logger.setLevel(logging.INFO)
            if not self._logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
                )
                handler.setFormatter(formatter)
                self._logger.addHandler(handler)

        # WebSocket and WebRTC configuration
        self._websocket_conn: Optional[WebSocketClientProtocol] = None
        self._server: str = server_url
        self._webrtc_pipeline_element_name: str = webrtc_pipeline_element_name
        self._webrtc: Optional[Gst.Element] = None
        self._connection_timeout: float = connection_timeout
        self._pipeline_started: bool = False

        # Threading and asyncio event loop management
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._shutdown_event: Optional[asyncio.Event] = None
        self._shutdown_complete: threading.Event = threading.Event()
        self._tasks: List[asyncio.Task] = []
        self._read_task: Optional[asyncio.Task] = None
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False

    @property
    def _connected(self) -> bool:
        """
        Checks if the WebSocket connection is active and not in a closing state.

        Args: None

        Returns:
            bool: True if connected, False otherwise.
        """
        if self._websocket_conn is None:
            return False
        try:
            return (
                self._websocket_conn.transport is not None
                and not self._websocket_conn.transport.is_closing()
            )
        except:
            return False

    def start(self) -> None:
        """
        Starts the WebRTC bridge handler in a new thread.

        Initializes and runs the asyncio event loop that manages the WebSocket
        connection and signaling.

        Args: None

        Returns: None
        """
        if self._running:
            self._logger.warning("WebRTC handler already running")
            return

        self._running = True
        self._shutdown_complete.clear()
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()

    def _run_event_loop(self) -> None:
        """
        The main entry point for the handler thread.

        This method sets up a new asyncio event loop and runs the connection
        manager task until a shutdown is signaled. It also handles the graceful
        cancellation of all running tasks upon exit.

        Args: None

        Returns: None
        """
        self._logger.info("WebRTC stream is starting...")
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._shutdown_event = asyncio.Event()

        try:
            connect_task = self._loop.create_task(self._connect_loop_task_fn())
            self._tasks = [connect_task]
            self._loop.run_until_complete(self._shutdown_event.wait())
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self._logger.error(f"Error in event loop: {e}")
        finally:
            self._logger.info("Cancelling outstanding tasks...")

            for task in self._tasks:
                if not task.done():
                    task.cancel()

            if self._read_task and not self._read_task.done():
                self._read_task.cancel()

            async def _gather_cancelled_tasks():
                await asyncio.gather(*self._tasks, return_exceptions=True)
                if self._read_task:
                    await asyncio.gather(self._read_task, return_exceptions=True)

            if self._loop.is_running():
                self._loop.run_until_complete(_gather_cancelled_tasks())

            self._logger.info("Closing event loop...")
            try:
                self._loop.close()
            except Exception as e:
                self._logger.error(f"Error closing loop: {e}")

            self._loop = None
            self._running = False
            self._shutdown_complete.set()
            self._logger.info("WebRTC stream has exited")

    async def _connect_loop_task_fn(self) -> None:
        """
        Manages the WebSocket connection lifecycle.

        This coroutine runs in a loop, attempting to connect to the signaling
        server. Once connected, it starts reading messages. If the connection
        is lost, it waits for a delay before attempting to reconnect.

        Args: None

        Returns: None
        """
        reconnect_delay = 5

        while not self._shutdown_event.is_set():
            try:
                if not self._connected:
                    self._logger.info("Attempting connection to signalling server")
                    await asyncio.wait_for(
                        self._connect(), timeout=self._connection_timeout
                    )

                    self._logger.info("Connection successful, now reading messages.")
                    await self._read_messages_task_fn()

                    if (
                        not self._shutdown_event.is_set()
                    ):  # Only wait if not shutting down
                        self._logger.info(
                            f"Waiting {reconnect_delay}s before reconnecting..."
                        )
                        await asyncio.sleep(reconnect_delay)
                else:
                    await asyncio.sleep(1)

            except asyncio.TimeoutError:
                self._logger.warning("Connection timeout")
                self._websocket_conn = None
                if not self._shutdown_event.is_set():
                    await asyncio.sleep(reconnect_delay)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.warning(f"connect_loop error: {repr(e)}")
                self._websocket_conn = None
                if not self._shutdown_event.is_set():
                    await asyncio.sleep(reconnect_delay)

    async def _connect(self) -> None:
        """
        Establishes a connection to the WebSocket signaling server.

        On the first successful connection, it also initializes the GStreamer pipeline.

        Args: None

        Returns: None
        """
        try:
            self._websocket_conn = await websockets.connect(self._server)
            if not self._pipeline_started:
                self._start_pipeline()
        except Exception as e:
            self._logger.error(f"Failed to connect: {e}")
            raise

    async def _read_messages_task_fn(self) -> None:
        """
        Reads incoming messages from the WebSocket connection.

        This coroutine continuously listens for messages and passes them to the
        message handler. It exits when the connection is closed or when a
        shutdown is initiated.

        Args: None

        Returns: None
        """
        try:
            self._logger.info("Starting to read messages from WebSocket")
            async for message in self._websocket_conn:
                if self._shutdown_event.is_set():
                    break
                try:
                    self._handle_message(message)
                except Exception as e:
                    self._logger.error(
                        f"Error handling message '{message[:100]}...': {e}"
                    )
        except websockets.exceptions.ConnectionClosed as e:
            self._logger.info(f"WebSocket connection closed: {e}")
        except asyncio.CancelledError:
            self._logger.info("Message reading cancelled")
            raise
        except Exception as e:
            self._logger.error(f"Error reading messages: {e}")
        finally:
            old_conn = self._websocket_conn
            self._websocket_conn = None
            if old_conn is not None:
                is_closed = getattr(old_conn, "closed", True)
                if callable(is_closed):
                    is_closed = is_closed()
                if not is_closed:
                    try:
                        await old_conn.close()
                    except:
                        pass
            self._logger.info("Message reading loop ended")

    def _handle_message(self, message) -> None:
        """
        Processes a raw message received from the signaling server.

        Args:
            message (str): The message string received from the WebSocket.

        Returns: None
        """
        self._logger.debug(f"Message: {message}")
        try:
            self._handle_sdp(message)
        except json.JSONDecodeError:
            self._logger.warning(f"Unknown message format: {message}")

    async def _send_sdp_offer(self, sdp_text: str) -> None:
        """
        Sends an SDP offer to the signaling server.

        Args:
            sdp_text (str): The SDP offer content as a string.

        Returns: None
        """
        try:
            if not self._connected:
                self._logger.error("Cannot send SDP: not connected to signaling server")
                return

            self._logger.info(f"Sending offer:\n{sdp_text[:200]}...")
            msg = json.dumps({"sdp": {"type": "offer", "sdp": sdp_text}})
            await self._websocket_conn.send(msg)
            self._logger.info("SDP offer sent successfully")

        except Exception as e:
            self._logger.error(f"Failed to send SDP offer: {e}")

    def _on_offer_created(self, promise, _, __) -> None:
        """
        Callback executed when a GStreamer Promise for an SDP offer is fulfilled.

        Args:
            promise (Gst.Promise): The promise containing the SDP offer reply.

        Returns: None
        """
        try:
            promise.wait()
            reply = promise.get_reply()
            offer = reply.get_value("offer")

            sdp_text = offer.sdp.as_text()

            promise = Gst.Promise.new()
            self._webrtc.emit("set-local-description", offer, promise)
            promise.interrupt()

            self._safe_run_coro(self._send_sdp_offer(sdp_text))

        except Exception as e:
            self._logger.error(f"Error in on_offer_created: {e}")

    def _send_ice_candidate_message(self, _, mlineindex, candidate) -> None:
        """
        Callback for the 'on-ice-candidate' signal from the webrtcbin element.

        Sends the generated ICE candidate to the signaling server.

        Args:
            _ (Gst.Element): The webrtcbin element that emitted the signal.
            mlineindex (int): The media line index of the candidate.
            candidate (str): The ICE candidate string.

        Returns: None
        """
        try:
            icemsg = json.dumps(
                {"ice": {"candidate": candidate, "sdpMLineIndex": mlineindex}}
            )

            if self._connected:
                self._safe_run_coro(self._websocket_conn.send(icemsg))

        except Exception as e:
            self._logger.error(f"Error sending ICE candidate: {e}")

    def _on_incoming_decodebin_stream(self, _, pad) -> None:
        """
        Callback for the 'pad-added' signal from a decodebin element.

        This is used to handle decoded streams from a remote peer. It links the
        newly decoded audio or video pad to an appropriate sink element.

        Args:
            _ (Gst.Element): The decodebin element that emitted the signal.
            pad (Gst.Pad): The new pad that has been added to the element.

        Returns: None
        """
        if not pad.has_current_caps():
            self._logger.info(f"{pad} has no caps, ignoring")
            return

        caps = pad.get_current_caps()
        assert len(caps)
        s = caps[0]
        name = s.get_name()

        if name.startswith("video"):
            q = Gst.ElementFactory.make("queue")
            conv = Gst.ElementFactory.make("videoconvert")
            sink = Gst.ElementFactory.make("autovideosink")
            self._pipeline.add(q, conv, sink)
            self._pipeline.sync_children_states()
            pad.link(q.get_static_pad("sink"))
            q.link(conv)
            conv.link(sink)
        elif name.startswith("audio"):
            q = Gst.ElementFactory.make("queue")
            conv = Gst.ElementFactory.make("audioconvert")
            resample = Gst.ElementFactory.make("audioresample")
            sink = Gst.ElementFactory.make("autoaudiosink")
            self._pipeline.add(q, conv, resample, sink)
            self._pipeline.sync_children_states()
            pad.link(q.get_static_pad("sink"))
            q.link(conv)
            conv.link(resample)
            resample.link(sink)

    def _on_incoming_stream(self, _, pad) -> None:
        """
        Callback for the 'pad-added' signal from the webrtcbin element.

        Handles incoming remote media streams by linking them to a new decodebin
        instance for processing.

        Args:
            _ (Gst.Element): The webrtcbin element.
            pad (Gst.Pad): The new source pad for the incoming stream.

        Returns: None
        """
        if pad.direction != Gst.PadDirection.SRC:
            return

        decodebin = Gst.ElementFactory.make("decodebin")
        decodebin.connect("pad-added", self._on_incoming_decodebin_stream)
        self._pipeline.add(decodebin)
        decodebin.sync_state_with_parent()
        self._webrtc.link(decodebin)

    def _start_pipeline(self) -> None:
        """
        Finds the webrtcbin element in the pipeline and connects its signals.

        This method should be called once the pipeline is ready to be connected
        to the signaling logic.

        Args: None

        Returns: None
        """

        if self._pipeline_started:
            self._logger.info("Pipeline already started")
            return

        try:

            self._webrtc = self._pipeline.get_by_name(
                self._webrtc_pipeline_element_name
            )
            if not self._webrtc:
                self._logger.error(
                    f"Could not find webrtcbin element with name '{self._webrtc_pipeline_element_name}'"
                )
                return

            try:
                self._webrtc.disconnect_by_func(self._send_ice_candidate_message)
                self._webrtc.disconnect_by_func(self._on_incoming_stream)
            except:
                pass

            self._webrtc.connect("on-ice-candidate", self._send_ice_candidate_message)
            self._webrtc.connect("pad-added", self._on_incoming_stream)

            current_state = self._pipeline.get_state(0)[1]
            self._logger.info(f"Pipeline state: {current_state.value_nick}")

            self._pipeline_started = True
            self._logger.info("WebRTC pipeline initialized successfully")

        except Exception as e:
            self._logger.error(f"Error starting pipeline: {e}")

    def _handle_sdp(self, message) -> None:
        """
        Parses SDP and ICE messages from the signaling server.

        This method handles SDP answers from the remote peer by setting them as
        the remote description on the webrtcbin element. It also handles incoming
        ICE candidates.

        Args:
            message (str): The JSON message string from the server.

        Returns: None
        """
        try:
            msg = json.loads(message)

            if "sdp" in msg:
                sdp = msg["sdp"]
                sdp_type = sdp.get("type", "")
                sdp_text = sdp.get("sdp", "")

                self._logger.debug(f"Received {sdp_type}:\n{sdp_text[:200]}...")

                if sdp_type == "answer":
                    if not self._webrtc:
                        self._logger.error(
                            "WebRTC element not initialized, cannot set remote description"
                        )
                        return

                    res, sdpmsg = GstSdp.SDPMessage.new()
                    GstSdp.sdp_message_parse_buffer(bytes(sdp_text.encode()), sdpmsg)
                    answer = GstWebRTC.WebRTCSessionDescription.new(
                        GstWebRTC.WebRTCSDPType.ANSWER, sdpmsg
                    )
                    promise = Gst.Promise.new()
                    self._webrtc.emit("set-remote-description", answer, promise)
                    promise.interrupt()
                    self._logger.info("Remote description set successfully")

            elif "ice" in msg:
                ice = msg["ice"]
                candidate = ice["candidate"]
                sdpmlineindex = ice["sdpMLineIndex"]

                if self._webrtc:
                    self._webrtc.emit("add-ice-candidate", sdpmlineindex, candidate)
                    self._logger.debug(f"Added ICE candidate: {candidate[:50]}...")
                else:
                    self._logger.warning(
                        "WebRTC element not ready, cannot add ICE candidate"
                    )

        except Exception as e:
            self._logger.error(f"Error handling SDP: {e}")

    def shutdown(self) -> None:
        """
        Initiates a graceful shutdown of the WebRTC handler.

        This method closes the WebSocket connection and signals the event loop
        thread to terminate.

        Args: None

        Returns: None
        """
        if not self._running:
            self._logger.info("WebRTC handler not running, nothing to shutdown")
            return

        future = self._safe_run_coro(self._async_shutdown())
        if future:
            try:
                future.result(timeout=5.0)
            except Exception as e:
                self._logger.error(f"Error during async shutdown: {e}")

    async def _async_shutdown(self) -> None:
        """
        Performs the asynchronous part of the shutdown process.

        Closes the WebSocket connection and sets the event to stop the loop.

        Args: None

        Returns: None
        """
        self._logger.info("Shutting down WebRTC handler...")

        if self._websocket_conn:
            try:
                await self._websocket_conn.close()
            except Exception as e:
                self._logger.warning(f"Error closing WebSocket connection: {e}")
            finally:
                self._websocket_conn = None

        self._shutdown_event.set()

    def handle_new_client(self) -> None:
        """
        Public API to manually trigger a new SDP offer.

        This is useful when a new peer connects and an offer needs to be sent
        to establish the WebRTC connection. It schedules the offer creation
        on the GStreamer main loop thread.

        Args: None

        Returns: None
        """
        GLib.timeout_add(100, self._do_create_offer)

    def _do_create_offer(self) -> bool:
        """
        Creates and emits an SDP offer from the webrtcbin element.

        This method is called via GLib's event loop to ensure it runs on the
        correct thread for GStreamer operations.

        Args: None

        Returns:
            bool: Always returns False to ensure the GLib timeout does not repeat.
        """
        if not self._webrtc:
            self._logger.error("Cannot create offer: webrtc element not initialized")
            return False

        self._logger.info("Manually creating offer...")
        promise = Gst.Promise.new_with_change_func(
            self._on_offer_created, self._webrtc, None
        )
        self._webrtc.emit("create-offer", None, promise)
        return False

    def join(self, timeout=None) -> None:
        """
        Waits for the handler thread to complete.

        This blocks the calling thread until the handler thread has terminated,
        or until the optional timeout occurs.

        Args:
            timeout (Optional[float]): The maximum time in seconds to wait.

        Returns: None
        """
        if hasattr(self, "_thread") and self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

        # Also wait for shutdown to complete
        if self._shutdown_complete:
            self._shutdown_complete.wait(timeout=timeout or 10.0)

    def _safe_run_coro(self, coro) -> Optional[asyncio.Future]:
        """
        Safely schedules a coroutine to run on the asyncio event loop from a
        different thread.

        Args:
            coro (Coroutine): The coroutine to execute.

        Returns:
            Optional[asyncio.Future]: A Future representing the execution of the
            coroutine, or None if the loop is not running.
        """
        try:
            if not self._loop or not self._loop.is_running():
                self._logger.error("Event loop not running, cannot schedule task")
                return None
            return asyncio.run_coroutine_threadsafe(coro, self._loop)
        except RuntimeError as e:
            self._logger.warning(f"Failed to schedule task: {e}")
            return None
