import asyncio
import json
import logging

import gi
import websockets

gi.require_version("Gst", "1.0")
from gi.repository import Gst

gi.require_version("GstWebRTC", "1.0")
from gi.repository import GstWebRTC

gi.require_version("GstSdp", "1.0")
from gi.repository import GstSdp

gi.require_version("GLib", "2.0")
from gi.repository import GLib


class MavWebRTC:
    
    def __init__(self, pipeline, config, webrtc_name="webrtc"):
        self.pipeline = pipeline
        self.logger = logging.getLogger("GSTREAMER_WEBRTC" + __name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.config = config
        self.conn = None
        self.server = "ws://0.0.0.0:8080/?topic=/camera0/debug_frame"
        self.webrtc_name = webrtc_name
        self.webrtc = None
        self.connection_timeout = 3.0  # seconds
        self.pipeline_started = False  # Track if pipeline has been started
        
        self.loop = None
        self.main_task = None
        self._shutdown_event = None
        self.tasks = []
        self.read_task = None

    @property
    def connected(self):
        if self.conn is None:
            return False
        try:
            #See if still connected
            return self.conn.transport is not None and not self.conn.transport.is_closing()
        except:
            #Assume disconnected
            return False

    def start(self):

        import threading
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()
        
    def _run_event_loop(self):
        #Run the asyncio event loop in a dedicated thread

        self.logger.info("WebRTC stream is starting...")
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self._shutdown_event = asyncio.Event()
        
        try:
            self.loop.run_until_complete(self.main())
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Error in event loop: {e}", exc_info=True)
        finally:
            #Cancel all pending tasks
            pending = asyncio.all_tasks(self.loop)
            for task in pending:
                task.cancel()
            if pending:
                self.loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            self.loop.close()
            self.logger.info("WebRTC stream has exited")

    async def main(self):
        #Async entry point

        try:
            #Create task for connection loop only
            #Message reading will be handled within the connection
            connect_task = asyncio.create_task(self.connect_loop())
            self.tasks = [connect_task]
            
            #Wait until shutdown is requested
            await self._shutdown_event.wait()
            
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}", exc_info=True)
        finally:
            #Cancel all tasks
            for task in self.tasks:
                if not task.done():
                    task.cancel()
            if self.read_task and not self.read_task.done():
                self.read_task.cancel()
            await asyncio.gather(*self.tasks, return_exceptions=True)
            if self.read_task:
                await asyncio.gather(self.read_task, return_exceptions=True)

    async def connect_loop(self):
        #Maintain connection with signalling server

        reconnect_delay = 5  # seconds between reconnection attempts
        
        while not self._shutdown_event.is_set():
            try:
                if not self.connected:
                    self.logger.info("Attempting connection to signalling server")
                    await asyncio.wait_for(self.connect(), timeout=self.connection_timeout)
                    
                    #Wait for the read task to complete (connection closed)
                    if self.read_task:
                        try:
                            await self.read_task
                        except asyncio.CancelledError:
                            break
                        except Exception as e:
                            self.logger.error(f"Read task error: {e}")
                    
                    #Connection closed, wait before reconnecting
                    self.logger.info(f"Waiting {reconnect_delay}s before reconnecting...")
                    await asyncio.sleep(reconnect_delay)
                else:
                    #Connection is active, just wait
                    await asyncio.sleep(1)
                    
            except asyncio.TimeoutError:
                self.logger.warning("Connection timeout")
                self.conn = None
                await asyncio.sleep(reconnect_delay)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.warning(f"connect_loop error: {repr(e)}")
                self.conn = None
                await asyncio.sleep(reconnect_delay)

    async def connect(self):
        try:
            self.conn = await websockets.connect(self.server)
            #Initialize webrtc element & signals
            if not self.pipeline_started:
                self.start_pipeline()
            #Start reading
            self.read_task = asyncio.create_task(self.read_messages())
        except Exception as e:
            self.logger.error(f"Failed to connect: {e}")
            raise

    async def read_messages(self):
        #Read and handle messages from the WebSocket connection

        try:
            self.logger.info("Starting to read messages from WebSocket")
            async for message in self.conn:
                try:
                    await self.handle_message(message)
                except Exception as e:
                    self.logger.error(f"Error handling message '{message[:100]}...': {e}", exc_info=True)
        except websockets.exceptions.ConnectionClosed as e:
            self.logger.info(f"WebSocket connection closed: {e}")
        except asyncio.CancelledError:
            self.logger.info("Message reading cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Error reading messages: {e}", exc_info=True)
        finally:
            old_conn = self.conn
            self.conn = None
            if old_conn is not None:
                is_closed = getattr(old_conn, "closed", True)
                if callable(is_closed):
                    is_closed = is_closed()
                if not is_closed:
                    try:
                        await old_conn.close()
                    except:
                        pass
            self.logger.info("Message reading loop ended")

    async def handle_message(self, message):
        #Handle messages from the signaling server

        self.logger.debug(f"Message: {message}")
        try:
            await self.handle_sdp(message)
        except json.JSONDecodeError:
            self.logger.warning(f"Unknown message format: {message}")

    def send_sdp_answer(self, answer):
        #Send SDP answer to signaling server

        try:
            text = answer.sdp.as_text()
            self.logger.info(f"Sending answer:\n{text[:200]}...")
            msg = json.dumps({"answer": {"type": "answer", "sdp": text}})
            
            if not self.connected:
                self.logger.error("Cannot send answer: not connected")
                return
                
            if self.loop and self.loop.is_running():
                async def _send():
                    try:
                        await self.conn.send(msg)
                        self.logger.info("Answer sent successfully")
                    except Exception as e:
                        self.logger.error(f"Failed to send answer: {e}")
                
                asyncio.run_coroutine_threadsafe(_send(), self.loop)
        except Exception as e:
            self.logger.error(f"Error in send_sdp_answer: {e}", exc_info=True)

    def send_sdp_offer(self, offer):
        #Send SDP offer to signaling server

        try:
            text = offer.sdp.as_text()
            self.logger.info(f"Sending offer:\n{text[:200]}...")
            msg = json.dumps({"sdp": {"type": "offer", "sdp": text}})
            
            #Check connection before sending
            if not self.connected:
                self.logger.error("Cannot send SDP: not connected to signaling server")
                return
                
            if self.loop and self.loop.is_running():
                #Send the message
                async def _send():
                    try:
                        await self.conn.send(msg)
                        self.logger.info("SDP offer sent successfully")
                    except Exception as e:
                        self.logger.error(f"Failed to send SDP offer: {e}")
                
                asyncio.run_coroutine_threadsafe(_send(), self.loop)
            else:
                self.logger.error(f"Cannot send SDP: event loop not running")
        except Exception as e:
            self.logger.error(f"Error in send_sdp_offer: {e}", exc_info=True)

    def on_offer_created(self, promise, _, __):
        #Handle offer creation

        try:
            promise.wait()
            reply = promise.get_reply()
            offer = reply.get_value("offer")
            
            #Set local description
            promise = Gst.Promise.new()
            self.webrtc.emit("set-local-description", offer, promise)
            promise.interrupt()
            
            #Send the offer
            self.send_sdp_offer(offer)
            
        except Exception as e:
            self.logger.error(f"Error in on_offer_created: {e}", exc_info=True)

    def on_negotiation_needed(self, element):
        #On Handle negotiation needed signal

        try:
            self.logger.info("Negotiation needed, creating offer...")
            promise = Gst.Promise.new_with_change_func(
                self.on_offer_created, element, None
            )
            element.emit("create-offer", None, promise)
        except Exception as e:
            self.logger.error(f"Error in on_negotiation_needed: {e}", exc_info=True)

    def send_ice_candidate_message(self, _, mlineindex, candidate):
        try:
            icemsg = json.dumps({"candidate": {
                "candidate": candidate,
                "sdpMLineIndex": mlineindex
            }})
            if self.loop and self.connected:
                asyncio.run_coroutine_threadsafe(self.conn.send(icemsg), self.loop)
        except Exception as e:
            self.logger.error(f"Error sending ICE candidate: {e}")

    def on_incoming_decodebin_stream(self, _, pad):
        #Handle incoming decodebin stream

        if not pad.has_current_caps():
            self.logger.info(f"{pad} has no caps, ignoring")
            return

        caps = pad.get_current_caps()
        assert len(caps)
        s = caps[0]
        name = s.get_name()
        
        if name.startswith("video"):
            q = Gst.ElementFactory.make("queue")
            conv = Gst.ElementFactory.make("videoconvert")
            sink = Gst.ElementFactory.make("autovideosink")
            self.pipeline.add(q, conv, sink)
            self.pipeline.sync_children_states()
            pad.link(q.get_static_pad("sink"))
            q.link(conv)
            conv.link(sink)
        elif name.startswith("audio"):
            q = Gst.ElementFactory.make("queue")
            conv = Gst.ElementFactory.make("audioconvert")
            resample = Gst.ElementFactory.make("audioresample")
            sink = Gst.ElementFactory.make("autoaudiosink")
            self.pipeline.add(q, conv, resample, sink)
            self.pipeline.sync_children_states()
            pad.link(q.get_static_pad("sink"))
            q.link(conv)
            conv.link(resample)
            resample.link(sink)

    def on_incoming_stream(self, _, pad):
        #Handle incoming stream

        if pad.direction != Gst.PadDirection.SRC:
            return

        decodebin = Gst.ElementFactory.make("decodebin")
        decodebin.connect("pad-added", self.on_incoming_decodebin_stream)
        self.pipeline.add(decodebin)
        decodebin.sync_state_with_parent()
        self.webrtc.link(decodebin)


    def start_pipeline(self):
        #Start the GStreamer pipeline

        try:
            if self.pipeline_started:
                self.logger.info("Pipeline already started")
                return
                
            self.webrtc = self.pipeline.get_by_name(self.webrtc_name)
            if not self.webrtc:
                self.logger.error(f"Could not find webrtcbin element with name '{self.webrtc_name}'")
                return
                
            self.webrtc.connect("on-negotiation-needed", self.on_negotiation_needed)
            self.webrtc.connect("on-ice-candidate", self.send_ice_candidate_message)
            self.webrtc.connect("pad-added", self.on_incoming_stream)
            
            #Pipeline should already be in PLAYING state from gstreamer_node.py
            current_state = self.pipeline.get_state(0)[1]
            self.logger.info(f"Pipeline state: {current_state.value_nick}")
            
            self.pipeline_started = True
            self.logger.info("WebRTC pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error starting pipeline: {e}", exc_info=True)

    async def handle_sdp(self, message):
        #Handle SDP messages

        try:
            msg = json.loads(message)
            
            if "sdp" in msg:
                sdp = msg["sdp"]
                sdp_type = sdp.get("type", "")
                sdp_text = sdp.get("sdp", "")
                
                self.logger.info(f"Received {sdp_type}:\n{sdp_text[:200]}...")
                
                if sdp_type == "answer":
                    if not self.webrtc:
                        self.logger.error("WebRTC element not initialized, cannot set remote description")
                        return
                        
                    res, sdpmsg = GstSdp.SDPMessage.new()
                    GstSdp.sdp_message_parse_buffer(bytes(sdp_text.encode()), sdpmsg)
                    answer = GstWebRTC.WebRTCSessionDescription.new(
                        GstWebRTC.WebRTCSDPType.ANSWER, sdpmsg
                    )
                    promise = Gst.Promise.new()
                    self.webrtc.emit("set-remote-description", answer, promise)
                    promise.interrupt()
                    self.logger.info("Remote description set successfully")
                    
            elif "ice" in msg:
                ice = msg["ice"]
                candidate = ice["candidate"]
                sdpmlineindex = ice["sdpMLineIndex"]
                
                if self.webrtc:
                    self.webrtc.emit("add-ice-candidate", sdpmlineindex, candidate)
                    self.logger.debug(f"Added ICE candidate: {candidate[:50]}...")
                else:
                    self.logger.warning("WebRTC element not ready, cannot add ICE candidate")
                    
        except Exception as e:
            self.logger.error(f"Error handling SDP: {e}", exc_info=True)

    def shutdown(self):
        #Shutdown the WebRTC handler

        if self.loop and self.loop.is_running():
            #Schedule shutdown in the event loop
            asyncio.run_coroutine_threadsafe(
                self._async_shutdown(), self.loop
            )

    async def _async_shutdown(self):
        #Async shutdown implementation
        
        self.logger.info("Shutting down WebRTC handler...")
        
        #Close WebSocket connection
        if self.conn:
            try:
                await self.conn.close()
            except Exception as e:
                self.logger.warning(f"Error closing WebSocket connection: {e}")
        
        #Signal shutdown
        self._shutdown_event.set()

    def join(self, timeout=None):
        #Wait for the handler thread to complete

        if hasattr(self, '_thread') and self._thread.is_alive():
            self._thread.join(timeout=timeout)