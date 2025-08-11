
import asyncio
import json
import logging
import ssl
import threading
import websockets

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
gi.require_version('GstWebRTC', '1.0')
from gi.repository import GstWebRTC
gi.require_version('GstSdp', '1.0')
from gi.repository import GstSdp
gi.require_version('GLib', '2.0')
from gi.repository import GLib

class MavWebRTC(threading.Thread):
    def __init__(self, pipeline, our_id, config):
        threading.Thread.__init__(self)
        self.daemon = True
        self.pipeline = pipeline
        self.logger = logging.getLogger('GSTREAMER_WEBRTC' + __name__)
        self.config = config
        self._should_shutdown = threading.Event()
        self.conn = None
        self.peer_id = None
        self.our_id = our_id
        self.server = 'ws://localhost:8000'
        self.webrtc = self.pipeline.get_by_name('webrtc')
        self.connection_timeout = 3.0 # seconds
    
    @property
    def connected(self):
        if self.conn:
            return True
        return False

    def run(self):
        self.logger.info("Webrtc stream is starting...")
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.main())
        self.loop.close()
        self.logger.info("Webrtc stream has exited")

    def shutdown(self):
            self._should_shutdown.set()

            for task in self.tasks:
                if not task.done():
                    task.cancel()

            if self.loop and not self.loop.is_closed():
                self.loop.stop()

            if self.conn:
                try:
                    asyncio.run(self.conn.close())
                except Exception as e:
                    self.logger.warning("Error closing WebSocket connection: {}".format(e))
    
    async def main(self):
        self.tasks = []
        connect_loop_task = asyncio.create_task(self.connect_loop_tasks())
        processing_loop_task = asyncio.create_task(self.processing_loop_tasks())
        self.tasks.append(connect_loop_task)
        self.tasks.append(processing_loop_task)
        await asyncio.gather(*self.tasks, return_exceptions=True)

    async def connect_loop_tasks(self):
        while not self._should_shutdown.is_set():
            await asyncio.sleep(1)
            await self.connect_loop()

    async def connect_loop(self):
        if not self.connected:
            try:
                self.logger.info("Starting peer connection with signalling server")
                await asyncio.wait_for(self.connect(), timeout=self.connection_timeout)
            except Exception as e:
                self.logger.warning("connect_loop error: {}".format(repr(e)))
                self.conn = None

    async def processing_loop_tasks(self):
        while not self._should_shutdown.is_set():
            await asyncio.sleep(2) 
            await self.processing_loop()
        
    async def connect(self):
        sslctx = ssl.create_default_context(purpose=ssl.Purpose.CLIENT_AUTH)
        self.conn = await websockets.connect(self.server)
        await self.conn.send('HELLO %d' % self.our_id)
        self.logger.info("WebRTC: registered with signalling server, peer id {}".format(self.our_id))

    async def setup_call(self):
        await self.conn.send('SESSION {}'.format(self.peer_id))

    def send_sdp_offer(self, offer):
        text = offer.sdp.as_text()
        self.logger.info('Sending offer:\n%s' % text)
        msg = json.dumps({'sdp': {'type': 'offer', 'sdp': text}})
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.conn.send(msg))
        
    def on_offer_created(self, promise, _, __):
        promise.wait()
        reply = promise.get_reply()
        offer = reply.get_value('offer')
        promise = Gst.Promise.new()
        self.webrtc.emit('set-local-description', offer, promise)
        promise.interrupt()
        self.send_sdp_offer(offer)

    def on_negotiation_needed(self, element):
        promise = Gst.Promise.new_with_change_func(self.on_offer_created, element, None)
        element.emit('create-offer', None, promise)

    def send_ice_candidate_message(self, _, mlineindex, candidate):
        icemsg = json.dumps({'ice': {'candidate': candidate, 'sdpMLineIndex': mlineindex}})
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.conn.send(icemsg))

    def on_incoming_decodebin_stream(self, _, pad):
        if not pad.has_current_caps():
            self.logger.info(pad, 'has no caps, ignoring')
            return

        caps = pad.get_current_caps()
        assert (len(caps))
        s = caps[0]
        name = s.get_name()
        if name.startswith('video'):
            q = Gst.ElementFactory.make('queue')
            conv = Gst.ElementFactory.make('videoconvert')
            sink = Gst.ElementFactory.make('autovideosink')
            self.pipe.add(q, conv, sink)
            self.pipe.sync_children_states()
            pad.link(q.get_static_pad('sink'))
            q.link(conv)
            conv.link(sink)
        elif name.startswith('audio'):
            q = Gst.ElementFactory.make('queue')
            conv = Gst.ElementFactory.make('audioconvert')
            resample = Gst.ElementFactory.make('audioresample')
            sink = Gst.ElementFactory.make('autoaudiosink')
            self.pipe.add(q, conv, resample, sink)
            self.pipe.sync_children_states()
            pad.link(q.get_static_pad('sink'))
            q.link(conv)
            conv.link(resample)
            resample.link(sink)

    def on_incoming_stream(self, _, pad):
        if pad.direction != Gst.PadDirection.SRC:
            return

        decodebin = Gst.ElementFactory.make('decodebin')
        decodebin.connect('pad-added', self.on_incoming_decodebin_stream)
        self.pipe.add(decodebin)
        decodebin.sync_state_with_parent()
        self.webrtc.link(decodebin)

    def start_pipeline(self):

        self.webrtc = self.pipeline.get_by_name('webrtc')
        self.webrtc.connect('on-negotiation-needed', self.on_negotiation_needed)
        self.webrtc.connect('on-ice-candidate', self.send_ice_candidate_message)
        self.webrtc.connect('pad-added', self.on_incoming_stream)
        self.logger.info("Setting WebRTC pipeline to active")
        self.pipeline.set_state(Gst.State.PLAYING)

    async def handle_sdp(self, message):
        assert (self.webrtc)
        msg = json.loads(message)
        if 'sdp' in msg:
            sdp = msg['sdp']
            assert(sdp['type'] == 'answer')
            sdp = sdp['sdp']
            self.logger.info('Received answer:\n%s' % sdp)
            res, sdpmsg = GstSdp.SDPMessage.new()
            GstSdp.sdp_message_parse_buffer(bytes(sdp.encode()), sdpmsg)
            answer = GstWebRTC.WebRTCSessionDescription.new(GstWebRTC.WebRTCSDPType.ANSWER, sdpmsg)
            promise = Gst.Promise.new()
            self.webrtc.emit('set-remote-description', answer, promise)
            promise.interrupt()
        elif 'ice' in msg:
            ice = msg['ice']
            candidate = ice['candidate']
            sdpmlineindex = ice['sdpMLineIndex']
            self.webrtc.emit('add-ice-candidate', sdpmlineindex, candidate)

    async def processing_loop(self):
        if self.connected and not self._should_shutdown.is_set():

            async for message in self.conn:
                self.logger.debug("Message: {}".format(message))
                if message.startswith('HELLO'):
                    self.logger.info("Received registration response from signalling server: {}".format(message))
                    self.start_pipeline()
                    #await self.setup_call()
                elif message == 'SESSION_OK':
                    self.logger.info("Received SESSION_OK, starting pipeline")
                    self.start_pipeline()
                elif message == 'SEND_SDP':
                    self.logger.info('Received SEND_SDP, starting pipeline')
                    self.start_pipeline()
                    self.on_negotiation_needed(self.webrtc)
                elif message.startswith('ERROR'):
                    self.logger.warning(message)
                    return 1
                else:
                    await self.handle_sdp(message)
            return 0
        else:
            return 1
