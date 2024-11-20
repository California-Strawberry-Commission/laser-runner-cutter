import asyncio
import fractions
import json
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, Optional, Set

import aiohttp_cors
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from av import VideoFrame
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from aioros2 import node, params, serve_nodes, start


class RosVideoStreamTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, topic, node, cv_bridge, logger):
        super().__init__()
        self._topic = topic
        self._node = node
        self._cv_bridge = cv_bridge
        self._logger = logger
        self._current_frame = None
        self._current_timestamp_millis = 0

        # TODO: support other message types
        self._subscription = node.create_subscription(
            Image, topic, self._sub_callback, qos_profile_sensor_data
        )

    def _sub_callback(self, msg):
        self._node.run_coroutine(self._frame_callback, msg)

    async def _frame_callback(self, msg):
        self._current_frame = self._cv_bridge.imgmsg_to_cv2(
            msg, desired_encoding="rgb8"
        )
        # ROS timestamps consist of two integers, one for seconds and one for nanoseconds
        self._current_timestamp_millis = (
            msg.header.stamp.sec * 1000 + msg.header.stamp.nanosec / 1e6
        )

    async def recv(self):
        frame = await self._wait_for_frame()

        # Create video frame
        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = int(self._current_timestamp_millis)
        video_frame.time_base = fractions.Fraction(1, 1000)

        return video_frame

    async def _wait_for_frame(self):
        while self._current_frame is None:
            await asyncio.sleep(0.1)

        return self._current_frame

    def destroy(self):
        self._logger.info(f"Destroying track for topic {self._topic}")
        self.stop()
        if self._subscription is not None:
            self._node.destroy_subscription(self._subscription)
            self._subscription = None


@dataclass
class WebRTCParams:
    host: str = "0.0.0.0"
    port: int = 8080


@node("webrtc_node")
class WebRTCNode:

    webrtc_params = params(WebRTCParams)

    @start
    async def start(self):
        self.cv_bridge = CvBridge()  # for converting image msg to numpy array
        self.pcs: Dict[uuid.UUID, RTCPeerConnection] = dict()
        self.tracks: Dict[str, RosVideoStreamTrack] = dict()  # topic -> track
        self.topic_connection_map: DefaultDict[str, Set[uuid.UUID]] = defaultdict(
            set
        )  # topic -> [peer connection ID]

        # Start WebSocket server
        asyncio.create_task(
            self.server_task(host=self.webrtc_params.host, port=self.webrtc_params.port)
        )

    async def server_task(self, host: str = "0.0.0.0", port: int = 8080):
        try:
            runner = self.create_server_runner()
            await runner.setup()
            site = web.TCPSite(runner, host=host, port=port)
            await site.start()

            self.log(f"Serving on {host}:{port}")

            # Keep the server running indefinitely
            while True:
                await asyncio.sleep(3600)
        finally:
            await runner.cleanup()

    def create_server_runner(self):

        async def websocket_handler(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            params = request.query
            topic = params.get("topic", "")

            pc = RTCPeerConnection()
            pc_id = uuid.uuid4()
            self.pcs[pc_id] = pc
            self.log(
                f"PeerConnection({pc_id}) created for {request.remote}. {len(self.pcs)} clients total."
            )

            @pc.on("datachannel")
            def on_datachannel(channel):
                @channel.on("message")
                def on_message(message):
                    if isinstance(message, str) and message.startswith("ping"):
                        channel.send("pong" + message[4:])

            @pc.on("connectionstatechange")
            async def on_connectionstatechange():
                self.log(
                    f"PeerConnection({pc_id}) connection state is <{pc.connectionState}>"
                )
                if pc.connectionState == "failed":
                    await self.close_pc(pc_id)

            pc.addTrack(self.get_track(topic, pc_id))

            try:
                async for msg in ws:
                    if msg.type == web.WSMsgType.TEXT:
                        data = json.loads(msg.data)

                        if "offer" in data:
                            offer = RTCSessionDescription(
                                sdp=data["offer"]["sdp"], type=data["offer"]["type"]
                            )

                            # Handle offer and generate answer
                            await pc.setRemoteDescription(offer)
                            answer = await pc.createAnswer()
                            await pc.setLocalDescription(answer)
                            await ws.send_json(
                                {
                                    "answer": {
                                        "sdp": pc.localDescription.sdp,
                                        "type": pc.localDescription.type,
                                    }
                                }
                            )
                    elif msg.type == web.WSMsgType.ERROR:
                        self.log_warn(
                            f"WebSocket connection closed with exception {ws.exception()}"
                        )
            except asyncio.CancelledError:
                self.log(f"PeerConnection({pc_id}): WebSocket connection was canceled")
            finally:
                self.log(f"PeerConnection({pc_id}): WebSocket connection closed")
                await ws.close()
                await self.close_pc(pc_id)

            return ws

        async def on_shutdown(app):
            # Close all peer connections
            coros = [pc.close() for pc in self.pcs.values()]
            await asyncio.gather(*coros)
            self.pcs.clear()
            self.tracks.clear()
            self.topic_connection_map.clear()

        app = web.Application()
        app.on_shutdown.append(on_shutdown)
        app.add_routes([web.get("/", websocket_handler)])

        # Set up CORS
        cors = aiohttp_cors.setup(
            app,
            defaults={
                "*": aiohttp_cors.ResourceOptions(
                    allow_credentials=True,
                    expose_headers="*",
                    allow_headers="*",
                )
            },
        )
        for route in list(app.router.routes()):
            cors.add(route)

        runner = web.AppRunner(app)
        return runner

    def get_track(self, topic: str, pc_id: uuid.UUID):
        self.topic_connection_map[topic].add(pc_id)
        self.log(f"PeerConnection({pc_id}): Getting track for topic {topic}")
        if topic in self.tracks:
            self.log(f"PeerConnection({pc_id}): Using cached track for topic {topic}")
            return self.tracks[topic]
        else:
            self.log(f"PeerConnection({pc_id}): Creating new track for topic {topic}")
            track = RosVideoStreamTrack(topic, self, self.cv_bridge, self.get_logger())
            self.tracks[topic] = track
            return track

    async def close_pc(self, pc_id: uuid.UUID):
        pc = self.pcs.pop(pc_id, None)
        if pc is None:
            return

        await pc.close()
        self.log(
            f"PeerConnection({pc_id}): Closed connection. {len(self.pcs)} clients total."
        )

        # Add a delay before cleaning up tracks, as destroying the subscription and resubscribing to the
        # same topic in quick succession results in an exception, likely related to threading.
        await asyncio.sleep(2.0)

        # Update topic_connection_map and tracks dicts
        topic = self.find_topic_by_pc_id(pc_id)
        if topic is not None:
            self.topic_connection_map[topic].discard(pc_id)
            # If there are no more connections for this topic, destroy the track
            if len(self.topic_connection_map[topic]) == 0:
                track = self.tracks.pop(topic, None)
                if track is not None:
                    track.destroy()

    def find_topic_by_pc_id(self, pc_id: uuid.UUID) -> Optional[str]:
        for topic, pc_ids in self.topic_connection_map.items():
            if pc_id in pc_ids:
                return topic
        return None


def main():
    serve_nodes(WebRTCNode())


if __name__ == "__main__":
    main()
