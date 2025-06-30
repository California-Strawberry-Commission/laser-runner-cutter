import asyncio
import fractions
import json
import logging
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

import aioros2


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

        loop = asyncio.get_running_loop()

        def sub_callback(msg):
            # This callback is called from another thread, so we need to use run_coroutine_threadsafe
            asyncio.run_coroutine_threadsafe(self._frame_callback(msg), loop)

        # TODO: support other message types
        self._subscription = node.create_subscription(
            Image, topic, sub_callback, qos_profile_sensor_data
        )

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


webrtc_params = aioros2.params(WebRTCParams)


class SharedState:
    logger: Optional[logging.Logger] = None
    cv_bridge = CvBridge()
    pcs: Dict[uuid.UUID, RTCPeerConnection] = dict()
    tracks: Dict[str, RosVideoStreamTrack] = dict()  # topic -> track
    topic_connection_map: DefaultDict[str, Set[uuid.UUID]] = defaultdict(
        set
    )  # topic -> [peer connection ID]


shared_state = SharedState()


@aioros2.start
async def start(node):
    shared_state.logger = node.get_logger()

    async def server_task(host: str = "0.0.0.0", port: int = 8080):
        try:
            runner = create_server_runner()
            await runner.setup()
            site = web.TCPSite(runner, host=host, port=port)
            await site.start()

            shared_state.logger.info(f"Serving on {host}:{port}")

            # Keep the server running indefinitely
            while True:
                await asyncio.sleep(3600)
        finally:
            await runner.cleanup()

    def create_server_runner():

        async def websocket_handler(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            params = request.query
            topic = params.get("topic", "")

            pc = RTCPeerConnection()
            pc_id = uuid.uuid4()
            shared_state.pcs[pc_id] = pc
            shared_state.logger.info(
                f"PeerConnection({pc_id}) created for {request.remote}. {len(shared_state.pcs)} clients total."
            )

            @pc.on("datachannel")
            def on_datachannel(channel):
                @channel.on("message")
                def on_message(message):
                    if isinstance(message, str) and message.startswith("ping"):
                        channel.send("pong" + message[4:])

            @pc.on("connectionstatechange")
            async def on_connectionstatechange():
                shared_state.logger.info(
                    f"PeerConnection({pc_id}) connection state is <{pc.connectionState}>"
                )
                if pc.connectionState == "failed":
                    await close_pc(pc_id)

            pc.addTrack(get_track(topic, pc_id))

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
                        shared_state.logger.warning(
                            f"WebSocket connection closed with exception {ws.exception()}"
                        )
            except asyncio.CancelledError:
                shared_state.logger.info(
                    f"PeerConnection({pc_id}): WebSocket connection was canceled"
                )
            finally:
                shared_state.logger.info(
                    f"PeerConnection({pc_id}): WebSocket connection closed"
                )
                await ws.close()
                await close_pc(pc_id)

            return ws

        async def on_shutdown(app):
            # Close all peer connections
            coros = [pc.close() for pc in shared_state.pcs.values()]
            await asyncio.gather(*coros)
            shared_state.pcs.clear()
            shared_state.tracks.clear()
            shared_state.topic_connection_map.clear()

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

    def get_track(topic: str, pc_id: uuid.UUID):
        shared_state.topic_connection_map[topic].add(pc_id)
        shared_state.logger.info(
            f"PeerConnection({pc_id}): Getting track for topic {topic}"
        )
        if topic in shared_state.tracks:
            shared_state.logger.info(
                f"PeerConnection({pc_id}): Using cached track for topic {topic}"
            )
            return shared_state.tracks[topic]
        else:
            shared_state.logger.info(
                f"PeerConnection({pc_id}): Creating new track for topic {topic}"
            )
            track = RosVideoStreamTrack(
                topic, node, shared_state.cv_bridge, shared_state.logger
            )
            shared_state.tracks[topic] = track
            return track

    async def close_pc(pc_id: uuid.UUID):
        pc = shared_state.pcs.pop(pc_id, None)
        if pc is None:
            return

        await pc.close()
        shared_state.logger.info(
            f"PeerConnection({pc_id}): Closed connection. {len(shared_state.pcs)} clients total."
        )

        # Add a delay before cleaning up tracks, as destroying the subscription and resubscribing to the
        # same topic in quick succession results in an exception, likely related to threading.
        await asyncio.sleep(2.0)

        # Update topic_connection_map and tracks dicts
        topic = find_topic_by_pc_id(pc_id)
        if topic is not None:
            shared_state.topic_connection_map[topic].discard(pc_id)
            # If there are no more connections for this topic, destroy the track
            if len(shared_state.topic_connection_map[topic]) == 0:
                track = shared_state.tracks.pop(topic, None)
                if track is not None:
                    track.destroy()

    def find_topic_by_pc_id(pc_id: uuid.UUID) -> Optional[str]:
        for topic, pc_ids in shared_state.topic_connection_map.items():
            if pc_id in pc_ids:
                return topic
        return None

    # Start WebSocket server
    asyncio.create_task(server_task(host=webrtc_params.host, port=webrtc_params.port))


def main():
    aioros2.run()


if __name__ == "__main__":
    main()
