import asyncio
import fractions
import os
import threading
import uuid

import aiohttp_cors
import numpy as np
import pyrealsense2 as rs
import rclpy
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import VIDEO_TIME_BASE
from av import VideoFrame
from rclpy.node import Node

ROOT = os.path.dirname(__file__)

pcs = set()


class RealSenseVideoStreamTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, logger):
        super().__init__()
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(
            rs.stream.color,
            1280,
            720,
            rs.format.rgb8,
            30,
        )
        self.profile = self.pipeline.start(config)
        self.logger = logger

    async def recv(self):
        # TODO: ROS node will be getting the latest frame via subscription
        await asyncio.sleep(0)  # frame timing
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # Convert RealSense frame to OpenCV format
        image = np.asanyarray(color_frame.get_data())

        # Create video frame
        video_frame = VideoFrame.from_ndarray(image, format="rgb24")
        video_frame.pts = int(color_frame.get_timestamp())
        video_frame.time_base = fractions.Fraction(1, 1000)

        return video_frame


def aiohttp_server(logger):

    async def handle_offer(request):
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = RTCPeerConnection()
        pc_id = f"PeerConnection({uuid.uuid4()})"
        pcs.add(pc)
        logger.info(f"{pc_id} Created for {request.remote}")

        @pc.on("datachannel")
        def on_datachannel(channel):
            @channel.on("message")
            def on_message(message):
                if isinstance(message, str) and message.startswith("ping"):
                    channel.send("pong" + message[4:])

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"{pc_id} Connection state is <{pc.connectionState}>")
            if pc.connectionState == "failed":
                await pc.close()
                pcs.discard(pc)

        pc.addTrack(track)

        # Handle offer and generate answer
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.json_response(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        )

    async def on_shutdown():
        # Close peer connections
        coros = [pc.close() for pc in pcs]
        await asyncio.gather(*coros)
        pcs.clear()

    track = RealSenseVideoStreamTrack(logger)

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.add_routes(
        [
            web.post("/offer", handle_offer),
        ]
    )

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


class WebRTCNode(Node):
    def __init__(self):
        super().__init__("webrtc_node")

        # Start the aiohttp server in a separate thread
        thread = threading.Thread(target=self.start_web_server, daemon=True)
        thread.start()

    def start_web_server(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        runner = aiohttp_server(self.get_logger())
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, host="0.0.0.0", port=8080)
        loop.run_until_complete(site.start())
        loop.run_forever()


def main(args=None):
    # Start ROS2 node
    rclpy.init(args=args)
    node = WebRTCNode()

    rclpy.spin(node)

    # Shutdown ROS2 node
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
