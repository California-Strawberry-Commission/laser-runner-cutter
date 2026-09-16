import asyncio
import os
import signal
import subprocess

from std_srvs.srv import Trigger

import aioros2

NODES = {
    "laser": "__node:=laser0",
    "control": "__node:=control0",
    "camera_detection": "__node:=camera_detection_container",
    "livekit": "__node:=livekit_whip_node",
    "rosbridge": "__node:=rosbridge_websocket",
}


@aioros2.service("~/restart_service", Trigger)
async def restart_service(node):
    node.get_logger().info("Restarting service...")
    # Delay so the service response is sent before SIGTERM reaches this node.
    # Inside a Docker container, the entrypoint is PID 1, so sending SIGTERM
    # to PID 1 will cause the container to exit. Docker then restarts the container
    # via restart: unless-stopped.
    asyncio.get_event_loop().call_later(1.0, lambda: os.kill(1, signal.SIGTERM))
    return {"success": True}


@aioros2.service("~/reboot_system", Trigger)
async def reboot_system(node):
    node.get_logger().info("Rebooting system...")
    # Delay so the service response is sent before reboot
    asyncio.get_event_loop().call_later(1.0, lambda: _trigger_reboot_system(node))
    return {"success": True}


def _trigger_reboot_system(node):
    try:
        subprocess.run(["reboot", "-f"], check=True)
    except subprocess.CalledProcessError as e:
        node.get_logger().error(f"Failed to reboot: {e}")


@aioros2.service("~/restart_laser", Trigger)
async def restart_laser(node):
    return _restart(node, "laser")


@aioros2.service("~/restart_control", Trigger)
async def restart_control(node):
    return _restart(node, "control")


@aioros2.service("~/restart_camera_detection", Trigger)
async def restart_camera_detection(node):
    return _restart(node, "camera_detection")


@aioros2.service("~/restart_livekit", Trigger)
async def restart_livekit(node):
    return _restart(node, "livekit")


@aioros2.service("~/restart_rosbridge", Trigger)
async def restart_rosbridge(node):
    return _restart(node, "rosbridge")


def _restart(node, name):
    match = NODES[name]
    result = subprocess.run(["pgrep", "-f", match], capture_output=True, text=True)
    pid = result.stdout.split()

    if not pid:
        message = f"No running process found for '{name}'"
        node.get_logger().error(message)
        return {"success": False, "message": message}

    pid = int(pid[0])
    node.get_logger().info(f"Restarting '{name}' (PID: {pid})...")
    # Delay so the service response is sent first.
    loop = asyncio.get_event_loop()
    loop.call_later(1.0, lambda: _kill(pid, signal.SIGTERM))
    # 5 second grace period since SIGTERM fires at 1.0
    loop.call_later(6.0, lambda: _kill(pid, signal.SIGKILL))

    return {"success": True, "message": f"Restarting '{name}'"}


def _kill(pid, sig):
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass


def main():
    aioros2.run()


if __name__ == "__main__":
    main()
