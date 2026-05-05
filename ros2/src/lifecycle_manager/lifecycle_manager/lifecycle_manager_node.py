import asyncio
import os
import signal
import subprocess

from std_srvs.srv import Trigger

import aioros2


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


def main():
    aioros2.run()


if __name__ == "__main__":
    main()
