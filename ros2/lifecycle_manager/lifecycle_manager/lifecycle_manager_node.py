import subprocess

from std_srvs.srv import Trigger

import aioros2


@aioros2.service("~/restart_service", Trigger)
async def restart_service(node):
    _trigger_restart_service(node)
    return {"success": True}


@aioros2.service("~/reboot_system", Trigger)
async def reboot_system(node):
    _trigger_reboot_system(node)
    return {"success": True}


def _trigger_restart_service(node):
    node.get_logger().info("Restarting service...")
    try:
        subprocess.run(
            ["sudo", "systemctl", "restart", "laser-runner-cutter-ros.service"],
            check=True,
        )
        node.get_logger().info("Service restart initiated successfully.")
    except subprocess.CalledProcessError as e:
        node.get_logger().error(f"Failed to restart service: {e}")


def _trigger_reboot_system(node):
    node.get_logger().info("Rebooting system...")
    try:
        subprocess.run(["sudo", "/sbin/reboot"], check=True)
    except subprocess.CalledProcessError as e:
        node.get_logger().error(f"Failed to reboot: {e}")


def main():
    aioros2.run()


if __name__ == "__main__":
    main()
