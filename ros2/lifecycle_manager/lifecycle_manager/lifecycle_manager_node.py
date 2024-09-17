import subprocess

from std_srvs.srv import Trigger

from aioros2 import node, result, serve_nodes, service


@node("lifecycle_manager_node")
class LifecycleManagerNode:

    @service("~/restart_service", Trigger)
    async def restart_service(self):
        self._trigger_restart_service()
        return result(success=True)

    @service("~/reboot_system", Trigger)
    async def reboot_system(self):
        self._trigger_reboot_system()
        return result(success=True)

    def _trigger_restart_service(self):
        self.log("Restarting service...")
        try:
            subprocess.run(
                ["sudo", "systemctl", "restart", "laser-runner-cutter-ros.service"],
                check=True,
            )
            self.log("Service restart initiated successfully.")
        except subprocess.CalledProcessError as e:
            self.log_error(f"Failed to restart service: {e}")

    def _trigger_reboot_system(self):
        self.log("Rebooting system...")
        try:
            subprocess.run(["sudo", "/sbin/reboot"], check=True)
        except subprocess.CalledProcessError as e:
            self.log_error(f"Failed to reboot: {e}")


def main():
    serve_nodes(LifecycleManagerNode())


if __name__ == "__main__":
    main()
