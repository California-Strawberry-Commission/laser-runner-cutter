import subprocess

from std_srvs.srv import Trigger

from aioros2 import node, result, serve_nodes, service


@node("lifecycle_manager_node")
class LifecycleManagerNode:

    @service("~/reboot", Trigger)
    async def reboot(self):
        self._trigger_reboot()
        return result(success=True)

    def _trigger_reboot(self):
        self.log("Rebooting system...")
        try:
            subprocess.run(["sudo", "/sbin/reboot"], check=True)
        except subprocess.CalledProcessError as e:
            self.log_error(f"Failed to reboot: {e}")


def main():
    serve_nodes(LifecycleManagerNode())


if __name__ == "__main__":
    main()
