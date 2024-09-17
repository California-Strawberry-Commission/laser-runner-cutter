import useROSNode from "@/lib/ros/useROSNode";

function triggerInputMapper() {
  return {};
}

function successOutputMapper(res: any): boolean {
  return res.success;
}

export default function useLifecycleManagerNode(nodeName: string) {
  const node = useROSNode(nodeName);

  const restart_service = node.useService(
    "~/restart_service",
    "std_srvs/Trigger",
    triggerInputMapper,
    successOutputMapper
  );

  const reboot_system = node.useService(
    "~/reboot_system",
    "std_srvs/Trigger",
    triggerInputMapper,
    successOutputMapper
  );

  return {
    ...node,
    restart_service,
    reboot_system,
  };
}
