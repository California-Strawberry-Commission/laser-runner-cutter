import useROSNode from "@/lib/ros/useROSNode";

function triggerInputMapper() {
  return {};
}

function successOutputMapper(res: any): boolean {
  return res.success;
}

export default function useLifecycleManagerNode(nodeName: string) {
  const node = useROSNode(nodeName);

  const reboot = node.useService(
    "~/reboot",
    "std_srvs/Trigger",
    triggerInputMapper,
    successOutputMapper
  );

  return {
    ...node,
    reboot,
  };
}
