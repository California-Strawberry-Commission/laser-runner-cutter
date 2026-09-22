import useROSNode from "@/lib/ros/useROSNode";
import { useCallback } from "react";

function triggerInputMapper() {
  return {};
}

function successOutputMapper(res: any): boolean {
  return res.success;
}

export default function useLifecycleManagerNode(nodeName: string) {
  const node = useROSNode(nodeName);

  const restartService = node.useService(
    "~/restart_service",
    "std_srvs/Trigger",
    triggerInputMapper,
    successOutputMapper,
  );

  const rebootSystem = node.useService(
    "~/reboot_system",
    "std_srvs/Trigger",
    triggerInputMapper,
    successOutputMapper,
  );

  const restartNode = node.useService(
    "~/restart_node",
    "lifecycle_manager_interfaces/RestartNode",
    useCallback((name: string) => ({ node_name: name }), []),
    successOutputMapper,
  );

  return {
    ...node,
    restartService,
    rebootSystem,
    restartNode,
  };
}
