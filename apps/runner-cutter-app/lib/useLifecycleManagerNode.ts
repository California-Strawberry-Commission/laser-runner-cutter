import useROSNode from "@/lib/ros/useROSNode";

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

  const restartLaser = node.useService(
    "~/restart_laser",
    "std_srvs/Trigger",
    triggerInputMapper,
    successOutputMapper,
  );

  const restartControl = node.useService(
    "~/restart_control",
    "std_srvs/Trigger",
    triggerInputMapper,
    successOutputMapper,
  );

  const restartCameraDetection = node.useService(
    "~/restart_camera_detection",
    "std_srvs/Trigger",
    triggerInputMapper,
    successOutputMapper,
  );

  const restartLivekit = node.useService(
    "~/restart_livekit",
    "std_srvs/Trigger",
    triggerInputMapper,
    successOutputMapper,
  );

  const restartRosbridge = node.useService(
    "~/restart_rosbridge",
    "std_srvs/Trigger",
    triggerInputMapper,
    successOutputMapper,
  );

  return {
    ...node,
    restartService,
    rebootSystem,
    restartLaser,
    restartControl,
    restartCameraDetection,
    restartLivekit,
    restartRosbridge,
  };
}
