import useROSNode, { ParamType } from "@/lib/ros/useROSNode";
import { useCallback } from "react";

export type State = {
  deviceState: DeviceState;
};

export enum DeviceState {
  DISCONNECTED,
  CONNECTING,
  STOPPED,
  PLAYING,
  DISCONNECTING,
}

function convertStateMessage(message: any): State {
  return {
    deviceState: message.device_state as DeviceState,
  };
}

function triggerInputMapper() {
  return {};
}

function successOutputMapper(res: any): boolean {
  return res.success;
}

export default function useLaserNode(nodeName: string) {
  const node = useROSNode(nodeName);
  const state = node.useSubscription(
    "~/state",
    "laser_control_interfaces/State",
    {
      deviceState: DeviceState.DISCONNECTED,
    },
    convertStateMessage
  );

  const startDevice = node.useService(
    "~/start_device",
    "std_srvs/Trigger",
    triggerInputMapper,
    successOutputMapper
  );

  const closeDevice = node.useService(
    "~/close_device",
    "std_srvs/Trigger",
    triggerInputMapper,
    successOutputMapper
  );

  const updatePath = node.usePublisher(
    "~/update_path",
    "laser_control_interfaces/PathUpdate",
    useCallback(
      (
        pathId: number,
        destination: { x: number; y: number },
        durationMs: number
      ) => ({
        path_id: pathId,
        destination,
        duration_ms: durationMs,
      }),
      []
    )
  );

  const removePath = node.useService(
    "~/remove_path",
    "laser_control_interfaces/RemovePath",
    useCallback(
      (pathId: number) => ({
        path_id: pathId,
      }),
      []
    ),
    successOutputMapper
  );

  const clearPaths = node.useService(
    "~/clear_paths",
    "std_srvs/Trigger",
    triggerInputMapper,
    successOutputMapper
  );

  const play = node.useService(
    "~/play",
    "std_srvs/Trigger",
    triggerInputMapper,
    successOutputMapper
  );

  const stop = node.useService(
    "~/stop",
    "std_srvs/Trigger",
    triggerInputMapper,
    successOutputMapper
  );

  const getColor = node.useGetParam<number[]>("color");
  const setColor = node.useSetParam(
    "color",
    ParamType.DOUBLE_ARRAY,
    useCallback((r: number, g: number, b: number) => [r, g, b, 0.0], [])
  );

  return {
    ...node,
    state,
    startDevice,
    closeDevice,
    updatePath,
    removePath,
    clearPaths,
    play,
    stop,
    getColor,
    setColor,
  };
}
