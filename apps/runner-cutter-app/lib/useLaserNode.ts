import useROSNode from "@/lib/ros/useROSNode";
import { useCallback } from "react";

export enum DeviceState {
  Disconnected,
  Connecting,
  Stopped,
  Playing,
}

function convertStateMessage(message: any): DeviceState {
  return message.data as DeviceState;
}

function triggerInputMapper() {
  return {};
}

function successOutputMapper(res: any): boolean {
  return res.success;
}

export default function useLaserNode(nodeName: string) {
  const node = useROSNode(nodeName);
  const state = node.useTopic(
    "~/state",
    "laser_control_interfaces/State",
    DeviceState.Disconnected,
    convertStateMessage
  );

  // TODO: Optimistically set device state to "connecting"
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

  const addPoint = node.useService(
    "~/add_point",
    "laser_control_interfaces/AddPoint",
    useCallback(
      (x: number, y: number) => ({
        point: {
          x: x,
          y: y,
        },
      }),
      []
    ),
    successOutputMapper
  );

  const clearPoints = node.useService(
    "~/clear_points",
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

  const setColor = node.useService(
    "~/set_color",
    "laser_control_interfaces/SetColor",
    useCallback(
      (r: number, g: number, b: number) => ({
        r: r,
        g: g,
        b: b,
        i: 0.0,
      }),
      []
    ),
    successOutputMapper
  );

  return {
    ...node,
    state,
    startDevice,
    closeDevice,
    addPoint,
    clearPoints,
    play,
    stop,
    setColor,
  };
}
