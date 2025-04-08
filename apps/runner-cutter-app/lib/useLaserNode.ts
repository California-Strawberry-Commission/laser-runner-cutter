import useROSNode from "@/lib/ros/useROSNode";
import { useCallback } from "react";

export type State = {
  deviceState: DeviceState;
};

export enum DeviceState {
  DISCONNECTED,
  CONNECTING,
  STOPPED,
  PLAYING,
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

  const path = node.usePublisher(
    "~/path",
    "laser_control_interfaces/Path",
    useCallback(
      (x: number, y: number) => ({
        point: {
          x: x,
          y: y,
        },
      }),
      []
    )
  );

  const setPoint = node.usePublisher(
    "~/path",
    "laser_control_interfaces/Path",
    useCallback(
      (x: number, y: number) => ({
        end: {
          x: x,
          y: y,
        },
        laser_on: true,
      }),
      []
    )
  );

  const clearPoints = node.usePublisher(
    "~/path",
    "laser_control_interfaces/Path",
    useCallback(
      () => ({
        laser_on: false,
      }),
      []
    )
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
    setPoint,
    clearPoints,
    play,
    stop,
    setColor,
  };
}
