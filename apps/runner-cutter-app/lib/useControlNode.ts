import useROSNode from "@/lib/ros/useROSNode";
import { useCallback } from "react";

export type State = {
  calibrated: boolean;
  state: string;
  tracks: Track[];
  normalizedLaserBounds: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
};
export type Track = {
  id: number;
  normalizedPixelCoords: { x: number; y: number };
  state: TrackState;
};
export enum TrackState {
  Pending,
  Active,
  Completed,
  Failed,
}

function convertStateMessage(message: any): State {
  return {
    calibrated: message.calibrated,
    state: message.state,
    tracks: message.tracks.map(convertTrackMessage),
    normalizedLaserBounds: {
      x: message.normalized_laser_bounds.w,
      y: message.normalized_laser_bounds.x,
      width: message.normalized_laser_bounds.y,
      height: message.normalized_laser_bounds.z,
    },
  };
}

function convertTrackMessage(message: any): Track {
  return {
    id: message.id,
    normalizedPixelCoords: message.normalized_pixel_coords,
    state: message.state as TrackState,
  };
}

function triggerInputMapper() {
  return {};
}

function successOutputMapper(res: any): boolean {
  return res.success;
}

export default function useControlNode(nodeName: string) {
  const node = useROSNode(nodeName);
  const state = node.useTopic(
    "~/state",
    "runner_cutter_control_interfaces/State",
    {
      calibrated: false,
      state: "idle",
      tracks: [],
      normalizedLaserBounds: { x: 0, y: 0, width: 0, height: 0 },
    },
    convertStateMessage
  );

  const calibrate = node.useService(
    "~/calibrate",
    "std_srvs/Trigger",
    triggerInputMapper,
    successOutputMapper
  );

  const addCalibrationPoint = node.useService(
    "~/add_calibration_points",
    "runner_cutter_control_interfaces/AddCalibrationPoints",
    useCallback(
      (normalizedX: number, normalizedY: number) => ({
        normalized_pixel_coords: [{ x: normalizedX, y: normalizedY }],
      }),
      []
    ),
    successOutputMapper
  );

  const manualTargetAimLaser = node.useService(
    "~/manual_target_aim_laser",
    "runner_cutter_control_interfaces/ManualTargetAimLaser",
    useCallback(
      (normalizedX: number, normalizedY: number) => ({
        normalized_pixel_coord: { x: normalizedX, y: normalizedY },
      }),
      []
    ),
    successOutputMapper
  );

  const startRunnerCutter = node.useService(
    "~/start_runner_cutter",
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

  return {
    ...node,
    state,
    calibrate,
    addCalibrationPoint,
    manualTargetAimLaser,
    startRunnerCutter,
    stop,
  };
}
