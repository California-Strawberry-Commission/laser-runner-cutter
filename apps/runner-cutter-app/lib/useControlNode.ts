import useROSNode, { ParamType } from "@/lib/ros/useROSNode";
import { useCallback } from "react";

export type State = {
  calibrated: boolean;
  state: string;
  normalizedLaserBounds: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
};

export type Track = {
  id: number;
  normalizedPixelCoord: { x: number; y: number };
  state: TrackState;
};

export enum TrackState {
  PENDING,
  ACTIVE,
  COMPLETED,
  FAILED,
}

function convertStateMessage(message: any): State {
  return {
    calibrated: message.calibrated,
    state: message.state,
    normalizedLaserBounds: {
      x: message.normalized_laser_bounds.w,
      y: message.normalized_laser_bounds.x,
      width: message.normalized_laser_bounds.y,
      height: message.normalized_laser_bounds.z,
    },
  };
}

function convertTracksMessage(message: any): Track[] {
  return message.tracks.map((track: any) => ({
    id: track.id,
    normalizedPixelCoord: {
      x: track.normalized_pixel_coord.x,
      y: track.normalized_pixel_coord.y,
    },
    state: track.state as TrackState,
  }));
}

function triggerInputMapper() {
  return {};
}

function successOutputMapper(res: any): boolean {
  return res.success;
}

export default function useControlNode(nodeName: string) {
  const node = useROSNode(nodeName);
  const state = node.useSubscription(
    "~/state",
    "runner_cutter_control_interfaces/State",
    {
      calibrated: false,
      state: "idle",
      normalizedLaserBounds: { x: 0, y: 0, width: 0, height: 0 },
    },
    convertStateMessage
  );

  const tracks = node.useSubscription(
    "~/tracks",
    "runner_cutter_control_interfaces/Tracks",
    [],
    convertTracksMessage
  );

  const calibrate = node.useService(
    "~/calibrate",
    "std_srvs/Trigger",
    triggerInputMapper,
    successOutputMapper
  );

  const saveCalibration = node.useService(
    "~/save_calibration",
    "std_srvs/Trigger",
    triggerInputMapper,
    successOutputMapper
  );

  const loadCalibration = node.useService(
    "~/load_calibration",
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

  const manualTargetLaser = node.useService(
    "~/manual_target_laser",
    "runner_cutter_control_interfaces/ManualTargetLaser",
    useCallback(
      (
        normalizedX: number,
        normalizedY: number,
        aim: boolean,
        burn: boolean
      ) => ({
        normalized_pixel_coord: { x: normalizedX, y: normalizedY },
        aim,
        burn,
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

  const startCircleFollower = node.useService(
    "~/start_circle_follower",
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

  const getTrackingLaserColor = node.useGetParam<number[]>(
    "tracking_laser_color"
  );
  const setTrackingLaserColor = node.useSetParam(
    "tracking_laser_color",
    ParamType.DOUBLE_ARRAY,
    useCallback((r: number, g: number, b: number) => [r, g, b, 0.0], [])
  );

  const getBurnLaserColor = node.useGetParam<number[]>("burn_laser_color");
  const setBurnLaserColor = node.useSetParam(
    "burn_laser_color",
    ParamType.DOUBLE_ARRAY,
    useCallback((r: number, g: number, b: number) => [r, g, b, 0.0], [])
  );

  const getBurnTimeSecs = node.useGetParam<number>("burn_time_secs");
  const setBurnTimeSecs = node.useSetParam<number>(
    "burn_time_secs",
    ParamType.DOUBLE
  );

  const getEnableAiming = node.useGetParam<boolean>("enable_aiming");
  const setEnableAiming = node.useSetParam<boolean>(
    "enable_aiming",
    ParamType.BOOL
  );

  const getTargetAttempts = node.useGetParam<number>("target_attempts");
  const setTargetAttempts = node.useSetParam<number>(
    "target_attempts",
    ParamType.INTEGER
  );

  const getAutoDisarmSecs = node.useGetParam<number>("auto_disarm_secs");
  const setAutoDisarmSecs = node.useSetParam<number>(
    "auto_disarm_secs",
    ParamType.DOUBLE
  );

  const getSaveDir = node.useGetParam<string>("save_dir");
  const setSaveDir = node.useSetParam<string>("save_dir", ParamType.STRING);

  return {
    ...node,
    state,
    tracks,
    calibrate,
    saveCalibration,
    loadCalibration,
    addCalibrationPoint,
    manualTargetLaser,
    startRunnerCutter,
    startCircleFollower,
    stop,
    getTrackingLaserColor,
    setTrackingLaserColor,
    getBurnLaserColor,
    setBurnLaserColor,
    getBurnTimeSecs,
    setBurnTimeSecs,
    getEnableAiming,
    setEnableAiming,
    getTargetAttempts,
    setTargetAttempts,
    getAutoDisarmSecs,
    setAutoDisarmSecs,
    getSaveDir,
    setSaveDir,
  };
}
