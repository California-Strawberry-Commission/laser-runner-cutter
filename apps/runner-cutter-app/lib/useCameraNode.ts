import useROSNode, { ParamType } from "@/lib/ros/useROSNode";
import { useCallback } from "react";

export enum DeviceState {
  DISCONNECTED,
  CONNECTING,
  STREAMING,
  DISCONNECTING,
}

export enum CaptureMode {
  CONTINUOUS,
  SINGLE_FRAME,
}

export type State = {
  deviceState: DeviceState;
  intervalCaptureActive: boolean;
  exposureUsRange: [number, number];
  gainDbRange: [number, number];
  colorDeviceTemperature: number;
  depthDeviceTemperature: number;
  colorWidth: number;
  colorHeight: number;
  depthWidth: number;
  depthHeight: number;
};

function convertStateMessage(message: any): State {
  return {
    deviceState: message.device_state as DeviceState,
    intervalCaptureActive: message.interval_capture_active,
    exposureUsRange: [message.exposure_us_range.x, message.exposure_us_range.y],
    gainDbRange: [message.gain_db_range.x, message.gain_db_range.y],
    colorDeviceTemperature: message.color_device_temperature,
    depthDeviceTemperature: message.depth_device_temperature,
    colorWidth: message.color_width,
    colorHeight: message.color_height,
    depthWidth: message.depth_width,
    depthHeight: message.depth_height,
  };
}

function triggerInputMapper() {
  return {};
}

function successOutputMapper(res: any): boolean {
  return res.success;
}

export default function useCameraNode(nodeName: string) {
  const node = useROSNode(nodeName);
  const state = node.useSubscription(
    "~/state",
    "camera_control_interfaces/State",
    {
      deviceState: DeviceState.DISCONNECTED,
      intervalCaptureActive: false,
      exposureUsRange: [0.0, 0.0],
      gainDbRange: [0.0, 0.0],
      colorDeviceTemperature: 0.0,
      depthDeviceTemperature: 0.0,
      colorWidth: 0,
      colorHeight: 0,
      depthWidth: 0,
      depthHeight: 0,
    },
    convertStateMessage
  );

  const startDevice = node.useService(
    "~/start_device",
    "camera_control_interfaces/StartDevice",
    useCallback(
      (captureMode: CaptureMode = CaptureMode.CONTINUOUS) => ({
        capture_mode: captureMode,
      }),
      []
    ),
    successOutputMapper
  );

  const closeDevice = node.useService(
    "~/close_device",
    "std_srvs/Trigger",
    triggerInputMapper,
    successOutputMapper
  );

  const acquireSingleFrame = node.useService(
    "~/acquire_single_frame",
    "camera_control_interfaces/AcquireSingleFrame",
    triggerInputMapper,
    useCallback(
      (_data: any) => _data.preview_image as { data: any; format: string },
      []
    )
  );

  const startIntervalCapture = node.useService(
    "~/start_interval_capture",
    "camera_control_interfaces/StartIntervalCapture",
    useCallback(
      (intervalSecs: number) => ({ interval_secs: intervalSecs }),
      []
    ),
    successOutputMapper
  );

  const stopIntervalCapture = node.useService(
    "~/stop_interval_capture",
    "std_srvs/Trigger",
    triggerInputMapper,
    successOutputMapper
  );

  const saveImage = node.useService(
    "~/save_image",
    "std_srvs/Trigger",
    triggerInputMapper,
    successOutputMapper
  );

  const getExposureUs = node.useGetParam<number>("exposure_us");
  const setExposureUs = node.useSetParam<number>("exposure_us", ParamType.DOUBLE);

  const getGainDb = node.useGetParam<number>("gain_db");
  const setGainDb = node.useSetParam<number>("gain_db", ParamType.DOUBLE);

  const getSaveDir = node.useGetParam<string>("save_dir");
  const setSaveDir = node.useSetParam<string>("save_dir", ParamType.STRING);

  const getImageCaptureIntervalSecs = node.useGetParam<number>(
    "image_capture_interval_secs",
    ParamType.DOUBLE
  );

  return {
    ...node,
    state,
    startDevice,
    closeDevice,
    acquireSingleFrame,
    startIntervalCapture,
    stopIntervalCapture,
    saveImage,
    getExposureUs,
    setExposureUs,
    getGainDb,
    setGainDb,
    getSaveDir,
    setSaveDir,
    getImageCaptureIntervalSecs,
  };
}
