import useROSNode from "@/lib/ros/useROSNode";
import { useCallback, useEffect, useState } from "react";
import expandTopicOrServiceName from "@/lib/ros/expandTopicName";

export type State = {
  deviceState: DeviceState;
  laserDetectionEnabled: boolean;
  runnerDetectionEnabled: boolean;
  recordingVideo: boolean;
  intervalCaptureActive: boolean;
  exposureUs: number;
  exposureUsRange: [number, number];
  gainDb: number;
  gainDbRange: [number, number];
  saveDirectory: string;
  imageCaptureIntervalSecs: number;
};
export enum DeviceState {
  Disconnected,
  Connecting,
  Streaming,
}

function convertStateMessage(message: any): State {
  return {
    deviceState: message.device_state.data as DeviceState,
    laserDetectionEnabled: message.laser_detection_enabled,
    runnerDetectionEnabled: message.runner_detection_enabled,
    recordingVideo: message.recording_video,
    intervalCaptureActive: message.interval_capture_active,
    exposureUs: message.exposure_us,
    exposureUsRange: [message.exposure_us_range.x, message.exposure_us_range.y],
    gainDb: message.gain_db,
    gainDbRange: [message.gain_db_range.x, message.gain_db_range.y],
    saveDirectory: message.save_directory,
    imageCaptureIntervalSecs: message.image_capture_interval_secs,
  };
}

function convertToLocalReadableTime(secondsSinceEpoch: number) {
  const date = new Date(secondsSinceEpoch * 1000);
  const hours = date.getHours().toString().padStart(2, "0");
  const minutes = date.getMinutes().toString().padStart(2, "0");
  const seconds = date.getSeconds().toString().padStart(2, "0");
  return `${hours}:${minutes}:${seconds}`;
}

function triggerInputMapper() {
  return {};
}

function successOutputMapper(res: any): boolean {
  return res.success;
}

export default function useCameraNode(nodeName: string) {
  const node = useROSNode(nodeName);
  const state = node.useTopic(
    "~/state",
    "camera_control_interfaces/State",
    {
      deviceState: DeviceState.Disconnected,
      laserDetectionEnabled: false,
      runnerDetectionEnabled: false,
      recordingVideo: false,
      intervalCaptureActive: false,
      exposureUs: 0.0,
      exposureUsRange: [0.0, 0.0],
      gainDb: 0.0,
      gainDbRange: [0.0, 0.0],
      saveDirectory: "",
      imageCaptureIntervalSecs: 0.0,
    },
    convertStateMessage
  );

  const [logMessages, setLogMessages] = useState<string[]>([]);

  const addLogMessage = useCallback(
    (logMessage: string) => {
      setLogMessages((prevLogMessages) => {
        const newLogMessages = [...prevLogMessages, logMessage];
        return newLogMessages.slice(-10);
      });
    },
    [setLogMessages]
  );

  // Subscription for log messages
  useEffect(() => {
    const logSub = node.ros.subscribe(
      expandTopicOrServiceName("~/log", nodeName),
      "rcl_interfaces/Log",
      (message) => {
        const timestamp_sec = parseInt(message["stamp"]["sec"]);
        const msg = `[${convertToLocalReadableTime(timestamp_sec)}] ${
          message["msg"]
        }`;
        addLogMessage(msg);
      }
    );

    return () => {
      logSub.unsubscribe();
    };
  }, [node.ros, nodeName, addLogMessage]);

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

  const setExposure = node.useService(
    "~/set_exposure",
    "camera_control_interfaces/SetExposure",
    useCallback((exposureUs: number) => ({ exposure_us: exposureUs }), []),
    successOutputMapper
  );

  const autoExposure = node.useService(
    "~/auto_exposure",
    "std_srvs/Trigger",
    triggerInputMapper,
    successOutputMapper
  );

  const setGain = node.useService(
    "~/set_gain",
    "camera_control_interfaces/SetGain",
    useCallback((gainDb: number) => ({ gain_db: gainDb }), []),
    successOutputMapper
  );

  const autoGain = node.useService(
    "~/auto_gain",
    "std_srvs/Trigger",
    triggerInputMapper,
    successOutputMapper
  );

  const setSaveDirectory = node.useService(
    "~/set_save_directory",
    "camera_control_interfaces/SetSaveDirectory",
    useCallback((saveDir: string) => ({ save_directory: saveDir }), []),
    successOutputMapper
  );

  const startLaserDetection = node.useService(
    "~/start_laser_detection",
    "std_srvs/Trigger",
    triggerInputMapper,
    successOutputMapper
  );

  const stopLaserDetection = node.useService(
    "~/stop_laser_detection",
    "std_srvs/Trigger",
    triggerInputMapper,
    successOutputMapper
  );

  const startRunnerDetection = node.useService(
    "~/start_runner_detection",
    "std_srvs/Trigger",
    triggerInputMapper,
    successOutputMapper
  );

  const stopRunnerDetection = node.useService(
    "~/stop_runner_detection",
    "std_srvs/Trigger",
    triggerInputMapper,
    successOutputMapper
  );

  const startRecordingVideo = node.useService(
    "~/start_recording_video",
    "std_srvs/Trigger",
    triggerInputMapper,
    successOutputMapper
  );

  const stopRecordingVideo = node.useService(
    "~/stop_recording_video",
    "std_srvs/Trigger",
    triggerInputMapper,
    successOutputMapper
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

  return {
    ...node,
    state,
    logMessages,
    startDevice,
    closeDevice,
    setExposure,
    autoExposure,
    setGain,
    autoGain,
    setSaveDirectory,
    startLaserDetection,
    stopLaserDetection,
    startRunnerDetection,
    stopRunnerDetection,
    startRecordingVideo,
    stopRecordingVideo,
    startIntervalCapture,
    stopIntervalCapture,
    saveImage,
  };
}
