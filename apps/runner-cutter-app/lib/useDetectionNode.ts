import useROSNode, { ParamType } from "@/lib/ros/useROSNode";
import { useCallback } from "react";

export enum DetectionType {
  LASER,
  RUNNER,
  CIRCLE,
}

export type State = {
  enabledDetectionTypes: DetectionType[];
  recordingVideo: boolean;
};

function convertStateMessage(message: any): State {
  return {
    // uint8[] from rosbridge server is encoded as a Base64 string, so we need to decode
    // it and convert each character into a number
    enabledDetectionTypes: atob(message.enabled_detection_types)
      .split("")
      .map((char: string) => char.charCodeAt(0))
      .filter((t: number) => t in DetectionType),
    recordingVideo: message.recording_video,
  };
}

function triggerInputMapper() {
  return {};
}

function successOutputMapper(res: any): boolean {
  return res.success;
}

export default function useDetectionNode(nodeName: string) {
  const node = useROSNode(nodeName);
  const state = node.useSubscription(
    "~/state",
    "detection_interfaces/State",
    {
      enabledDetectionTypes: [],
      recordingVideo: false,
    },
    convertStateMessage
  );

  const startDetection = node.useService(
    "~/start_detection",
    "detection_interfaces/StartDetection",
    useCallback(
      (detectionType: DetectionType) => ({ detection_type: detectionType }),
      []
    ),
    successOutputMapper
  );

  const stopDetection = node.useService(
    "~/stop_detection",
    "detection_interfaces/StopDetection",
    useCallback(
      (detectionType: DetectionType) => ({ detection_type: detectionType }),
      []
    ),
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

  const getSaveDir = node.useGetParam<string>("save_dir");
  const setSaveDir = node.useSetParam<string>("save_dir", ParamType.STRING);

  return {
    ...node,
    state,
    startDetection,
    stopDetection,
    startRecordingVideo,
    stopRecordingVideo,
    getSaveDir,
    setSaveDir,
  };
}
