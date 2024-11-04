"use client";

import FramePreviewWithOverlay from "@/components/camera/frame-preview-with-overlay";
import RunnerCutterCard, {
  RunnerCutterState,
} from "@/components/runner-cutter/runner-cutter-card";
import useCameraNode, {
  DeviceState as CameraDeviceState,
  DetectionType,
} from "@/lib/useCameraNode";
import useControlNode from "@/lib/useControlNode";

export default function Controls() {
  const cameraNode = useCameraNode("/camera0");
  const controlNode = useControlNode("/control0");

  let circleFollowerState = RunnerCutterState.UNAVAILABLE;
  if (controlNode.state.state === "idle") {
    if (cameraNode.state.enabledDetectionTypes.includes(DetectionType.CIRCLE)) {
      circleFollowerState = RunnerCutterState.TRACKING;
    } else {
      circleFollowerState = RunnerCutterState.IDLE;
    }
  } else if (controlNode.state.state === "circle_follower") {
    circleFollowerState = RunnerCutterState.ARMED;
  }

  return (
    <div className="flex flex-col gap-4 items-center">
      <div className="flex flex-row items-center gap-4">
        <RunnerCutterCard
          runnerCutterState={circleFollowerState}
          onTrackClick={() => cameraNode.startDetection(DetectionType.CIRCLE)}
          onTrackStopClick={() =>
            cameraNode.stopDetection(DetectionType.CIRCLE)
          }
          onArmClick={() => controlNode.startCircleFollower()}
          onArmStopClick={() => controlNode.stop()}
        />
      </div>
      <FramePreviewWithOverlay
        className="w-full h-[360px]"
        topicName="/camera0/debug_frame"
        enableStream={
          cameraNode.state.deviceState === CameraDeviceState.STREAMING
        }
        enableOverlay
        overlayText={`State: ${controlNode.state.state}`}
        overlayNormalizedRect={controlNode.state.normalizedLaserBounds}
      />
    </div>
  );
}
