"use client";

import FramePreviewWithOverlay from "@/components/camera/frame-preview-with-overlay";
import CalibrationCard, {
  CalibrationState,
} from "@/components/runner-cutter/calibration-card";
import DeviceCard, {
  DeviceState,
  convertCameraNodeDeviceState,
  convertLaserNodeDeviceState,
} from "@/components/runner-cutter/device-card";
import CircleFollowerCard, {
  CircleFollowerState,
} from "@/components/track/circle-follower-card";
import { Card, CardContent } from "@/components/ui/card";
import useCameraNode, {
  DeviceState as CameraDeviceState,
  DetectionType,
} from "@/lib/useCameraNode";
import useControlNode from "@/lib/useControlNode";
import useLaserNode from "@/lib/useLaserNode";

export default function Controls() {
  const cameraNode = useCameraNode("/camera0");
  const laserNode = useLaserNode("/laser0");
  const controlNode = useControlNode("/control0");

  const cameraDeviceState = convertCameraNodeDeviceState(cameraNode);
  const laserDeviceState = convertLaserNodeDeviceState(laserNode);

  let calibrationState = CalibrationState.UNAVAILABLE;
  if (
    controlNode.connected &&
    cameraDeviceState === DeviceState.CONNECTED &&
    laserDeviceState === DeviceState.CONNECTED
  ) {
    if (controlNode.state.state === "calibration") {
      calibrationState = CalibrationState.CALIBRATING;
    } else {
      calibrationState = controlNode.state.calibrated
        ? CalibrationState.CALIBRATED
        : CalibrationState.UNCALIBRATED;
    }
  }

  let circleFollowerState = CircleFollowerState.UNAVAILABLE;
  if (
    controlNode.connected &&
    cameraDeviceState === DeviceState.CONNECTED &&
    laserDeviceState === DeviceState.CONNECTED &&
    calibrationState === CalibrationState.CALIBRATED
  ) {
    if (controlNode.state.state === "idle") {
      if (
        cameraNode.state.enabledDetectionTypes.includes(DetectionType.CIRCLE)
      ) {
        circleFollowerState = CircleFollowerState.TRACKING;
      } else {
        circleFollowerState = CircleFollowerState.IDLE;
      }
    } else if (controlNode.state.state === "circle_follower") {
      circleFollowerState = CircleFollowerState.FOLLOWING;
    }
  }

  return (
    <div className="flex flex-col gap-4 items-center">
      <Card>
        <CardContent className="p-4 flex flex-row items-center gap-4">
          <DeviceCard
            deviceName="Camera"
            deviceState={cameraDeviceState}
            onConnectClick={() => cameraNode.startDevice()}
            onDisconnectClick={() => cameraNode.closeDevice()}
          />
          <DeviceCard
            deviceName="Laser"
            deviceState={laserDeviceState}
            onConnectClick={() => laserNode.startDevice()}
            onDisconnectClick={() => laserNode.closeDevice()}
          />
          <CalibrationCard
            calibrationState={calibrationState}
            disabled={
              calibrationState !== CalibrationState.CALIBRATING &&
              controlNode.state.state !== "idle"
            }
            onCalibrateClick={() => controlNode.calibrate()}
            onStopClick={() => controlNode.stop()}
            onSaveClick={() => controlNode.saveCalibration()}
            onLoadClick={() => controlNode.loadCalibration()}
          />
          <CircleFollowerCard
            circleFollowerState={circleFollowerState}
            onTrackClick={() => cameraNode.startDetection(DetectionType.CIRCLE)}
            onTrackStopClick={() =>
              cameraNode.stopDetection(DetectionType.CIRCLE)
            }
            onFollowClick={() => controlNode.startCircleFollower()}
            onFollowStopClick={() => controlNode.stop()}
          />
        </CardContent>
      </Card>
      <FramePreviewWithOverlay
        className="w-full h-[480px]"
        topicName="/camera0/debug_frame"
        enableStream={
          cameraNode.state.deviceState === CameraDeviceState.STREAMING
        }
        enableOverlay
        overlayText={`State: ${controlNode.state.state}`}
        overlayNormalizedRect={controlNode.state.normalizedLaserBounds}
        showRotateButton
      />
    </div>
  );
}
