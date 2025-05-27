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
import { Card, CardContent } from "@/components/ui/card";
import useCameraNode, {
  DeviceState as CameraDeviceState,
} from "@/lib/useCameraNode";
import useControlNode from "@/lib/useControlNode";
import useLaserNode from "@/lib/useLaserNode";
import { useCallback } from "react";

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

  const onImageClick = useCallback(
    (normalizedX: number, normalizedY: number) => {
      if (controlNode.state.state !== "idle") {
        return;
      }

      controlNode.manualTargetLaser(normalizedX, normalizedY, true, false);
    },
    [controlNode]
  );

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
        </CardContent>
      </Card>
      <p className="text-center">
        Click on the image below to attempt to aim the laser to that point.
      </p>
      <FramePreviewWithOverlay
        className="w-full h-[480px]"
        topicName="/camera0/debug_frame"
        enableStream={
          cameraNode.state.deviceState === CameraDeviceState.STREAMING
        }
        onImageClick={onImageClick}
        showRotateButton
      />
    </div>
  );
}
