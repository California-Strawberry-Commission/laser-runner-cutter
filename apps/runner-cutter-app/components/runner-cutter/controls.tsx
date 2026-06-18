"use client";

import FramePreviewLiveKit from "@/components/camera/frame-preview-livekit";
import CalibrationCard, {
  CalibrationState,
} from "@/components/runner-cutter/calibration-card";
import DeviceCard, {
  DeviceState,
  convertCameraNodeDeviceState,
  convertLaserNodeDeviceState,
} from "@/components/runner-cutter/device-card";
import NodeStatusBar from "@/components/runner-cutter/node-status-bar";
import RunnerCutterCard, {
  RunnerCutterMode,
  RunnerCutterState,
} from "@/components/runner-cutter/runner-cutter-card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import useROS from "@/lib/ros/useROS";
import useCameraNode, {
  DeviceState as CameraDeviceState,
} from "@/lib/useCameraNode";
import useControlNode, { TrackState } from "@/lib/useControlNode";
import useDetectionNode, { DetectionType } from "@/lib/useDetectionNode";
import useLaserNode from "@/lib/useLaserNode";
import useLifecycleManagerNode from "@/lib/useLifecycleManagerNode";
import { enumToLabel } from "@/lib/utils";
import { AlertCircleIcon } from "lucide-react";
import { useCallback, useMemo, useState } from "react";

const DEVICE_TEMPERATURE_ALERT_THRESHOLD = 70.0;

export default function Controls({
  lifecycleManagerNodeName,
  cameraNodeName,
  detectionNodeName,
  laserNodeName,
  controlNodeName,
}: {
  lifecycleManagerNodeName: string;
  cameraNodeName: string;
  detectionNodeName: string;
  laserNodeName: string;
  controlNodeName: string;
}) {
  const { connected: rosConnected } = useROS();

  const lifecycleManagerNode = useLifecycleManagerNode(
    lifecycleManagerNodeName,
  );
  const cameraNode = useCameraNode(cameraNodeName);
  const detectionNode = useDetectionNode(detectionNodeName);
  const laserNode = useLaserNode(laserNodeName);
  const controlNode = useControlNode(controlNodeName);

  const [manualMode, setManualMode] = useState<boolean>(false);

  const onImageClick = useCallback(
    (normalizedX: number, normalizedY: number) => {
      if (controlNode.state.state !== "idle") {
        return;
      }

      if (manualMode) {
        controlNode.manualTargetLaser(normalizedX, normalizedY, true, true);
      } else {
        controlNode.addCalibrationPoint(normalizedX, normalizedY);
      }
    },
    [controlNode, manualMode],
  );

  const nodeInfos = useMemo(() => {
    const rosbridgeNodeInfo = {
      name: "Rosbridge",
      connected: rosConnected,
    };
    return [
      rosbridgeNodeInfo,
      lifecycleManagerNode,
      cameraNode,
      detectionNode,
      laserNode,
      controlNode,
    ];
  }, [
    rosConnected,
    lifecycleManagerNode,
    cameraNode,
    detectionNode,
    laserNode,
    controlNode,
  ]);

  const deviceTemperatureAlert =
    cameraNode.state.colorDeviceTemperature >=
      DEVICE_TEMPERATURE_ALERT_THRESHOLD ||
    cameraNode.state.depthDeviceTemperature >=
      DEVICE_TEMPERATURE_ALERT_THRESHOLD ? (
      <Alert variant="destructive">
        <AlertCircleIcon />
        <AlertTitle>Camera temperatures are high.</AlertTitle>
        <AlertDescription>
          <p>
            Color device: {cameraNode.state.colorDeviceTemperature}°C, depth
            device: {cameraNode.state.depthDeviceTemperature}°C. Please check
            that there is sufficient ventilation and/or other means of heat
            management.
          </p>
        </AlertDescription>
      </Alert>
    ) : null;

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

  let runnerCutterState = RunnerCutterState.UNAVAILABLE;
  if (
    controlNode.connected &&
    cameraDeviceState === DeviceState.CONNECTED &&
    laserDeviceState === DeviceState.CONNECTED &&
    calibrationState === CalibrationState.CALIBRATED
  ) {
    if (controlNode.state.state === "idle") {
      if (
        detectionNode.state.enabledDetectionTypes.includes(DetectionType.RUNNER)
      ) {
        runnerCutterState = RunnerCutterState.TRACKING;
      } else if (manualMode) {
        runnerCutterState = RunnerCutterState.ARMED_MANUAL;
      } else {
        runnerCutterState = RunnerCutterState.IDLE;
      }
    } else if (controlNode.state.state === "runner_cutter") {
      runnerCutterState = RunnerCutterState.ARMED_AUTO;
    }
  }

  let stateStr = enumToLabel(controlNode.state.state);
  if (
    runnerCutterState !== RunnerCutterState.UNAVAILABLE &&
    runnerCutterState !== RunnerCutterState.IDLE
  ) {
    stateStr = enumToLabel(RunnerCutterState[runnerCutterState]);
  }
  const framePreviewOverlayText = `State: ${stateStr}`;

  let framePreviewOverlaySubtext;
  if (runnerCutterState === RunnerCutterState.ARMED_AUTO) {
    const trackStateCounts = {
      [TrackState.PENDING]: 0,
      [TrackState.ACTIVE]: 0,
      [TrackState.COMPLETED]: 0,
      [TrackState.FAILED]: 0,
    };
    controlNode.tracks.forEach((track) => (trackStateCounts[track.state] += 1));
    framePreviewOverlaySubtext = `Pending: ${
      trackStateCounts[TrackState.PENDING]
    }, Active: ${trackStateCounts[TrackState.ACTIVE]}, Completed: ${
      trackStateCounts[TrackState.COMPLETED]
    }, Failed: ${trackStateCounts[TrackState.FAILED]}`;
  } else if (runnerCutterState === RunnerCutterState.IDLE) {
    framePreviewOverlaySubtext = "Click image to add a calibration point";
  } else if (runnerCutterState === RunnerCutterState.ARMED_MANUAL) {
    framePreviewOverlaySubtext = "Click image to burn that point";
  }

  return (
    <div className="flex flex-col gap-4 items-center">
      <Card className="w-full">
        <CardHeader className="p-4">
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg">Control Panel</CardTitle>
            <NodeStatusBar
              nodeInfos={nodeInfos}
              onRestartNodes={() => lifecycleManagerNode.restartService()}
              onRebootSystem={() => lifecycleManagerNode.rebootSystem()}
              restartDisabled={!lifecycleManagerNode.connected}
            />
          </div>
        </CardHeader>
        <CardContent className="p-4 pt-0 flex flex-row flex-wrap items-center justify-center gap-4">
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
          <RunnerCutterCard
            runnerCutterState={runnerCutterState}
            onStartClick={(mode) => {
              switch (mode) {
                case RunnerCutterMode.TRACKING_ONLY:
                  detectionNode.startDetection(
                    DetectionType.RUNNER,
                    controlNode.state.normalizedLaserBounds,
                  );
                  break;
                case RunnerCutterMode.AUTO:
                  controlNode.startRunnerCutter();
                  break;
                case RunnerCutterMode.MANUAL:
                  setManualMode(true);
                  break;
              }
            }}
            onStopClick={() => {
              switch (runnerCutterState) {
                case RunnerCutterState.TRACKING:
                  detectionNode.stopDetection(DetectionType.RUNNER);
                  break;
                case RunnerCutterState.ARMED_AUTO:
                  controlNode.stop();
                  break;
                case RunnerCutterState.ARMED_MANUAL:
                  setManualMode(false);
                  break;
              }
            }}
          />
        </CardContent>
      </Card>
      {deviceTemperatureAlert}
      <FramePreviewLiveKit
        className="w-full h-90"
        topicName={`${detectionNodeName}/debug/image`}
        enableStream={
          cameraNode.state.deviceState === CameraDeviceState.STREAMING
        }
        onImageClick={onImageClick}
        enableOverlay
        overlayText={framePreviewOverlayText}
        overlaySubtext={framePreviewOverlaySubtext}
        overlayNormalizedRect={controlNode.state.normalizedLaserBounds}
        showRotateButton
      />
    </div>
  );
}
