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
import NodesCarousel from "@/components/runner-cutter/nodes-carousel";
import RunnerCutterCard, {
  RunnerCutterMode,
  RunnerCutterState,
} from "@/components/runner-cutter/runner-cutter-card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import useROS from "@/lib/ros/useROS";
import useCameraNode, {
  DeviceState as CameraDeviceState,
  DetectionType,
} from "@/lib/useCameraNode";
import useControlNode, { TrackState } from "@/lib/useControlNode";
import useLaserNode from "@/lib/useLaserNode";
import useLifecycleManagerNode from "@/lib/useLifecycleManagerNode";
import { enumToLabel } from "@/lib/utils";
import { AlertCircleIcon } from "lucide-react";
import { useCallback, useMemo, useState } from "react";

const DEVICE_TEMPERATURE_ALERT_THRESHOLD = 70.0;

export default function Controls({
  lifecycleManagerNodeName,
  cameraNodeName,
  laserNodeName,
  controlNodeName,
}: {
  lifecycleManagerNodeName: string;
  cameraNodeName: string;
  laserNodeName: string;
  controlNodeName: string;
}) {
  const { connected: rosConnected } = useROS();

  const lifecycleManagerNode = useLifecycleManagerNode(
    lifecycleManagerNodeName
  );
  const cameraNode = useCameraNode(cameraNodeName);
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
    [controlNode, manualMode]
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
      laserNode,
      controlNode,
    ];
  }, [rosConnected, lifecycleManagerNode, cameraNode, laserNode, controlNode]);

  const restartServiceDialog = (
    <Dialog>
      <DialogTrigger asChild>
        <div className="pointer-events-none w-full h-full flex flex-col justify-center">
          <Button
            className="pointer-events-auto w-full"
            disabled={!lifecycleManagerNode.connected}
            variant="destructive"
          >
            Restart Service
          </Button>
        </div>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Are you absolutely sure?</DialogTitle>
          <DialogDescription>
            This will restart the ROS 2 nodes, and may take a few minutes for it
            to come back up. You can also choose to reboot the host machine,
            which will take longer.
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <DialogClose asChild>
            <Button
              variant="destructive"
              onClick={() => {
                lifecycleManagerNode.restart_service();
              }}
            >
              Restart Nodes
            </Button>
          </DialogClose>
          <DialogClose asChild>
            <Button
              variant="destructive"
              onClick={() => {
                lifecycleManagerNode.reboot_system();
              }}
            >
              Reboot System
            </Button>
          </DialogClose>
          <DialogClose asChild>
            <Button>Cancel</Button>
          </DialogClose>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );

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
        cameraNode.state.enabledDetectionTypes.includes(DetectionType.RUNNER)
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
      <NodesCarousel className="w-full" nodeInfos={nodeInfos}>
        {restartServiceDialog}
      </NodesCarousel>
      <Card>
        <CardHeader className="p-4">
          <CardTitle className="text-lg">Control Panel</CardTitle>
        </CardHeader>
        <CardContent className="p-4 pt-0 flex flex-row items-center gap-4">
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
            onStartClick={(mode: RunnerCutterMode) => {
              switch (mode) {
                case RunnerCutterMode.TRACKING_ONLY:
                  cameraNode.startDetection(DetectionType.RUNNER);
                  break;
                case RunnerCutterMode.AUTO:
                  controlNode.startRunnerCutter();
                  break;
                case RunnerCutterMode.MANUAL:
                  setManualMode(true);
                  break;
                default:
                  break;
              }
            }}
            onStopClick={() => {
              switch (runnerCutterState) {
                case RunnerCutterState.TRACKING:
                  cameraNode.stopDetection(DetectionType.RUNNER);
                  break;
                case RunnerCutterState.ARMED_AUTO:
                  controlNode.stop();
                  break;
                case RunnerCutterState.ARMED_MANUAL:
                  setManualMode(false);
                  break;
                default:
                  break;
              }
            }}
          />
        </CardContent>
      </Card>
      {deviceTemperatureAlert}
      <FramePreviewWithOverlay
        className="w-full h-[360px]"
        topicName={`/detection_node/overlay_frame`}
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
