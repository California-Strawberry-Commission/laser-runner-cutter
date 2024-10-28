"use client";

import FramePreviewWithOverlay from "@/components/camera/frame-preview-with-overlay";
import CalibrationCard, {
  CalibrationState,
} from "@/components/runner-cutter/calibration-card";
import DeviceCard, {
  DeviceState,
} from "@/components/runner-cutter/device-card";
import NodesCarousel from "@/components/runner-cutter/nodes-carousel";
import RunnerCutterCard, {
  RunnerCutterState,
} from "@/components/runner-cutter/runner-cutter-card";
import { Button } from "@/components/ui/button";
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
import useControlNode from "@/lib/useControlNode";
import useLaserNode, {
  DeviceState as LaserDeviceState,
} from "@/lib/useLaserNode";
import useLifecycleManagerNode from "@/lib/useLifecycleManagerNode";
import { useCallback, useMemo } from "react";

export default function Controls() {
  const { connected: rosConnected } = useROS();

  const lifecycleManagerNode = useLifecycleManagerNode("/lifecycle_manager");
  const cameraNode = useCameraNode("/camera0");
  const laserNode = useLaserNode("/laser0");
  const controlNode = useControlNode("/control0");

  const onImageClick = useCallback(
    (normalizedX: number, normalizedY: number) => {
      if (controlNode.state.state !== "idle") {
        return;
      }

      controlNode.addCalibrationPoint(normalizedX, normalizedY);
    },
    [controlNode]
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
        <div className="w-full h-full flex flex-col justify-center">
          <Button
            className="w-full"
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

  let cameraDeviceState = DeviceState.UNAVAILABLE;
  if (cameraNode.connected) {
    switch (cameraNode.state.deviceState) {
      case CameraDeviceState.DISCONNECTED:
        cameraDeviceState = DeviceState.DISCONNECTED;
        break;
      case CameraDeviceState.CONNECTING:
        cameraDeviceState = DeviceState.CONNECTING;
        break;
      case CameraDeviceState.STREAMING:
        cameraDeviceState = DeviceState.CONNECTED;
        break;
      default:
        break;
    }
  }

  let laserDeviceState = DeviceState.UNAVAILABLE;
  if (laserNode.connected) {
    switch (laserNode.state.deviceState) {
      case LaserDeviceState.DISCONNECTED:
        laserDeviceState = DeviceState.DISCONNECTED;
        break;
      case LaserDeviceState.CONNECTING:
        laserDeviceState = DeviceState.CONNECTING;
        break;
      case LaserDeviceState.PLAYING:
      case LaserDeviceState.STOPPED:
        laserDeviceState = DeviceState.CONNECTED;
        break;
      default:
        break;
    }
  }

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
      } else {
        runnerCutterState = RunnerCutterState.IDLE;
      }
    } else if (controlNode.state.state === "runner_cutter") {
      runnerCutterState = RunnerCutterState.ARMED;
    }
  }

  return (
    <div className="flex flex-col gap-4 items-center">
      <NodesCarousel className="w-full" nodeInfos={nodeInfos}>
        {restartServiceDialog}
      </NodesCarousel>
      <div className="flex flex-row items-center gap-4">
        <DeviceCard
          className="w-36"
          deviceName="Camera"
          deviceState={cameraDeviceState}
          onConnectClick={() => cameraNode.startDevice()}
          onDisconnectClick={() => cameraNode.closeDevice()}
        />
        <DeviceCard
          className="w-36"
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
          onTrackClick={() => cameraNode.startDetection(DetectionType.RUNNER)}
          onTrackStopClick={() =>
            cameraNode.stopDetection(DetectionType.RUNNER)
          }
          onArmClick={() => controlNode.startRunnerCutter()}
          onArmStopClick={() => controlNode.stop()}
        />
      </div>
      <FramePreviewWithOverlay
        className="w-full h-[360px]"
        topicName="/camera0/debug_frame"
        enableStream={
          cameraNode.state.deviceState === CameraDeviceState.STREAMING
        }
        onImageClick={onImageClick}
        enableOverlay
        overlayText={`State: ${controlNode.state.state}`}
        overlayNormalizedRect={controlNode.state.normalizedLaserBounds}
      />
    </div>
  );
}
