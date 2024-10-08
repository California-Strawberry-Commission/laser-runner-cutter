"use client";

import FramePreview from "@/components/camera/frame-preview";
import DeviceCard, { DeviceState } from "@/components/setup/device-card";
import CalibrationCard, {
  CalibrationState,
} from "@/components/setup/calibration-card";
import NodesCarousel from "@/components/setup/nodes-carousel";
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
} from "@/lib/useCameraNode";
import useControlNode from "@/lib/useControlNode";
import useLaserNode, {
  DeviceState as LaserDeviceState,
} from "@/lib/useLaserNode";
import useLifecycleManagerNode from "@/lib/useLifecycleManagerNode";
import { useCallback, useMemo, useState } from "react";
import Overlay from "@/components/runner-cutter/overlay";

export default function Controls() {
  const { connected: rosConnected } = useROS();

  const lifecycleManagerNode = useLifecycleManagerNode("/lifecycle_manager");
  const cameraNode = useCameraNode("/camera0");
  const laserNode = useLaserNode("/laser0");
  const controlNode = useControlNode("/control0");

  const [framePreviewSize, setFramePreviewSize] = useState({
    width: 0,
    height: 0,
  });

  const onFramePreviewSizeChanged = useCallback(
    (width: number, height: number) => {
      if (
        width !== framePreviewSize.width ||
        height !== framePreviewSize.height
      ) {
        setFramePreviewSize({ width, height });
      }
    },
    [framePreviewSize, setFramePreviewSize]
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

  let cameraDeviceState = DeviceState.Unavailable;
  if (cameraNode.connected) {
    switch (cameraNode.state.deviceState) {
      case CameraDeviceState.Disconnected:
        cameraDeviceState = DeviceState.Disconnected;
        break;
      case CameraDeviceState.Connecting:
        cameraDeviceState = DeviceState.Connecting;
        break;
      case CameraDeviceState.Streaming:
        cameraDeviceState = DeviceState.Connected;
        break;
      default:
        break;
    }
  }

  let laserDeviceState = DeviceState.Unavailable;
  if (laserNode.connected) {
    switch (laserNode.state) {
      case LaserDeviceState.Disconnected:
        laserDeviceState = DeviceState.Disconnected;
        break;
      case LaserDeviceState.Connecting:
        laserDeviceState = DeviceState.Connecting;
        break;
      default:
        laserDeviceState = DeviceState.Connected;
        break;
    }
  }

  let calibrationState = CalibrationState.Unavailable;
  if (
    controlNode.connected &&
    cameraDeviceState === DeviceState.Connected &&
    laserDeviceState === DeviceState.Connected
  ) {
    if (controlNode.state.state === "idle") {
      calibrationState = controlNode.state.calibrated
        ? CalibrationState.Calibrated
        : CalibrationState.Uncalibrated;
    } else {
      calibrationState = CalibrationState.Busy;
    }
  }

  return (
    <div className="flex flex-col gap-4 items-center">
      <div className="flex flex-row gap-4 items-center">
        <NodesCarousel className="w-[600px]" nodeInfos={nodeInfos}>
          {restartServiceDialog}
        </NodesCarousel>
      </div>
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
          onCalibrateClick={() => controlNode.calibrate()}
          onStopClick={() => controlNode.stop()}
          onSaveClick={() => controlNode.save_calibration()}
          onLoadClick={() => controlNode.load_calibration()}
        />
      </div>
      <p className="text-center">
        After calibration, click on the image below to fire the laser at that
        point and add a calibration point.
      </p>
      <div className="relative flex items-center" style={{ height: 360 }}>
        <FramePreview
          height={360}
          topicName={"/camera0/debug_frame"}
          onComponentSizeChanged={onFramePreviewSizeChanged}
          onImageClick={(event: any) => {
            if (controlNode.state.state !== "idle") {
              return;
            }

            const boundingRect = event.target.getBoundingClientRect();
            const x = Math.round(event.clientX - boundingRect.left);
            const y = Math.round(event.clientY - boundingRect.top);
            const normalizedX = x / boundingRect.width;
            const normalizedY = y / boundingRect.height;
            controlNode.addCalibrationPoint(normalizedX, normalizedY);
          }}
        />
        <Overlay
          width={framePreviewSize.width}
          height={framePreviewSize.height}
          state={controlNode.state.state}
          tracks={controlNode.state.tracks}
          normalizedRect={controlNode.state.normalizedLaserBounds}
        />
      </div>
    </div>
  );
}
