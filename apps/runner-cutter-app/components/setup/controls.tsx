"use client";

import FramePreview from "@/components/camera/frame-preview";
import NodeCards from "@/components/nodes/node-cards";
import { Button } from "@/components/ui/button";
import useROS from "@/lib/ros/useROS";
import useCameraNode from "@/lib/useCameraNode";
import useControlNode from "@/lib/useControlNode";
import useLaserNode from "@/lib/useLaserNode";
import { Loader2 } from "lucide-react";
import { useMemo } from "react";

export default function Controls() {
  const { nodeInfo: rosbridgeNodeInfo } = useROS();
  const {
    nodeInfo: controlNodeInfo,
    controlState,
    calibrate,
  } = useControlNode("/control0");
  const {
    nodeInfo: cameraNodeInfo,
    deviceState: cameraState,
    startDevice: connectCamera,
    closeDevice: disconnectCamera,
  } = useCameraNode("/camera0");
  const {
    nodeInfo: laserNodeInfo,
    laserState,
    startDevice: connectLaser,
    closeDevice: disconnectLaser,
  } = useLaserNode("/laser0");

  const nodeInfos = useMemo(() => {
    return [rosbridgeNodeInfo, controlNodeInfo, cameraNodeInfo, laserNodeInfo];
  }, [rosbridgeNodeInfo, controlNodeInfo, cameraNodeInfo, laserNodeInfo]);

  let cameraButton = null;
  const enableCameraButton = cameraNodeInfo.connected;
  if (cameraState === "disconnected") {
    cameraButton = (
      <Button disabled={!enableCameraButton} onClick={() => connectCamera()}>
        Connect Camera
      </Button>
    );
  } else if (cameraState === "connecting") {
    cameraButton = (
      <Button disabled>
        <Loader2 className="h-4 w-4 animate-spin" />
      </Button>
    );
  } else {
    cameraButton = (
      <Button
        disabled={!enableCameraButton}
        variant="destructive"
        onClick={() => disconnectCamera()}
      >
        Disconnect Camera
      </Button>
    );
  }

  let laserButton = null;
  const enableLaserButton = laserNodeInfo.connected;
  if (laserState === "disconnected") {
    laserButton = (
      <Button disabled={!enableLaserButton} onClick={() => connectLaser()}>
        Connect Laser
      </Button>
    );
  } else if (laserState === "connecting") {
    laserButton = (
      <Button disabled>
        <Loader2 className="h-4 w-4 animate-spin" />
      </Button>
    );
  } else {
    laserButton = (
      <Button
        disabled={!enableLaserButton}
        variant="destructive"
        onClick={() => disconnectLaser()}
      >
        Disconnect Laser
      </Button>
    );
  }

  const enableCalibrationButton =
    controlNodeInfo.connected &&
    controlState === "idle" &&
    cameraState === "streaming" &&
    laserState === "stopped";

  return (
    <div className="flex flex-col gap-4 items-center">
      <NodeCards nodeInfos={nodeInfos} />
      <div className="flex flex-row items-center gap-4">
        {cameraButton}
        {laserButton}
        <Button
          disabled={!enableCalibrationButton}
          onClick={() => {
            calibrate();
          }}
        >
          Start Calibration
        </Button>
      </div>
      <FramePreview topicName={"/camera0/debug_frame"} />
    </div>
  );
}
