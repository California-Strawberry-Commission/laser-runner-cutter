"use client";

import FramePreview from "@/components/camera/frame-preview";
import NodeCards from "@/components/nodes/node-cards";
import { Button } from "@/components/ui/button";
import useROS from "@/lib/ros/useROS";
import useCameraNode from "@/lib/useCameraNode";
import useControlNode from "@/lib/useControlNode";
import useLaserNode from "@/lib/useLaserNode";

export default function Controls() {
  const { nodeInfo: rosbridgeNodeInfo } = useROS();
  const {
    nodeInfo: controlNodeInfo,
    controlState,
    calibrate,
    startRunnerCutter,
    stop,
  } = useControlNode("/control0");
  const { nodeInfo: cameraNodeInfo, frameSrc } = useCameraNode("/camera0");
  const { nodeInfo: laserNodeInfo } = useLaserNode("/laser0");

  const nodeInfos = [
    rosbridgeNodeInfo,
    controlNodeInfo,
    cameraNodeInfo,
    laserNodeInfo,
  ];
  const disableButtons =
    !rosbridgeNodeInfo.connected ||
    !controlNodeInfo.connected ||
    controlState !== "idle";

  return (
    <div className="flex flex-col gap-4 items-center">
      <NodeCards nodeInfos={nodeInfos} />
      <div className="flex flex-row items-center gap-4">
        <Button
          disabled={disableButtons}
          onClick={() => {
            calibrate();
          }}
        >
          Calibrate
        </Button>
        <Button
          disabled={disableButtons}
          onClick={() => {
            startRunnerCutter();
          }}
        >
          Start Cutter
        </Button>
        <Button
          disabled={!rosbridgeNodeInfo.connected || !controlNodeInfo.connected}
          variant="destructive"
          onClick={() => {
            stop();
          }}
        >
          Stop
        </Button>
      </div>
      <FramePreview frameSrc={frameSrc} />
    </div>
  );
}
