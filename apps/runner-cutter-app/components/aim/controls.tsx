"use client";

import FramePreview from "@/components/camera/frame-preview";
import NodeCards from "@/components/nodes/node-cards";
import { Button } from "@/components/ui/button";
import useROS from "@/lib/ros/useROS";
import useCameraNode from "@/lib/useCameraNode";
import useControlNode from "@/lib/useControlNode";
import useLaserNode from "@/lib/useLaserNode";
import { useMemo, useState } from "react";

export default function Controls() {
  const { nodeInfo: rosbridgeNodeInfo } = useROS();
  const {
    nodeInfo: controlNodeInfo,
    controlState,
    calibrate,
    manualTargetAimLaser,
  } = useControlNode("/control0");
  const { nodeInfo: cameraNodeInfo, frameSrc } = useCameraNode("/camera0");
  const { nodeInfo: laserNodeInfo } = useLaserNode("/laser0");
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });

  const onImageLoad = (event: any) => {
    const { naturalWidth: width, naturalHeight: height } = event.target;
    setImageSize({ width, height });
  };

  const onImageClick = (event: any) => {
    if (controlState !== "idle") {
      return;
    }

    const boundingRect = event.target.getBoundingClientRect();
    const x = Math.round(event.clientX - boundingRect.left);
    const y = Math.round(event.clientY - boundingRect.top);
    // Scale x, y from rendered size to actual image size
    const scaledX = (imageSize.width / boundingRect.width) * x;
    const scaledY = (imageSize.height / boundingRect.height) * y;
    manualTargetAimLaser(scaledX, scaledY);
  };

  const nodeInfos = useMemo(() => {
    return [rosbridgeNodeInfo, controlNodeInfo, cameraNodeInfo, laserNodeInfo];
  }, [rosbridgeNodeInfo, controlNodeInfo, cameraNodeInfo, laserNodeInfo]);
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
          Start Calibration
        </Button>
      </div>
      <p className="text-center">
        Click on the image below to attempt to aim the laser to that point.
      </p>
      <FramePreview
        frameSrc={frameSrc}
        onImageLoad={onImageLoad}
        onImageClick={onImageClick}
      />
    </div>
  );
}
