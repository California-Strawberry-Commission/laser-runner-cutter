"use client";

import FramePreview from "@/components/camera/frame-preview";
import { Button } from "@/components/ui/button";
import useROS from "@/lib/ros/useROS";
import useControlNode from "@/lib/useControlNode";

export default function Controls() {
  const { nodeInfo: rosbridgeNodeInfo } = useROS();
  const {
    nodeInfo: controlNodeInfo,
    controlState,
    calibrate,
    addCalibrationPoint,
  } = useControlNode("/control0");

  const onImageClick = (event: any) => {
    if (controlState !== "idle") {
      return;
    }

    const boundingRect = event.target.getBoundingClientRect();
    const x = Math.round(event.clientX - boundingRect.left);
    const y = Math.round(event.clientY - boundingRect.top);
    const normalizedX = x / boundingRect.width;
    const normalizedY = y / boundingRect.height;
    addCalibrationPoint(normalizedX, normalizedY);
  };

  const disableButtons =
    !rosbridgeNodeInfo.connected ||
    !controlNodeInfo.connected ||
    controlState !== "idle";

  return (
    <div className="flex flex-col gap-4 items-center">
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
        After calibration, click on the image below to fire the laser at that
        point and add a calibration point.
      </p>
      <FramePreview
        height={600}
        topicName={"/camera0/debug_frame"}
        onImageClick={onImageClick}
      />
    </div>
  );
}
