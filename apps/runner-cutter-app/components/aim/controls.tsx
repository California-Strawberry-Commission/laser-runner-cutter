"use client";

import FramePreview from "@/components/camera/frame-preview";
import useControlNode from "@/lib/useControlNode";
import { useState } from "react";

export default function Controls() {
  const {
    nodeInfo: controlNodeInfo,
    controlState,
    manualTargetAimLaser,
  } = useControlNode("/control0");
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
    const normalizedX = x / boundingRect.width;
    const normalizedY = y / boundingRect.height;
    manualTargetAimLaser(normalizedX, normalizedY);
  };

  return (
    <div className="flex flex-col gap-4 items-center">
      <p className="text-center">
        Click on the image below to attempt to aim the laser to that point.
      </p>
      <FramePreview
        height={600}
        topicName={"/camera0/debug_frame"}
        onImageLoad={onImageLoad}
        onImageClick={onImageClick}
      />
    </div>
  );
}
