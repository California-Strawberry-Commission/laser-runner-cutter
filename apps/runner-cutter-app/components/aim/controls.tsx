"use client";

import FramePreview from "@/components/camera/frame-preview";
import useControlNode from "@/lib/useControlNode";

export default function Controls() {
  const { nodeState: controlNodeState, manualTargetAimLaser } =
    useControlNode("/control0");

  const onImageClick = (event: any) => {
    if (controlNodeState.state !== "idle") {
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
        onImageClick={onImageClick}
      />
    </div>
  );
}
