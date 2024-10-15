"use client";

import FramePreviewWithOverlay from "@/components/camera/frame-preview-with-overlay";
import useCameraNode, {
  DeviceState as CameraDeviceState,
} from "@/lib/useCameraNode";
import useControlNode from "@/lib/useControlNode";
import { useCallback } from "react";

export default function Controls() {
  const cameraNode = useCameraNode("/camera0");
  const controlNode = useControlNode("/control0");

  const onImageClick = useCallback(
    (normalizedX: number, normalizedY: number) => {
      if (controlNode.state.state !== "idle") {
        return;
      }

      controlNode.manualTargetAimLaser(normalizedX, normalizedY);
    },
    [controlNode]
  );

  return (
    <div className="flex flex-col gap-4 items-center">
      <p className="text-center">
        Click on the image below to attempt to aim the laser to that point.
      </p>
      <FramePreviewWithOverlay
        className="w-full h-[600px]"
        topicName="/camera0/debug_frame"
        enableStream={
          cameraNode.state.deviceState === CameraDeviceState.STREAMING
        }
        onImageClick={onImageClick}
      />
    </div>
  );
}
