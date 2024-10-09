"use client";

import FramePreviewWithOverlay from "@/components/camera/frame-preview-with-overlay";
import useCameraNode, {
  DeviceState as CameraDeviceState,
} from "@/lib/useCameraNode";
import useControlNode from "@/lib/useControlNode";

export default function Controls() {
  const cameraNode = useCameraNode("/camera0");
  const controlNode = useControlNode("/control0");

  const onImageClick = (event: any) => {
    if (controlNode.state.state !== "idle") {
      return;
    }

    const boundingRect = event.target.getBoundingClientRect();
    const x = Math.round(event.clientX - boundingRect.left);
    const y = Math.round(event.clientY - boundingRect.top);
    const normalizedX = x / boundingRect.width;
    const normalizedY = y / boundingRect.height;
    controlNode.manualTargetAimLaser(normalizedX, normalizedY);
  };

  return (
    <div className="flex flex-col gap-4 items-center">
      <p className="text-center">
        Click on the image below to attempt to aim the laser to that point.
      </p>
      <FramePreviewWithOverlay
        className="w-full h-[600px]"
        topicName="/camera0/debug_frame"
        enableStream={
          cameraNode.state.deviceState === CameraDeviceState.Streaming
        }
        onImageClick={onImageClick}
      />
    </div>
  );
}
