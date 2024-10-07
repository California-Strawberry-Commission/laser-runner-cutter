"use client";

import FramePreview from "@/components/camera/frame-preview";
import { Button } from "@/components/ui/button";
import useControlNode from "@/lib/useControlNode";
import Overlay from "@/components/runner-cutter/overlay";
import { useCallback, useState } from "react";

export default function Controls() {
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

  const onImageClick = (event: any) => {
    if (controlNode.state.state !== "idle") {
      return;
    }

    const boundingRect = event.target.getBoundingClientRect();
    const x = Math.round(event.clientX - boundingRect.left);
    const y = Math.round(event.clientY - boundingRect.top);
    const normalizedX = x / boundingRect.width;
    const normalizedY = y / boundingRect.height;
    controlNode.addCalibrationPoint(normalizedX, normalizedY);
  };

  const disableButtons =
    !controlNode.connected || controlNode.state.state !== "idle";

  return (
    <div className="flex flex-col gap-4 items-center">
      <div className="flex flex-row items-center gap-4">
        <Button
          disabled={disableButtons}
          onClick={() => {
            controlNode.calibrate();
          }}
        >
          Start Calibration
        </Button>
        <Button
          disabled={disableButtons || !controlNode.state.calibrated}
          onClick={() => {
            controlNode.save_calibration();
          }}
        >
          Save
        </Button>
        <Button
          disabled={disableButtons}
          onClick={() => {
            controlNode.load_calibration();
          }}
        >
          Load
        </Button>
      </div>
      <p className="text-center">
        After calibration, click on the image below to fire the laser at that
        point and add a calibration point.
      </p>
      <div className="relative flex items-center" style={{ height: 600 }}>
        <FramePreview
          height={600}
          topicName={"/camera0/debug_frame"}
          onComponentSizeChanged={onFramePreviewSizeChanged}
          onImageClick={onImageClick}
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
