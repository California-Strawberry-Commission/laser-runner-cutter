"use client";

import FramePreview from "@/components/camera/frame-preview";
import Overlay from "@/components/runner-cutter/overlay";
import { Button } from "@/components/ui/button";
import useROS from "@/lib/ros/useROS";
import useControlNode from "@/lib/useControlNode";
import React, { useCallback, useState } from "react";

export default function Controls() {
  const { nodeInfo: rosbridgeNodeInfo } = useROS();
  const {
    nodeInfo: controlNodeInfo,
    nodeState: controlNodeState,
    startRunnerCutter,
    stop,
  } = useControlNode("/control0");
  const [framePreviewSize, setFramePreviewSize] = useState({
    width: 0,
    height: 0,
  });

  const disableButtons =
    !rosbridgeNodeInfo.connected ||
    !controlNodeInfo.connected ||
    controlNodeState.state !== "idle";

  const onFramePreviewSizeChanged = useCallback(
    (width: number, height: number) => {
      if (
        width !== framePreviewSize.width ||
        height !== framePreviewSize.height
      ) {
        setFramePreviewSize({ width, height });
      }
    },
    [setFramePreviewSize]
  );

  // TODO: convert controlNodeState.tracks to markers and pass into Overlay

  return (
    <div className="flex flex-col gap-4 items-center">
      <div className="flex flex-row items-center gap-4">
        <Button
          disabled={disableButtons}
          onClick={() => {
            startRunnerCutter();
          }}
        >
          Start
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
      <div className="relative flex items-center" style={{ height: 600 }}>
        <FramePreview
          height={600}
          topicName={"/camera0/debug_frame"}
          onSizeChanged={onFramePreviewSizeChanged}
        />
        <Overlay
          width={framePreviewSize.width}
          height={framePreviewSize.height}
          tracks={controlNodeState.tracks}
        />
      </div>
    </div>
  );
}
