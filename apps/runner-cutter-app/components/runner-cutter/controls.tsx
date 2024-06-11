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
    startRunnerCutter,
    stop,
  } = useControlNode("/control0");

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
      <FramePreview topicName={"/camera0/debug_frame"} />
    </div>
  );
}
