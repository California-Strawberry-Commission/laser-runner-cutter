"use client";

import FramePreview from "@/components/camera/frame-preview";
import Overlay from "@/components/runner-cutter/overlay";
import { Button } from "@/components/ui/button";
import useROS from "@/lib/ros/useROS";
import useControlNode from "@/lib/useControlNode";

export default function Controls() {
  const { connected: rosConnected } = useROS();
  const controlNode = useControlNode("/control0");

  const disableButtons =
    !rosConnected ||
    !controlNode.connected ||
    controlNode.state.state !== "idle";

  return (
    <div className="flex flex-col gap-4 items-center">
      <div className="flex flex-row items-center gap-4">
        <Button
          disabled={disableButtons}
          onClick={() => {
            controlNode.startRunnerCutter();
          }}
        >
          Start
        </Button>
        <Button
          disabled={!rosConnected || !controlNode.connected}
          variant="destructive"
          onClick={() => {
            controlNode.stop();
          }}
        >
          Stop
        </Button>
      </div>
      <div className="relative flex items-center" style={{ height: 600 }}>
        <FramePreview height={600} topicName={"/camera0/debug_frame"} />
        <Overlay
          tracks={controlNode.state.tracks}
          normalizedRect={controlNode.state.normalizedLaserBounds}
        />
      </div>
    </div>
  );
}
