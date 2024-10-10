"use client";

import FramePreviewWithOverlay from "@/components/camera/frame-preview-with-overlay";
import { Button } from "@/components/ui/button";
import useCameraNode, {
  DeviceState as CameraDeviceState,
} from "@/lib/useCameraNode";
import useControlNode from "@/lib/useControlNode";

export type Track = {
  id: number;
  normalizedPixelCoords: { x: number; y: number };
  state: TrackState;
};

export enum TrackState {
  Pending,
  Active,
  Completed,
  Failed,
}

function convertTracksMessage(message: any): Track[] {
  return message.tracks.map((track: any) => {
    return {
      id: track.id,
      normalizedPixelCoords: track.normalized_pixel_coords,
      state: track.state as TrackState,
    };
  });
}

export default function Controls() {
  const cameraNode = useCameraNode("/camera0");
  const controlNode = useControlNode("/control0");

  const tracks = controlNode.useTopic(
    "~/tracks",
    "runner_cutter_control_interfaces/Tracks",
    [],
    convertTracksMessage
  );

  const disableButtons =
    !controlNode.connected || controlNode.state.state !== "idle";

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
          disabled={!controlNode.connected}
          variant="destructive"
          onClick={() => {
            controlNode.stop();
          }}
        >
          Stop
        </Button>
      </div>
      <FramePreviewWithOverlay
        className="w-full h-[600px]"
        topicName="/camera0/debug_frame"
        enableStream={
          cameraNode.state.deviceState === CameraDeviceState.Streaming
        }
        enableOverlay
        overlayText={`State: ${controlNode.state.state}`}
        overlayNormalizedRect={controlNode.state.normalizedLaserBounds}
        overlayTracks={tracks}
      />
    </div>
  );
}
