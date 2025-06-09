"use client";

import ColorPicker from "@/components/laser/color-picker";
import DeviceCard, {
  DeviceState,
  convertLaserNodeDeviceState,
} from "@/components/runner-cutter/device-card";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { InputWithLabel } from "@/components/ui/input-with-label";
import useLaserNode, {
  DeviceState as LaserDeviceState,
} from "@/lib/useLaserNode";
import useRgbColor from "@/lib/useRgbColor";
import { useEffect, useState } from "react";

export default function Controls({ laserNodeName }: { laserNodeName: string }) {
  const laserNode = useLaserNode(laserNodeName);
  // Normalized to [0, 1]
  const [destinationX, setDestinationX] = useState<number>(0.0);
  const [destinationY, setDestinationY] = useState<number>(0.0);
  const [color, setColor] = useRgbColor({ r: 0.0, g: 0.0, b: 0.0 });
  const [durationMs, setDurationMs] = useState<number>(1000.0);

  useEffect(() => {
    async function fetchParams() {
      if (laserNode.connected) {
        const result = await laserNode.getColor();
        if (result.length >= 3) {
          setColor({
            r: result[0],
            g: result[1],
            b: result[2],
          });
        }
      }
    }

    fetchParams();
    // We intentionally did not add laserNode to deps
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [setColor, laserNode.connected]);

  const laserDeviceState = convertLaserNodeDeviceState(laserNode);
  const disableButtons = laserDeviceState !== DeviceState.CONNECTED;

  let playbackButton = null;
  switch (laserNode.state.deviceState) {
    case LaserDeviceState.STOPPED:
      playbackButton = (
        <Button disabled={disableButtons} onClick={() => laserNode.play()}>
          Start Laser
        </Button>
      );
      break;
    case LaserDeviceState.PLAYING:
      playbackButton = (
        <Button disabled={disableButtons} onClick={() => laserNode.stop()}>
          Stop Laser
        </Button>
      );
      break;
    default:
      playbackButton = <Button disabled>Laser Disconnected</Button>;
      break;
  }

  return (
    <div className="flex flex-col gap-4 items-center">
      <div className="flex flex-row gap-4 items-center">
        <DeviceCard
          className="h-fit"
          deviceName="Laser"
          deviceState={laserDeviceState}
          onConnectClick={() => laserNode.startDevice()}
          onDisconnectClick={() => laserNode.closeDevice()}
        />
        <Card>
          <CardContent className="p-4 flex flex-col items-center gap-4">
            <div className="flex flex-row items-center gap-4">
              <ColorPicker
                className="w-[200px]"
                color={color}
                onColorChange={(color) => {
                  setColor(color);
                  laserNode.setColor(color.r, color.g, color.b);
                }}
              />
              {playbackButton}
            </div>
            <div className="flex flex-row items-center gap-4">
              <InputWithLabel
                className="flex-none w-16"
                type="number"
                id="destinationX"
                name="destinationX"
                label="Dest X"
                step={0.1}
                value={destinationX}
                onChange={(str) => {
                  const value = Number(str);
                  if (!isNaN(value)) {
                    setDestinationX(value);
                  }
                }}
              />
              <InputWithLabel
                className="flex-none w-16"
                type="number"
                id="destinationY"
                name="destinationY"
                label="Dest Y"
                step={0.1}
                value={destinationY}
                onChange={(str) => {
                  const value = Number(str);
                  if (!isNaN(value)) {
                    setDestinationY(value);
                  }
                }}
              />
              <InputWithLabel
                className="flex-none w-20"
                type="number"
                id="durationMs"
                name="durationMs"
                label="Duration (ms)"
                step={100}
                value={durationMs}
                onChange={(str) => {
                  const value = Number(str);
                  if (!isNaN(value)) {
                    setDurationMs(value);
                  }
                }}
              />
              <Button
                disabled={disableButtons}
                onClick={() =>
                  laserNode.updatePath(
                    1,
                    { x: destinationX, y: destinationY },
                    durationMs
                  )
                }
              >
                Update Dest
              </Button>
              <Button
                disabled={disableButtons}
                onClick={() =>
                  laserNode.updatePath(
                    1,
                    { x: Math.random(), y: Math.random() },
                    durationMs
                  )
                }
              >
                Random Dest
              </Button>
              <Button
                disabled={disableButtons}
                onClick={() => laserNode.clearPaths()}
              >
                Clear Path
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
