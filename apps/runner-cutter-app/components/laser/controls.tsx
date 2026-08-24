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

type Waypoint = {
  x: number;
  y: number;
  delayMs: number;
};

// Converts a delay in milliseconds from now into a ROS timestamp
// (epoch seconds + nanoseconds).
function delayMsToTimestamp(delayMs: number): { sec: number; nanosec: number } {
  const epochMs = Date.now() + delayMs;
  const sec = Math.floor(epochMs / 1000);
  const nanosec = Math.round((epochMs - sec * 1000) * 1e6);
  return { sec, nanosec };
}

export default function Controls({ laserNodeName }: { laserNodeName: string }) {
  const laserNode = useLaserNode(laserNodeName);
  const [color, setColor] = useRgbColor({ r: 0.0, g: 0.0, b: 0.0 });
  // destinationX and destinationY are normalized to [0, 1]
  const [destinationX, setDestinationX] = useState<number>(0.0);
  const [destinationY, setDestinationY] = useState<number>(0.0);
  const [waypointDelayMs, setWaypointDelayMs] = useState<number>(1000.0);
  const [waypoints, setWaypoints] = useState<Waypoint[]>([]);

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
    // We intentionally omit laserNode object to avoid re-running on every
    // render. `.connected` is the signal we care about.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [laserNode.connected]);

  const laserDeviceState = convertLaserNodeDeviceState(laserNode);
  const disableButtons = laserDeviceState !== DeviceState.CONNECTED;

  let playbackButton = null;
  switch (laserNode.state.deviceState) {
    case LaserDeviceState.STOPPED:
      playbackButton = (
        <Button disabled={disableButtons} onClick={() => laserNode.play()}>
          Arm Laser
        </Button>
      );
      break;
    case LaserDeviceState.PLAYING:
      playbackButton = (
        <Button disabled={disableButtons} onClick={() => laserNode.stop()}>
          Disarm Laser
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
                onColorChange={(newColor) => {
                  setColor(newColor);
                  laserNode.setColor(newColor.r, newColor.g, newColor.b);
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
                  const value = parseFloat(str);
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
                  const value = parseFloat(str);
                  if (!isNaN(value)) {
                    setDestinationY(value);
                  }
                }}
              />
              <InputWithLabel
                className="flex-none w-20"
                type="number"
                id="waypointDelayMs"
                name="waypointDelayMs"
                label="Delay (ms)"
                step={100}
                value={waypointDelayMs}
                onChange={(str) => {
                  const value = parseFloat(str);
                  if (!isNaN(value)) {
                    setWaypointDelayMs(value);
                  }
                }}
              />
              <Button
                disabled={disableButtons}
                onClick={() =>
                  setWaypoints((prev) => [
                    ...prev,
                    {
                      x: destinationX,
                      y: destinationY,
                      delayMs: waypointDelayMs,
                    },
                  ])
                }
              >
                Add Waypoint
              </Button>
            </div>
            {waypoints.length > 0 && (
              <div className="flex flex-col gap-2 w-full">
                {waypoints.map((waypoint, index) => (
                  <div key={index} className="flex flex-row items-center gap-4">
                    <span className="w-6 text-sm">{index + 1}.</span>
                    <span className="flex-1 text-sm">
                      X: {waypoint.x.toFixed(3)}
                    </span>
                    <span className="flex-1 text-sm">
                      Y: {waypoint.y.toFixed(3)}
                    </span>
                    <span className="flex-1 text-sm">
                      Delay: {waypoint.delayMs}ms
                    </span>
                    <Button
                      variant="destructive"
                      onClick={() =>
                        setWaypoints((prev) =>
                          prev.filter((_, i) => i !== index),
                        )
                      }
                    >
                      Remove
                    </Button>
                  </div>
                ))}
              </div>
            )}
            {waypoints.length > 0 && (
              <div className="flex flex-row justify-center gap-4">
                <Button
                  disabled={disableButtons}
                  onClick={() => {
                    waypoints.forEach((waypoint) => {
                      laserNode.updatePath(
                        1,
                        { x: waypoint.x, y: waypoint.y },
                        delayMsToTimestamp(waypoint.delayMs),
                      );
                    });
                  }}
                >
                  Play
                </Button>
                <Button
                  disabled={disableButtons}
                  onClick={() => {
                    laserNode.clearPaths();
                  }}
                >
                  Stop/Reset
                </Button>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
