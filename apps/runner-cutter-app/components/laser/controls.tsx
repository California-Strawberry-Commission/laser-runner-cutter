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
import { useEffect, useState } from "react";

function hexToRgb(hexColor: string) {
  hexColor = hexColor.replace("#", "");
  const r = parseInt(hexColor.substring(0, 2), 16) / 255.0;
  const g = parseInt(hexColor.substring(2, 4), 16) / 255.0;
  const b = parseInt(hexColor.substring(4, 6), 16) / 255.0;
  if (isNaN(r) || isNaN(g) || isNaN(b)) {
    return null;
  } else {
    return { r, g, b };
  }
}

function rgbToHex(r: number, g: number, b: number) {
  return (
    "#" +
    [r, g, b]
      .map((x) =>
        Math.round(x * 255)
          .toString(16)
          .padStart(2, "0")
      )
      .join("")
      .toLowerCase()
  );
}

export default function Controls() {
  // TODO: add ability to select node name
  const [nodeName, setNodeName] = useState<string>("/laser0");
  const laserNode = useLaserNode(nodeName);
  const [laserColor, setLaserColor] = useState<string>("#000000");
  const [startX, setStartX] = useState<string>("0.0");
  const [startY, setStartY] = useState<string>("0.0");
  const [endX, setEndX] = useState<string>("1.0");
  const [endY, setEndY] = useState<string>("1.0");
  const [durationMs, setDurationMs] = useState<string>("1000");

  useEffect(() => {
    async function fetchParams() {
      if (laserNode.connected) {
        const result = await laserNode.getColor();
        if (result.length >= 3) {
          setLaserColor(rgbToHex(result[0], result[1], result[2]));
        }
      }
    }

    fetchParams();
  }, [laserNode.connected]);

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
            <div className="flex flex-row gap-4 items-center">
              <InputWithLabel
                className="flex-none w-16"
                type="number"
                id="startX"
                name="startX"
                label="Start X"
                step={0.1}
                value={startX}
                onChange={(str) => {
                  const value = Number(str);
                  if (!isNaN(value)) {
                    setStartX(str);
                  }
                }}
              />
              <InputWithLabel
                className="flex-none w-16"
                type="number"
                id="startY"
                name="startY"
                label="Start Y"
                step={0.1}
                value={startY}
                onChange={(str) => {
                  const value = Number(str);
                  if (!isNaN(value)) {
                    setStartY(str);
                  }
                }}
              />
              <InputWithLabel
                className="flex-none w-16"
                type="number"
                id="endX"
                name="endX"
                label="End X"
                step={0.1}
                value={endX}
                onChange={(str) => {
                  const value = Number(str);
                  if (!isNaN(value)) {
                    setEndX(str);
                  }
                }}
              />
              <InputWithLabel
                className="flex-none w-16"
                type="number"
                id="endY"
                name="endY"
                label="End Y"
                step={0.1}
                value={endY}
                onChange={(str) => {
                  const value = Number(str);
                  if (!isNaN(value)) {
                    setEndY(str);
                  }
                }}
              />
              <InputWithLabel
                className="flex-none w-20"
                type="number"
                id="durationMs"
                name="durationMs"
                label="Duration (ms)"
                step={0.1}
                value={durationMs}
                onChange={(str) => {
                  const value = Number(str);
                  if (!isNaN(value)) {
                    setDurationMs(str);
                  }
                }}
              />
              <Button
                disabled={disableButtons}
                onClick={() =>
                  laserNode.setPath(
                    { x: Number(startX), y: Number(startY) },
                    { x: Number(endX), y: Number(endY) },
                    Number(durationMs)
                  )
                }
              >
                Set Path
              </Button>
              <Button
                disabled={disableButtons}
                onClick={() => laserNode.clearPath()}
              >
                Clear Path
              </Button>
            </div>
            <div className="flex flex-row items-center gap-4">
              <ColorPicker
                color={laserColor}
                onColorChange={(color: string) => {
                  setLaserColor(color);
                  const rgb = hexToRgb(color);
                  if (rgb) {
                    laserNode.setColor(rgb.r, rgb.g, rgb.b);
                  }
                }}
              />
              {playbackButton}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
