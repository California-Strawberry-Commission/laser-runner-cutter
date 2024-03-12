"use client";

import { useEffect, useState } from "react";
import useROS from "@/lib/ros/useROS";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import ColorPicker from "@/components/laser/color-picker";
import useLaserNodeState, { LASER_STATES } from "@/lib/useLaserNode";

function hexToRgb(hexColor: string) {
  hexColor = hexColor.replace("#", "");
  const r = parseInt(hexColor.substring(0, 2), 16) / 255.0;
  const g = parseInt(hexColor.substring(2, 4), 16) / 255.0;
  const b = parseInt(hexColor.substring(4, 6), 16) / 255.0;
  return { r, g, b };
}

export default function Controls() {
  const { ros } = useROS();
  const [rosConnected, setRosConnected] = useState<boolean>(false);
  // TODO: add ability to select laser node name
  const [nodeName, setNodeName] = useState<string>("/laser0");
  const { laserState, addPoint, clearPoints, play, stop, setColor } =
    useLaserNodeState(nodeName);
  const [laserColor, setLaserColor] = useState<string>("#ff0000");
  const [x, setX] = useState<number>(0);
  const [y, setY] = useState<number>(0);

  useEffect(() => {
    // TODO: add listener for rosbridge connection state
    setRosConnected(ros.isConnected);
  }, []);

  let laserButton = null;
  if (laserState === 1) {
    laserButton = (
      <Button disabled={!rosConnected} onClick={() => play()}>
        Start Laser
      </Button>
    );
  } else if (laserState === 2) {
    laserButton = (
      <Button disabled={!rosConnected} onClick={() => stop()}>
        Stop Laser
      </Button>
    );
  }

  return (
    <div className="flex flex-col gap-4 items-center">
      <div className="flex flex-col items-center">
        <p className="text-center">{`Rosbridge: ${
          rosConnected ? "connected" : "disconnected"
        }`}</p>
        <p className="text-center">{`Laser (${nodeName}): ${LASER_STATES[laserState]}`}</p>
      </div>
      <div className="flex flex-row gap-4 items-center">
        <Input
          className="flex-none w-20"
          type="number"
          id="x"
          name="x"
          placeholder="x"
          step={0.1}
          value={x.toString()}
          onChange={(event) => {
            const value = Number(event.target.value);
            setX(isNaN(value) ? 0 : value);
          }}
        />
        <Input
          className="flex-none w-20"
          type="number"
          id="y"
          name="y"
          placeholder="y"
          step={0.1}
          value={y.toString()}
          onChange={(event) => {
            const value = Number(event.target.value);
            setY(isNaN(value) ? 0 : value);
          }}
        />
        <Button disabled={!rosConnected} onClick={() => addPoint(x, y)}>
          Add Point
        </Button>
        <Button disabled={!rosConnected} onClick={() => clearPoints()}>
          Clear Points
        </Button>
      </div>
      <div className="flex flex-row items-center gap-4">
        <ColorPicker
          color={laserColor}
          onColorChange={(color: string) => {
            setLaserColor(color);
            const rgb = hexToRgb(color);
            setColor(rgb.r, rgb.g, rgb.b);
          }}
        />
      </div>
      <div className="flex flex-row items-center gap-4">{laserButton}</div>
    </div>
  );
}
