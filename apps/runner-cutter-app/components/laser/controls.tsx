"use client";

import { useContext, useEffect, useState } from "react";
import ROSContext from "@/lib/ros/ROSContext";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import ColorPicker from "@/components/laser/color-picker";
import useLaserNode, { LASER_STATES } from "@/lib/useLaserNode";

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

export default function Controls() {
  const ros = useContext(ROSContext);
  const [rosConnected, setRosConnected] = useState<boolean>(false);
  // TODO: add ability to select laser node name
  const [nodeName, setNodeName] = useState<string>("/laser0");
  const {
    nodeConnected,
    laserState,
    addPoint,
    clearPoints,
    play,
    stop,
    setColor,
  } = useLaserNode(nodeName);
  const [laserColor, setLaserColor] = useState<string>("#ff0000");
  const [x, setX] = useState<string>("0");
  const [y, setY] = useState<string>("0");

  useEffect(() => {
    ros.onStateChange(() => {
      setRosConnected(ros.ros.isConnected);
    });
    setRosConnected(ros.ros.isConnected);
  }, [setRosConnected]);

  let laserButton = null;
  if (laserState === 1) {
    laserButton = (
      <Button disabled={!rosConnected || !nodeConnected} onClick={() => play()}>
        Start Laser
      </Button>
    );
  } else if (laserState === 2) {
    laserButton = (
      <Button disabled={!rosConnected || !nodeConnected} onClick={() => stop()}>
        Stop Laser
      </Button>
    );
  } else {
    laserButton = <Button disabled={true}>Laser Disconnected</Button>;
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
          value={x}
          onChange={(event) => {
            const value = Number(event.target.value);
            if (!isNaN(value)) {
              setX(event.target.value);
            }
          }}
        />
        <Input
          className="flex-none w-20"
          type="number"
          id="y"
          name="y"
          placeholder="y"
          step={0.1}
          value={y}
          onChange={(event) => {
            const value = Number(event.target.value);
            if (!isNaN(value)) {
              setY(event.target.value);
            }
          }}
        />
        <Button
          disabled={!rosConnected || !nodeConnected}
          onClick={() => addPoint(Number(x), Number(y))}
        >
          Add Point
        </Button>
        <Button
          disabled={!rosConnected || !nodeConnected}
          onClick={() => clearPoints()}
        >
          Clear Points
        </Button>
      </div>
      <div className="flex flex-row items-center gap-4">
        <ColorPicker
          color={laserColor}
          onColorChange={(color: string) => {
            setLaserColor(color);
            const rgb = hexToRgb(color);
            if (rgb) {
              setColor(rgb.r, rgb.g, rgb.b);
            }
          }}
        />
      </div>
      <div className="flex flex-row items-center gap-4">{laserButton}</div>
    </div>
  );
}
