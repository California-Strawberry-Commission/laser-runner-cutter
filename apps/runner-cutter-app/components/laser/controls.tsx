"use client";

import ColorPicker from "@/components/laser/color-picker";
import NodeCards from "@/components/nodes/node-cards";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import useROS from "@/lib/ros/useROS";
import useLaserNode from "@/lib/useLaserNode";
import { useMemo, useState } from "react";

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
  const { nodeInfo: rosbridgeNodeInfo } = useROS();
  // TODO: add ability to select node name
  const [nodeName, setNodeName] = useState<string>("/laser0");
  const { nodeInfo, laserState, addPoint, clearPoints, play, stop, setColor } =
    useLaserNode(nodeName);
  const [laserColor, setLaserColor] = useState<string>("#ff0000");
  const [x, setX] = useState<string>("0");
  const [y, setY] = useState<string>("0");

  const nodeInfos = useMemo(() => {
    return [rosbridgeNodeInfo, nodeInfo];
  }, [rosbridgeNodeInfo, nodeInfo]);
  const disableButtons = !rosbridgeNodeInfo.connected || !nodeInfo.connected;

  let laserButton = null;
  if (laserState === "stopped") {
    laserButton = (
      <Button disabled={disableButtons} onClick={() => play()}>
        Start Laser
      </Button>
    );
  } else if (laserState === "playing") {
    laserButton = (
      <Button disabled={disableButtons} onClick={() => stop()}>
        Stop Laser
      </Button>
    );
  } else {
    laserButton = <Button disabled={true}>Laser Disconnected</Button>;
  }

  return (
    <div className="flex flex-col gap-4 items-center">
      <NodeCards nodeInfos={nodeInfos} />
      <div className="flex flex-row gap-4 items-center">
        <Input
          className="flex-none w-20"
          type="number"
          id="x"
          name="x"
          placeholder="x"
          step={0.1}
          value={x}
          onChange={(str) => {
            const value = Number(str);
            if (!isNaN(value)) {
              setX(str);
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
          onChange={(str) => {
            const value = Number(str);
            if (!isNaN(value)) {
              setY(str);
            }
          }}
        />
        <Button
          disabled={disableButtons}
          onClick={() => addPoint(Number(x), Number(y))}
        >
          Add Point
        </Button>
        <Button disabled={disableButtons} onClick={() => clearPoints()}>
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
