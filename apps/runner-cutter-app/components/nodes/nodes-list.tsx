"use client";

import NodeCards from "@/components/nodes/node-cards";
import useROS from "@/lib/ros/useROS";
import useCameraNode from "@/lib/useCameraNode";
import useControlNode from "@/lib/useControlNode";
import useLaserNode from "@/lib/useLaserNode";
import { useMemo } from "react";

export type Node = {
  name: string;
  connected: boolean;
  status: {};
};

export default function NodesList() {
  const { connected: rosConnected } = useROS();
  const controlNode = useControlNode("/control0");
  const cameraNode = useCameraNode("/camera0");
  const laserNode = useLaserNode("/laser0");

  const nodeInfos = useMemo(() => {
    const rosbridgeNodeInfo = {
      name: "Rosbridge",
      connected: rosConnected,
    };
    return [rosbridgeNodeInfo, controlNode, cameraNode, laserNode];
  }, [rosConnected, controlNode, cameraNode, laserNode]);

  return <NodeCards nodeInfos={nodeInfos} />;
}
