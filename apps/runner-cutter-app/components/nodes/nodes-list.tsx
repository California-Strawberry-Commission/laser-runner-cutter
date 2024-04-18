"use client";

import NodeCards from "@/components/nodes/node-cards";
import useROS from "@/lib/ros/useROS";
import useCameraNode from "@/lib/useCameraNode";
import useControlNode from "@/lib/useControlNode";
import useLaserNode from "@/lib/useLaserNode";

export type Node = {
  name: string;
  connected: boolean;
  status: {};
};

export default function NodesList() {
  const { nodeInfo: rosbridgeNodeInfo } = useROS();
  const { nodeInfo: controlNodeInfo } = useControlNode("/control0");
  const { nodeInfo: cameraNodeInfo } = useCameraNode("/camera0");
  const { nodeInfo: laserNodeInfo } = useLaserNode("/laser0");

  const nodeInfos = [
    rosbridgeNodeInfo,
    controlNodeInfo,
    cameraNodeInfo,
    laserNodeInfo,
  ];

  return <NodeCards nodeInfos={nodeInfos} />;
}
