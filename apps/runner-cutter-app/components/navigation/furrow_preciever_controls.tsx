"use client";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import useROS from "@/lib/ros/useROS";
import useFurrowPerceiverNode from "@/lib/useFurrowPerceiverNode";
import useLaserNode from "@/lib/useFurrowPerceiverNode";
import { useState } from "react";

export default function Controls() {
  const { nodeInfo: rosbridgeNodeInfo } = useROS();
  // TODO: add ability to select node name
  const [nodeName, setNodeName] = useState<string>("/furrow0");
  const { nodeInfo } = useFurrowPerceiverNode(nodeName);

  const disableButtons = !rosbridgeNodeInfo.connected || !nodeInfo.connected;

  let playbackButton = null;

  return (<div>
    <p>{nodeInfo.connected ? "CONN" : "DISCONN"}</p>
  </div>);
}
