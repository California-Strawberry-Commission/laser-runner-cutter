"use client";

import { Input } from "@/components/ui/input";
import useROS from "@/lib/ros/useROS";
import useFurrowPerceiverNode from "@/lib/useFurrowPerceiverNode";
import useLaserNode from "@/lib/useFurrowPerceiverNode";
import { useState } from "react";
import { InputWithLabel } from "@/components/ui/input-with-label";
import { Button } from "@/components/ui/button";

export default function FurrowPercieverControls() {
  const { connected: rosConnected } = useROS();

  // TODO: add ability to select node name
  const [nodeName, setNodeName] = useState<string>("/furrow0");

  const { nodeInfo } = useFurrowPerceiverNode(nodeName);

  const disableButtons = !rosConnected || !nodeInfo.connected;

  let playbackButton = null;

  return (
    <div className="flex gap-2 mb-2">
      FPC
      <p>{nodeInfo.connected ? "CONN" : "DISCONN"}</p>
    </div>
  );
}
