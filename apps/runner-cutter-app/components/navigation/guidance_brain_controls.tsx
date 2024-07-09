"use client";

import { Input } from "@/components/ui/input";
import useROS from "@/lib/ros/useROS";
import useLaserNode from "@/lib/useFurrowPerceiverNode";
import { useState } from "react";
import { InputWithLabel } from "@/components/ui/input-with-label";
import { Button } from "@/components/ui/button";
import useGuidanceBrainNode from "@/lib/useGuidanceBrainNode";

export default function GuidanceBrainControls() {
    const { nodeInfo: rosbridgeNodeInfo } = useROS();

    // TODO: add ability to select node name
    const [nodeName, setNodeName] = useState<string>("/brain0");

    const node = useGuidanceBrainNode(nodeName);

    const disableButtons = !rosbridgeNodeInfo.connected || !node.nodeInfo.connected;

    let playbackButton = null;

    <input type="number">lsls</input>

    return (<div className="flex gap-2 mb-2">
        <InputWithLabel
            className="flex-none w-24 rounded-r-none"
            type="number"
            id="exposure"
            label="Speed (ft/min)"
            step={10}
            value={0}
            onChange={(str) => {
                const value = Number(str);
            }}
        />
        <InputWithLabel
            className="flex-none w-24 rounded-r-none"
            type="number"
            id="exposure"
            label="P Gain"
            step={10}
            value={0}
            onChange={(str) => {
                const value = Number(str);
            }}
        />
        <InputWithLabel
            className="flex-none w-24 rounded-r-none"
            type="number"
            id="exposure"
            label="I Gain"
            step={10}
            value={0}
            onChange={(str) => {
                const value = Number(str);
            }}
        />
        <InputWithLabel
            className="flex-none w-24 rounded-r-none"
            type="number"
            id="exposure"
            label="D Gain"
            step={10}
            value={0}
            onChange={(str) => {
                const value = Number(str);
            }}
        />

        <Button
            disabled={disableButtons}
            onClick={() => {
                node.setActive(!node.nodeState.active)
            }}
        >
            {node.nodeState.active ? "Deactivate" : "Activate"}
        </Button>

        <p>{node.nodeInfo.connected ? "CONN" : "DISCONN"}</p>
    </div>);
}
