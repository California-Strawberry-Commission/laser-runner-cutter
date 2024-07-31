"use client";

import { Input } from "@/components/ui/input";
import useROS from "@/lib/ros/useROS";
import useLaserNode from "@/lib/useFurrowPerceiverNode";
import { useEffect, useState } from "react";
import { InputWithLabel } from "@/components/ui/input-with-label";
import { Button } from "@/components/ui/button";
import useGuidanceBrainNode from "@/lib/useGuidanceBrainNode";

export default function GuidanceBrainControls() {
    const node = useGuidanceBrainNode("/brain0");

    const disableButtons = !node.connected;

    return (<div className="flex flex-col gap-2 mb-2 mt-2">
        <div className="flex gap-2">
            <InputWithLabel
                className="flex-none w-24 rounded-r-none"
                type="number"
                id="exposure"
                label="Speed (ft/min)"
                step={1}
                min={0}
                max={30}
                value={node.state.speed}
                onChange={(str) => node.setSpeed(Number(str))}
            />
            <InputWithLabel
                className="flex-none w-24 rounded-r-none"
                type="number"
                id="exposure"
                label="P Gain"
                step={5}
                min={0}
                max={100}
                value={node.state.follower_pid.p}
                onChange={(str) => node.setP(Number(str))}
            />

            {/* <InputWithLabel
            className="flex-none w-24 rounded-r-none"
            type="number"
            id="exposure"
            label="I Gain"
            step={10}
            value={node.state.follower_pid.i}
            onChange={(str) => node.setI(Number(str))}
        />
        <InputWithLabel
            className="flex-none w-24 rounded-r-none"
            type="number"
            id="exposure"
            label="D Gain"
            step={10}
            value={node.state.follower_pid.d}
            onChange={(str) => node.setD(Number(str))}
        /> */}

            <Button
                disabled={disableButtons}
                onClick={() => {
                    node.setActive(!node.state.guidance_active)
                }}
            >
                {node.state.guidance_active ? "Deactivate" : "Activate"}
            </Button>

        </div>


        <p>{node.connected ? "CONN" : "DISCONN"}</p>

        <pre>
            {JSON.stringify(node.state, null, 2)}
        </pre>
    </div>);
}
