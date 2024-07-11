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


    const [pid, setPid] = useState(node.state.follower_pid);
    useEffect(() => {
        setPid(node.state.follower_pid)
    }, [node.state.follower_pid])


    const disableButtons = !node.connected;

    <input type="number">lsls</input>

    return (<div className="flex gap-2 mb-2">
        <InputWithLabel
            className="flex-none w-24 rounded-r-none"
            type="number"
            id="exposure"
            label="Speed (ft/min)"
            step={10}
            value={pid.p}
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
            value={pid.p}
            onChange={(str) => {
                setPid({...pid, p: Number(str)})
            }}
        />
        <InputWithLabel
            className="flex-none w-24 rounded-r-none"
            type="number"
            id="exposure"
            label="I Gain"
            step={10}
            value={pid.i}
            onChange={(str) => {
                setPid({...pid, i: Number(str)})
            }}
        />
        <InputWithLabel
            className="flex-none w-24 rounded-r-none"
            type="number"
            id="exposure"
            label="D Gain"
            step={10}
            value={pid.d}
            onChange={(str) => {
                setPid({...pid, d: Number(str)})
            }}
        />


        <Button
            disabled={disableButtons}
            onClick={() => {
                node.setPID(pid.p, pid.i, pid.d);
            }}
        >
            set pid
        </Button>
        
        <Button
            disabled={disableButtons}
            onClick={() => {
                node.setActive(!node.state.guidance_active)
            }}
        >
            {node.state.guidance_active ? "Deactivate" : "Activate"}
        </Button>

        <p>{node.connected ? "CONN" : "DISCONN"}</p>
    </div>);
}
