"use client";

import { Input } from "@/components/ui/input";
import useROS from "@/lib/ros/useROS";
import useFurrowPerceiverNode from "@/lib/useFurrowPerceiverNode";
import useLaserNode from "@/lib/useFurrowPerceiverNode";
import { useState } from "react";
import { InputWithLabel } from "@/components/ui/input-with-label";
import { Button } from "@/components/ui/button";

export default function FurrowPercieverControls() {
  const f1 = useFurrowPerceiverNode("/furrow0");
  const f2 = useFurrowPerceiverNode("/furrow1");

  return (
    <div className="flex gap-2 mb-2">
      <InputWithLabel
        className="flex-none w-24 rounded-r-none"
        type="number"
        label="Forwards Offset"
        step={1}
        min={-300}
        max={300}
        value={f1.state.guidance_offset}
        onChange={(str) => f1.setGuidanceOffset(Number(str))}
      />
      <InputWithLabel
        className="flex-none w-24 rounded-r-none"
        type="number"
        label="Backward Offset"
        step={1}
        min={-300}
        max={300}
        value={f2.state.guidance_offset}
        onChange={(str) => f2.setGuidanceOffset(Number(str))}
      />
    </div>
  );
}
