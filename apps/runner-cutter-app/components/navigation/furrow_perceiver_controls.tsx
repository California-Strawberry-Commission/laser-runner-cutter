"use client";

import useFurrowPerceiverNode from "@/lib/useFurrowPerceiverNode";
import { InputWithLabel } from "@/components/ui/input-with-label";

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
        onChange={(str) => {
          const value = Number(str);
          if (!isNaN(value)) {
            f1.setGuidanceOffset(value);
          }
        }}
      />
      <InputWithLabel
        className="flex-none w-24 rounded-r-none"
        type="number"
        label="Backward Offset"
        step={1}
        min={-300}
        max={300}
        value={f2.state.guidance_offset}
        onChange={(str) => {
          const value = Number(str);
          if (!isNaN(value)) {
            f2.setGuidanceOffset(value);
          }
        }}
      />
    </div>
  );
}
