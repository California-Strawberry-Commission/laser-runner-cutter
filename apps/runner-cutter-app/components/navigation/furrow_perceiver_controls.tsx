"use client";

import FramePreviewWithOverlay from "@/components/camera/frame-preview-with-overlay";
import { InputWithLabel } from "@/components/ui/input-with-label";
import useFurrowPerceiverNode from "@/lib/useFurrowPerceiverNode";

export default function FurrowPercieverControls() {
  const f1 = useFurrowPerceiverNode("/furrow_perceiver_forward");
  const f2 = useFurrowPerceiverNode("/furrow_perceiver_backward");

  return (
    <div className="flex items-center justify-center flex-col gap-4">
      <div className="flex flex-row gap-4">
        <FramePreviewWithOverlay
          className="w-[360px] h-[270px]"
          topicName="/furrow_perceiver_forward/debug_img"
          enableStream
        />
        <FramePreviewWithOverlay
          className="w-[360px] h-[270px]"
          topicName="/furrow_perceiver_backward/debug_img"
          enableStream
        />
      </div>
      <div className="flex flex-row gap-2 mb-2">
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
    </div>
  );
}
