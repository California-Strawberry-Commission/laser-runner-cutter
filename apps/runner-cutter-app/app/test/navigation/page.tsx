import FramePreview from "@/components/camera/frame-preview";
import FurrowPercieverControls from "@/components/navigation/furrow_perceiver_controls";
import GuidanceBrainControls from "@/components/navigation/guidance_brain_controls";

export default function Laser() {
  return (
    <main className="flex flex-col h-full gap-4">
      <h1 className="text-3xl font-bold">Navigation Test</h1>
      <div className="items-center justify-center">

        <FramePreview topicName="/furrow0/debug_img" />
        <FramePreview topicName="/furrow1/debug_img" />
        <FurrowPercieverControls />
        <GuidanceBrainControls />

      </div>
    </main>
  );
}
