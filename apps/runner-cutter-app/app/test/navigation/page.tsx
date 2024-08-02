import FramePreview from "@/components/camera/frame-preview";
import FurrowPercieverControls from "@/components/navigation/furrow_preciever_controls";
import GuidanceBrainControls from "@/components/navigation/guidance_brain_controls";

export default function Laser() {
  return (
    <main className="flex flex-col h-full gap-4">
      <h1 className="text-3xl font-bold">Navigation Test</h1>
      <div className="items-center justify-center">
        <FurrowPercieverControls></FurrowPercieverControls>
        <FramePreview topicName="/furrow0/debug_img"></FramePreview>
        <GuidanceBrainControls></GuidanceBrainControls>
      </div>
    </main>
  );
}
