import FurrowPercieverControls from "@/components/navigation/furrow_perceiver_controls";
import GuidanceBrainControls from "@/components/navigation/guidance_brain_controls";

export default function Navigation() {
  return (
    <main className="flex flex-col h-full gap-4">
      <h1 className="text-3xl font-bold">Navigation Test</h1>
      <div className="flex flex-col items-center justify-center gap-4">
        <FurrowPercieverControls />
        <GuidanceBrainControls />
      </div>
    </main>
  );
}
