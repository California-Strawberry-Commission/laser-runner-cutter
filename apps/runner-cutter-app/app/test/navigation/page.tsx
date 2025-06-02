import FurrowPercieverControls from "@/components/navigation/furrow_perceiver_controls";
import GuidanceBrainControls from "@/components/navigation/guidance_brain_controls";
import {
  FURROW_PERCEIVER_BACKWARD_NODE_NAME,
  FURROW_PERCEIVER_FORWARD_NODE_NAME,
  GUIDANCE_BRAIN_NODE_NAME,
} from "@/constants/node_names";

export default function Navigation() {
  return (
    <main className="flex flex-col h-full gap-4">
      <h1 className="text-3xl font-bold">Navigation Test</h1>
      <div className="flex flex-col items-center justify-center gap-4">
        <FurrowPercieverControls
          forwardNodeName={FURROW_PERCEIVER_FORWARD_NODE_NAME}
          backwardNodeName={FURROW_PERCEIVER_BACKWARD_NODE_NAME}
        />
        <GuidanceBrainControls
          guidanceBrainNodeName={GUIDANCE_BRAIN_NODE_NAME}
        />
      </div>
    </main>
  );
}
