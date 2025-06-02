import Controls from "@/components/settings/controls";
import { CAMERA_NODE_NAME, CONTROL_NODE_NAME } from "@/constants/node_names";

export default function Settings() {
  return (
    <main className="flex flex-col h-full gap-4">
      <h1 className="text-3xl font-bold">Settings</h1>
      <div className="items-center justify-center">
        <Controls
          cameraNodeName={CAMERA_NODE_NAME}
          controlNodeName={CONTROL_NODE_NAME}
        />
      </div>
    </main>
  );
}
