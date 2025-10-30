import Controls from "@/components/camera/controls";
import { CAMERA_NODE_NAME, DETECTION_NODE_NAME } from "@/constants/node_names";

export default function Camera() {
  return (
    <main className="flex flex-col h-full gap-4">
      <h1 className="text-3xl font-bold">Camera Test</h1>
      <div className="items-center justify-center">
        <Controls cameraNodeName={CAMERA_NODE_NAME} detectionNodeName={DETECTION_NODE_NAME} />
      </div>
    </main>
  );
}
