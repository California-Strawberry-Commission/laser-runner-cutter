import Controls from "@/components/runner-cutter/controls";
import {
  CAMERA_NODE_NAME,
  CONTROL_NODE_NAME,
  DETECTION_NODE_NAME,
  LASER_NODE_NAME,
  LIFECYCLE_MANAGER_NODE_NAME,
} from "@/constants/node_names";

export default function Home() {
  return (
    <main className="flex flex-col h-full gap-4">
      <div className="items-center justify-center">
        <Controls
          lifecycleManagerNodeName={LIFECYCLE_MANAGER_NODE_NAME}
          cameraNodeName={CAMERA_NODE_NAME}
          detectionNodeName={DETECTION_NODE_NAME}
          controlNodeName={CONTROL_NODE_NAME}
          laserNodeName={LASER_NODE_NAME}
        />
      </div>
    </main>
  );
}
