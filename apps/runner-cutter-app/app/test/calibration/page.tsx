import Controls from "@/components/calibration/controls";
import {
  CAMERA_NODE_NAME,
  CONTROL_NODE_NAME,
  LASER_NODE_NAME,
} from "@/constants/node_names";

export default function Calibration() {
  return (
    <main className="flex flex-col h-full gap-4">
      <h1 className="text-3xl font-bold">Calibration/Aim Test</h1>
      <div className="items-center justify-center">
        <Controls
          cameraNodeName={CAMERA_NODE_NAME}
          controlNodeName={CONTROL_NODE_NAME}
          laserNodeName={LASER_NODE_NAME}
        />
      </div>
    </main>
  );
}
