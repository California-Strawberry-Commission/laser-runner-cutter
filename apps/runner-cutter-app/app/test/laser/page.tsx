import Controls from "@/components/laser/controls";
import { LASER_NODE_NAME } from "@/constants/node_names";

export default function Laser() {
  return (
    <main className="flex flex-col h-full gap-4">
      <h1 className="text-3xl font-bold">Laser Test</h1>
      <div className="items-center justify-center">
        <Controls laserNodeName={LASER_NODE_NAME} />
      </div>
    </main>
  );
}
