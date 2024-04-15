import NodesList from "@/components/nodes/nodes-list";

export default function Nodes() {
  return (
    <main className="flex flex-col h-full gap-4">
      <h1 className="text-3xl font-bold">Nodes</h1>
      <NodesList />
    </main>
  );
}
