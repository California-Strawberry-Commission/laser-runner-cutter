import Controls from "@/components/setup/controls";

export default function Nodes() {
  return (
    <main className="flex flex-col h-full gap-4">
      <h1 className="text-3xl font-bold">Setup</h1>
      <div className="items-center justify-center">
        <Controls />
      </div>
    </main>
  );
}
