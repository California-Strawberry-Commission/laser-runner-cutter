import Controls from "@/components/runner-cutter/controls";

export default function Cutter() {
  return (
    <main className="flex flex-col h-full gap-4">
      <h1 className="text-3xl font-bold">Laser Runner Cutter</h1>
      <div className="items-center justify-center">
        <Controls />
      </div>
    </main>
  );
}
