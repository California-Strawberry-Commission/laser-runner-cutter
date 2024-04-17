import Controls from "@/components/laser/controls";

export default function Laser() {
  return (
    <main className="flex flex-col h-full gap-4">
      <h1 className="text-3xl font-bold">Laser Test</h1>
      <div className="items-center justify-center">
        <Controls />
      </div>
    </main>
  );
}
