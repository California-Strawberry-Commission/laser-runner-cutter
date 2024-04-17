import Controls from "@/components/aim/controls";

export default function Aim() {
  return (
    <main className="flex flex-col h-full gap-4">
      <h1 className="text-3xl font-bold">Aim Test</h1>
      <div className="items-center justify-center">
        <Controls />
      </div>
    </main>
  );
}
