import Controls from "@/components/strobe/controls";

export default function Strobe() {
  return (
    <main className="flex flex-col h-full gap-4">
      <h1 className="text-3xl font-bold">Strobe Test</h1>
      <div className="items-center justify-center">
        <Controls />
      </div>
    </main>
  );
}
