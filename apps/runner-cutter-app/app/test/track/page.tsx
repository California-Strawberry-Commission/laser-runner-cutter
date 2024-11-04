import Controls from "@/components/track/controls";

export default function Track() {
  return (
    <main className="flex flex-col h-full gap-4">
      <h1 className="text-3xl font-bold">Track Test</h1>
      <div className="items-center justify-center">
        <Controls />
      </div>
    </main>
  );
}
