import Controls from "@/components/runner-cutter/controls";

export default function Home() {
  return (
    <main className="flex flex-col h-full gap-4">
      <div className="items-center justify-center">
        <Controls />
      </div>
    </main>
  );
}
