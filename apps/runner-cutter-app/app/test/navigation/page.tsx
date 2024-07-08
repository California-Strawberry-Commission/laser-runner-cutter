import Controls from "@/components/navigation/furrow_preciever_controls";

export default function Laser() {
  return (
    <main className="flex flex-col h-full gap-4">
      <h1 className="text-3xl font-bold">Navigation Test</h1>
      <div className="items-center justify-center">
        <Controls></Controls>
      </div>
    </main>
  );
}
