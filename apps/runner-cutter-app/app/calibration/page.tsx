import Controls from "@/components/calibration/controls";

export default function Calibration() {
  return (
    <main className="flex flex-col h-full gap-4">
      <h1 className="text-3xl font-bold">Calibration</h1>
      <div className="items-center justify-center">
        <Controls />
      </div>
    </main>
  );
}
