import Controls from "@/components/laser/controls";

export default function Laser() {
  return (
    <main className="flex flex-col min-h-screen items-center justify-center p-4 gap-4">
      <h1 className="text-5xl font-bold text-center">Laser Test</h1>
      <Controls />
    </main>
  );
}