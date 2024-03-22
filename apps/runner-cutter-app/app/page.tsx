import Controls from "@/components/controls";

export default function Home() {
  return (
    <main className="flex flex-col min-h-screen items-center justify-center p-4">
      <h1 className="text-5xl font-bold text-center">Laser Runner Cutter</h1>
      <Controls />
    </main>
  );
}
