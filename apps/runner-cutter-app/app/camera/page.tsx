import Controls from "@/components/camera/controls";

export default function Camera() {
  return (
    <main className="flex flex-col min-h-screen items-center justify-center gap-4">
      <h1 className="text-5xl font-bold text-center">Camera Control</h1>
      <Controls />
    </main>
  );
}
