import Controls from "@/components/camera/controls";

export default function Camera() {
  return (
    <main className="flex flex-col h-full gap-4">
      <h1 className="text-3xl font-bold">Camera Test</h1>
      <div className="items-center justify-center">
        <Controls />
      </div>
    </main>
  );
}
