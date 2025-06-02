import Controls from "@/components/settings/controls";

export default function Settings() {
  return (
    <main className="flex flex-col h-full gap-4">
      <h1 className="text-3xl font-bold">Settings</h1>
      <div className="items-center justify-center">
        <Controls />
      </div>
    </main>
  );
}
