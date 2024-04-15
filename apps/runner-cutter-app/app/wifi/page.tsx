import NetworkSelector from "@/components/wifi/network-selector";

export default function Wifi() {
  return (
    <main className="flex flex-col h-full gap-4">
      <h1 className="text-3xl font-bold">Wi-Fi Manager</h1>
      <NetworkSelector />
    </main>
  );
}
