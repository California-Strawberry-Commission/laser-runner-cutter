import NetworkSelector from "@/components/wifi/network-selector";

export default function Wifi() {
  return (
    <main className="flex flex-col min-h-screen p-4 gap-4">
      <h1 className="text-3xl font-bold">Wi-Fi Manager</h1>
      <NetworkSelector />
    </main>
  );
}
