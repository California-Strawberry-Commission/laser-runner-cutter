export const dynamic = "force-dynamic";

import * as wifi from "node-wifi";

// Absolutely necessary even to set interface to null
wifi.init({ iface: null });

/**
 * POST /api/wifi/connect
 *
 * Connects the host machine to a Wi-Fi network via `node-wifi`.
 */
export async function POST(request: Request) {
  const { ssid, password } = await request.json();

  if (!ssid) {
    return Response.json({ error: "ssid is required" }, { status: 400 });
  }

  try {
    await wifi.connect({ ssid, password });
    return new Response(null, { status: 200 });
  } catch (err) {
    console.error("Wi-Fi connect failed:", err);
    return Response.json(
      { error: err instanceof Error ? err.message : "Connection failed" },
      { status: 400 },
    );
  }
}
