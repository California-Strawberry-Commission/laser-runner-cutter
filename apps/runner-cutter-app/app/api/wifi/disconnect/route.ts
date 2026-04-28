export const dynamic = "force-dynamic";

import * as wifi from "node-wifi";

// Absolutely necessary even to set interface to null
wifi.init({ iface: null });

/**
 * POST /api/wifi/disconnect
 *
 * Disconnects the host machine from its current Wi-Fi network via `node-wifi`.
 */
export async function POST(_request: Request) {
  try {
    await wifi.disconnect();
    return new Response(null, { status: 200 });
  } catch (err) {
    console.error("Wi-Fi disconnect failed:", err);
    return Response.json(
      { error: err instanceof Error ? err.message : "Disconnection failed" },
      { status: 500 },
    );
  }
}
