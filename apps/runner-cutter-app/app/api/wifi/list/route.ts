export const dynamic = "force-dynamic";

import * as wifi from "node-wifi";

// Absolutely necessary even to set interface to null
wifi.init({ iface: null });

type NetworkEntry = {
  ssid: string;
  signal: number;
  security: string;
  connected: boolean;
}

/**
 * GET /api/wifi/list
 *
 * Scans for nearby Wi-Fi networks and returns a deduplicated list,
 * keeping the highest-quality entry per SSID.
 */
export async function GET(_request: Request) {
  try {
    const currentConnections = await wifi.getCurrentConnections();
    const connectedSsids = new Set(currentConnections.map((conn) => conn.ssid));
    const networks = await wifi.scan();

    const result: Record<string, NetworkEntry> = {};
    networks.forEach((network) => {
      const { ssid, quality, security } = network;
      if (ssid.trim() && (!(ssid in result) || quality > result[ssid].signal)) {
        result[ssid] = {
          ssid,
          signal: quality,
          security,
          connected: connectedSsids.has(ssid),
        };
      }
    });

    return Response.json(Object.values(result));
  } catch (err) {
    console.error("Wi-Fi list failed:", err);
    return Response.json(
      { error: err instanceof Error ? err.message : "Scan failed" },
      { status: 500 }
    );
  }
}
