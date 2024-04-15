export const dynamic = "force-dynamic";
const wifi = require("node-wifi");

// Initialize wifi module
// Absolutely necessary even to set interface to null
wifi.init({
  iface: null, // network interface, choose a random wifi interface if set to null
});

export async function GET(request: Request) {
  try {
    const currentConnections = await wifi.getCurrentConnections();
    const connectedSsids = new Set(
      currentConnections.map((conn: any) => conn.ssid)
    );
    const networks = await wifi.scan();
    const result: any = {};
    networks.forEach((network: any) => {
      const { ssid, quality, security } = network;
      if (
        ssid.trim().length > 0 &&
        (!(ssid in result) || quality > result[ssid].quality)
      ) {
        result[ssid] = {
          ssid,
          signal: quality,
          security,
          connected: connectedSsids.has(ssid),
        };
      }
    });
    return Response.json(Object.values(result));
  } catch (error) {
    return new Response(null, {
      status: 500,
    });
  }
}
