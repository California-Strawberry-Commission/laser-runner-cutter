export const dynamic = "force-dynamic";
const wifi = require("node-wifi");

// Initialize wifi module
// Absolutely necessary even to set interface to null
wifi.init({
  iface: null, // network interface, choose a random wifi interface if set to null
});

export async function POST(request: Request) {
  try {
    await wifi.disconnect();
    return new Response(null, {
      status: 200,
    });
  } catch (error) {
    return new Response(null, {
      status: 500,
    });
  }
}
