export const dynamic = "force-dynamic";
import { AccessToken } from "livekit-server-sdk";
import { NextRequest, NextResponse } from "next/server";

export async function GET(req: NextRequest) {
  const room = req.nextUrl.searchParams.get("room");
  const identity = req.nextUrl.searchParams.get("identity");

  if (!room || !identity) {
    return NextResponse.json(
      { error: "room & identity required" },
      { status: 400 }
    );
  }

  const apiKey = process.env.LIVEKIT_API_KEY ?? "devkey";
  const apiSecret =
    process.env.LIVEKIT_API_SECRET ??
    "b5470c2bd57b77f98d05bb1d2be204696f83d68a12072804dc186b6eaeea1904";

  const accessToken = new AccessToken(apiKey, apiSecret, {
    identity,
    ttl: "10m",
  });

  // Subscribe-only permissions
  accessToken.addGrant({
    room,
    roomJoin: true,
    canSubscribe: true,
    canPublish: false,
    canPublishData: false,
  });

  const token = await accessToken.toJwt();
  return NextResponse.json(
    { token },
    { headers: { "Cache-Control": "no-store" } }
  );
}
