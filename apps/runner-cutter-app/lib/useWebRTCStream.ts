import { useEffect, useRef, useState } from "react";

export default function useWebRTCStream(
  topicName: string = "",
  enableStream: boolean = false
) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [connected, setConnected] = useState(false);

  // Unfortunately, with SSR, this needed for code that should only run on client side. Otherwise
  // we will get an error when enableStream is true on initial render since the server and
  // client "src" values will not match.
  useEffect(() => {
    if (typeof window !== "undefined" && enableStream && topicName) {
      const signalingServerUrl =
        process.env.NEXT_PUBLIC_VIDEO_SERVER_URL ??
        `ws://${window.location.hostname}:8080`;
      const streamUrl = `${signalingServerUrl}?topic=${topicName}`;
      const socket = new WebSocket(streamUrl);
      const pc = new RTCPeerConnection({
        iceServers: [], // No STUN/TURN servers needed
      });

      socket.onmessage = async (event) => {
        const data = JSON.parse(event.data);

        if (data.answer) {
          await pc.setRemoteDescription(data.answer);
        } else if (data.candidate) {
          await pc.addIceCandidate(data.candidate);
        }
      };

      socket.onopen = async () => {
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        socket.send(JSON.stringify({ offer: pc.localDescription }));
      };

      // From https://stackoverflow.com/questions/50002099/webrtc-one-way-video-call
      pc.addTransceiver("video");
      // This step seems to be optional:
      pc.getTransceivers().forEach((t) => (t.direction = "recvonly"));

      pc.ontrack = (event) => {
        if (event.track.kind === "video") {
          console.log(`Video track received`);
          if (videoRef.current) {
            videoRef.current.srcObject = event.streams[0];
          }
          setConnected(true);
        }
      };

      pc.oniceconnectionstatechange = (event) => {
        console.log(`oniceconnectionstatechange: ${pc.iceConnectionState}`);
      };

      pc.onsignalingstatechange = (event) => {
        console.log(`onsignalingstatechange: ${pc.signalingState}`);
      };

      const startTime = Date.now();
      pc.onconnectionstatechange = (event) => {
        console.log(`onconnectionstatechange: ${pc.connectionState}`);
        if (pc.iceConnectionState === "connected") {
          console.log(`Connected in ${Date.now() - startTime} ms`);
        }
      };

      return () => {
        socket.close();
        pc.close();
      };
    }
  }, [topicName, enableStream]);

  return { videoRef, connected };
}
