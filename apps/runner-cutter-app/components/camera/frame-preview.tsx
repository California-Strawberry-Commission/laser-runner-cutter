"use client";

import React, { useEffect, useRef, useState } from "react";

export default function FramePreview({
  topicName,
  height = 360,
  onLoaded,
  onComponentSizeChanged,
  onClick,
}: {
  topicName?: string;
  height?: number;
  quality?: number;
  onLoaded?: (videoWidth: number, videoHeight: number) => void;
  onComponentSizeChanged?: (width: number, height: number) => void;
  onClick?: React.MouseEventHandler<HTMLVideoElement>;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [componentSize, setComponentSize] = useState({ width: 0, height: 0 });
  const [isLoading, setIsLoading] = useState(true);

  // Unfortunately, with SSR, this needed for code that should only run on client side
  // TODO: move this to a separate custom hook
  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    const pc = new RTCPeerConnection({
      iceServers: [], // No STUN/TURN servers needed
    });
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

    const signalingServerUrl =
      process.env.NEXT_PUBLIC_VIDEO_SERVER_URL ??
      `ws://${window.location.hostname}:8080`;
    const socket = new WebSocket(`${signalingServerUrl}?topic=${topicName}`);

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

    return () => {
      socket.close();
      pc.close();
    };
  }, [topicName]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    const video = videoRef.current;
    if (video === null) {
      return;
    }

    const updateSize = () => {
      const video = videoRef.current;
      if (video === null) {
        return;
      }

      if (
        componentSize.width !== video.offsetWidth ||
        componentSize.height !== video.offsetHeight
      ) {
        setComponentSize({
          width: video.offsetWidth,
          height: video.offsetHeight,
        });
        onComponentSizeChanged &&
          onComponentSizeChanged(video.offsetWidth, video.offsetHeight);
      }
    };

    window.addEventListener("resize", updateSize);

    return () => {
      window.removeEventListener("resize", updateSize);
    };
  }, [height, componentSize, setComponentSize, onComponentSizeChanged]);

  return (
    <div className="relative bg-gray-600">
      {isLoading && (
        <div
          className="absolute"
          style={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            zIndex: 10,
          }}
        >
          <p className="text-white">Waiting for stream...</p>
        </div>
      )}
      <video
        ref={videoRef}
        className="w-auto max-h-full max-w-full object-fill"
        autoPlay={true}
        playsInline={true}
        muted={true}
        style={{ height }}
        onWaiting={() => {
          setIsLoading(true);
        }}
        onCanPlay={() => {
          setIsLoading(false);
        }}
        onPlaying={() => {
          setIsLoading(false);
        }}
        onLoadedMetadata={(event) => {
          const { videoWidth, videoHeight, offsetWidth, offsetHeight } =
            event.currentTarget;
          onLoaded && onLoaded(videoWidth, videoHeight);

          if (
            componentSize.width !== offsetWidth ||
            componentSize.height !== offsetHeight
          ) {
            setComponentSize({
              width: offsetWidth,
              height: offsetHeight,
            });
            onComponentSizeChanged &&
              onComponentSizeChanged(offsetWidth, offsetHeight);
          }
        }}
        onClick={onClick}
      />
    </div>
  );
}
