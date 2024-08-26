"use client";

import React, { useEffect, useRef, useState } from "react";

export default function FramePreview({
  topicName,
  height = 360,
  onImageLoad,
  onImageClick,
  onImageSizeChanged,
  onComponentSizeChanged,
}: {
  topicName?: string;
  height?: number;
  quality?: number;
  onImageLoad?: React.EventHandler<
    React.SyntheticEvent<HTMLImageElement, Event>
  >;
  onImageClick?: React.MouseEventHandler<HTMLImageElement>;
  onImageSizeChanged?: (width: number, height: number) => void;
  onComponentSizeChanged?: (width: number, height: number) => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const peerConnection = useRef<RTCPeerConnection | null>(null);

  const [streamUrl, setStreamUrl] = useState<string>();
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [componentSize, setComponentSize] = useState({ width: 0, height: 0 });

  // Unfortunately, with SSR, this needed for code that should only run on client side
  useEffect(() => {
    console.log(`FRAME PREVIEW EFFECT: ${topicName}`);

    if (typeof window === "undefined") {
      return;
    }

    const initWebRTC = async () => {
      const pc = new RTCPeerConnection();
      peerConnection.current = pc;
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

      pc.onconnectionstatechange = (event) => {
        console.log(`onconnectionstatechange: ${pc.connectionState}`);
      };

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      const signalingServerUrl =
        process.env.NEXT_PUBLIC_VIDEO_SERVER_URL ??
        `http://${window.location.hostname}:8080`;
      const response = await fetch(
        `${signalingServerUrl}/offer?topic=${topicName}`,
        {
          method: "POST",
          body: JSON.stringify(pc.localDescription),
          headers: {
            "Content-Type": "application/json",
          },
        }
      );
      const answer = await response.json();
      await pc.setRemoteDescription(answer);
    };

    // Start the WebRTC connection
    initWebRTC();

    return () => {
      if (peerConnection.current) {
        peerConnection.current.close();
        peerConnection.current = null;
      }
    };
  }, [topicName]);

  /*
  useEffect(() => {
    const img = imageRef.current;
    if (img) {
      const updateSize = () => {
        if (imgRef.current) {
          const { naturalWidth, naturalHeight, offsetWidth, offsetHeight } =
            imgRef.current;
          if (
            imageSize.width !== naturalWidth ||
            imageSize.height !== naturalHeight
          ) {
            setImageSize({
              width: naturalWidth,
              height: naturalHeight,
            });
            onImageSizeChanged &&
              onImageSizeChanged(naturalWidth, naturalHeight);
          }
          if (
            componentSize.width !== offsetWidth ||
            componentSize.height !== offsetHeight
          ) {
            setComponentSize({ width: offsetWidth, height: offsetHeight });
            onComponentSizeChanged &&
              onComponentSizeChanged(offsetWidth, offsetHeight);
          }
        }
      };

      if (img.complete) {
        updateSize();
      }

      window.addEventListener("resize", updateSize);
      img.addEventListener("load", updateSize);

      return () => {
        window.removeEventListener("resize", updateSize);
        img.removeEventListener("load", updateSize);
      };
    }
  }, [
    height,
    streamUrl,
    imageSize,
    setImageSize,
    componentSize,
    setComponentSize,
    onImageSizeChanged,
    onComponentSizeChanged,
  ]);
  */

  return (
    <video
      ref={videoRef}
      className="w-auto max-h-full max-w-full"
      autoPlay={true}
      playsInline={true}
      muted={true}
      style={{ height }}
      // onLoad={onImageLoad}
      // onClick={onImageClick}
    />
  );
}
