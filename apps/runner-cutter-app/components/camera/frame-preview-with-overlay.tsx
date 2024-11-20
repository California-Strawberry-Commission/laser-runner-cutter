"use client";

import { cn } from "@/lib/utils";
import React, { useCallback, useEffect, useRef, useState } from "react";
import { Track, TrackState } from "@/lib/useControlNode";

function getVideoSizeAndOffset(videoElement: HTMLVideoElement) {
  const rect = videoElement.getBoundingClientRect();

  // The dimensions of the video element (including the padding caused by object-contain)
  const containerWidth = rect.width;
  const containerHeight = rect.height;

  // Get the actual video size based on its intrinsic dimensions and container's aspect ratio
  const naturalWidth = videoElement.videoWidth;
  const naturalHeight = videoElement.videoHeight;
  const aspectRatio = naturalWidth / naturalHeight;

  let renderedWidth, renderedHeight;
  if (containerWidth / containerHeight > aspectRatio) {
    // Video is constrained by height
    renderedHeight = containerHeight;
    renderedWidth = renderedHeight * aspectRatio;
  } else {
    // Video is constrained by width
    renderedWidth = containerWidth;
    renderedHeight = renderedWidth / aspectRatio;
  }

  // Calculate the top-left corner of the video inside the container
  const offsetX = (containerWidth - renderedWidth) / 2;
  const offsetY = (containerHeight - renderedHeight) / 2;

  return { renderedWidth, renderedHeight, offsetX, offsetY };
}

export default function FramePreviewWithOverlay({
  topicName,
  enableStream = false,
  onImageClick,
  enableOverlay = false,
  overlayText,
  overlayNormalizedRect,
  overlayTracks,
  className,
}: {
  topicName?: string;
  enableStream?: boolean;
  onImageClick?: (normalizedX: number, normalizedY: number) => void;
  enableOverlay?: boolean;
  overlayText?: string;
  overlayNormalizedRect?: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  overlayTracks?: Track[];
  className?: string;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isLoading, setIsLoading] = useState(true);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [canvasStyle, setCanvasStyle] = useState({
    width: 0,
    height: 0,
    top: 0,
    left: 0,
  });

  let streamUrl = "";
  if (typeof window !== "undefined" && enableStream && topicName) {
    const signalingServerUrl =
      process.env.NEXT_PUBLIC_VIDEO_SERVER_URL ??
      `ws://${window.location.hostname}:8080`;
    streamUrl = `${signalingServerUrl}?topic=${topicName}`;
  }

  const renderOverlay = streamUrl && enableOverlay && !isLoading;

  // Unfortunately, with SSR, this needed for code that should only run on client side
  // TODO: move this to a separate custom hook
  useEffect(() => {
    if (typeof window === "undefined" || !streamUrl) {
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

    const socket = new WebSocket(streamUrl);

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
  }, [streamUrl]);

  // Handle video click and calculate normalized coordinates
  const handleVideoClick = useCallback(
    (e: React.MouseEvent<HTMLVideoElement, MouseEvent>) => {
      if (!videoRef.current || !streamUrl) {
        return;
      }

      const videoElement = videoRef.current;
      const { renderedWidth, renderedHeight, offsetX, offsetY } =
        getVideoSizeAndOffset(videoElement);
      const rect = videoElement.getBoundingClientRect();

      // Get the click position relative to the image
      const clickX = e.clientX - rect.left - offsetX;
      const clickY = e.clientY - rect.top - offsetY;

      // Normalize the coordinates
      const normalizedX = clickX / renderedWidth;
      const normalizedY = clickY / renderedHeight;

      // Only call the callback if the click was within the image bounds
      if (
        onImageClick &&
        normalizedX >= 0 &&
        normalizedX <= 1 &&
        normalizedY >= 0 &&
        normalizedY <= 1
      ) {
        onImageClick(normalizedX, normalizedY);
      }
    },
    [streamUrl, onImageClick]
  );

  // Update canvas position and size to exactly match the rendered image
  const updateCanvasSize = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) {
      return;
    }

    const videoElement = videoRef.current;
    const { renderedWidth, renderedHeight, offsetX, offsetY } =
      getVideoSizeAndOffset(videoElement);
    console.log(
      `setCanvasStyle(renderedWidth: ${renderedWidth}, renderedHeight: ${renderedHeight}, offsetX: ${offsetX}, offsetY: ${offsetY})`
    );
    setCanvasStyle({
      width: renderedWidth,
      height: renderedHeight,
      top: offsetY,
      left: offsetX,
    });

    const canvasElement = canvasRef.current;
    const ctx = canvasElement.getContext("2d");
    if (!ctx) {
      return;
    }

    canvasElement.width = renderedWidth;
    canvasElement.height = renderedHeight;

    // Clear the canvas
    ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    // Draw rect first, as it requires clearing part of the canvas
    if (
      overlayNormalizedRect &&
      overlayNormalizedRect.width > 0.0 &&
      overlayNormalizedRect.height > 0.0
    ) {
      // Draw a semi-opaque red overlay on the entire canvas
      ctx.fillStyle = "rgba(255, 0, 0, 0.1)";
      ctx.fillRect(0, 0, canvasElement.width, canvasElement.height);

      // Clear the specified rectangle
      ctx.clearRect(
        overlayNormalizedRect.x * canvasElement.width,
        overlayNormalizedRect.y * canvasElement.height,
        overlayNormalizedRect.width * canvasElement.width,
        overlayNormalizedRect.height * canvasElement.height
      );
    }

    // Draw text
    if (overlayText) {
      ctx.font = "16px sans-serif";
      ctx.fillStyle = "white";
      ctx.fillText(`${overlayText}`, 10, 25);
    }

    // Draw tracks
    if (overlayTracks) {
      const markerSize = 14;
      overlayTracks.forEach((track) => {
        const x = track.normalizedPixelCoords.x * canvasElement.width;
        const y = track.normalizedPixelCoords.y * canvasElement.height;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x - markerSize / 2, y - markerSize / 2);
        ctx.lineTo(x + markerSize / 2, y + markerSize / 2);
        ctx.moveTo(x + markerSize / 2, y - markerSize / 2);
        ctx.lineTo(x - markerSize / 2, y + markerSize / 2);
        ctx.strokeStyle =
          track.state === TrackState.COMPLETED ? "green" : "red";
        ctx.stroke();
      });
    }
  }, [setCanvasStyle, overlayText, overlayNormalizedRect, overlayTracks]);

  useEffect(() => {
    updateCanvasSize();
    window.addEventListener("resize", updateCanvasSize);

    return () => {
      window.removeEventListener("resize", updateCanvasSize);
    };
  }, [updateCanvasSize, renderOverlay]);

  return (
    <div className={cn("relative", className)}>
      <video
        ref={videoRef}
        className="w-full h-full object-contain bg-black"
        autoPlay
        playsInline
        muted
        onWaiting={() => {
          setIsLoading(true);
        }}
        onCanPlay={() => {
          setIsLoading(false);
        }}
        onPlaying={() => {
          setIsLoading(false);
        }}
        onClick={handleVideoClick}
      />
      {renderOverlay && (
        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            top: `${canvasStyle.top}px`,
            left: `${canvasStyle.left}px`,
            width: `${canvasStyle.width}px`,
            height: `${canvasStyle.height}px`,
            pointerEvents: "none",
          }}
          className="absolute inset-0 pointer-events-none"
        />
      )}
      {!enableStream && (
        <div className="absolute inset-0 flex items-center justify-center text-white">
          <span className="text-white text-lg">Stream unavailable</span>
        </div>
      )}
    </div>
  );
}
