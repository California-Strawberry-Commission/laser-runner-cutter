"use client";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { RemoteTrack, Room, RoomEvent, Track as LkTrack } from "livekit-client";
import { useCallback, useEffect, useRef, useState } from "react";
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

function getOrMakeIdentity() {
  // Stable per-browser identity
  const key = "lk-identity";
  let id = typeof window !== "undefined" ? localStorage.getItem(key) : null;
  if (!id) {
    id = `viewer-${Math.random().toString(36).slice(2, 10)}`;
    try {
      localStorage.setItem(key, id);
    } catch {}
  }
  return id!;
}

export default function FramePreviewLiveKit({
  topicName,
  enableStream = false,
  onImageClick,
  enableOverlay = false,
  overlayText,
  overlaySubtext,
  overlayNormalizedRect,
  overlayTracks,
  showRotateButton = false,
  className,
}: {
  topicName?: string;
  enableStream?: boolean;
  onImageClick?: (normalizedX: number, normalizedY: number) => void;
  enableOverlay?: boolean;
  overlayText?: string;
  overlaySubtext?: string;
  overlayNormalizedRect?: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  overlayTracks?: Track[];
  showRotateButton?: boolean;
  className?: string;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [streamStatus, setStreamStatus] = useState<
    "idle" | "connecting" | "connected" | "error"
  >("idle");
  const [streamErrorMessage, setStreamErrorMessage] = useState<string | null>(
    null
  );
  const [rotate180, setRotate180] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [canvasStyle, setCanvasStyle] = useState({
    width: 0,
    height: 0,
    top: 0,
    left: 0,
  });

  const serverUrl =
    typeof window !== "undefined" && enableStream
      ? process.env.NEXT_PUBLIC_LIVEKIT_URL ??
        `ws://${window.location.hostname}:7880`
      : "";
  const identity = getOrMakeIdentity();

  useEffect(() => {
    if (!topicName || !serverUrl) {
      return;
    }

    let cancelled = false;

    const room = new Room({
      adaptiveStream: true,
      dynacast: true,
    });

    const onTrackSubscribed = (track: RemoteTrack) => {
      if (track.kind === LkTrack.Kind.Video && videoRef.current) {
        track.attach(videoRef.current);
      }
    };

    const onTrackUnsubscribed = (track: RemoteTrack) => {
      if (track.kind === LkTrack.Kind.Video && videoRef.current) {
        track.detach(videoRef.current);
      }
    };

    // Try to attach the first available remote video track
    const tryAttachVideoTrack = () => {
      if (!videoRef.current) {
        return false;
      }

      for (const participant of room.remoteParticipants.values()) {
        for (const pub of participant.trackPublications.values()) {
          if (pub.kind === LkTrack.Kind.Video) {
            if (pub.isSubscribed && pub.videoTrack) {
              pub.videoTrack.attach(videoRef.current);
              return true;
            }
            try {
              // Ensure we subscribe, then TrackSubscribed will fire
              pub.setSubscribed(true);
            } catch {}
          }
        }
      }
      return false;
    };

    (async () => {
      setStreamStatus("connecting");
      console.log("Fetching JWT...");

      try {
        const res = await fetch(
          `/api/token?room=${encodeURIComponent(
            topicName
          )}&identity=${encodeURIComponent(identity)}`,
          { cache: "no-store" }
        );
        if (!res.ok) {
          throw new Error(`Token endpoint error: ${res.status}`);
        }
        const { token } = await res.json();

        console.log("JWT obtained. Connecting to LiveKit room...");
        room
          .on(RoomEvent.TrackSubscribed, onTrackSubscribed)
          .on(RoomEvent.TrackUnsubscribed, onTrackUnsubscribed);
        await room.connect(serverUrl, token);
        if (cancelled) {
          return;
        }

        // In case the publisher was already in the room
        tryAttachVideoTrack();
        console.log("Connected to LiveKit room");
        setStreamStatus("connected");
      } catch (e: any) {
        if (!cancelled) {
          setStreamErrorMessage(e?.message ?? String(e));
          setStreamStatus("error");
        }
      }
    })();

    return () => {
      cancelled = true;
      try {
        room.disconnect();
      } catch {}
      if (videoRef.current) {
        // Clear any attached stream
        (videoRef.current as any).srcObject = null;
      }
    };
  }, [topicName, serverUrl, identity]);

  // Handle image click and calculate normalized coordinates
  const handleVideoClick = useCallback(
    (e: React.MouseEvent<HTMLVideoElement, MouseEvent>) => {
      if (!videoRef.current || streamStatus !== "connected") {
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
      let normalizedX = clickX / renderedWidth;
      let normalizedY = clickY / renderedHeight;
      if (rotate180) {
        normalizedX = 1 - normalizedX;
        normalizedY = 1 - normalizedY;
      }

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
    [streamStatus, rotate180, onImageClick]
  );

  // Update canvas position and size to exactly match the rendered image
  const updateCanvasSize = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) {
      return;
    }

    const videoElement = videoRef.current;
    const { renderedWidth, renderedHeight, offsetX, offsetY } =
      getVideoSizeAndOffset(videoElement);
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
      const rectX = rotate180
        ? 1 - overlayNormalizedRect.x - overlayNormalizedRect.width
        : overlayNormalizedRect.x;
      const rectY = rotate180
        ? 1 - overlayNormalizedRect.y - overlayNormalizedRect.height
        : overlayNormalizedRect.y;
      ctx.clearRect(
        rectX * canvasElement.width,
        rectY * canvasElement.height,
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
    if (overlaySubtext) {
      ctx.font = "12px sans-serif";
      ctx.fillStyle = "white";
      ctx.fillText(`${overlaySubtext}`, 10, 45);
    }

    // Draw tracks
    if (overlayTracks) {
      const markerSize = 14;
      overlayTracks.forEach((track) => {
        const x = track.normalizedPixelCoord.x * canvasElement.width;
        const y = track.normalizedPixelCoord.y * canvasElement.height;
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
  }, [
    rotate180,
    setCanvasStyle,
    overlayText,
    overlaySubtext,
    overlayNormalizedRect,
    overlayTracks,
  ]);

  // Render a stream status message if not connected. Otherwise, render the overlay.
  let streamStatusMessage = "";
  if (streamStatus === "connecting") {
    streamStatusMessage = "Stream connecting...";
  } else if (streamStatus === "error") {
    streamStatusMessage = `Stream error: ${streamErrorMessage}`;
  } else if (streamStatus === "idle") {
    streamStatusMessage = "Stream unavailable";
  } else if (isLoading) {
    streamStatusMessage = "Stream loading...";
  }
  const renderOverlay = enableOverlay && !streamStatusMessage;

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
        className={cn(
          "w-full h-full object-contain bg-black",
          rotate180 ? "rotate-180" : null
        )}
        autoPlay
        playsInline
        muted // required for autoplay in some browsers
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
            top: `${canvasStyle.top}px`,
            left: `${canvasStyle.left}px`,
            width: `${canvasStyle.width}px`,
            height: `${canvasStyle.height}px`,
          }}
          className="absolute inset-0 pointer-events-none"
        />
      )}
      {streamStatusMessage && (
        <div className="absolute inset-0 flex items-center justify-center text-white">
          <span className="text-white text-lg">{streamStatusMessage}</span>
        </div>
      )}
      {showRotateButton && enableStream && (
        <Button
          className="absolute bottom-4 right-4"
          variant="secondary"
          onClick={() => {
            setRotate180(!rotate180);
          }}
        >
          Rotate
        </Button>
      )}
    </div>
  );
}
