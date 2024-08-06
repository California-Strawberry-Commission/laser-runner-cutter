import React, { useRef, useEffect } from "react";
import { Track, TrackState } from "@/lib/useControlNode";

export default function Overlay({
  tracks,
  normalizedRect,
}: {
  tracks?: Track[];
  normalizedRect?: { x: number; y: number; width: number; height: number };
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (canvasRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        return;
      }

      // TODO: Set size to match base image
      canvas.width = 800;
      canvas.height = 600;

      // Clear the canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw rect
      if (
        normalizedRect &&
        normalizedRect.width > 0.0 &&
        normalizedRect.height > 0.0
      ) {
        // Draw a semi-opaque red overlay on the entire canvas
        ctx.fillStyle = "rgba(255, 0, 0, 0.1)";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Clear the specified rectangle
        ctx.clearRect(
          normalizedRect.x * canvas.width,
          normalizedRect.y * canvas.height,
          normalizedRect.width * canvas.width,
          normalizedRect.height * canvas.height
        );
      }

      // Draw markers
      if (tracks) {
        const markerSize = 14;
        tracks.forEach((track) => {
          const x = track.normalizedPixelCoords.x * canvas.width;
          const y = track.normalizedPixelCoords.y * canvas.height;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(x - markerSize / 2, y - markerSize / 2);
          ctx.lineTo(x + markerSize / 2, y + markerSize / 2);
          ctx.moveTo(x + markerSize / 2, y - markerSize / 2);
          ctx.lineTo(x - markerSize / 2, y + markerSize / 2);
          ctx.strokeStyle =
            track.state === TrackState.Completed ? "green" : "red";
          ctx.stroke();
        });
      }
    }
  }, [tracks, normalizedRect]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 pointer-events-none w-full h-full"
    />
  );
}
