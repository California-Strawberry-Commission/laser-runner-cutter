import React, { useRef, useEffect } from "react";
import { Track, TrackState } from "@/lib/useControlNode";

export default function Overlay({
  width,
  height,
  tracks,
}: {
  width: number;
  height: number;
  tracks?: Track[];
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (canvasRef.current) {
      const canvas = canvasRef.current;
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        return;
      }

      // Clear the canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (!tracks) {
        return;
      }

      // Draw markers
      ctx.font = "16px sans-serif";
      const textXOffset = 18;
      const textYOffset = 13;
      const markerSize = 14;
      tracks.forEach((track) => {
        const x = track.normalizedPixelCoords.x * width;
        const y = track.normalizedPixelCoords.y * height;
        const text = track.id.toString();
        const textX = x + textXOffset;
        const textY = y + textYOffset;
        switch (track.state) {
          case TrackState.Pending:
            ctx.fillStyle = "yellow";
            ctx.fillRect(x, y, markerSize, markerSize);
            ctx.fillText(text, textX, textY);
            break;
          case TrackState.Active:
            ctx.beginPath();
            ctx.arc(x, y, markerSize / 2, 0, 2 * Math.PI);
            ctx.fillStyle = "white";
            ctx.fill();
            ctx.fillText(text, textX, textY);
            break;
          case TrackState.Completed:
            ctx.beginPath();
            ctx.arc(x, y, markerSize / 2, 0, 2 * Math.PI);
            ctx.fillStyle = "green";
            ctx.fill();
            ctx.fillText(text, textX, textY);
            break;
          case TrackState.OutOfLaserBounds:
            ctx.fillStyle = "red";
            ctx.fillRect(x, y, markerSize, markerSize);
            ctx.fillText(text, textX, textY);
            break;
          default:
            break;
        }
      });
    }
  }, [width, height, tracks]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0"
      style={{ pointerEvents: "none" }}
    />
  );
}
