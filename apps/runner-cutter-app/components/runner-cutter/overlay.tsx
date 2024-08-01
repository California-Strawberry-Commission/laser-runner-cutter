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
      const markerSize = 14;
      tracks.forEach((track) => {
        const x = track.normalizedPixelCoords.x * width;
        const y = track.normalizedPixelCoords.y * height;
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
  }, [width, height, tracks]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0"
      style={{ pointerEvents: "none" }}
    />
  );
}
