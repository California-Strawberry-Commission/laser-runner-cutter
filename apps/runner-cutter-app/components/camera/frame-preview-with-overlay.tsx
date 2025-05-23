"use client";

import { Track, TrackState } from "@/lib/useControlNode";
import useVideoServerStreamUrl from "@/lib/useVideoServerStreamUrl";
import { cn } from "@/lib/utils";
import React, { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";

function getImageSizeAndOffset(imgElement: HTMLImageElement) {
  const rect = imgElement.getBoundingClientRect();

  // The dimensions of the image element (including the padding caused by object-contain)
  const containerWidth = rect.width;
  const containerHeight = rect.height;

  // Get the actual image size based on its intrinsic dimensions and container's aspect ratio
  const naturalWidth = imgElement.naturalWidth;
  const naturalHeight = imgElement.naturalHeight;
  const aspectRatio = naturalWidth / naturalHeight;

  let renderedWidth, renderedHeight;
  if (containerWidth / containerHeight > aspectRatio) {
    // Image is constrained by height
    renderedHeight = containerHeight;
    renderedWidth = renderedHeight * aspectRatio;
  } else {
    // Image is constrained by width
    renderedWidth = containerWidth;
    renderedHeight = renderedWidth / aspectRatio;
  }

  // Calculate the top-left corner of the image inside the container
  const offsetX = (containerWidth - renderedWidth) / 2;
  const offsetY = (containerHeight - renderedHeight) / 2;

  return { renderedWidth, renderedHeight, offsetX, offsetY };
}

export default function FramePreviewWithOverlay({
  topicName,
  quality = 30,
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
  quality?: number;
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
  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [canvasStyle, setCanvasStyle] = useState({
    width: 0,
    height: 0,
    top: 0,
    left: 0,
  });
  const [rotate180, setRotate180] = useState(false);

  const streamUrl = useVideoServerStreamUrl(topicName, quality, enableStream);

  const renderOverlay = streamUrl && enableOverlay;

  // Handle image click and calculate normalized coordinates
  const handleImageClick = useCallback(
    (e: React.MouseEvent<HTMLImageElement, MouseEvent>) => {
      if (!imgRef.current || !streamUrl) {
        return;
      }

      const imgElement = imgRef.current;
      const { renderedWidth, renderedHeight, offsetX, offsetY } =
        getImageSizeAndOffset(imgElement);
      const rect = imgElement.getBoundingClientRect();

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
    [streamUrl, rotate180, onImageClick]
  );

  // Update canvas position and size to exactly match the rendered image
  const updateCanvasSize = useCallback(() => {
    if (!imgRef.current || !canvasRef.current) {
      return;
    }

    const imgElement = imgRef.current;
    const { renderedWidth, renderedHeight, offsetX, offsetY } =
      getImageSizeAndOffset(imgElement);
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

  useEffect(() => {
    updateCanvasSize();
    window.addEventListener("resize", updateCanvasSize);

    let imgRefValue = null;
    if (imgRef.current) {
      imgRef.current.addEventListener("load", updateCanvasSize);
      imgRefValue = imgRef.current;
    }

    return () => {
      window.removeEventListener("resize", updateCanvasSize);
      if (imgRefValue) {
        imgRefValue.removeEventListener("load", updateCanvasSize);
      }
    };
  }, [updateCanvasSize, renderOverlay]);

  // Note: we render <img> even when streamUrl is empty in order to properly close
  // the connection to the stream and avoid potential cases of multiple simultaneous
  // connections on rerenders.
  return (
    <div className={cn("relative", className)}>
      <img
        ref={imgRef}
        src={streamUrl || undefined}
        alt="Camera Stream"
        className={cn(
          "w-full h-full object-contain bg-black",
          rotate180 ? "rotate-180" : null
        )}
        onClick={handleImageClick}
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
      {!enableStream && (
        <div className="absolute inset-0 flex items-center justify-center text-white">
          <span className="text-white text-lg">Stream unavailable</span>
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
