import React, { useEffect, useRef } from "react";

export default function FramePreview({
  topicName,
  height = 360,
  onImageLoad,
  onImageClick,
  onSizeChanged,
}: {
  topicName?: string;
  height?: number;
  onImageLoad?: React.EventHandler<
    React.SyntheticEvent<HTMLImageElement, Event>
  >;
  onImageClick?: React.MouseEventHandler<HTMLImageElement>;
  onSizeChanged?: (width: number, height: number) => void;
}) {
  const imgRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    const updateSize = () => {
      if (onSizeChanged && imgRef.current) {
        onSizeChanged(imgRef.current.offsetWidth, imgRef.current.offsetHeight);
      }
    };

    // Initial size update
    updateSize();

    // Update size on window resize
    window.addEventListener("resize", updateSize);
    return () => window.removeEventListener("resize", updateSize);
  }, [onSizeChanged]);

  const videoServer =
    process.env.NEXT_PUBLIC_VIDEO_SERVER_URL ?? "http://localhost:8080";
  const streamUrl = `${videoServer}/stream?topic=${topicName}`;

  return (
    <img
      ref={imgRef}
      src={streamUrl}
      alt="Camera Color Frame"
      className="w-auto max-h-full max-w-full"
      style={{ height }}
      onLoad={onImageLoad}
      onClick={onImageClick}
    />
  );
}
