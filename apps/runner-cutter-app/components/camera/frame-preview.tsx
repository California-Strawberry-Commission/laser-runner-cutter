"use client";

import React, { useEffect, useRef, useState } from "react";

export default function FramePreview({
  topicName,
  height = 360,
  onImageLoad,
  onImageClick,
}: {
  topicName?: string;
  height?: number;
  onImageLoad?: React.EventHandler<
    React.SyntheticEvent<HTMLImageElement, Event>
  >;
  onImageClick?: React.MouseEventHandler<HTMLImageElement>;
}) {
  const imgRef = useRef<HTMLImageElement>(null);
  const [streamUrl, setStreamUrl] = useState<string>();

  // Unfortunately, with SSR, this needed for code that should only run on client side
  useEffect(() => {
    if (typeof window !== "undefined") {
      const videoServer =
        process.env.NEXT_PUBLIC_VIDEO_SERVER_URL ??
        `http://${window.location.hostname}:8080`;
      setStreamUrl(`${videoServer}/stream?topic=${topicName}`);
    }
  }, []);

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
