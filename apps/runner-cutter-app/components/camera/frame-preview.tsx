"use client";

import React, { useEffect, useRef, useState } from "react";

export default function FramePreview({
  topicName,
  height = 360,
  quality = 30,
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
  const imgRef = useRef<HTMLImageElement>(null);
  const [streamUrl, setStreamUrl] = useState<string>();
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [componentSize, setComponentSize] = useState({ width: 0, height: 0 });

  // Unfortunately, with SSR, this needed for code that should only run on client side
  useEffect(() => {
    if (typeof window !== "undefined") {
      const videoServer =
        process.env.NEXT_PUBLIC_VIDEO_SERVER_URL ??
        `http://${window.location.hostname}:8080`;
      setStreamUrl(
        `${videoServer}/stream?topic=${topicName}&quality=${quality}`
      );
    }
  }, [topicName, quality]);

  useEffect(() => {
    const img = imgRef.current;
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
