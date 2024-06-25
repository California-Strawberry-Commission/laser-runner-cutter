import React from "react";

export default function FramePreview({
  topicName,
  height = 360,
  onImageLoad,
  onImageClick,
}: {
  topicName?: string;
  width?: number;
  height?: number;
  onImageLoad?: React.EventHandler<
    React.SyntheticEvent<HTMLImageElement, Event>
  >;
  onImageClick?: React.MouseEventHandler<HTMLImageElement>;
}) {
  const videoServer =
    process.env.NEXT_PUBLIC_VIDEO_SERVER_URL ?? `http://${window.location.hostname}:8080`;
  const streamUrl = `${videoServer}/stream?topic=${topicName}`;

  return (
    <img
      src={streamUrl}
      alt="Camera Color Frame"
      className="w-auto max-h-full max-w-full"
      style={{ height }}
      onLoad={onImageLoad}
      onClick={onImageClick}
    />
  );
}
