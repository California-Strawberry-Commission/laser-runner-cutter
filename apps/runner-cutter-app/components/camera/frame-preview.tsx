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
    process.env.NEXT_PUBLIC_VIDEO_SERVER_URL ?? "http://localhost:8080";
  const streamUrl = `${videoServer}/stream?topic=${topicName}`;

  return (
    <div className="flex flex-col w-full items-center" style={{ height }}>
      {topicName ? (
        <img
          src={streamUrl}
          alt="Camera Color Frame"
          className="h-full w-auto max-h-full max-w-full"
          onLoad={onImageLoad}
          onClick={onImageClick}
        />
      ) : (
        <div
          className="flex h-full bg-slate-400 justify-center items-center"
          style={{ width: height * 1.78 }}
        >
          <p>No camera found</p>
        </div>
      )}
    </div>
  );
}
