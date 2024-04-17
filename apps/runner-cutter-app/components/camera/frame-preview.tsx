import React from "react";

export default function FramePreview({
  frameSrc,
  height = 360,
  onImageLoad,
  onImageClick,
}: {
  frameSrc?: string;
  height?: number;
  onImageLoad?: React.EventHandler<
    React.SyntheticEvent<HTMLImageElement, Event>
  >;
  onImageClick?: React.MouseEventHandler<HTMLImageElement>;
}) {
  return (
    <div className="flex flex-col w-full items-center" style={{ height }}>
      {frameSrc ? (
        <img
          src={frameSrc}
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
