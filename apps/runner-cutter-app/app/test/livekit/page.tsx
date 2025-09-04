import FramePreviewLiveKit from "@/components/camera/frame-preview-livekit";

export default function LiveKit() {
  return (
    <main className="flex flex-col h-full gap-4">
      <h1 className="text-3xl font-bold">LiveKit Test</h1>
      <div className="items-center justify-center">
        <FramePreviewLiveKit
          className="w-full h-[480px]"
          topicName="/camera0/debug_frame"
          enableStream
          showRotateButton
          enableOverlay
          overlayText="Overlay text"
          overlaySubtext="Overlay subtext"
        />
      </div>
    </main>
  );
}
