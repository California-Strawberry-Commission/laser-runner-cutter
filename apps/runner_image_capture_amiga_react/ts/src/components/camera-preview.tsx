import { useEffect, useRef, useState } from "react";

export default function CameraPreview() {
  const webSocket = useRef<WebSocket | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isWebSocketOpen, setIsWebSocketOpen] = useState<boolean>(false);
  const [frameWidth, setFrameWidth] = useState<number>(0);
  const [frameHeight, setFrameHeight] = useState<number>(0);

  useEffect(() => {
    const startWebSocket = () => {
      const ws = new WebSocket(
        `ws://${window.location.hostname}:8042/camera/preview`
      );

      ws.onopen = () => {
        setIsWebSocketOpen(true);
      };

      ws.onmessage = (event) => {
        const img = new Image();
        img.src = `data:image/jpeg;base64,${event.data}`;
        img.onload = () => {
          if (canvasRef.current == null) {
            return;
          }

          const ctx = canvasRef.current.getContext("2d");
          if (ctx == null) {
            return;
          }

          ctx.clearRect(
            0,
            0,
            canvasRef.current.width,
            canvasRef.current.height
          );
          ctx.drawImage(
            img,
            0,
            0,
            canvasRef.current.width,
            canvasRef.current.height
          );

          setFrameWidth(img.width);
          setFrameHeight(img.height);
        };
      };

      ws.onclose = () => {
        setIsWebSocketOpen(false);
        setTimeout(startWebSocket, 1000);
      };

      webSocket.current = ws;
    };

    startWebSocket();

    return () => {
      if (webSocket.current) {
        webSocket.current.close();
      }
    };
  }, []);

  return (
    <div className="relative w-[848px] h-[480px] flex justify-center items-center">
      {isWebSocketOpen ? (
        <div className="absolute top-4 left-4 z-10 text-white bg-black bg-opacity-50">
          <p>
            {frameWidth}x{frameHeight}
          </p>
        </div>
      ) : (
        <div className="absolute inset-0 bg-gray-200 flex justify-center items-center">
          <p>Camera not available</p>
        </div>
      )}
      <canvas ref={canvasRef} width={848} height={480} />
    </div>
  );
}
