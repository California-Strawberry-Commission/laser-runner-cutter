import { useEffect, useRef, useState } from "react";

export default function CameraPreview() {
  const webSocket = useRef<WebSocket | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isWebSocketOpen, setIsWebSocketOpen] = useState(false);

  useEffect(() => {
    const startWebSocket = () => {
      const ws = new WebSocket(
        `ws://${window.location.hostname}:8042/camera_preview`
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
      {isWebSocketOpen ? null : (
        <div className="absolute z-10 inset-0 bg-gray-200 flex justify-center items-center">
          <p>Camera not available</p>
        </div>
      )}
      <canvas ref={canvasRef} width={848} height={480} />
    </div>
  );
}
