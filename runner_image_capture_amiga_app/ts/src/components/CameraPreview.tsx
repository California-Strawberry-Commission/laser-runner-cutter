import { useEffect, useRef } from "react";

export default function CameraPreview() {
  const websocket = useRef<WebSocket | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    console.log("USE EFFECT");
    const startWebsocket = () => {
      const ws = new WebSocket(`ws://${window.location.hostname}:8042/camera`);

      ws.onopen = (event) => {
        console.log("Camera WebSocket connection opened:", event);
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

      ws.onclose = (event) => {
        console.log("Camera WebSocket connection closed:", event);
        setTimeout(startWebsocket, 1000);
      };

      websocket.current = ws;
    };

    startWebsocket();

    return () => {
      if (websocket.current) {
        websocket.current.close();
      }
    };
  });

  return <canvas ref={canvasRef} width={848} height={480} />;
}
