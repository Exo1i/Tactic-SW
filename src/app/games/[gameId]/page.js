"use client";
import { useEffect, useRef, useState, use } from "react";

export default function GamePage({ params }) {
  const { gameId } = use(params);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const [status, setStatus] = useState("Connecting...");
  const [output, setOutput] = useState(null);
  const [rawFrame, setRawFrame] = useState(null);
  const [processedFrame, setProcessedFrame] = useState(null);

  useEffect(() => {
    let animationId;
    let stopped = false;

    // Use back camera if available
    navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" } }
    }).then((stream) => {
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    });

    const ws = new WebSocket(`ws://localhost:8000/ws/${gameId}`);
    wsRef.current = ws;

    ws.onopen = () => setStatus("Connected");
    ws.onclose = () => setStatus("Disconnected");
    ws.onerror = () => setStatus("Error");
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setOutput(data);
        if (data.raw_frame) setRawFrame(`data:image/jpeg;base64,${data.raw_frame}`);
        if (data.processed_frame) setProcessedFrame(`data:image/jpeg;base64,${data.processed_frame}`);
      } catch {
        setOutput(event.data);
      }
    };

    // Use requestAnimationFrame for lower latency
    function sendFrame() {
      if (stopped) return;
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ws = wsRef.current;
      if (video && canvas && ws && ws.readyState === 1) {
        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob((blob) => {
          if (blob) {
            blob.arrayBuffer().then((buffer) => {
              if (ws.readyState === 1) ws.send(buffer);
            });
          }
        }, "image/jpeg", 0.7);
      }
      animationId = requestAnimationFrame(sendFrame);
    }
    animationId = requestAnimationFrame(sendFrame);

    return () => {
      stopped = true;
      if (animationId) cancelAnimationFrame(animationId);
      ws.close();
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach((t) => t.stop());
      }
    };
  }, [gameId]);

  return (
    <div className="flex flex-col items-center gap-4 p-4">
      <h2 className="text-xl font-semibold">Playing: {gameId.replace("-", " ")}</h2>
      <div className="flex flex-row gap-8">
        <div>
          <div className="mb-2 text-center">Raw Camera</div>
          <video ref={videoRef} autoPlay playsInline width={320} height={240} style={{ display: "block" }} />
          <canvas ref={canvasRef} width={320} height={240} style={{ display: "none" }} />
        </div>
        <div>
          <div className="mb-2 text-center">Raw Frame (from backend)</div>
          {rawFrame && <img src={rawFrame} width={320} height={240} alt="Raw Frame" />}
        </div>
        <div>
          <div className="mb-2 text-center">Processed Frame</div>
          {processedFrame && <img src={processedFrame} width={320} height={240} alt="Processed Frame" />}
        </div>
      </div>
      <div>Status: {status}</div>
      <div className="mt-4">
        <pre>{JSON.stringify(output, null, 2)}</pre>
      </div>
    </div>
  );
}
