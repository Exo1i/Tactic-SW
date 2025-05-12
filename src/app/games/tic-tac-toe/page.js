"use client";
import { useState, useRef, useEffect } from "react";
import { initializeVideoSource } from "@/utils/cameraUtils";

// --- Settings Modal with input focus ---
const SettingsModalComponent = ({
  showSettings,
  setShowSettings,
  cameraSettings,
  handleCameraSettingsChange,
  ipCameraInputRef,
  setAppliedCameraSettings,
}) => {
  if (!showSettings) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-40">
      <div className="bg-white p-6 rounded-lg shadow-lg w-full max-w-md relative">
        <button
          className="absolute top-2 right-2 text-gray-400 hover:text-gray-700"
          onClick={() => setShowSettings(false)}
          aria-label="Close"
        >
          ×
        </button>
        <h3 className="text-lg font-medium mb-3">Camera Settings</h3>
        <div className="flex items-center mb-3">
          <input
            type="checkbox"
            id="useIpCamera"
            checked={cameraSettings.useIpCamera}
            onChange={(e) =>
              handleCameraSettingsChange({
                ...cameraSettings,
                useIpCamera: e.target.checked,
              })
            }
            className="mr-2"
          />
          <label htmlFor="useIpCamera">Use IP Camera</label>
        </div>
        {cameraSettings.useIpCamera && (
          <div className="mb-3">
            <label htmlFor="ipCameraAddress" className="block mb-1">
              IP Camera URL:
            </label>
            <input
              ref={ipCameraInputRef}
              type="text"
              id="ipCameraAddress"
              value={cameraSettings.ipCameraAddress}
              onChange={(e) =>
                handleCameraSettingsChange({
                  ...cameraSettings,
                  ipCameraAddress: e.target.value,
                })
              }
              placeholder="http://camera-ip:port/stream"
              className="w-full p-2 border rounded"
            />
            <small className="text-gray-500">
              Example: http://192.168.1.100:8080/video
            </small>
          </div>
        )}
        <button
          onClick={() => {
            setAppliedCameraSettings(cameraSettings);
            setShowSettings(false);
          }}
          className="px-3 py-1 bg-green-500 text-white rounded hover:bg-green-600"
        >
          Apply Settings
        </button>
      </div>
    </div>
  );
};

export default function TicTacToePage() {
  const gameId = "tic-tac-toe";
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const ipCamImgRef = useRef(null);
  const ipCameraInputRef = useRef(null);

  const [status, setStatus] = useState("Connecting...");
  const [output, setOutput] = useState(null);
  const [rawFrame, setRawFrame] = useState(null);
  const [processedFrame, setProcessedFrame] = useState(null);
  const [birdViewFrame, setBirdViewFrame] = useState(null);
  const [showSettings, setShowSettings] = useState(false);
  const [cameraSettings, setCameraSettings] = useState({
    useIpCamera: false,
    ipCameraAddress: "",
  });
  const [appliedCameraSettings, setAppliedCameraSettings] = useState({
    useIpCamera: false,
    ipCameraAddress: "",
  });

  // Tic Tac Toe arguments
  const [tttArgs, setTttArgs] = useState({
    model: "games/tic-tac-toe/data/model.h5",
    zoom: 0.5, // Changed from 1.0 to 0.5 for a wider view
    check_interval: 10.0,
  });
  const [tttStarted, setTttStarted] = useState(false);

  // Loading indicators
  const [isCameraLoading, setIsCameraLoading] = useState(false);
  const [isBackendLoading, setIsBackendLoading] = useState(false);

  const sendNextFrameRef = useRef(true);
  const lastSentRef = useRef(0);
  const minFrameInterval = 100; // 10 FPS

  // Winner/result state
  const [winner, setWinner] = useState(null);

  useEffect(() => {
    if (!tttStarted) return;
    let stopped = false;

    const initCamera = async () => {
      setIsCameraLoading(true);
      if (appliedCameraSettings.useIpCamera) {
        if (videoRef.current && videoRef.current.srcObject) {
          videoRef.current.srcObject.getTracks().forEach((t) => t.stop());
          videoRef.current.srcObject = null;
        }
        setTimeout(() => setIsCameraLoading(false), 1000);
      } else {
        await initializeVideoSource(videoRef.current, appliedCameraSettings);
        setIsCameraLoading(false);
      }
    };

    initCamera();

    const ws = new WebSocket(`ws://localhost:8000/ws/tic-tac-toe`);
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus("Connected");
      // Send config as first message
      ws.send(JSON.stringify(tttArgs));
    };
    ws.onclose = () => setStatus("Disconnected");
    ws.onerror = () => setStatus("Error");

    ws.onmessage = (event) => {
      setIsBackendLoading(false);
      sendNextFrameRef.current = true;
      try {
        const data = JSON.parse(event.data);
        setOutput(data);
        if (data.raw_frame)
          setRawFrame(`data:image/jpeg;base64,${data.raw_frame}`);
        if (data.processed_frame)
          setProcessedFrame(`data:image/jpeg;base64,${data.processed_frame}`);
        if (data.bird_view_frame)
          setBirdViewFrame(`data:image/jpeg;base64,${data.bird_view_frame}`);
        else
          setBirdViewFrame(null); // Clear if no bird view available

        // Winner/result logic
        let winnerVal = null;
        if (data.game_state && data.game_state.winner) {
          winnerVal = data.game_state.winner;
        } else if (data.game_state && data.game_state.board_status === "complete") {
          winnerVal = data.game_state.winner || "Draw";
        }
        setWinner(winnerVal);
      } catch {
        setOutput(event.data);
      }
    };

    function sendFrame() {
      if (stopped) return;
      const now = Date.now();
      if (!sendNextFrameRef.current) {
        requestAnimationFrame(sendFrame);
        return;
      }
      if (now - lastSentRef.current < minFrameInterval) {
        requestAnimationFrame(sendFrame);
        return;
      }
      lastSentRef.current = now;

      const ws = wsRef.current;
      const canvas = canvasRef.current;
      let sourceEl = appliedCameraSettings.useIpCamera
        ? ipCamImgRef.current
        : videoRef.current;
      if (sourceEl && canvas && ws && ws.readyState === 1) {
        const ctx = canvas.getContext("2d");
        try {
          ctx.drawImage(sourceEl, 0, 0, canvas.width, canvas.height);
          canvas.toBlob(
            (blob) => {
              if (blob) {
                setIsBackendLoading(true);
                sendNextFrameRef.current = false;
                blob.arrayBuffer().then((buffer) => {
                  if (ws.readyState === 1) ws.send(buffer);
                });
              }
            },
            "image/jpeg",
            0.7
          );
        } catch (e) {}
      }
      requestAnimationFrame(sendFrame);
    }
    sendNextFrameRef.current = true;
    requestAnimationFrame(sendFrame);

    return () => {
      stopped = true;
      ws.close();
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach((t) => t.stop());
      }
    };
  }, [tttStarted, appliedCameraSettings, tttArgs]);

  const handleCameraSettingsChange = (newSettings) => {
    setCameraSettings(newSettings);
  };

  // Focus input field when settings modal opens
  useEffect(() => {
    if (
      showSettings &&
      cameraSettings.useIpCamera &&
      ipCameraInputRef.current
    ) {
      setTimeout(() => {
        ipCameraInputRef.current.focus();
      }, 100);
    }
  }, [showSettings, cameraSettings.useIpCamera]);

  // --- Winner/Result Display ---
  const WinnerBanner = () =>
    winner ? (
      <div className="mb-6 flex flex-col items-center">
        <div className="text-4xl mb-2 animate-bounce">
          {winner === "Draw" || winner === null
            ? "🤝"
            : winner === "X"
            ? "❌"
            : winner === "O"
            ? "⭕"
            : "🏁"}
        </div>
        <div
          className={`text-2xl font-bold ${
            winner === "Draw"
              ? "text-gray-700"
              : winner === "X"
              ? "text-red-700"
              : winner === "O"
              ? "text-blue-700"
              : "text-purple-700"
          }`}
        >
          {winner === "Draw" || winner === null
            ? "It's a Draw!"
            : winner === "X"
            ? "You Win! (X)"
            : winner === "O"
            ? "Computer Wins! (O)"
            : `Winner: ${winner}`}
        </div>
      </div>
    ) : null;

  return (
    <div className="flex flex-col items-center gap-4 p-4 min-h-screen bg-gradient-to-br from-blue-100 via-white to-yellow-100">
      <SettingsModalComponent
        showSettings={showSettings}
        setShowSettings={setShowSettings}
        cameraSettings={cameraSettings}
        handleCameraSettingsChange={handleCameraSettingsChange}
        ipCameraInputRef={ipCameraInputRef}
        setAppliedCameraSettings={setAppliedCameraSettings}
      />
      <div className="w-full max-w-5xl bg-white rounded-2xl shadow-2xl p-8 mt-4 border border-gray-200">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-6">
          <div className="flex items-center gap-3">
            <h1 className="text-3xl font-extrabold tracking-tight text-blue-900 drop-shadow">
              Tic Tac Toe
            </h1>
          </div>
          <div className="flex items-center gap-4">
            <span
              className={`px-2 py-1 rounded font-semibold text-xs shadow ${
                status === "Connected"
                  ? "bg-green-100 text-green-700 border border-green-300"
                  : status === "Disconnected"
                  ? "bg-red-100 text-red-700 border border-red-300"
                  : "bg-yellow-100 text-yellow-700 border border-yellow-300"
              }`}
            >
              {status}
            </span>
            <button
              onClick={() => setShowSettings(true)}
              className="px-3 py-1 bg-blue-600 text-white rounded hover:bg-blue-700 shadow transition"
            >
              Camera Settings
            </button>
            {!tttStarted ? (
              <button
                onClick={() => setTttStarted(true)}
                className="px-3 py-1 bg-green-600 text-white rounded hover:bg-green-700 shadow transition"
              >
                Start Game
              </button>
            ) : (
              <button
                onClick={() => {
                  setTttStarted(false);
                  setStatus("Disconnected");
                  setOutput(null);
                  setRawFrame(null);
                  setProcessedFrame(null);
                  setBirdViewFrame(null);
                  setWinner(null);
                }}
                className="px-3 py-1 bg-red-600 text-white rounded hover:bg-red-700 shadow transition"
              >
                Stop Game
              </button>
            )}
          </div>
        </div>
        {appliedCameraSettings.useIpCamera && (
          <div className="mb-2 text-red-600 text-sm">
            Note: IP camera streams must support MJPEG and allow CORS. If you
            see a blank image, check your camera's settings.
          </div>
        )}
        {!window.isSecureContext && (
          <div className="mb-2 text-red-600 text-sm">
            Warning: Device camera access requires HTTPS in most browsers.
          </div>
        )}

        {/* Winner/Result Banner */}
        <WinnerBanner />

        <div className="w-full max-w-lg bg-gray-50 rounded-xl shadow p-4 mb-8">
          <h2 className="text-lg font-semibold mb-2">Game Arguments</h2>
          <div className="mb-2">
            <label className="block mb-1">Model Path:</label>
            <input
              type="text"
              value={tttArgs.model}
              onChange={e => setTttArgs(a => ({ ...a, model: e.target.value }))}
              className="w-full p-2 border rounded"
            />
          </div>
          <div className="mb-2">
            <label className="block mb-1">Zoom:</label>
            <input
              type="number"
              step="0.1"
              min="0.2" // Changed from 1 to 0.2 to allow smaller values
              value={tttArgs.zoom}
              onChange={e => setTttArgs(a => ({ ...a, zoom: parseFloat(e.target.value) }))}
              className="w-full p-2 border rounded"
            />
            <small className="text-gray-500">
              Lower values (0.2-0.5) show more of the paper, higher values zoom in.
            </small>
          </div>
          <div className="mb-2">
            <label className="block mb-1">Check Interval (seconds):</label>
            <input
              type="number"
              step="0.1"
              min="0.1"
              value={tttArgs.check_interval}
              onChange={e => setTttArgs(a => ({ ...a, check_interval: parseFloat(e.target.value) }))}
              className="w-full p-2 border rounded"
            />
          </div>
        </div>

        <h2 className="text-xl font-semibold mb-4">
          Playing: Tic Tac Toe
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="flex flex-col items-center">
            <div className="mb-2 text-center font-semibold text-blue-800">
              📷 Raw Camera
            </div>
            <div className="relative w-[320px] h-[240px] rounded-xl overflow-hidden border-2 border-blue-200 bg-black flex items-center justify-center shadow-lg">
              {appliedCameraSettings.useIpCamera ? (
                <img
                  ref={ipCamImgRef}
                  src={appliedCameraSettings.ipCameraAddress}
                  alt="IP Camera"
                  width={320}
                  height={240}
                  className="object-contain"
                  crossOrigin="anonymous"
                  onLoad={() => setIsCameraLoading(false)}
                  onError={() => setStatus("IP Camera Error")}
                />
              ) : (
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  width={320}
                  height={240}
                  className="object-contain"
                  style={{ background: "#222" }}
                />
              )}
              {isCameraLoading && (
                <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-60 text-white text-lg">
                  Loading...
                </div>
              )}
            </div>
            <canvas
              ref={canvasRef}
              width={320}
              height={240}
              style={{ display: "none" }}
            />
          </div>
          <div className="flex flex-col items-center">
            <div className="mb-2 text-center font-semibold text-green-800">
              🟩 Bird's Eye View
            </div>
            <div className="relative w-[320px] h-[240px] rounded-xl overflow-hidden border-2 border-green-200 bg-black flex items-center justify-center shadow-lg">
              {birdViewFrame ? (
                <img
                  src={birdViewFrame}
                  width={320}
                  height={240}
                  alt="Bird's Eye View"
                  className="object-contain"
                />
              ) : (
                <span className="text-gray-400">No paper detected</span>
              )}
              {isBackendLoading && (
                <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-60 text-white text-lg">
                  Processing...
                </div>
              )}
            </div>
          </div>
          <div className="flex flex-col items-center">
            <div className="mb-2 text-center font-semibold text-purple-800">
              🧠 Processed Frame
            </div>
            <div className="relative w-[320px] h-[240px] rounded-xl overflow-hidden border-2 border-purple-200 bg-black flex items-center justify-center shadow-lg">
              {processedFrame ? (
                <img
                  src={processedFrame}
                  width={320}
                  height={240}
                  alt="Processed Frame"
                  className="object-contain"
                />
              ) : (
                <span className="text-gray-400">No frame</span>
              )}
              {isBackendLoading && (
                <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-60 text-white text-lg">
                  Processing...
                </div>
              )}
            </div>
          </div>
        </div>
        <div className="mt-8 p-4 bg-gray-50 rounded-lg border border-gray-200 shadow-inner">
          <h3 className="text-lg font-semibold mb-2 text-gray-700 flex items-center gap-2">
            <span>Game Data</span>
            <span className="text-xs bg-gray-200 px-2 py-0.5 rounded text-gray-600">
              Debug
            </span>
          </h3>
          <pre className="text-sm overflow-x-auto whitespace-pre-wrap break-words max-h-56">
            {output ? JSON.stringify(output, null, 2) : "No data"}
          </pre>
        </div>
      </div>
    </div>
  );
}
