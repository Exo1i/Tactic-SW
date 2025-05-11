"use client";
import { useState, useRef, useEffect } from "react";
import { initializeVideoSource } from "@/utils/cameraUtils";

// Define SettingsModalComponent outside ShellGamePage
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

export default function ShellGamePage() {
  const gameId = "shell-game";
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const ipCamImgRef = useRef(null);

  const [status, setStatus] = useState("Connecting...");
  const [output, setOutput] = useState(null);
  const [rawFrame, setRawFrame] = useState(null);
  const [processedFrame, setProcessedFrame] = useState(null);
  const [livefeedFrame, setLivefeedFrame] = useState(null);
  const [showSettings, setShowSettings] = useState(false);
  const [cameraSettings, setCameraSettings] = useState({
    useIpCamera: false,
    ipCameraAddress: "",
  });
  const [appliedCameraSettings, setAppliedCameraSettings] = useState({
    useIpCamera: false,
    ipCameraAddress: "",
  });

  // Loading indicators
  const [isCameraLoading, setIsCameraLoading] = useState(false);

  // Only send a new frame after backend responds
  const sendNextFrameRef = useRef(true);
  // Add a ref to track last sent time for FPS limiting
  const lastSentRef = useRef(0);
  // Set your desired FPS here (e.g., 10 or 20)
  const minFrameInterval = 30; // ms (10 FPS). Use 50 for 20 FPS.

  const [isGameStarted, setIsGameStarted] = useState(false);
  const [processedStreamUrl, setProcessedStreamUrl] = useState(null);
  const [debugState, setDebugState] = useState(null);
  const [cupResult, setCupResult] = useState(null);

  // Add ref for IP camera input field
  const ipCameraInputRef = useRef(null);

  // Add state for secure context check
  const [isSecureContext, setIsSecureContext] = useState(true);

  // Check secure context after component mounts
  useEffect(() => {
    setIsSecureContext(window.isSecureContext === true);
  }, []);

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

  // Handle Escape key to close modal
  useEffect(() => {
    const handleEscapeKey = (e) => {
      if (e.key === "Escape" && showSettings) {
        setShowSettings(false);
      }
    };

    window.addEventListener("keydown", handleEscapeKey);
    return () => window.removeEventListener("keydown", handleEscapeKey);
  }, [showSettings]);

  useEffect(() => {
    if (!isGameStarted) return;

    const ws = new WebSocket(`ws://localhost:8000/ws/${gameId}`);
    wsRef.current = ws;

    ws.onopen = () => setStatus("Connected");
    ws.onclose = () => setStatus("Disconnected");
    ws.onerror = () => setStatus("Error");

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "livefeed" && data.payload) {
          setLivefeedFrame(`data:image/jpeg;base64,${data.payload}`);
          return;
        }
        setOutput(data);
        if (data.raw_frame)
          setRawFrame(`data:image/jpeg;base64,${data.raw_frame}`);
        if (data.processed_frame)
          setProcessedFrame(`data:image/jpeg;base64,${data.processed_frame}`);
      } catch {
        setOutput(event.data);
      }
    };

    setProcessedStreamUrl("http://localhost:8000/stream/shell-game");

    return () => {
      // Removed erroneous `stopped = true;` as `stopped` is not defined in this scope
      // and ws.close() handles the necessary cleanup for the WebSocket.
      if (wsRef.current) {
        wsRef.current.close();
      }
      setProcessedStreamUrl(null);
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach((t) => t.stop());
      }
    };
  }, [gameId, appliedCameraSettings, isGameStarted]);

  // Poll debug/game state from backend
  useEffect(() => {
    if (!isGameStarted) {
      setDebugState(null);
      setCupResult(null);
      return;
    }
    let stopped = false;
    async function pollDebug() {
      while (!stopped) {
        try {
          const res = await fetch("http://localhost:8000/shell-game/debug");
          if (res.ok) {
            const data = await res.json();
            setDebugState(data);
            // React to game end and show result
            if (data.status === "ended" && data.cup_name_result) {
              setCupResult(data.cup_name_result);
            }
          }
        } catch (e) {
          setDebugState({ error: "Failed to fetch debug state" });
        }
        await new Promise((r) => setTimeout(r, 400)); // Poll every 400ms
      }
    }
    pollDebug();
    return () => {
      stopped = true;
    };
  }, [isGameStarted]);

  const handleCameraSettingsChange = (newSettings) => {
    setCameraSettings(newSettings);
  };

  return (
    <div className="flex flex-col items-center gap-4 p-4 min-h-screen bg-gradient-to-br from-gray-100 to-gray-300">
      <SettingsModalComponent
        showSettings={showSettings}
        setShowSettings={setShowSettings}
        cameraSettings={cameraSettings}
        handleCameraSettingsChange={handleCameraSettingsChange}
        ipCameraInputRef={ipCameraInputRef}
        setAppliedCameraSettings={setAppliedCameraSettings}
      />
      <div className="w-full max-w-4xl bg-white rounded-xl shadow-lg p-6 mt-4">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-4">
          <h1 className="text-2xl font-bold mb-2 md:mb-0">Shell Game</h1>
          <div className="flex items-center gap-4">
            <span
              className={`px-2 py-1 rounded text-xs ${
                status === "Connected"
                  ? "bg-green-100 text-green-700"
                  : status === "Disconnected"
                  ? "bg-red-100 text-red-700"
                  : "bg-yellow-100 text-yellow-700"
              }`}
            >
              {status}
            </span>
            <button
              onClick={() => setShowSettings(true)}
              className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600"
            >
              Camera Settings
            </button>
            {!isGameStarted ? (
              <button
                onClick={() => setIsGameStarted(true)}
                className="px-3 py-1 bg-green-600 text-white rounded hover:bg-green-700"
              >
                Start Game
              </button>
            ) : (
              <button
                onClick={() => {
                  setIsGameStarted(false);
                  setStatus("Disconnected");
                  setOutput(null);
                  setRawFrame(null);
                  setProcessedFrame(null);
                }}
                className="px-3 py-1 bg-red-600 text-white rounded hover:bg-red-700"
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
        {!isSecureContext && (
          <div className="mb-2 text-red-600 text-sm">
            Warning: Device camera access requires HTTPS in most browsers.
          </div>
        )}
        <div className="grid grid-cols-1 md:grid-cols-3">
          <div className="flex flex-col items-center">
            <div className="mb-2 text-center font-medium">Live Cam Preview</div>
            <div className="relative w-[320px] h-[240px] rounded-lg overflow-hidden border-2 border-gray-300 bg-black flex items-center justify-center">
              {livefeedFrame ? (
                <img
                  src={livefeedFrame}
                  width={320}
                  height={240}
                  alt="Livefeed Frame"
                  className="object-contain"
                />
              ) : (
                <span className="text-gray-400">No livefeed</span>
              )}
            </div>
          </div>

          <div className="flex flex-col items-center">
            <div className="mb-2 text-center font-medium">Processed Frame</div>
            <div className="relative w-[320px] h-[240px] rounded-lg overflow-hidden border-2 border-gray-300 bg-black flex items-center justify-center">
              {processedStreamUrl ? (
                <img
                  src={processedStreamUrl}
                  width={320}
                  height={240}
                  alt="Processed Stream"
                  className="object-contain"
                  style={{ background: "#222" }}
                />
              ) : (
                <span className="text-gray-400">No frame</span>
              )}
            </div>
          </div>
        </div>
        {cupResult && (
          <div className="mt-6 p-4 bg-green-100 border border-green-400 rounded-lg text-center text-xl font-bold text-green-800 animate-pulse">
            🎉 The ball is under the{" "}
            <span className="uppercase">{cupResult}</span> cup!
          </div>
        )}
        <div className="mt-6 p-4 bg-gray-50 rounded-lg border">
          <h3 className="text-lg font-medium mb-2">Game Data</h3>
          <pre className="text-sm overflow-x-auto">
            {debugState ? JSON.stringify(debugState, null, 2) : "No data"}
          </pre>
        </div>
      </div>
    </div>
  );
}
