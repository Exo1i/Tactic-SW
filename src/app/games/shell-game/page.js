"use client";
import { useState, useRef, useEffect } from "react";
import { initializeVideoSource } from "@/utils/cameraUtils";

export default function ShellGamePage() {
  const gameId = "shell-game";
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const ipCamImgRef = useRef(null);
  const ipInputRef = useRef(null);

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

  useEffect(() => {
    if (showSettings && cameraSettings.useIpCamera && ipInputRef.current) {
      ipInputRef.current.focus();
    }
  }, [showSettings, cameraSettings.useIpCamera]);

  useEffect(() => {
    if (!isGameStarted) return;

    const ws = new WebSocket(`ws://localhost:8000/ws/${gameId}`);
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus("Connected");
      // Send config with IP camera URL if enabled
      if (
        appliedCameraSettings.useIpCamera &&
        appliedCameraSettings.ipCameraAddress
      ) {
        const config = {
          ip_camera_url: appliedCameraSettings.ipCameraAddress,
        };
        console.log("[ShellGame] Sending config to backend:", config);
        ws.send(JSON.stringify(config));
      }
    };
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
      ws.close();
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

    // Save IP camera address to local storage
    if (typeof newSettings.ipCameraAddress === "string") {
      localStorage.setItem("ipCameraAddress", newSettings.ipCameraAddress);
    }
  };

  // Helper to save IP to local storage and apply settings
  const saveIpAndApplySettings = () => {
    if (typeof cameraSettings.ipCameraAddress === "string") {
      localStorage.setItem("ipCameraAddress", cameraSettings.ipCameraAddress);
    }
    setAppliedCameraSettings(cameraSettings);
    setShowSettings(false);
  };

  // Modal overlay for settings
  const SettingsModal = () =>
    showSettings && (
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
                ref={ipInputRef}
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
                autoFocus
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    saveIpAndApplySettings();
                  }
                }}
              />
              <small className="text-gray-500">
                Example: http://192.168.1.100:8080/video
              </small>
            </div>
          )}
          <button
            onClick={saveIpAndApplySettings}
            className="px-3 py-1 bg-green-500 text-white rounded hover:bg-green-600"
          >
            Apply Settings
          </button>
        </div>
      </div>
    );

  return (
    <div className="flex flex-col items-center gap-4 p-4 min-h-screen bg-gradient-to-br from-blue-100 via-white to-green-100">
      <SettingsModal />
      <div className="w-full max-w-5xl bg-white rounded-2xl shadow-2xl p-8 mt-4 border border-gray-200">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-6">
          <div className="flex items-center gap-3">
            <h1 className="text-3xl font-extrabold tracking-tight text-blue-900 drop-shadow">
              Shell Game
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
            {!isGameStarted ? (
              <button
                onClick={() => setIsGameStarted(true)}
                className="px-3 py-1 bg-green-600 text-white rounded hover:bg-green-700 shadow transition"
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
                  setCupResult(null);
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

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="flex flex-col items-center">
            <div className="mb-2 text-center font-semibold text-blue-800">
              📷 Live Cam Preview
            </div>
            <div className="relative w-[320px] h-[240px] rounded-xl overflow-hidden border-2 border-blue-200 bg-black flex items-center justify-center shadow-lg">
              {livefeedFrame ? (
                <img
                  src={livefeedFrame}
                  width={320}
                  height={240}
                  alt="Livefeed Frame"
                  className="object-contain"
                  draggable={false}
                  style={{ userSelect: "none" }}
                />
              ) : (
                <span className="text-gray-400">No livefeed</span>
              )}
            </div>
          </div>

          <div className="flex flex-col items-center">
            <div className="mb-2 text-center font-semibold text-green-800">
              🕵️‍♂️ Processed Frame
            </div>
            <div className="relative w-[320px] h-[240px] rounded-xl overflow-hidden border-2 border-green-200 bg-black flex items-center justify-center shadow-lg">
              {processedStreamUrl ? (
                <img
                  src={processedStreamUrl}
                  width={320}
                  height={240}
                  alt="Processed Stream"
                  className="object-contain"
                  draggable={false}
                  style={{ background: "#222", userSelect: "none" }}
                />
              ) : (
                <span className="text-gray-400">No frame</span>
              )}
            </div>
          </div>

          <div className="flex flex-col items-center">
            <div className="mb-2 text-center font-semibold text-purple-800">
              🏆 Game Status
            </div>
            <div className="relative w-full min-h-[120px] rounded-xl bg-gradient-to-t from-purple-50 to-purple-100 border-2 border-purple-200 flex flex-col justify-center items-center p-4 shadow-inner">
              {cupResult ? (
                <div className="flex flex-col items-center">
                  <span className="text-4xl mb-2 animate-bounce">🎉</span>
                  <span className="text-2xl font-bold text-purple-900">
                    The ball is under the{" "}
                    <span className="uppercase underline decoration-wavy decoration-pink-500">
                      {cupResult}
                    </span>{" "}
                    cup!
                  </span>
                </div>
              ) : (
                <span className="text-gray-500 text-lg">
                  {debugState?.status === "waiting"
                    ? "Waiting for cups to be detected..."
                    : "Game in progress…"}
                </span>
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
            {debugState ? JSON.stringify(debugState, null, 2) : "No data"}
          </pre>
        </div>
      </div>
    </div>
  );
}
