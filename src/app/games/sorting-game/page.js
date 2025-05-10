"use client";
import { useState, useRef, useEffect } from "react";
import { initializeVideoSource } from "@/utils/cameraUtils";

export default function SortingGamePage() {
  const gameId = "sorting-game";
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const ipCamImgRef = useRef(null);

  const [status, setStatus] = useState("Connecting...");
  const [output, setOutput] = useState(null);
  const [mainFrame, setMainFrame] = useState(null);
  const [warpedFrame, setWarpedFrame] = useState(null);
  const [showSettings, setShowSettings] = useState(false);
  const [cameraSettings, setCameraSettings] = useState({
    useIpCamera: false,
    ipCameraAddress: "",
  });
  const [appliedCameraSettings, setAppliedCameraSettings] = useState({
    useIpCamera: false,
    ipCameraAddress: "",
  });

  // Game states
  const [boardDetected, setBoardDetected] = useState(false);
  const [gameStarted, setGameStarted] = useState(false);
  const [gameCompleted, setGameCompleted] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");
  const [gridShapes, setGridShapes] = useState([
    ["Unknown", "Unknown", "Unknown", "Unknown"],
    ["Unknown", "Unknown", "Unknown", "Unknown"]
  ]);
  const [requiredSwaps, setRequiredSwaps] = useState([]);
  const [currentMove, setCurrentMove] = useState(null);

  // Loading indicators
  const [isCameraLoading, setIsCameraLoading] = useState(false);
  const [isBackendLoading, setIsBackendLoading] = useState(false);

  // Send frame control
  const sendNextFrameRef = useRef(true);
  const lastSentRef = useRef(0);
  const minFrameInterval = 30; // ~30 FPS

  useEffect(() => {
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

    const ws = new WebSocket(`ws://localhost:8000/ws/${gameId}`);
    wsRef.current = ws;

    ws.onopen = () => setStatus("Connected");
    ws.onclose = () => {
      setStatus("Disconnected");
      // Reset game states if connection is lost
      setGameStarted(false);
      setBoardDetected(false);
    };
    ws.onerror = () => setStatus("Error");

    ws.onmessage = (event) => {
      setIsBackendLoading(false);
      sendNextFrameRef.current = true;
      try {
        const data = JSON.parse(event.data);
        setOutput(data);
        
        // Update UI based on game state
        if (data.frame) {
          setMainFrame(`data:image/jpeg;base64,${data.frame}`);
        }
        if (data.warped_frame) {
          setWarpedFrame(`data:image/jpeg;base64,${data.warped_frame}`);
        }
        
        // Update game states
        setBoardDetected(data.board_detected || false);
        setGameStarted(data.game_started || false);
        setGameCompleted(data.game_completed || false);
        setStatusMessage(data.status_message || "");
        
        if (data.grid_shapes) {
          setGridShapes(data.grid_shapes);
        }
        
        if (data.required_swaps) {
          setRequiredSwaps(data.required_swaps);
        }
        
        setCurrentMove(data.current_move);
      } catch (error) {
        console.error("Error parsing WebSocket message:", error);
        setOutput(error.message);
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
        } catch (e) {
          // Drawing may fail if image is not loaded yet
        }
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
  }, [gameId, appliedCameraSettings]);

  const handleCameraSettingsChange = (newSettings) => {
    setCameraSettings(newSettings);
  };

  // Game control functions
  const handleStartGame = () => {
    if (wsRef.current && wsRef.current.readyState === 1) {
      wsRef.current.send(JSON.stringify({ action: "start_game" }));
    }
  };

  const handleExecuteSwap = () => {
    if (wsRef.current && wsRef.current.readyState === 1) {
      wsRef.current.send(JSON.stringify({ action: "execute_swap" }));
    }
  };

  const handleResetGame = () => {
    if (wsRef.current && wsRef.current.readyState === 1) {
      wsRef.current.send(JSON.stringify({ action: "reset_game" }));
    }
  };

  // Settings Modal component
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

  // Shape visualization component
  const ShapeVisualizer = ({ shape, matched }) => {
    let bgColor = "bg-gray-200"; // Default Unknown
    let shapeName = shape || "Unknown";
    
    // Set background color based on shape
    if (shape === "Circle") bgColor = "bg-blue-500";
    else if (shape === "Square") bgColor = "bg-red-500";
    else if (shape === "Triangle") bgColor = "bg-green-500";
    else if (shape === "Pentagon") bgColor = "bg-yellow-500";
    
    return (
      <div className={`flex flex-col items-center justify-center ${matched ? 'opacity-70' : ''}`}>
        <div className={`w-12 h-12 ${bgColor} rounded-md flex items-center justify-center`}>
          <span className="text-white font-bold text-xs">{shapeName}</span>
        </div>
      </div>
    );
  };

  return (
    <div className="flex flex-col items-center gap-4 p-4 min-h-screen bg-gradient-to-br from-gray-100 to-gray-300">
      <SettingsModal />
      <div className="w-full max-w-4xl bg-white rounded-xl shadow-lg p-6 mt-4">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-4">
          <h1 className="text-2xl font-bold mb-2 md:mb-0">Shape Sorting Game</h1>
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
          </div>
        </div>
        
        {/* Status message */}
        <div className="mb-4 p-3 bg-gray-50 rounded-lg border border-gray-200">
          <p className="text-gray-700">{statusMessage}</p>
        </div>
        
        {/* Game Controls */}
        <div className="flex flex-wrap gap-3 mb-6">
          <button
            onClick={handleStartGame}
            disabled={!boardDetected || gameStarted || gameCompleted}
            className={`px-4 py-2 ${
              !boardDetected || gameStarted || gameCompleted
                ? "bg-gray-400 cursor-not-allowed"
                : "bg-green-600 hover:bg-green-700"
            } text-white rounded`}
          >
            Start Game
          </button>
          <button
            onClick={handleExecuteSwap}
            disabled={!gameStarted || gameCompleted || requiredSwaps.length === 0}
            className={`px-4 py-2 ${
              !gameStarted || gameCompleted || requiredSwaps.length === 0
                ? "bg-gray-400 cursor-not-allowed"
                : "bg-blue-600 hover:bg-blue-700"
            } text-white rounded`}
          >
            Execute Swap ({requiredSwaps.length})
          </button>
          <button
            onClick={handleResetGame}
            className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded"
          >
            Reset Game
          </button>
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
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Camera Feeds */}
          <div className="flex flex-col items-center">
            <div className="mb-2 text-center font-medium">Live Camera Feed</div>
            <div className="relative w-full aspect-video rounded-lg overflow-hidden border-2 border-gray-300 bg-black flex items-center justify-center">
              {appliedCameraSettings.useIpCamera ? (
                <img
                  ref={ipCamImgRef}
                  src={appliedCameraSettings.ipCameraAddress}
                  alt="IP Camera"
                  className="w-full h-full object-contain"
                  crossOrigin="anonymous"
                  onLoad={() => setIsCameraLoading(false)}
                  onError={() => setStatus("IP Camera Error")}
                />
              ) : (
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  className="w-full h-full object-contain"
                  style={{ background: "#222" }}
                />
              )}
              {isCameraLoading && (
                <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-60 text-white text-lg">
                  Loading Camera...
                </div>
              )}
              {mainFrame && (
                <img
                  src={mainFrame}
                  alt="Main Frame"
                  className="absolute inset-0 w-full h-full object-contain"
                />
              )}
            </div>
            <canvas
              ref={canvasRef}
              width={320}
              height={240}
              style={{ display: "none" }}
            />
          </div>
          
          {/* Detected Board */}
          <div className="flex flex-col items-center">
            <div className="mb-2 text-center font-medium">Detected Board</div>
            <div className="relative w-full aspect-video rounded-lg overflow-hidden border-2 border-gray-300 bg-black flex items-center justify-center">
              {warpedFrame ? (
                <img
                  src={warpedFrame}
                  alt="Warped Board"
                  className="w-full h-full object-contain"
                />
              ) : (
                <span className="text-gray-400">
                  {boardDetected ? "Processing board..." : "No board detected"}
                </span>
              )}
              {isBackendLoading && (
                <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-60 text-white text-lg">
                  Processing...
                </div>
              )}
            </div>
          </div>
        </div>
        
        {/* Grid Visualization */}
        <div className="mt-6 p-4 bg-gray-50 rounded-lg border">
          <h3 className="text-lg font-medium mb-4">Detected Shapes</h3>
          <div className="grid grid-cols-4 gap-4">
            {/* Top row (Target) */}
            {gridShapes[0].map((shape, index) => (
              <div 
                key={`top-${index}`}
                className="p-2 border rounded flex justify-center items-center bg-blue-50"
              >
                <ShapeVisualizer shape={shape} matched={false} />
                <div className="ml-2 text-xs text-gray-500">
                  <div>Target</div>
                  <div>({index})</div>
                </div>
              </div>
            ))}
            
            {/* Bottom row (To match) */}
            {gridShapes[1].map((shape, index) => {
              const targetShape = gridShapes[0][index];
              const isMatched = shape === targetShape && shape !== "Unknown";
              const isPartOfSwap = requiredSwaps.some(
                ([[row, col]]) => row === 1 && col === index
              );
              
              return (
                <div 
                  key={`bottom-${index}`}
                  className={`p-2 border rounded flex justify-center items-center ${
                    isMatched 
                      ? 'bg-green-50 border-green-500' 
                      : isPartOfSwap 
                      ? 'bg-yellow-50 border-yellow-500' 
                      : 'bg-gray-50'
                  }`}
                >
                  <ShapeVisualizer shape={shape} matched={isMatched} />
                  <div className="ml-2 text-xs text-gray-500">
                    <div>{isMatched ? "Matched" : "Current"}</div>
                    <div>({index})</div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
        
        {/* Current Move Display */}
        {currentMove && (
          <div className="mt-4 p-3 bg-yellow-50 border border-yellow-300 rounded-lg">
            <h3 className="font-medium">Current Move</h3>
            <p>Moving from position ({currentMove[0][0]}, {currentMove[0][1]}) to ({currentMove[1][0]}, {currentMove[1][1]})</p>
          </div>
        )}
        
        {/* Game Data for debugging */}
        <div className="mt-6 p-4 bg-gray-50 rounded-lg border">
          <h3 className="text-lg font-medium mb-2">Game Data</h3>
          <pre className="text-xs overflow-x-auto bg-gray-100 p-2 rounded">
            {output ? JSON.stringify(output, null, 2) : "No data"}
          </pre>
        </div>
      </div>
    </div>
  );
}
