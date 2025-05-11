"use client";
import { useState, useRef, useEffect } from "react";
import { initializeVideoSource } from "@/utils/cameraUtils";
import ShapeVisualizer from "./ShapeVisualizer";

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
  const [statusMessage, setStatusMessage] = useState(
    "Please position the board and press Start Game when ready."
  );
  const [gridShapes, setGridShapes] = useState([
    ["Unknown", "Unknown", "Unknown", "Unknown"],
    ["Unknown", "Unknown", "Unknown", "Unknown"],
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

  // Add loading states for game actions
  const [isGameStarting, setIsGameStarting] = useState(false);
  const [isGameResetting, setIsGameResetting] = useState(false);

  // Add a state for IP camera validation
  const [ipCameraStatus, setIpCameraStatus] = useState("idle"); // 'idle', 'testing', 'success', 'error'

  useEffect(() => {
    let stopped = false;

    const initCamera = async () => {
      setIsCameraLoading(true);
      if (appliedCameraSettings.useIpCamera) {
        if (videoRef.current && videoRef.current.srcObject) {
          videoRef.current.srcObject.getTracks().forEach((t) => t.stop());
          videoRef.current.srcObject = null;
        }

        // Add a timer to detect if the IP camera failed to load
        const ipCamLoadTimer = setTimeout(() => {
          if (isCameraLoading) {
            setStatus("IP Camera Error");
            setIsCameraLoading(false);
            setStatusMessage(
              "Could not connect to IP camera. Please check settings."
            );
          }
        }, 5000);

        // Clear the timer if component unmounts or camera loads successfully
        return () => clearTimeout(ipCamLoadTimer);
      } else {
        try {
          await initializeVideoSource(videoRef.current, appliedCameraSettings);
          setIsCameraLoading(false);
        } catch (error) {
          console.error("Error initializing camera:", error);
          setStatus("Camera Error");
          setStatusMessage(
            "Error initializing camera. Please check permissions."
          );
          setIsCameraLoading(false);
        }
      }
    };

    initCamera();

    const ws = new WebSocket(`ws://localhost:8000/ws/${gameId}`);
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus("Connected");
      setStatusMessage(
        "Please position the board and press Start Game when ready."
      );
    };

    ws.onclose = () => {
      setStatus("Disconnected");
      // Reset game states if connection is lost
      setGameStarted(false);
      setBoardDetected(false);
      setStatusMessage("Connection lost. Please refresh the page.");
    };

    ws.onerror = () => setStatus("Error");

    ws.onmessage = (event) => {
      setIsBackendLoading(false);
      sendNextFrameRef.current = true;
      try {
        const data = JSON.parse(event.data);
        setOutput(data);

        // Track previous state to detect transitions
        const wasBoardDetected = boardDetected;

        // Update UI based on game state
        if (data.frame) {
          setMainFrame(`data:image/jpeg;base64,${data.frame}`);
        }
        if (data.warped_frame) {
          setWarpedFrame(`data:image/jpeg;base64,${data.warped_frame}`);
          setBoardDetected(true);
        } else {
          setBoardDetected(
            typeof data.board_detected === "boolean"
              ? data.board_detected
              : wasBoardDetected
          );
        }

        setGameStarted(data.game_started || false);
        setGameCompleted(data.game_completed || false);

        if (data.status_message) {
          setStatusMessage(data.status_message);
        } else {
          if (!wasBoardDetected && boardDetected) {
            setStatusMessage("Board detected! Press Start Game to begin.");
          }
        }

        // Ensure gridShapes is always updated from backend if present
        if (
          data.grid_shapes &&
          Array.isArray(data.grid_shapes) &&
          data.grid_shapes.length === 2 &&
          Array.isArray(data.grid_shapes[0]) &&
          Array.isArray(data.grid_shapes[1]) &&
          data.grid_shapes[0].length === 4 &&
          data.grid_shapes[1].length === 4
        ) {
          setGridShapes([[...data.grid_shapes[0]], [...data.grid_shapes[1]]]);
        }
      } catch (error) {
        // ...existing error handling...
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

  // Add a test connection function for IP camera
  const testIpCameraConnection = () => {
    if (!cameraSettings.ipCameraAddress) {
      setIpCameraStatus("error");
      return;
    }

    setIpCameraStatus("testing");
    const testImg = new Image();
    testImg.onload = () => {
      setIpCameraStatus("success");
    };
    testImg.onerror = () => {
      setIpCameraStatus("error");
    };
    testImg.src = cameraSettings.ipCameraAddress;

    // Set a timeout in case the connection hangs
    setTimeout(() => {
      if (ipCameraStatus === "testing") {
        setIpCameraStatus("error");
      }
    }, 5000);
  };

  // Enhanced game control functions with loading states
  const handleStartGame = () => {
    if (wsRef.current && wsRef.current.readyState === 1 && !isGameStarting) {
      setIsGameStarting(true);
      setStatusMessage("Starting new game...");
      wsRef.current.send(JSON.stringify({ action: "start_game" }));

      // Clear any previous state that might be lingering
      setRequiredSwaps([]);
      setCurrentMove(null);

      // The WebSocket message handler will update game state when the backend responds

      // Fallback timeout in case the backend doesn't respond
      setTimeout(() => {
        if (isGameStarting) {
          setIsGameStarting(false);
          setStatusMessage("Game start timed out. Please try again.");
        }
      }, 5000);
    }
  };

  const handleExecuteSwap = () => {
    if (wsRef.current && wsRef.current.readyState === 1) {
      wsRef.current.send(JSON.stringify({ action: "execute_swap" }));
    }
  };

  const handleResetGame = () => {
    if (wsRef.current && wsRef.current.readyState === 1 && !isGameResetting) {
      setIsGameResetting(true);
      setStatusMessage("Resetting game...");
      wsRef.current.send(JSON.stringify({ action: "reset_game" }));

      // Reset local game state
      setGameStarted(false);
      setGameCompleted(false);
      setGridShapes([
        ["Unknown", "Unknown", "Unknown", "Unknown"],
        ["Unknown", "Unknown", "Unknown", "Unknown"],
      ]);
      setRequiredSwaps([]);
      setCurrentMove(null);

      // Fallback timeout in case the backend doesn't respond
      setTimeout(() => {
        if (isGameResetting) {
          setIsGameResetting(false);
          setStatusMessage("Game reset completed.");
        }
      }, 5000);
    }
  };

  // Updated Settings Modal component with test connection functionality
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
              <div className="flex">
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
                  className={`w-full p-2 border rounded-l ${
                    ipCameraStatus === "error"
                      ? "border-red-500"
                      : ipCameraStatus === "success"
                      ? "border-green-500"
                      : "border-gray-300"
                  }`}
                />
                <button
                  onClick={testIpCameraConnection}
                  disabled={
                    ipCameraStatus === "testing" ||
                    !cameraSettings.ipCameraAddress
                  }
                  className={`px-3 py-2 border border-l-0 rounded-r ${
                    ipCameraStatus === "testing"
                      ? "bg-gray-200 cursor-wait"
                      : "bg-blue-500 hover:bg-blue-600 text-white"
                  }`}
                >
                  {ipCameraStatus === "testing" ? "Testing..." : "Test"}
                </button>
              </div>
              {ipCameraStatus === "error" && (
                <p className="text-red-500 text-sm mt-1">
                  Could not connect to camera. Please check the URL.
                </p>
              )}
              {ipCameraStatus === "success" && (
                <p className="text-green-500 text-sm mt-1">
                  Successfully connected to camera.
                </p>
              )}
              <small className="text-gray-500 block mt-1">
                Example: http://192.168.1.100:8080/video
              </small>
              <small className="text-gray-500 block mt-1">
                Make sure your camera supports MJPEG and CORS.
              </small>
            </div>
          )}
          <div className="flex justify-between">
            <button
              onClick={() => {
                setShowSettings(false);
                setIpCameraStatus("idle");
              }}
              className="px-3 py-1 bg-gray-300 text-gray-700 rounded hover:bg-gray-400"
            >
              Cancel
            </button>
            <button
              onClick={() => {
                setAppliedCameraSettings(cameraSettings);
                setShowSettings(false);
                setIpCameraStatus("idle");
              }}
              className={`px-3 py-1 rounded ${
                cameraSettings.useIpCamera && ipCameraStatus === "error"
                  ? "bg-gray-400 cursor-not-allowed"
                  : "bg-green-500 hover:bg-green-600 text-white"
              }`}
              disabled={
                cameraSettings.useIpCamera && ipCameraStatus === "error"
              }
            >
              Apply Settings
            </button>
          </div>
        </div>
      </div>
    );

  // For debugging - add this at the end of the component before the return statement
  useEffect(() => {
    console.log("Button state:", {
      boardDetected,
      gameStarted,
      gameCompleted,
      isGameStarting,
      isGameResetting,
      buttonDisabled:
        !boardDetected ||
        gameStarted ||
        gameCompleted ||
        isGameStarting ||
        isGameResetting,
    });
  }, [
    boardDetected,
    gameStarted,
    gameCompleted,
    isGameStarting,
    isGameResetting,
  ]);

  return (
    <div className="flex flex-col items-center gap-4 p-4 min-h-screen bg-gradient-to-br from-gray-100 to-gray-300">
      <SettingsModal />
      <div className="w-full max-w-4xl bg-white rounded-xl shadow-lg p-6 mt-4">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-4">
          <h1 className="text-2xl font-bold mb-2 md:mb-0">
            Shape Sorting Game
          </h1>
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
          {/* Add debug info */}
          <p className="text-xs text-gray-500 mt-1">
            Board Detected: {boardDetected ? "Yes" : "No"} | Game Started:{" "}
            {gameStarted ? "Yes" : "No"} | Game Completed:{" "}
            {gameCompleted ? "Yes" : "No"}
          </p>
        </div>

        {/* Game Controls */}
        <div className="flex flex-wrap gap-3 mb-6">
          <button
            onClick={() => {
              // Force update board detection before starting game
              setBoardDetected(true);
              handleStartGame();
            }}
            disabled={
              gameStarted || gameCompleted || isGameStarting || isGameResetting
              // Removed !boardDetected condition temporarily for troubleshooting
            }
            className={`px-4 py-2 ${
              gameStarted || gameCompleted || isGameStarting || isGameResetting
                ? "bg-gray-400 cursor-not-allowed"
                : "bg-green-600 hover:bg-green-700"
            } text-white rounded flex items-center`}
          >
            {isGameStarting ? (
              <>
                <svg
                  className="animate-spin -ml-1 mr-2 h-4 w-4 text-white"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  ></circle>
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  ></path>
                </svg>
                Starting...
              </>
            ) : (
              <>
                <svg
                  className="mr-2 h-4 w-4 text-white"
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 20 20"
                  fill="currentColor"
                >
                  <path
                    fillRule="evenodd"
                    d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z"
                    clipRule="evenodd"
                  />
                </svg>
                Start Game
              </>
            )}
          </button>
          <button
            onClick={handleExecuteSwap}
            disabled={
              !gameStarted ||
              gameCompleted ||
              requiredSwaps.length === 0 ||
              isGameResetting
            }
            className={`px-4 py-2 ${
              !gameStarted ||
              gameCompleted ||
              requiredSwaps.length === 0 ||
              isGameResetting
                ? "bg-gray-400 cursor-not-allowed"
                : "bg-blue-600 hover:bg-blue-700"
            } text-white rounded`}
          >
            Execute Swap ({requiredSwaps.length})
          </button>
          <button
            onClick={handleResetGame}
            disabled={isGameResetting}
            className={`px-4 py-2 ${
              isGameResetting
                ? "bg-gray-400 cursor-not-allowed"
                : "bg-red-600 hover:bg-red-700"
            } text-white rounded flex items-center`}
          >
            {isGameResetting ? (
              <>
                <svg
                  className="animate-spin -ml-1 mr-2 h-4 w-4 text-white"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  ></circle>
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  ></path>
                </svg>
                Resetting...
              </>
            ) : (
              "Reset Game"
            )}
          </button>
        </div>

        {/* Add explicit board detection button for troubleshooting */}
        <div className="mb-4">
          <button
            onClick={() => setBoardDetected(true)}
            className="px-3 py-1 bg-purple-500 text-white rounded hover:bg-purple-600 text-sm"
          >
            Force Board Detection
          </button>
          <span className="ml-2 text-xs text-gray-500">
            (Use this if Start Game is disabled)
          </span>
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
                  onLoad={() => {
                    setIsCameraLoading(false);
                    setStatus("Connected");
                  }}
                  onError={() => {
                    setStatus("IP Camera Error");
                    setIsCameraLoading(false);
                    setStatusMessage(
                      "IP camera connection failed. Please check the URL."
                    );
                  }}
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
                  <div className="text-center">
                    <svg
                      className="animate-spin mx-auto h-8 w-8 text-white mb-2"
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      ></circle>
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      ></path>
                    </svg>
                    <p>Connecting to camera...</p>
                  </div>
                </div>
              )}
              {status === "IP Camera Error" && !isCameraLoading && (
                <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-60 text-white text-lg">
                  <div className="text-center p-4 max-w-sm">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="mx-auto h-12 w-12 text-red-500 mb-2"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                      />
                    </svg>
                    <p className="mb-2">Failed to connect to IP camera.</p>
                    <button
                      onClick={() => setShowSettings(true)}
                      className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600"
                    >
                      Check Settings
                    </button>
                  </div>
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
                      ? "bg-green-50 border-green-500"
                      : isPartOfSwap
                      ? "bg-yellow-50 border-yellow-500"
                      : "bg-gray-50"
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
            <p>
              Moving from position ({currentMove[0][0]}, {currentMove[0][1]}) to
              ({currentMove[1][0]}, {currentMove[1][1]})
            </p>
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
