// --- START OF FILE page.jsx ---
"use client";
import React, { useState, useRef, useEffect } from "react";

const COLOR_OPTIONS = [
  { name: "Red", value: "red", color: "#ef4444" },
  { name: "Green", value: "green", color: "#22c55e" },
  { name: "Yellow", value: "yellow", color: "#fde047" },
  // Add more if your get_color_name supports them
];

export default function ShooterGamePage() {
  const [selectedColor, setSelectedColor] = useState("red");
  const [shooterOffsetX, setShooterOffsetX] = useState(5);
  const [shooterOffsetY, setShooterOffsetY] = useState(18);
  const [focalLength, setFocalLength] = useState(580);
  const [kpX, setKpX] = useState(0.05);
  const [kpY, setKpY] = useState(0.05);
  const [cameraSource, setCameraSource] = useState("0"); // "0" for default webcam, or IP URL
  const [enableBackendPreview, setEnableBackendPreview] = useState(false);

  const [gameStarted, setGameStarted] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState("Disconnected");
  const [processedFrame, setProcessedFrame] = useState(null);
  const [balloonsInfo, setBalloonsInfo] = useState([]); // To display balloon data
  const wsRef = useRef(null);

  useEffect(() => {
    return () => {
      // Cleanup on unmount
      if (wsRef.current) {
        console.log("Closing WebSocket on unmount");
        wsRef.current.close();
      }
    };
  }, []);

  const handleStartGame = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      console.log(
        "Game already started or WebSocket open, not starting again."
      );
      return;
    }

    const config = {
      action: "start", // This might not be strictly needed by backend GameSession if config is always first
      color: selectedColor,
      shooterOffsetX,
      shooterOffsetY,
      focalLength,
      kp_x: kpX,
      kp_y: kpY,
      enable_local_preview: enableBackendPreview,
      camera_source: isNaN(parseInt(cameraSource))
        ? cameraSource
        : parseInt(cameraSource), // Send as int if number
    };
    console.log("Attempting to start game with config:", config);

    wsRef.current = new WebSocket(`ws://localhost:8000/ws/shooter-game`);

    wsRef.current.onopen = () => {
      setConnectionStatus("Connected");
      console.log("WebSocket connected, sending configuration...");
      wsRef.current.send(JSON.stringify(config));
      setGameStarted(true); // Assuming sending config means game is trying to start
      setProcessedFrame(null); // Clear previous frame
      setBalloonsInfo([]);
    };

    wsRef.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.status === "error") {
          console.error("Backend Error:", data.message);
          setConnectionStatus(`Error: ${data.message.substring(0, 100)}`);
          // Optionally stop game on critical backend error
          // setGameStarted(false);
          return;
        }

        if (data.processed_frame) {
          // Backend now sends 'processed_frame'
          setProcessedFrame(`data:image/jpeg;base64,${data.processed_frame}`);
        }
        if (data.balloons_info) {
          setBalloonsInfo(data.balloons_info);
        }
        if (data.status && data.status.includes("_acknowledged")) {
          console.log("Command acknowledged by backend:", data.status);
        }
      } catch (error) {
        console.error(
          "Error processing message from backend:",
          error,
          "Data:",
          event.data
        );
      }
    };

    wsRef.current.onclose = (event) => {
      setConnectionStatus(`Disconnected (Code: ${event.code})`);
      setGameStarted(false);
      setProcessedFrame(null);
      console.log("WebSocket connection closed.", event.reason);
    };

    wsRef.current.onerror = (error) => {
      setConnectionStatus("Connection Error");
      setGameStarted(false); // Assume game cannot run if WebSocket errors out
      console.error("WebSocket error:", error);
    };
  };

  const sendCommand = (command) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      console.log("Sending command:", command);
      wsRef.current.send(JSON.stringify(command));
    } else {
      console.warn("WebSocket not open, cannot send command:", command);
    }
  };

  const handleEndGame = () => sendCommand({ action: "stop" });
  const handleEmergency = () => sendCommand({ action: "emergency" });

  return (
    <div className="flex flex-col items-center gap-4 p-6 min-h-screen bg-gray-100">
      <h1 className="text-3xl font-bold mb-2">Shooter Game (Backend Camera)</h1>

      <div className="p-3 bg-white shadow rounded-lg w-full max-w-md">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-semibold">Status</h2>
          <span
            className={`px-2 py-1 rounded text-xs font-medium ${
              connectionStatus.startsWith("Connected")
                ? "bg-green-100 text-green-700"
                : connectionStatus.startsWith("Error") ||
                  connectionStatus.startsWith("Connection Error")
                ? "bg-red-100 text-red-700"
                : "bg-yellow-100 text-yellow-700"
            }`}
          >
            {connectionStatus}
          </span>
        </div>

        <h3 className="text-md font-semibold mb-1">Target Color:</h3>
        <div className="flex gap-2 mb-3">
          {COLOR_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              type="button"
              onClick={() => setSelectedColor(opt.value)}
              className={`p-1 border-2 rounded-md ${
                selectedColor === opt.value
                  ? "border-blue-500 ring-2 ring-blue-300"
                  : "border-gray-300"
              }`}
              disabled={gameStarted}
            >
              <div
                className="w-10 h-10 rounded"
                style={{ backgroundColor: opt.color }}
              ></div>
              <span className="text-xs mt-1">{opt.name}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="p-3 bg-white shadow rounded-lg w-full max-w-md grid grid-cols-2 gap-3">
        {[
          {
            label: "Shooter Offset X (cm)",
            value: shooterOffsetX,
            setter: setShooterOffsetX,
            type: "number",
          },
          {
            label: "Shooter Offset Y (cm)",
            value: shooterOffsetY,
            setter: setShooterOffsetY,
            type: "number",
          },
          {
            label: "Focal Length (px)",
            value: focalLength,
            setter: setFocalLength,
            type: "number",
          },
          {
            label: "Kp X",
            value: kpX,
            setter: setKpX,
            type: "number",
            step: "0.001",
          },
          {
            label: "Kp Y",
            value: kpY,
            setter: setKpY,
            type: "number",
            step: "0.001",
          },
          {
            label: "Camera Source (0, 1,.. or URL)",
            value: cameraSource,
            setter: setCameraSource,
            type: "text",
          },
        ].map((input) => (
          <label key={input.label} className="flex flex-col text-sm">
            {input.label}:
            <input
              type={input.type}
              value={input.value}
              step={input.step || "any"}
              onChange={(e) =>
                input.setter(
                  input.type === "number"
                    ? Number(e.target.value)
                    : e.target.value
                )
              }
              className="mt-1 p-1.5 border rounded-md text-sm"
              disabled={gameStarted}
            />
          </label>
        ))}
        <label className="col-span-2 flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={enableBackendPreview}
            onChange={(e) => setEnableBackendPreview(e.target.checked)}
            className="form-checkbox h-4 w-4 text-blue-600"
            disabled={gameStarted}
          />
          Enable Backend Local Preview
        </label>
      </div>

      <div className="flex gap-3 mt-2">
        <button
          className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:bg-gray-400"
          onClick={handleStartGame}
          disabled={gameStarted}
        >
          Start Game
        </button>
        <button
          className="px-4 py-2 bg-orange-500 text-white rounded hover:bg-orange-600 disabled:bg-gray-400"
          onClick={handleEndGame}
          disabled={!gameStarted}
        >
          End Game
        </button>
        <button
          className="px-4 py-2 bg-red-700 text-white rounded hover:bg-red-800 disabled:bg-gray-400"
          onClick={handleEmergency}
          disabled={!gameStarted}
        >
          Emergency Stop
        </button>
      </div>

      <div className="mt-4 w-full max-w-2xl">
        <h3 className="text-center font-semibold mb-1">
          Processed Feed (from Backend)
        </h3>
        <div className="w-full aspect-[4/3] rounded bg-black flex items-center justify-center overflow-hidden border-2 border-gray-300 shadow-lg">
          {processedFrame ? (
            <img
              src={processedFrame}
              alt="Processed feed"
              className="max-w-full max-h-full object-contain"
            />
          ) : (
            <span className="text-gray-400">
              {gameStarted ? "Waiting for feed..." : "Game not started"}
            </span>
          )}
        </div>
      </div>

      {balloonsInfo.length > 0 && (
        <div className="mt-3 p-2 bg-gray-200 rounded w-full max-w-2xl text-xs">
          <h4 className="font-semibold mb-1">Detected Balloons:</h4>
          <ul>
            {balloonsInfo.slice(0, 5).map(
              (
                b,
                i // Show first 5
              ) => (
                <li key={i}>
                  {b.label} ({b.color}) - Conf: {b.conf.toFixed(2)} - Depth:{" "}
                  {b.depth_cm}cm - Pos: ({b.pos[0]},{b.pos[1]})
                </li>
              )
            )}
          </ul>
        </div>
      )}
    </div>
  );
}
// --- END OF FILE page.jsx ---
