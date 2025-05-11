"use client";
import React, { useState, useRef, useEffect, useCallback } from "react";
import styles from "./ShootingGame.module.css";

// Define SettingsModalComponent outside ShootingGamePage
const SettingsModalComponent = ({
  showSettings,
  setShowSettings,
  cameraSettings,
  setCameraSettings, // Pass the full setter
  ipCameraInputRef,
  applySettingsCallback, // Renamed for clarity
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
            id="useIpCameraShooting" // Unique ID
            checked={cameraSettings.useIpCamera}
            onChange={(e) =>
              setCameraSettings({
                // Use the passed setter
                ...cameraSettings,
                useIpCamera: e.target.checked,
              })
            }
            className="mr-2"
          />
          <label htmlFor="useIpCameraShooting">Use IP Camera</label>
        </div>
        {cameraSettings.useIpCamera && (
          <div className="mb-3">
            <label htmlFor="ipCameraAddressShooting" className="block mb-1">
              {" "}
              {/* Unique ID */}
              IP Camera URL:
            </label>
            <input
              ref={ipCameraInputRef}
              type="text"
              id="ipCameraAddressShooting" // Unique ID
              value={cameraSettings.ipCameraAddress}
              onChange={(e) =>
                setCameraSettings({
                  // Use the passed setter
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
          onClick={applySettingsCallback}
          className="px-3 py-1 bg-green-500 text-white rounded hover:bg-green-600"
        >
          Apply Settings
        </button>
      </div>
    </div>
  );
};

const ShootingGamePage = () => {
  const [offsetX, setOffsetX] = useState(4);
  const [offsetY, setOffsetY] = useState(18);
  const [focalLength, setFocalLength] = useState(580);
  const [targetColor, setTargetColor] = useState("yellow");
  const [noBalloonTimeout, setNoBalloonTimeout] = useState(10);
  const [ws, setWs] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [gameState, setGameState] = useState(null);
  const [statusMessage, setStatusMessage] = useState(
    "Not Connected. Configure and Start."
  );
  const [isGameRunning, setIsGameRunning] = useState(false);
  const [streamUrl, setStreamUrl] = useState(null);
  const ipCameraInputRef = useRef(null); // Ref for IP camera input in modal
  const [isSecureContext, setIsSecureContext] = useState(true);

  // Camera settings modal state (remains in parent)
  const [showSettings, setShowSettings] = useState(false);
  const [cameraSettings, setCameraSettings] = useState({
    useIpCamera: false,
    ipCameraAddress: "",
  });
  // Applied settings can also be managed if needed, or directly use cameraSettings
  // For this game, it seems settings are applied immediately or on game start.

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

  // Handle Escape key press for modal
  useEffect(() => {
    const handleEscapeKey = (e) => {
      if (e.key === "Escape" && showSettings) {
        setShowSettings(false);
      }
    };

    window.addEventListener("keydown", handleEscapeKey);
    return () => window.removeEventListener("keydown", handleEscapeKey);
  }, [showSettings]);

  const backendUrl =
    process.env.NEXT_PUBLIC_BACKEND_WS_URL || "ws://localhost:8000";
  const backendHttpUrl =
    process.env.NEXT_PUBLIC_BACKEND_HTTP_URL || "http://localhost:8000";

  const connectWebSocket = useCallback(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      console.log("WebSocket already open.");
      return ws;
    }
    console.log("Attempting to connect WebSocket...");
    const newWs = new WebSocket(`${backendUrl}/ws/target-shooter`);

    newWs.onopen = () => {
      console.log("Target Shooter WebSocket connected");
      setIsConnected(true);
      setStatusMessage(
        'Connected. Send initial config via "Start Shoot" button.'
      );
      setWs(newWs);
    };

    newWs.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.game_state) {
          setGameState(data.game_state);
        }
        if (data.message) {
          setStatusMessage(data.message);
        }
        if (data.status === "ended") {
          setStatusMessage(`Game Ended: ${data.message}`);
          setIsGameRunning(false);
        } else if (
          data.status === "ok" ||
          data.status === "command_processed"
        ) {
          if (!isGameRunning) setIsGameRunning(true);
        }
      } catch (error) {
        console.error("Error processing message from backend:", error);
        setStatusMessage("Error processing backend message.");
      }
    };

    newWs.onerror = (error) => {
      console.error("WebSocket error:", error);
      setStatusMessage("WebSocket error. Check console.");
      setIsConnected(false);
      setIsGameRunning(false);
    };

    newWs.onclose = () => {
      console.log("Target Shooter WebSocket disconnected");
      setIsConnected(false);
      setIsGameRunning(false);
      setStatusMessage("Disconnected. Reconnect to play.");
      setWs(null);
    };
    return newWs;
  }, [backendUrl, ws, isGameRunning]);

  const handleStartShoot = async () => {
    setStatusMessage("Starting game with current settings...");

    let currentWs = ws;
    if (!currentWs || currentWs.readyState !== WebSocket.OPEN) {
      currentWs = connectWebSocket();
      setWs(currentWs);
    }

    const sendConfigWhenReady = (socket) => {
      if (socket.readyState === WebSocket.OPEN) {
        const initialConfig = {
          action: "initial_config",
          focal_length: parseFloat(focalLength),
          laser_offset_cm_x: parseFloat(offsetX),
          laser_offset_cm_y: parseFloat(offsetY),
          target_color: targetColor,
          no_balloon_timeout: parseFloat(noBalloonTimeout),
        };
        socket.send(JSON.stringify(initialConfig));
        setStatusMessage(
          "Configuration sent. Backend will start streaming frames."
        );
        setIsGameRunning(true);
        setStreamUrl(`${backendHttpUrl}/stream/target-shooter?${Date.now()}`);
      } else if (socket.readyState === WebSocket.CONNECTING) {
        console.log("WebSocket is connecting, waiting to send config...");
        setTimeout(() => sendConfigWhenReady(socket), 200);
      } else {
        setStatusMessage("WebSocket not open. Cannot send configuration.");
        console.error(
          "WebSocket not open for sending config. State: " + socket.readyState
        );
        setIsGameRunning(false);
      }
    };
    sendConfigWhenReady(currentWs);
  };

  const sendCommandToBackend = (command) => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(command));
      console.log("Sent command:", command);
    } else {
      setStatusMessage("Not connected. Cannot send command.");
      console.warn(
        "Attempted to send command while WebSocket is not open:",
        command
      );
    }
  };

  const handleEndGame = () => {
    setStatusMessage("Ending game...");
    sendCommandToBackend({ action: "end_game" });
    setIsGameRunning(false);
    setStreamUrl(null);
  };

  const handleEmergency = () => {
    setStatusMessage("Emergency stop initiated...");
    sendCommandToBackend({ action: "emergency_stop" });
    setIsGameRunning(false);
    setStreamUrl(null);
  };

  useEffect(() => {
    return () => {
      if (ws) {
        console.log("Closing WebSocket on component unmount.");
        ws.close();
      }
      setStreamUrl(null);
    };
  }, []);

  // Apply camera settings (called from modal)
  const applyModalSettings = () => {
    // Logic to apply cameraSettings, e.g., re-initiate connection or send to backend
    // For now, just closes the modal. The actual application of these settings
    // would depend on how the Shooting Game uses IP cameras (not fully implemented here).
    console.log("Applying camera settings:", cameraSettings);
    // If settings affect the WebSocket or stream, handle that here.
    // For example, if IP camera is used for the main feed:
    // if (cameraSettings.useIpCamera && cameraSettings.ipCameraAddress) {
    //   setStreamUrl(cameraSettings.ipCameraAddress); // Or a modified URL
    // } else {
    //   // Revert to default/device camera logic if applicable
    // }
    setShowSettings(false);
  };

  return (
    <div className={styles.container}>
      <SettingsModalComponent
        showSettings={showSettings}
        setShowSettings={setShowSettings}
        cameraSettings={cameraSettings}
        setCameraSettings={setCameraSettings} // Pass the full setter
        ipCameraInputRef={ipCameraInputRef}
        applySettingsCallback={applyModalSettings}
      />

      <h1>Target Shooter Game</h1>
      {!isSecureContext && (
        <div className="mb-2 text-red-600 text-sm p-2 bg-red-100 border border-red-300 rounded">
          Warning: Device camera access (if used by this game for non-IP camera
          mode) requires HTTPS in most browsers.
        </div>
      )}
      {cameraSettings.useIpCamera && cameraSettings.ipCameraAddress && (
        <div className="mb-2 text-blue-600 text-sm p-2 bg-blue-100 border border-blue-300 rounded">
          Using IP Camera: {cameraSettings.ipCameraAddress}
        </div>
      )}
      <div className={styles.controlsConfig}>
        <div className={styles.inputGroup}>
          <label>Laser Offset X (cm):</label>
          <input
            type="number"
            value={offsetX}
            onChange={(e) => setOffsetX(e.target.value)}
            disabled={isGameRunning}
          />
        </div>
        <div className={styles.inputGroup}>
          <label>Laser Offset Y (cm):</label>
          <input
            type="number"
            value={offsetY}
            onChange={(e) => setOffsetY(e.target.value)}
            disabled={isGameRunning}
          />
        </div>
        <div className={styles.inputGroup}>
          <label>Focal Length (px):</label>
          <input
            type="number"
            value={focalLength}
            onChange={(e) => setFocalLength(e.target.value)}
            disabled={isGameRunning}
          />
        </div>
        <div className={styles.inputGroup}>
          <label>Target Color:</label>
          <select
            value={targetColor}
            onChange={(e) => setTargetColor(e.target.value)}
            disabled={isGameRunning}
          >
            <option value="yellow">Yellow</option>
            <option value="red">Red</option>
            <option value="green">Green</option>
          </select>
        </div>
        <div className={styles.inputGroup}>
          <label>No Balloon Timeout (s):</label>
          <input
            type="number"
            value={noBalloonTimeout}
            onChange={(e) => setNoBalloonTimeout(e.target.value)}
            disabled={isGameRunning}
          />
        </div>
      </div>

      <div className={styles.actionButtons}>
        <button onClick={handleStartShoot} disabled={isGameRunning}>
          Start Shoot
        </button>
        <button
          onClick={handleEndGame}
          disabled={!isGameRunning && !isConnected}
        >
          End Game
        </button>
        <button
          onClick={handleEmergency}
          disabled={!isConnected}
          className={styles.emergencyButton}
        >
          Emergency Stop
        </button>
      </div>

      <p className={styles.statusMessage}>Status: {statusMessage}</p>

      <div className={styles.videoContainer}>
        <div className={styles.videoFeed}>
          <h2>Processed Feed from Backend</h2>
          {isGameRunning && streamUrl ? (
            <img
              src={streamUrl}
              alt="Processed MJPEG stream"
              className={styles.videoElement}
              style={{ background: "#222" }}
            />
          ) : (
            <div className={styles.noFramePlaceholder}>
              {isGameRunning
                ? "Waiting for processed frame..."
                : "Game not started or no feed."}
            </div>
          )}
        </div>
      </div>

      {gameState && (
        <div className={styles.gameState}>
          <h2>Game State</h2>
          <p>
            Pan: {gameState.pan?.toFixed(1)} | Tilt:{" "}
            {gameState.tilt?.toFixed(1)}
          </p>
          <p>Depth: {gameState.depth_cm?.toFixed(1)} cm</p>
          <p>Target Color: {gameState.target_color}</p>
          <p>Timeout Setting: {gameState.no_balloon_timeout_setting}s</p>
          <p>Shots Fired (Angles Saved): {gameState.shot_angles_count}</p>
          <p>
            KP_X: {gameState.kp_x?.toFixed(3)} | KP_Y:{" "}
            {gameState.kp_y?.toFixed(3)}
          </p>
          {gameState.game_over_timeout && (
            <p style={{ color: "red" }}>GAME OVER DUE TO TIMEOUT</p>
          )}
          {gameState.game_requested_stop && !gameState.game_over_timeout && (
            <p style={{ color: "orange" }}>GAME STOPPED BY USER</p>
          )}
        </div>
      )}
    </div>
  );
};

export default ShootingGamePage;
