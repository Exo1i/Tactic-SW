"use client";
import React, { useState, useRef, useEffect, useCallback } from 'react';
import styles from './ShootingGame.module.css'; 

const ShootingGamePage = () => {
  const [offsetX, setOffsetX] = useState(4); 
  const [offsetY, setOffsetY] = useState(18); 
  const [focalLength, setFocalLength] = useState(580); 
  const [targetColor, setTargetColor] = useState('red');
  const [noBalloonTimeout, setNoBalloonTimeout] = useState(1000); // Changed default to 10 seconds
  const [ws, setWs] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [gameState, setGameState] = useState(null);
  const [statusMessage, setStatusMessage] = useState('Not Connected. Configure and Start.');
  const [isGameRunning, setIsGameRunning] = useState(false); 
  const [streamUrl, setStreamUrl] = useState(null); 

  const backendUrl = process.env.NEXT_PUBLIC_BACKEND_WS_URL || 'ws://localhost:8000';
  const backendHttpUrl = process.env.NEXT_PUBLIC_BACKEND_HTTP_URL || 'http://localhost:8000';

  const connectWebSocket = useCallback(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      console.log("WebSocket already open.");
      return ws;
    }
    console.log("Attempting to connect WebSocket...");
    const newWs = new WebSocket(`${backendUrl}/ws/target-shooter`);

    newWs.onopen = () => {
      console.log('Target Shooter WebSocket connected');
      setIsConnected(true);
      setStatusMessage('Connected. Send initial config via "Start Shoot" button.');
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
        if (data.status === 'ended') {
          setStatusMessage(`Game Ended: ${data.message}`);
          setIsGameRunning(false);
          setStreamUrl(null); // Clear stream URL when game ends
        } else if (data.status === 'ok' || data.status === 'command_processed') {
            if (isGameRunning && data.message) {
                // Example: setStatusMessage(data.message);
            }
        }
      } catch (error) {
        console.error('Error processing message from backend:', error);
        setStatusMessage('Error processing backend message.');
      }
    };

    newWs.onerror = (error) => {
      console.error('WebSocket error:', error);
      setStatusMessage('WebSocket error. Check console.');
      setIsConnected(false);
      setIsGameRunning(false);
    };

    newWs.onclose = () => {
      console.log('Target Shooter WebSocket disconnected');
      setIsConnected(false);
      setIsGameRunning(false);
      setStatusMessage('Disconnected. Reconnect to play.');
      setWs(null);
    };
    return newWs;
  }, [backendUrl, ws, isGameRunning]);

  const handleStartShoot = async () => {
    setStatusMessage('Starting game with current settings...');
    
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
        setStatusMessage('Configuration sent. Backend will start streaming frames.');
        setIsGameRunning(true);
        
        // Add a timestamp and random number to force reload and avoid caching
        const timestamp = Date.now();
        const random = Math.floor(Math.random() * 10000);
        setStreamUrl(`${backendHttpUrl}/stream/target-shooter?t=${timestamp}&r=${random}`);
        
        console.log(`Stream URL set to: ${backendHttpUrl}/stream/target-shooter?t=${timestamp}&r=${random}`);
      } else if (socket.readyState === WebSocket.CONNECTING) {
        console.log("WebSocket is connecting, waiting to send config...");
        setTimeout(() => sendConfigWhenReady(socket), 200);
      } else {
        setStatusMessage("WebSocket not open. Cannot send configuration.");
        console.error("WebSocket not open for sending config. State: " + socket.readyState);
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
      console.warn("Attempted to send command while WebSocket is not open:", command);
    }
  };

  const handleEndGame = () => {
    setStatusMessage('Ending game...');
    sendCommandToBackend({ action: 'end_game' });
    setIsGameRunning(false);
    setStreamUrl(null);
  };

  const handleEmergency = () => {
    setStatusMessage('Emergency stop initiated...');
    sendCommandToBackend({ action: 'emergency_stop' });
    setIsGameRunning(false);
    setStreamUrl(null);
  };

  const handleReloadStream = useCallback(() => {
    if (isGameRunning) {
      const timestamp = Date.now();
      const random = Math.floor(Math.random() * 10000);
      const newStreamUrl = `${backendHttpUrl}/stream/target-shooter?t=${timestamp}&r=${random}`;
      console.log(`Reloading stream with new URL: ${newStreamUrl}`);
      setStreamUrl(newStreamUrl);
      setStatusMessage("Attempting to reload stream...");
    }
  }, [isGameRunning, backendHttpUrl]);

  useEffect(() => {
    return () => {
      if (ws) {
        console.log("Closing WebSocket on component unmount.");
        ws.close();
      }
      setStreamUrl(null);
    };
  }, []);

  return (
    <div className={styles.container}>
      <h1>Target Shooter Game</h1>
      <div className={styles.controlsConfig}>
        <div className={styles.inputGroup}>
          <label>Laser Offset X (cm):</label>
          <input type="number" value={offsetX} onChange={(e) => setOffsetX(e.target.value)} disabled={isGameRunning} />
        </div>
        <div className={styles.inputGroup}>
          <label>Laser Offset Y (cm):</label>
          <input type="number" value={offsetY} onChange={(e) => setOffsetY(e.target.value)} disabled={isGameRunning} />
        </div>
        <div className={styles.inputGroup}>
          <label>Focal Length (px):</label>
          <input type="number" value={focalLength} onChange={(e) => setFocalLength(e.target.value)} disabled={isGameRunning} />
        </div>
        <div className={styles.inputGroup}>
          <label>Target Color:</label>
          <select value={targetColor} onChange={(e) => setTargetColor(e.target.value)} disabled={isGameRunning}>
            <option value="yellow">Yellow</option>
            <option value="red">Red</option>
            <option value="green">Green</option>
          </select>
        </div>
        <div className={styles.inputGroup}>
          <label>No Balloon Timeout (s):</label>
          <input type="number" value={noBalloonTimeout} onChange={(e) => setNoBalloonTimeout(e.target.value)} disabled={isGameRunning} />
        </div>
      </div>

      <div className={styles.actionButtons}>
        <button onClick={handleStartShoot} disabled={isGameRunning}>Start Shoot</button>
        <button onClick={handleEndGame} disabled={!isGameRunning && !isConnected}>End Game</button>
        <button onClick={handleEmergency} disabled={!isConnected} className={styles.emergencyButton}>Emergency Stop</button>
      </div>
      
      <p className={styles.statusMessage}>Status: {statusMessage}</p>

      <div className={styles.videoContainer}>
        <div className={styles.videoFeed}>
          <h2>Processed Feed from Backend</h2>
          {isGameRunning && streamUrl ? (
            <React.Fragment>
              <img
                src={streamUrl}
                alt="Processed MJPEG stream"
                className={styles.videoElement}
                style={{ background: "#222" }}
                onLoad={() => {
                  console.log("Stream loaded successfully.");
                }}
                onError={(e) => {
                  console.error("Error loading stream:", e);
                  setStatusMessage("Error loading video stream. Attempting to reload...");
                  setTimeout(handleReloadStream, 3000);
                }}
              />
              <p style={{fontSize: '0.8rem', color: '#666'}}>
                Stream URL: {streamUrl.split('?')[0]}
              </p>
            </React.Fragment>
          ) : (
            <div className={styles.noFramePlaceholder}>
              {isGameRunning ? "Waiting for processed frame..." : "Game not started or no feed."}
            </div>
          )}
        </div>
      </div>

      {gameState && (
        <div className={styles.gameState}>
          <h2>Game State</h2>
          <p>Pan: {gameState.pan?.toFixed(1)} | Tilt: {gameState.tilt?.toFixed(1)}</p>
          <p>Depth: {gameState.depth_cm?.toFixed(1)} cm</p>
          <p>Target Color: {gameState.target_color}</p>
          <p>Timeout Setting: {gameState.no_balloon_timeout_setting}s</p>
          <p>Shots Fired (Angles Saved): {gameState.shot_angles_count}</p>
          <p>KP_X: {gameState.kp_x?.toFixed(3)} | KP_Y: {gameState.kp_y?.toFixed(3)}</p>
          {gameState.game_over_timeout && <p style={{color: 'red'}}>GAME OVER DUE TO TIMEOUT</p>}
          {gameState.game_requested_stop && !gameState.game_over_timeout && <p style={{color: 'orange'}}>GAME STOPPED BY USER</p>}
        </div>
      )}
    </div>
  );
};

export default ShootingGamePage;

