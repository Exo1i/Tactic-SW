"use client";
import React, { useState, useRef, useEffect, useCallback } from 'react';
import styles from './ShootingGame.module.css'; 

const ShootingGamePage = () => {
  const [offsetX, setOffsetX] = useState(4); 
  const [offsetY, setOffsetY] = useState(18); 
  const [focalLength, setFocalLength] = useState(580); 
  const [targetColor, setTargetColor] = useState('yellow');
  const [noBalloonTimeout, setNoBalloonTimeout] = useState(10); // Added timeout setting

  const [ws, setWs] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [processedFrame, setProcessedFrame] = useState(null);
  const [gameState, setGameState] = useState(null);
  const [statusMessage, setStatusMessage] = useState('Not Connected. Configure and Start.');
  const [isGameRunning, setIsGameRunning] = useState(false); // Track if game logic is active on backend

  // Removed videoRef, canvasRef, mediaStreamRef, frameIntervalRef

  const backendUrl = process.env.NEXT_PUBLIC_BACKEND_WS_URL || 'ws://localhost:8000';

  const connectWebSocket = useCallback(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      console.log("WebSocket already open.");
      // If already connected and trying to start, send config again if needed
      // This path might be hit if user clicks "Start" multiple times without disconnecting
      return ws;
    }
    console.log("Attempting to connect WebSocket...");
    const newWs = new WebSocket(`${backendUrl}/ws/target-shooter`);

    newWs.onopen = () => {
      console.log('Target Shooter WebSocket connected');
      setIsConnected(true);
      setStatusMessage('Connected. Send initial config via "Start Shoot" button.');
      setWs(newWs);
      // Automatically send initial config if game was intended to start
      // This logic is now primarily in handleStartShoot
    };

    newWs.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.processed_frame) {
          setProcessedFrame(`data:image/jpeg;base64,${data.processed_frame}`);
        }
        if (data.game_state) {
          setGameState(data.game_state);
        }
        if (data.message) {
          setStatusMessage(data.message);
        }
        if (data.status === 'ended') {
          setStatusMessage(`Game Ended: ${data.message}`);
          setIsGameRunning(false); // Mark game as not running
          // Optionally close WS here or let user do it via "End Game"
        } else if (data.status === 'ok' || data.status === 'command_processed') {
            if (!isGameRunning && data.processed_frame) setIsGameRunning(true); // If we get frames, game is running
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
      setProcessedFrame(null); // Clear frame on disconnect
    };
    return newWs;
  }, [backendUrl, ws, isGameRunning]); // Added isGameRunning

  // Removed startVideo, stopVideo, sendFrame callbacks

  const handleStartShoot = async () => {
    setStatusMessage('Starting game with current settings...');
    
    let currentWs = ws;
    if (!currentWs || currentWs.readyState !== WebSocket.OPEN) {
      currentWs = connectWebSocket(); // connectWebSocket now returns the newWs instance
      setWs(currentWs); // Ensure ws state is updated for subsequent operations
    }
    
    const sendConfigWhenReady = (socket) => {
      if (socket.readyState === WebSocket.OPEN) {
        const initialConfig = {
            action: "initial_config",
            focal_length: parseFloat(focalLength),
            laser_offset_cm_x: parseFloat(offsetX),
            laser_offset_cm_y: parseFloat(offsetY),
            target_color: targetColor,
            no_balloon_timeout: parseFloat(noBalloonTimeout), // Send new timeout
            // kp_x, kp_y can also be sent if you want them configurable from frontend
        };
        socket.send(JSON.stringify(initialConfig));
        setStatusMessage('Configuration sent. Backend will start streaming frames.');
        setIsGameRunning(true); // Assume backend will start streaming
      } else if (socket.readyState === WebSocket.CONNECTING) {
        console.log("WebSocket is connecting, waiting to send config...");
        setTimeout(() => sendConfigWhenReady(socket), 200); // Retry after a short delay
      } else {
        setStatusMessage("WebSocket not open. Cannot send configuration.");
        console.error("WebSocket not open for sending config. State: " + socket.readyState);
        setIsGameRunning(false);
      }
    };
    
    // If ws was just created by connectWebSocket, it might still be connecting.
    // So, we use sendConfigWhenReady to handle this.
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

  // stopGameFlow is no longer needed as frontend doesn't manage camera

  const handleEndGame = () => {
    setStatusMessage('Ending game...');
    sendCommandToBackend({ action: 'end_game' });
    setIsGameRunning(false);
    // Backend will stop sending frames. WS might close from backend or stay open for new game.
    // For a clean stop, explicitly close from client too after command.
    if(ws) {
        // ws.close(); // Or let backend close it after processing 'end_game'
    }
  };

  const handleEmergency = () => {
    setStatusMessage('Emergency stop initiated...');
    sendCommandToBackend({ action: 'emergency_stop' });
    setIsGameRunning(false);
    if(ws) {
        // ws.close(); // Or let backend close it
    }
  };

  useEffect(() => {
    // Attempt to connect when component mounts if not already connected
    // This is optional, user can click "Start Shoot" to connect and send config
    // if (!ws) {
    //   connectWebSocket();
    // }

    return () => {
      if (ws) {
        console.log("Closing WebSocket on component unmount.");
        ws.close();
      }
    };
  }, []); // ws removed from dependency array to prevent re-connect loops if ws state changes elsewhere

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
        <div className={styles.inputGroup}> {/* Added Timeout input */}
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
        {/* Removed Local Webcam Feed */}
        <div className={styles.videoFeed}>
          <h2>Processed Feed from Backend</h2>
          {processedFrame ? (
            <img src={processedFrame} alt="Processed frame" className={styles.videoElement} />
          ) : (
            <div className={styles.noFramePlaceholder}>{isGameRunning ? "Waiting for processed frame..." : "Game not started or no feed."}</div>
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

