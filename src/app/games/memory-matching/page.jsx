"use client"
import React, { useState, useEffect, useRef, useCallback } from 'react';
import './page.css';

const GRID_ROWS = 2;
const GRID_COLS = 4;
const CARD_COUNT = GRID_ROWS * GRID_COLS;

// Basic colors for fallback rendering (ensure keys match backend color names)
const colorMap = {
    red: '#FF0000',
    yellow: '#FFFF00',
    green: '#00FF00',
    blue: '#0000FF',
    // Add colors for YOLO objects if desired for fallback
    orange: '#FFA500',
    apple: '#FF0000', // Reusing red
    cat: '#0000FF',   // Reusing blue
    car: '#00FFFF',
    umbrella: '#008000', // Darker green
    banana: '#FFFFE0', // Light yellow
    'fire hydrant': '#FF4500', // OrangeRed
    person: '#808080', // Gray
};

export default function MemoryGame() {
    const [gameVersion, setGameVersion] = useState(null); // 'color', 'yolo', or null
    const [isConnected, setIsConnected] = useState(false);
    const [videoSrc, setVideoSrc] = useState('');
    const [gameState, setGameState] = useState(null); // Will hold card states, etc.
    const [message, setMessage] = useState('Select game version to start.');
    const [isGameOver, setIsGameOver] = useState(false);
    const websocket = useRef(null);

    const connectWebSocket = useCallback((version) => {
        if (websocket.current) {
            websocket.current.close();
        }
        setMessage(`Connecting to ${version} game...`);
        setIsGameOver(false); // Reset game over state

        // Use ws:// for local dev, wss:// for production with HTTPS
        const wsUrl = `ws://${window.location.hostname}:8000/ws/${version}`; // Adjust hostname/port if needed
        websocket.current = new WebSocket(wsUrl);

        websocket.current.onopen = () => {
            console.log('WebSocket Connected');
            setIsConnected(true);
            setMessage(`${version.charAt(0).toUpperCase() + version.slice(1)} game connected. Waiting for start...`);
        };

        websocket.current.onclose = () => {
            console.log('WebSocket Disconnected');
            setIsConnected(false);
            setVideoSrc(''); // Clear video on disconnect
            setGameState(null); // Clear game state
            // Only reset gameVersion if not already ended by game_over
            if (!isGameOver) {
                 setMessage('Disconnected. Select game version to reconnect.');
                 setGameVersion(null);
            }
             websocket.current = null;
        };

        websocket.current.onerror = (error) => {
            console.error('WebSocket Error:', error);
            setMessage(`WebSocket Error: ${error.message || 'Connection failed.'}`);
            setIsConnected(false);
            // Consider closing and allowing reconnect here
        };

        websocket.current.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                // console.log("WS Message:", data.type); // Debug: log message types

                switch (data.type) {
                    case 'frame':
                        // Prepend the necessary header for base64 images
                        setVideoSrc(`data:image/jpeg;base64,${data.payload}`);
                        break;
                    case 'game_state':
                         console.log("Game State Update:", data.payload);
                         // If payload contains card_states directly (adjust if nested differently)
                         if (data.payload?.card_states || typeof data.payload === 'object') {
                            setGameState(data.payload);
                         } else {
                            console.warn("Received game_state without expected structure:", data.payload)
                         }
                         break;
                    case 'arm_status':
                        console.log("Arm Status:", data.payload);
                        setMessage(`Arm action '${data.payload?.action}' ${data.payload?.success ? 'succeeded' : 'failed'}.`);
                        break;
                     case 'cards_hidden':
                         console.log("Cards hidden:", data.payload);
                         // Optionally update UI to show cards flipping back briefly
                         setMessage(`Cards ${data.payload.join(' & ')} flipped back.`);
                         // The main game_state update should reflect the cards being unflipped visually
                         break;
                    case 'message':
                        console.log("Message:", data.payload);
                        setMessage(data.payload);
                        break;
                    case 'game_over':
                        console.log("Game Over:", data.payload);
                        setMessage(`Game Over! ${data.payload}`);
                        setIsGameOver(true);
                        // Maybe close WS after a delay? Or keep it open to view final state.
                        // setTimeout(() => websocket.current?.close(), 5000);
                        break;
                    case 'error':
                        console.error('Game Error:', data.payload);
                        setMessage(`Error: ${data.payload}`);
                        // Decide if error is fatal and close connection / reset UI
                        // if (/* error is fatal */) {
                        //     websocket.current?.close();
                        //     setGameVersion(null);
                        // }
                        break;
                    default:
                        console.warn('Unknown message type:', data.type);
                }
            } catch (error) {
                console.error('Failed to parse WebSocket message:', error);
                // Handle non-JSON messages if necessary
                // console.log("Raw WS Data:", event.data);
            }
        };
    }, [isGameOver]); // Add isGameOver dependency

    useEffect(() => {
        if (gameVersion) {
            connectWebSocket(gameVersion);
        }
        // Cleanup function to close WebSocket when component unmounts or gameVersion changes
        return () => {
            websocket.current?.close();
        };
    }, [gameVersion, connectWebSocket]); // Rerun effect if gameVersion changes

    const handleVersionSelect = (version) => {
        if (!isConnected) { // Allow selection only if not connected or after disconnect
             setGameVersion(version);
        }
    };

    const renderCardContent = (cardData) => {
        if (!cardData || !cardData.isFlippedBefore) {
            return <span className="card-back">?</span>; // Hidden card
        }
        if (gameVersion === 'yolo' && cardData.object) {
            // Try to show color based on object name, or just the name
            const color = colorMap[cardData.object.toLowerCase()];
            return (
                <div className="card-face" style={{ backgroundColor: color || '#CCCCCC' }}>
                    {cardData.object}
                </div>
            );
        }
        if (gameVersion === 'color' && cardData.color) {
             const color = colorMap[cardData.color.toLowerCase()];
            return (
                <div className="card-face" style={{ backgroundColor: color || '#CCCCCC' }}>
                    {cardData.color}
                </div>
            );
        }
        // Fallback if flipped but no data (shouldn't happen often)
        return <div className="card-face" style={{ backgroundColor: '#AAAAAA' }}></div>;
    };

    return (
        <div className="App">
            <h1>Memory Puzzle Game</h1>
            <div className="status-bar">
                Status: {isConnected ? 'Connected' : 'Disconnected'} | Message: {message}
            </div>

            {!gameVersion && !isConnected && (
                <div className="version-selector">
                    <button onClick={() => handleVersionSelect('yolo')}>Start YOLO Version</button>
                    <button onClick={() => handleVersionSelect('color')}>Start Color Version</button>
                </div>
            )}

            {gameVersion && (
                 <div className="game-area">
                     <div className="video-feed">
                         <h2>Live Camera Feed ({gameVersion})</h2>
                         {videoSrc ? (
                             <img src={videoSrc} alt="Live feed" />
                         ) : (
                             <div className="video-placeholder">Connecting to camera...</div>
                         )}
                     </div>

                     <div className="game-board-area">
                         <h2>Game Board</h2>
                          <p>Pairs Found: {gameState?.pairs_found ?? 0} / {CARD_COUNT / 2}</p>
                         <div className="game-grid">
                             {Array.from({ length: CARD_COUNT }).map((_, index) => {
                                // Get state for this specific card index
                                const cardState = gameState?.card_states ? gameState.card_states[index] : null;
                                 const isCurrentlyFlipped = gameState?.current_flipped?.includes(index);
                                 return (
                                     <div
                                        key={index}
                                        className={`card ${cardState?.isFlippedBefore ? 'flipped' : ''} ${isCurrentlyFlipped ? 'currently-active' : ''}`}
                                    >
                                         {renderCardContent(cardState)}
                                     </div>
                                 );
                             })}
                         </div>
                          {isGameOver && <button onClick={() => { setIsGameOver(false); setGameVersion(null); setMessage("Select game version.")}}>Play Again?</button>}
                     </div>
                 </div>
            )}
        </div>
    );
}

