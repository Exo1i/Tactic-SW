"use client";
import React, { useState, useEffect, useRef, useCallback } from 'react';
import "../../globals.css"; // Ensure this path is correct for your project structure

// --- Constants ---
const GRID_ROWS = 2;
const GRID_COLS = 4; // Ensure this matches backend's GRID_COLS for grid-cols-X class
const CARD_COUNT = GRID_ROWS * GRID_COLS;
const BOARD_DETECT_WIDTH = 400; // Must match backend if used for frontend calculations
const BOARD_DETECT_HEIGHT = 200; // Must match backend

// Map backend names to Tailwind colors or hex codes
const colorMap = {
    red: '#ef4444',    // Tailwind red-500
    yellow: '#facc15', // Tailwind yellow-400
    green: '#22c55e',  // Tailwind green-500
    blue: '#3b82f6',   // Tailwind blue-500
    orange: '#f97316', // Tailwind orange-500
    apple: '#dc2626',  // Tailwind red-600
    cat: '#a855f7',    // Tailwind purple-500
    car: '#06b6d4',    // Tailwind cyan-500
    umbrella: '#16a34a',// Tailwind green-600
    banana: '#fde047', // Tailwind yellow-300
    'fire hydrant': '#e11d48', // Tailwind rose-600
    person: '#78716c', // Tailwind stone-500
    'detect_fail': '#6b7280', // Tailwind gray-500 (for error/unknown states)
    'perma_fail': '#4b5563', // Tailwind gray-600 (for permanent detection failure)
    'unknown': '#9ca3af',   // Tailwind gray-400
    'default_flipped': '#d1d5db', // Tailwind gray-300 (default background for flipped cards if no specific color)
    'black': '#1f2937', // Tailwind gray-800 (for card back if detected as 'black')
};

export default function MemoryGame() {
    const [gameVersion, setGameVersion] = useState(null); // 'color', 'yolo', or null
    const [isConnected, setIsConnected] = useState(false);
    const [videoSrc, setVideoSrc] = useState('');
    const [transformedVideoSrc, setTransformedVideoSrc] = useState('');
    const [gameState, setGameState] = useState({ // Initialize with a default structure
        card_states: {},
        pairs_found: 0,
        current_flipped: [],
    });
    const [message, setMessage] = useState('Select game version to start.');
    const [lastMessageTime, setLastMessageTime] = useState(0);
    const [isGameOver, setIsGameOver] = useState(false);
    const [showError, setShowError] = useState(null);
    const websocket = useRef(null);

    const updateMessage = useCallback((newMessage, isError = false) => {
        const now = Date.now();
        if (isError || now - lastMessageTime > 300) {
            setMessage(newMessage);
            setLastMessageTime(now);
            if (isError) {
                setShowError(newMessage);
                setTimeout(() => setShowError(null), 7000); // Longer display for errors
            } else if (showError && !isError) { // Clear error only if current message is not an error
                setShowError(null);
            }
        }
    }, [lastMessageTime, showError]); // Added showError dependency

    const resetGameStates = () => {
        setIsConnected(false);
        setVideoSrc('');
        setTransformedVideoSrc('');
        setGameState({ card_states: {}, pairs_found: 0, current_flipped: [] });
        setIsGameOver(false);
        setShowError(null);
        // message will be updated by connect or disconnect handlers
    };

    const connectWebSocket = useCallback((version) => {
        if (websocket.current && websocket.current.readyState === WebSocket.OPEN) {
            console.log("WebSocket already open for", version);
            // Potentially send a re-init or new game command if backend supports it
            // For now, we assume main.py handles new connection as a new game session
            return;
        }
        if (websocket.current) {
            websocket.current.close(1000, "Starting new connection");
        }

        resetGameStates(); // Reset UI states before new connection
        updateMessage(`Connecting to ${version} game...`);

        const backendWsHost = process.env.NEXT_PUBLIC_BACKEND_WS_URL || `${window.location.hostname}:8000`;
        const wsProtocol = window.location.protocol === "https:" ? "wss://" : "ws://";
        const wsUrl = `${wsProtocol}${backendWsHost}/ws/${version}`;

        console.log(`Attempting to connect to: ${wsUrl}`);
        websocket.current = new WebSocket(wsUrl);

        websocket.current.onopen = () => {
            console.log('WebSocket Connected to', version);
            setIsConnected(true);
            updateMessage(`${version.charAt(0).toUpperCase() + version.slice(1)} game connected. Initializing...`);
        };

        websocket.current.onclose = (event) => {
            console.log(`WebSocket Disconnected: Code=${event.code}, Reason=${event.reason}`);
            const stillGameOver = isGameOver; // Capture state before reset
            resetGameStates(); // Reset UI states

            if (event.code === 1000 && stillGameOver) { // Normal close after game over
                updateMessage(`Game Over! Disconnected. Select version to play again.`);
            } else if (event.code === 1000) { // Normal close by client/cleanup
                updateMessage("Disconnected. Select game version to start.");
            } else if (!stillGameOver) { // Abnormal close and not game over
                updateMessage(`Disconnected (Code: ${event.code}). Select game to reconnect.`, true);
            } else { // Abnormal close but game was over
                 updateMessage(`Game Over! Disconnected (Code: ${event.code}). Select version to play again.`);
            }
            setGameVersion(null); // Allow re-selection
            websocket.current = null;
        };

        websocket.current.onerror = (errorEvent) => {
            console.error('WebSocket Error:', errorEvent);
            // Attempt to get a more specific message if available
            let specificError = "Connection failed.";
            if (errorEvent && errorEvent.message) {
                specificError = errorEvent.message;
            } else if (websocket.current && websocket.current.url) {
                specificError = `Failed to connect to ${websocket.current.url}`;
            }
            updateMessage(`WebSocket Error: ${specificError}`, true);
            resetGameStates();
            setGameVersion(null); // Allow re-selection on error
            if (websocket.current) { // Ensure it's fully nulled out after error
                websocket.current.onclose = null; // Prevent further onclose calls
                websocket.current.onerror = null;
                // websocket.current.close(); // It's likely already closed or in error state
                websocket.current = null;
            }
        };

        websocket.current.onmessage = (event) => {
            console.log("Raw WS Data Received:", event.data); // <--- ADD THIS LINE
            try {
                const data = JSON.parse(event.data);
                console.log("Parsed WS Data:", data); // Log parsed data too
        
                // Check if data or data.type is null/undefined immediately after parsing
                if (!data || typeof data.type === 'undefined') {
                    console.error("Parsed data is missing 'type' property!", data);
                    // Handle this specific error? Maybe updateMessage?
                    updateMessage("Received invalid message format from server.", true);
                    // Don't proceed with the switch statement if type is missing
                    return;
                }
        
                switch (data.type) {
                    case 'frame_update':
                        if (data.payload?.frame) {
                            setVideoSrc(`data:image/jpeg;base64,${data.payload.frame}`);
                        }
                        if (data.payload?.transformed_frame) {
                            setTransformedVideoSrc(`data:image/jpeg;base64,${data.payload.transformed_frame}`);
                        } else {
                            // Optional: Clear if not sent consistently to indicate loss of board detection
                            // setTransformedVideoSrc('');
                        }
                        break;
                    case 'game_state':
                        // console.log("Game State Update:", data.payload);
                        if (data.payload && typeof data.payload === 'object' && data.payload.card_states) {
                            setGameState(prevState => ({ ...prevState, ...data.payload }));
                        } else {
                            console.warn("Received game_state with unexpected structure:", data.payload);
                        }
                        break;
                    case 'arm_status':
                        updateMessage(`Arm: ${data.payload?.action} ${data.payload?.success ? 'OK ✅' : 'Failed ❌'}`);
                        break;
                    case 'cards_hidden': // This message might be redundant if game_state updates are sufficient
                        // updateMessage(`Cards ${data.payload.join(' & ')} returned.`);
                        break;
                    case 'message':
                        updateMessage(data.payload);
                        break;
                    case 'game_over':
                        updateMessage(`Game Over! ${data.payload}`);
                        setIsGameOver(true);
                        // Backend should manage WS closure or keep alive. Client can close after a delay if desired.
                        // setTimeout(() => {
                        //     if (websocket.current && websocket.current.readyState === WebSocket.OPEN) {
                        //         websocket.current.close(1000, "Game ended");
                        //     }
                        // }, 5000);
                        break;
                    case 'error':
                        updateMessage(`Server Error: ${data.payload}`, true);
                        // Fatal errors from backend might require client to disconnect and reset
                        if (data.payload.toLowerCase().includes("serial") || data.payload.toLowerCase().includes("critical") || data.payload.toLowerCase().includes("model failed")) {
                            if (websocket.current && websocket.current.readyState === WebSocket.OPEN) {
                                websocket.current.close(1007, "Fatal server error"); // 1007: Invalid frame payload data
                            }
                            setGameVersion(null); // Force re-selection
                        }
                        break;
                        default:
                            console.warn('Unknown message type:', data.type, data.payload); // Log payload too
                    }
                } catch (error) {
                    console.error('Failed to parse WebSocket message:', error);
                    // Raw data already logged above
                    updateMessage("Error processing message from server.", true);
                }
            };
    }, [isGameOver, updateMessage]); // isGameOver, updateMessage are dependencies

    useEffect(() => {
        if (gameVersion) {
            connectWebSocket(gameVersion);
        }
        return () => {
            if (websocket.current) {
                console.log("Closing WebSocket via main effect cleanup.");
                websocket.current.onclose = null;
                websocket.current.onerror = null;
                if (websocket.current.readyState === WebSocket.OPEN || websocket.current.readyState === WebSocket.CONNECTING) {
                   websocket.current.close(1000, "Client unmounting or changing game version");
                }
                websocket.current = null;
                resetGameStates(); // Ensure UI is clean
            }
        };
    }, [gameVersion, connectWebSocket]);

    const handleVersionSelect = (version) => {
        if (gameVersion === version && isConnected) return; // Already connected to this version

        if (websocket.current) { // If there's an existing connection, close it first
            websocket.current.close(1000, "Changing game version");
            // connectWebSocket will be called by useEffect when gameVersion changes
        }
        setGameVersion(version); // This will trigger the useEffect
        setIsGameOver(false); // Reset game over state if selecting a new version
    };

    const handlePlayAgain = () => {
        // This will trigger the cleanup in useEffect, then re-trigger connection
        // by setting gameVersion to null then back to a (potentially new) version.
        // Or, simply allow selecting a version again.
        const currentVersion = gameVersion;
        setGameVersion(null); // Trigger cleanup
        updateMessage("Select game version to start.");
        // If you want to immediately restart the same version, uncomment below:
        // setTimeout(() => {
        //     setGameVersion(currentVersion);
        // }, 100);
    };

    const renderCardContent = (cardIndex) => {
        const cardState = gameState?.card_states?.[cardIndex] ?? {};
        const isCurrentlyFlipped = gameState?.current_flipped?.includes(cardIndex);
        const isMatched = cardState?.isMatched;

        // Card is considered "face up" if it's currently selected by the arm (in current_flipped)
        // OR if it has been flipped before AND is not yet matched.
        // This allows showing previously seen cards if the game logic requires it.
        // For this memory game, usually, only current_flipped and matched are primary visual drivers.
        // The backend's game_state (card_states.isFlippedBefore) dictates memory.
        const showFaceUp = isCurrentlyFlipped || (cardState.isFlippedBefore && !isMatched);

        let cardFaceContent = null;
        let cardFaceBgStyle = { backgroundColor: colorMap.default_flipped };

        if (showFaceUp) {
            let itemKey = null; // Used to look up color or image
            let itemDisplayName = "";

            if (gameVersion === 'yolo' && cardState.object) {
                itemKey = cardState.object.toLowerCase();
                itemDisplayName = cardState.object;
                 if (itemKey === DETECTION_PERMANENT_FAIL_STATE.toLowerCase() || itemKey === 'detect_fail') {
                    itemKey = 'detect_fail'; // Normalize for colorMap
                    itemDisplayName = "Detection Failed";
                }

                if (colorMap[itemKey]) { // Use color as fallback BG if image fails or for non-image items
                    cardFaceBgStyle = { backgroundColor: colorMap[itemKey] };
                }
                
                const imageName = itemKey.replace(/ /g, '_'); // Replace spaces for filenames
                cardFaceContent = (
                    <div
                        className="absolute inset-0 w-full h-full flex items-center justify-center rounded-lg text-center p-1 backface-hidden"
                        style={{ backgroundColor: 'white', transform: 'rotateY(180deg)' }} // White BG for images
                    >
                        {itemKey === 'detect_fail' ? (
                            <span className="text-xs sm:text-sm md:text-base font-semibold text-gray-700 break-words p-1">
                                {itemDisplayName}
                            </span>
                        ) : (
                            <img
                                src={`/images/${imageName}.png`}
                                alt={itemDisplayName}
                                className="max-h-[75%] max-w-[75%] object-contain"
                                onError={(e) => { e.target.style.display = 'none'; /* Hide if image fails to load */ }}
                            />
                        )}
                    </div>
                );
            } else if (gameVersion === 'color' && cardState.color) {
                itemKey = cardState.color.toLowerCase();
                itemDisplayName = cardState.color;
                if (itemKey === DETECTION_PERMANENT_FAIL_STATE.toLowerCase() || itemKey === 'detect_fail') {
                    itemKey = 'detect_fail';
                    itemDisplayName = "Detection Failed";
                } else if (itemKey === 'black') {
                    itemDisplayName = "Card Back"; // Or leave empty
                }


                if (colorMap[itemKey]) {
                    cardFaceBgStyle = { backgroundColor: colorMap[itemKey] };
                }
                cardFaceContent = (
                    <div
                        className="absolute inset-0 w-full h-full flex items-center justify-center rounded-lg backface-hidden"
                        style={{ ...cardFaceBgStyle, transform: 'rotateY(180deg)' }}
                    >
                        {itemKey === 'detect_fail' && (
                             <span className="text-xs sm:text-sm md:text-base font-semibold text-white break-words p-1">
                                {itemDisplayName}
                            </span>
                        )}
                        {/* For color cards, usually no text, just the background color. */}
                    </div>
                );
            } else if (cardState.isFlippedBefore) { // Fallback if no specific content but was flipped
                cardFaceBgStyle = { backgroundColor: colorMap.unknown };
                cardFaceContent = (
                     <div
                        className="absolute inset-0 w-full h-full flex items-center justify-center rounded-lg text-center p-1 backface-hidden"
                        style={{ ...cardFaceBgStyle, transform: 'rotateY(180deg)' }}
                    >
                        <span className="text-xs sm:text-sm md:text-base font-semibold text-gray-700 break-words">
                            ?
                        </span>
                    </div>
                );
            }
        }

        return (
            <div key={cardIndex} className="perspective w-full aspect-square">
                <div
                    className={`relative w-full h-full transition-transform duration-700 ease-in-out preserve-3d rounded-lg
                                ${showFaceUp ? 'rotate-y-180' : ''}
                                ${isMatched ? 'opacity-30 scale-90 pointer-events-none' : ''}
                                ${isCurrentlyFlipped && !isMatched ? 'ring-4 ring-yellow-400 ring-offset-1 scale-105 shadow-2xl z-20' : 'shadow-lg'}
                              `}
                >
                    {/* Card Back */}
                    <div className="absolute inset-0 w-full h-full bg-gradient-to-br from-blue-500 to-indigo-600 rounded-lg flex items-center justify-center backface-hidden card-back-pattern">
                        {/* SVG or complex pattern can go here, or use CSS ::before/::after */}
                        <span className="text-4xl text-blue-200 opacity-50 font-bold">?</span>
                    </div>
                    {/* Card Face (rendered if showFaceUp is true and content exists) */}
                    {cardFaceContent}
                </div>
            </div>
        );
    };


    return (
        <div className="min-h-screen bg-gray-900 text-gray-100 flex flex-col items-center p-2 sm:p-4">
            <h1 className="text-3xl sm:text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-500 mb-4 sm:mb-6">Memory Puzzle Game</h1>

            <div className="w-full max-w-6xl bg-gray-800 text-gray-300 p-3 rounded-lg shadow-xl mb-4 text-xs sm:text-sm flex flex-wrap justify-between items-center gap-x-4 gap-y-2">
                <span>Status:
                    <span className={`ml-1 font-semibold ${isConnected ? 'text-green-400' : 'text-red-400'}`}>
                        {isConnected ? 'Connected' : 'Disconnected'}
                    </span>
                    {isConnected && gameVersion && ` (${gameVersion.charAt(0).toUpperCase() + gameVersion.slice(1)})`}
                 </span>
                 <span className="text-center flex-grow mx-2 truncate font-medium text-gray-100" title={message}>{message}</span>
                 <span>Pairs: <strong className="text-yellow-400">{gameState?.pairs_found ?? 0}</strong> / {CARD_COUNT / 2}</span>
            </div>

            {showError && (
                <div className="w-full max-w-6xl bg-red-700 border border-red-500 text-red-100 px-4 py-2 rounded-md shadow-lg mb-4 text-sm" role="alert">
                    <strong className="font-bold">Error: </strong>
                    <span className="block sm:inline">{showError}</span>
                </div>
            )}

            {(!gameVersion || isGameOver) && (
                <div className="flex flex-col sm:flex-row items-center justify-center space-y-3 sm:space-y-0 sm:space-x-4 mb-6 p-4 bg-gray-800 rounded-lg shadow-xl">
                    {!isGameOver ? (
                         <>
                            <button
                                onClick={() => handleVersionSelect('yolo')}
                                disabled={!!gameVersion && isConnected && !isGameOver}
                                className="px-6 py-3 text-lg bg-gradient-to-r from-purple-600 to-indigo-600 text-white rounded-md shadow-lg hover:from-purple-700 hover:to-indigo-700 disabled:opacity-60 disabled:cursor-not-allowed transition-all transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-indigo-400 focus:ring-opacity-75 w-full sm:w-auto"
                            >
                                Start YOLO Version
                            </button>
                            <button
                                onClick={() => handleVersionSelect('color')}
                                disabled={!!gameVersion && isConnected && !isGameOver}
                                className="px-6 py-3 text-lg bg-gradient-to-r from-teal-500 to-cyan-600 text-white rounded-md shadow-lg hover:from-teal-600 hover:to-cyan-700 disabled:opacity-60 disabled:cursor-not-allowed transition-all transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-cyan-400 focus:ring-opacity-75 w-full sm:w-auto"
                            >
                                Start Color Version
                            </button>
                         </>
                     ) : (
                         <button
                            onClick={handlePlayAgain}
                            className="px-8 py-4 text-xl bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-md shadow-xl hover:from-green-600 hover:to-emerald-700 transition-all transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-emerald-400 focus:ring-opacity-75"
                         >
                            Play Again?
                         </button>
                    )}
                </div>
            )}

            {gameVersion && !isGameOver && ( // Render game area only if version selected and not game over
                 <div className="flex flex-col lg:flex-row gap-4 sm:gap-6 w-full max-w-6xl">
                     <div className="flex flex-col gap-4 sm:gap-6 lg:w-1/2">
                         <div className="bg-gray-800 p-3 sm:p-4 rounded-lg shadow-xl border border-gray-700">
                             <h2 className="text-xl sm:text-2xl font-semibold mb-3 text-transparent bg-clip-text bg-gradient-to-r from-sky-400 to-blue-500">Live Camera</h2>
                             <div className="w-full aspect-video bg-gray-700 border border-gray-600 rounded overflow-hidden">
                                {videoSrc ? (
                                    <img src={videoSrc} alt="Live feed" className="w-full h-full object-cover" />
                                ) : (
                                    <div className="w-full h-full flex items-center justify-center text-gray-400">
                                        {isConnected ? "Waiting for camera..." : "Connecting..."}
                                    </div>
                                )}
                             </div>
                         </div>

                          <div className="bg-gray-800 p-3 sm:p-4 rounded-lg shadow-xl border border-gray-700">
                             <h2 className="text-xl sm:text-2xl font-semibold mb-3 text-transparent bg-clip-text bg-gradient-to-r from-lime-400 to-green-500">Detected Board</h2>
                             <div
                                 className="w-full bg-gray-700 border border-gray-600 rounded overflow-hidden relative"
                                 // Aspect ratio based on backend's board detection dimensions
                                 style={{ paddingTop: `${(BOARD_DETECT_HEIGHT / BOARD_DETECT_WIDTH) * 100}%` }}
                             >
                                {transformedVideoSrc ? (
                                    <img src={transformedVideoSrc} alt="Transformed board" className="absolute inset-0 w-full h-full object-contain" />
                                ) : (
                                    <div className="absolute inset-0 flex items-center justify-center text-gray-400 text-xs p-2 text-center">
                                        {videoSrc && isConnected ? 'Waiting for board detection...' : (isConnected ? 'Camera feed needed' : 'Connecting...')}
                                    </div>
                                )}
                             </div>
                         </div>
                     </div>

                     <div className="bg-gray-800 p-3 sm:p-4 rounded-lg shadow-xl border border-gray-700 lg:w-1/2">
                         <h2 className="text-xl sm:text-2xl font-semibold mb-3 text-transparent bg-clip-text bg-gradient-to-r from-yellow-400 to-orange-500">Game Board</h2>
                         <div className={`grid grid-cols-${GRID_COLS} gap-2 sm:gap-3 lg:gap-4`}>
                             {Array.from({ length: CARD_COUNT }).map((_, index) => renderCardContent(index) )}
                         </div>
                     </div>
                 </div>
            )}

             {(!gameVersion || (!isConnected && !isGameOver)) && ( // Show this only if no game version or disconnected (and not already game over)
                 <div className="mt-10 text-gray-500 text-lg">
                     Please select a game version to begin.
                 </div>
             )}
        </div>
    );
}

// For Tailwind JIT, ensure these classes are discoverable or safelisted in tailwind.config.js
// if GRID_COLS is dynamic or not a standard Tailwind number:
// e.g. for GRID_COLS = 4, 'grid-cols-4' is standard.
// The 3D flip animation classes:
const _dynamicTailwindClasses = 'perspective preserve-3d rotate-y-180 backface-hidden';