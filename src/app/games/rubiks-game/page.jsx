// app/page.jsx
"use client";
import React, { useState, useRef, useEffect, useCallback } from "react";

// Helper constant
const COLOR_NAMES = ["W", "R", "G", "Y", "O", "B"];
const MIN_FRAME_INTERVAL = 100; // ms, roughly 10 FPS. Adjust as needed.
const JPEG_QUALITY = 0.75; // 0.0 to 1.0

export default function RubiksSolverPage() {
    const gameId = "rubiks";
    const videoRef = useRef(null);
    const canvasRef = useRef(null); 
    const wsRef = useRef(null);
    const ipCamImgRef = useRef(null);

    const [isClient, setIsClient] = useState(false);
    const [status, setStatus] = useState("Idle");
    const [processedFrame, setProcessedFrame] = useState(null);
    const [showSettings, setShowSettings] = useState(false);
    
    const [serialPort, setSerialPort] = useState("COM7"); 
    const [cameraSettings, setCameraSettings] = useState({
        useIpCamera: true, // Let's default to true if that's your primary use case
        ipCameraAddress: "http://192.168.1.9:8080/video",
    });
    const [appliedCameraSettings, setAppliedCameraSettings] = useState({
        useIpCamera: true,
        ipCameraAddress: "http://192.168.1.9:8080/video",
    });

    const [gameState, setGameState] = useState({
        mode: "idle",
        calibration_step: 0,
        scan_index: 0,
        solve_move_index: 0,
        total_solve_moves: 0,
        status_message: "Initialize game.",
        error_message: null,
        serial_connected: false,
        current_color_calibrating: null,
        solution_preview: null,
    });
    const [gameStarted, setGameStarted] = useState(false);
    const [isCameraLoading, setIsCameraLoading] = useState(false); // For local preview loading
    
    const frameSenderHandle = useRef(null); 
    const lastFrameSentTime = useRef(0);

    useEffect(() => {
        setIsClient(true);
    }, []);

    // Effect for WebSocket connection and local camera
    useEffect(() => {
        if (!isClient || !gameStarted) {
            if (wsRef.current) {
                wsRef.current.close();
                wsRef.current = null;
            }
            if (frameSenderHandle.current) {
                cancelAnimationFrame(frameSenderHandle.current);
                frameSenderHandle.current = null;
            }
            if (videoRef.current && videoRef.current.srcObject) {
                videoRef.current.srcObject.getTracks().forEach((t) => t.stop());
                videoRef.current.srcObject = null;
            }
            // Set status when game stops
            if (gameStarted === false && status !== "Idle" && !status.includes("Error") && !status.includes("Disconnected")) {
                setStatus("Disconnected (Game Stopped)");
            }
            return;
        }

        // --- Setup Local Camera Preview ---
        setIsCameraLoading(true); // Set loading true when starting camera setup
        let localStream = null;
        const setupLocalCamera = async () => {
            // Stop any existing stream
            if (videoRef.current && videoRef.current.srcObject) {
                videoRef.current.srcObject.getTracks().forEach((t) => t.stop());
                videoRef.current.srcObject = null;
            }
            // Clear IP cam src if switching to local
            if (ipCamImgRef.current && !appliedCameraSettings.useIpCamera) {
                 ipCamImgRef.current.src = ""; 
            }


            if (appliedCameraSettings.useIpCamera) {
                console.log("Attempting to use IP Camera:", appliedCameraSettings.ipCameraAddress);
                // For IP Camera, loading is handled by img onload/onerror
                // The img tag with key={appliedCameraSettings.ipCameraAddress} handles src update
                // setIsCameraLoading will be set to false by onLoad/onError of the img tag
            } else { // Local webcam
                console.log("Attempting to use local webcam.");
                try {
                    localStream = await navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 } });
                    if (videoRef.current) {
                        videoRef.current.srcObject = localStream;
                        videoRef.current.onloadedmetadata = () => {
                            console.log("Local webcam metadata loaded.");
                            setIsCameraLoading(false);
                        };
                        videoRef.current.onerror = (e) => {
                            console.error("Local video error:", e);
                            setGameState(prev => ({ ...prev, error_message: "Local camera error." }));
                            setIsCameraLoading(false);
                        }
                    } else { setIsCameraLoading(false); }
                } catch (err) {
                    console.error("Local camera getUserMedia error:", err);
                    setGameState(prev => ({ ...prev, error_message: "Local camera access failed. Check permissions." }));
                    setIsCameraLoading(false);
                }
            }
        };
        setupLocalCamera();

        // --- Setup WebSocket ---
        const ws = new WebSocket(`ws://localhost:8000/ws/${gameId}`);
        wsRef.current = ws;
        setStatus("Connecting to WebSocket...");

        ws.onopen = () => {
            setStatus("WebSocket Connected");
            console.log("WebSocket opened. Sending initial config.");
            const initialConfig = {
                serial_port: serialPort,
                serial_baudrate: 9600,
                video_source: appliedCameraSettings.useIpCamera ? appliedCameraSettings.ipCameraAddress : "0" 
            };
            ws.send(JSON.stringify(initialConfig));
            
            lastFrameSentTime.current = 0; 
            if (frameSenderHandle.current) cancelAnimationFrame(frameSenderHandle.current); 
            
            const sendFrame = () => {
                if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN || !gameStarted) {
                    if(frameSenderHandle.current) cancelAnimationFrame(frameSenderHandle.current);
                    frameSenderHandle.current = null;
                    return;
                }

                const now = Date.now();
                if (now - lastFrameSentTime.current < MIN_FRAME_INTERVAL) {
                    frameSenderHandle.current = requestAnimationFrame(sendFrame);
                    return;
                }
                lastFrameSentTime.current = now;

                const canvas = canvasRef.current;
                let sourceElement = appliedCameraSettings.useIpCamera ? ipCamImgRef.current : videoRef.current;
                
                if (sourceElement && canvas &&
                    ( (sourceElement.tagName === "VIDEO" && sourceElement.readyState >= 2 /* HAVE_CURRENT_DATA or more */) ||
                      (sourceElement.tagName === "IMG" && sourceElement.complete && sourceElement.naturalWidth > 0 && sourceElement.naturalHeight > 0) ) // Check naturalWidth/Height for img
                   ) {
                    
                    // Ensure canvas dimensions match the source element if it's a video or loaded image
                    const sWidth = sourceElement.videoWidth || sourceElement.naturalWidth;
                    const sHeight = sourceElement.videoHeight || sourceElement.naturalHeight;

                    if (sWidth && sHeight) { // Only draw if dimensions are valid
                        canvas.width = sWidth;
                        canvas.height = sHeight;
                        const ctx = canvas.getContext("2d", { alpha: false });
                        try {
                            ctx.drawImage(sourceElement, 0, 0, canvas.width, canvas.height);
                            canvas.toBlob(
                                (blob) => {
                                    if (blob && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                                        // console.log("Sending frame blob of size:", blob.size); 
                                        wsRef.current.send(blob); 
                                    }
                                },
                                "image/jpeg",
                                JPEG_QUALITY
                            );
                        } catch (e) {
                            console.error("Error drawing source to canvas or sending frame:", e);
                        }
                    } else {
                        // console.warn("Source element dimensions not ready for drawing.");
                    }
                } else {
                    // console.warn("Source element or canvas not ready for sendFrame.");
                }
                frameSenderHandle.current = requestAnimationFrame(sendFrame);
            };
            frameSenderHandle.current = requestAnimationFrame(sendFrame);
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                // console.log("Received from backend:", data); 
                setGameState(prev => ({
                    ...prev,
                    mode: data.mode !== undefined ? data.mode : prev.mode,
                    calibration_step: data.calibration_step !== undefined ? data.calibration_step : prev.calibration_step,
                    scan_index: data.scan_index !== undefined ? data.scan_index : prev.scan_index,
                    solve_move_index: data.solve_move_index !== undefined ? data.solve_move_index : prev.solve_move_index,
                    total_solve_moves: data.total_solve_moves !== undefined ? data.total_solve_moves : prev.total_solve_moves,
                    status_message: data.status_message !== undefined ? data.status_message : prev.status_message,
                    error_message: data.error_message !== undefined ? data.error_message : null, // Clear if not present
                    serial_connected: data.serial_connected !== undefined ? data.serial_connected : false,
                    current_color_calibrating: data.current_color_calibrating !== undefined ? data.current_color_calibrating : prev.current_color_calibrating,
                    solution_preview: data.solution_preview !== undefined ? data.solution_preview : prev.solution_preview,
                }));
                if (data.processed_frame) {
                    setProcessedFrame(`data:image/jpeg;base64,${data.processed_frame}`);
                } else if (data.processed_frame === null && gameState.mode !== "error") { 
                     // If backend explicitly sends null for frame (e.g. after command), 
                     // don't clear it if we were already showing an error frame.
                     // This logic might need refinement based on how you want to handle "no frame" vs "error frame".
                     // For now, if backend sends null and we are not in error mode, we can clear.
                     // setProcessedFrame(null); // Or keep the last good frame.
                }
            } catch (e) {
                console.error("Error parsing message from backend:", e, event.data);
                setGameState(prev => ({ ...prev, error_message: "Received invalid data from backend."}));
            }
        };

        ws.onclose = (event) => { 
            setStatus(`WebSocket Closed (Code: ${event.code})`);
            console.log(`WebSocket closed: Code ${event.code}, Reason: ${event.reason}`);
            if (frameSenderHandle.current) {
                cancelAnimationFrame(frameSenderHandle.current);
                frameSenderHandle.current = null;
            }
        };
        ws.onerror = (err_event) => { 
            setStatus("WebSocket Error"); 
            console.error("WebSocket Error Event:", err_event);
            if (frameSenderHandle.current) {
                cancelAnimationFrame(frameSenderHandle.current);
                frameSenderHandle.current = null;
            }
        };

        return () => { 
            console.log("Cleaning up RubiksPage useEffect dependencies.");
            if (frameSenderHandle.current) {
                cancelAnimationFrame(frameSenderHandle.current);
                frameSenderHandle.current = null;
            }
            if (wsRef.current) {
                console.log("Closing WebSocket connection from cleanup.");
                wsRef.current.onopen = null; // Remove handlers before closing
                wsRef.current.onmessage = null;
                wsRef.current.onerror = null;
                wsRef.current.onclose = null;
                wsRef.current.close();
                wsRef.current = null;
            }
            if (localStream) {
                localStream.getTracks().forEach((t) => t.stop());
            }
             if (videoRef.current && videoRef.current.srcObject) { 
                videoRef.current.srcObject.getTracks().forEach(t => t.stop());
                videoRef.current.srcObject = null;
            }
        };
    // Make sure serialPort and appliedCameraSettings are stable or memoized if they cause re-runs too often
    }, [isClient, gameStarted, appliedCameraSettings.useIpCamera, appliedCameraSettings.ipCameraAddress, serialPort]);


    if (!isClient) return <div className="flex justify-center items-center min-h-screen"><p className="text-xl">Loading Client...</p></div>;

    const handleSendCommand = (commandPayload) => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            // console.log("Sending command:", commandPayload);
            wsRef.current.send(JSON.stringify(commandPayload));
        } else {
            setGameState(prev => ({ ...prev, error_message: "WebSocket not connected. Cannot send command."}));
            console.warn("Attempted to send command, but WebSocket is not open.", commandPayload);
        }
    };
    
    const SettingsModal = () =>
        showSettings && (
            <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-60 p-4">
                <div className="bg-white p-6 rounded-lg shadow-xl w-full max-w-md relative">
                    <button onClick={() => setShowSettings(false)} className="absolute top-3 right-3 text-gray-400 hover:text-gray-600 text-2xl font-bold">×</button>
                    <h3 className="text-xl font-semibold text-gray-800 mb-5">Settings</h3>
                    <div className="mb-4">
                        <label htmlFor="serialPort" className="block mb-1.5 text-sm font-medium text-gray-700">Arduino Serial Port:</label>
                        <input type="text" id="serialPort" value={serialPort} onChange={(e) => setSerialPort(e.target.value)} placeholder="e.g., COM7 or /dev/ttyUSB0" className="w-full p-2.5 border border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500"/>
                    </div>
                    <div className="flex items-center mb-4">
                        <input type="checkbox" id="useIpCamera" checked={cameraSettings.useIpCamera} onChange={(e) => setCameraSettings({ ...cameraSettings, useIpCamera: e.target.checked, ipCameraAddress: cameraSettings.useIpCamera ? "" : appliedCameraSettings.ipCameraAddress || "http://192.168.1.9:8080/video"})} className="mr-2 h-4 w-4 text-indigo-600 border-gray-300 rounded focus:ring-indigo-500"/>
                        <label htmlFor="useIpCamera" className="text-sm font-medium text-gray-700">Use IP Camera</label>
                    </div>
                    {cameraSettings.useIpCamera && (
                        <div className="mb-5">
                            <label htmlFor="ipCameraAddress" className="block mb-1.5 text-sm font-medium text-gray-700">IP Camera URL:</label>
                            <input type="text" id="ipCameraAddress" value={cameraSettings.ipCameraAddress} onChange={(e) => setCameraSettings({ ...cameraSettings, ipCameraAddress: e.target.value })} placeholder="http://camera-ip:port/video" className="w-full p-2.5 border border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500"/>
                        </div>
                    )}
                    <button
                        onClick={() => {
                            setAppliedCameraSettings(cameraSettings); // This will trigger useEffect
                            setShowSettings(false);
                        }}
                        className="w-full px-4 py-2.5 bg-indigo-600 text-white font-semibold rounded-md shadow-sm hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                    >Apply & Restart Connection</button>
                </div>
            </div>
        );

    return (
        <div className="flex flex-col items-center gap-4 p-4 min-h-screen bg-gradient-to-br from-slate-200 to-slate-400 font-sans">
            <SettingsModal />
            <div className="w-full max-w-5xl bg-white rounded-xl shadow-2xl p-6 md:p-8 mt-5">
                <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-6 pb-4 border-b border-gray-200">
                    <h1 className="text-3xl font-bold text-slate-800 mb-3 md:mb-0">Rubik's Cube Solver</h1>
                    <div className="flex flex-wrap items-center gap-3">
                        <span title={status} className={`px-3 py-1.5 rounded-full text-xs font-semibold truncate max-w-[150px] md:max-w-xs ${status.includes("Connected") ? "bg-green-100 text-green-800" : status.includes("Error") || status.includes("Disconnected") || status.includes("Closed") ? "bg-red-100 text-red-800" : "bg-yellow-100 text-yellow-800"}`}>
                            {status}
                        </span>
                         <span className={`px-3 py-1.5 rounded-full text-xs font-semibold ${gameState.serial_connected ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"}`}>
                            Arduino: {gameState.serial_connected ? "Ok" : "Off"}
                        </span>
                        <button onClick={() => setShowSettings(true)} className="px-4 py-2 bg-indigo-500 text-white text-sm font-medium rounded-md shadow-sm hover:bg-indigo-600 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2">Settings</button>
                    </div>
                </div>

                <div className="w-full bg-slate-50 rounded-lg shadow p-5 mb-6">
                    <h2 className="text-xl font-semibold text-slate-700 mb-4">Controls</h2>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                        <button
                            className={`w-full px-4 py-2.5 rounded-md font-semibold shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-2 transition-colors duration-150 ${!gameStarted ? "bg-green-500 text-white hover:bg-green-600 focus:ring-green-500" : "bg-red-500 text-white hover:bg-red-600 focus:ring-red-500"}`}
                            onClick={() => setGameStarted(!gameStarted)}
                        >
                            {gameStarted ? "Stop Game" : "Start Game"}
                        </button>
                        <button className="w-full px-4 py-2.5 bg-amber-500 text-white rounded-md font-semibold shadow-sm hover:bg-amber-600 focus:ring-amber-400 disabled:opacity-60 disabled:cursor-not-allowed" onClick={() => handleSendCommand({ mode: "calibrating" })} disabled={!gameStarted || gameState.mode === 'solving' || gameState.mode === 'scrambling'}>Calibrate</button>
                        <button className="w-full px-4 py-2.5 bg-sky-500 text-white rounded-md font-semibold shadow-sm hover:bg-sky-600 focus:ring-sky-400 disabled:opacity-60 disabled:cursor-not-allowed" onClick={() => handleSendCommand({ mode: "scanning" })} disabled={!gameStarted || gameState.mode === 'solving' || gameState.mode === 'scrambling'}>Scan Cube</button>
                        <button className="w-full px-4 py-2.5 bg-purple-500 text-white rounded-md font-semibold shadow-sm hover:bg-purple-600 focus:ring-purple-400 disabled:opacity-60 disabled:cursor-not-allowed" onClick={() => handleSendCommand({ action: "scramble_cube" })} disabled={!gameStarted || gameState.mode === 'solving' || gameState.mode === 'scrambling' || gameState.mode !== 'idle'}>Scramble</button>
                    </div>
                    {gameState.mode === "calibrating" && gameStarted && (
                        <div className="mt-4">
                            <button className="w-full px-4 py-2.5 bg-teal-500 text-white rounded-md font-semibold shadow-sm hover:bg-teal-600 focus:ring-teal-400" onClick={() => handleSendCommand({ action: "calibrate_color" })}>
                                Capture Color ({gameState.current_color_calibrating || (gameState.calibration_step < COLOR_NAMES.length ? COLOR_NAMES[gameState.calibration_step] : 'Done')})
                            </button>
                        </div>
                    )}
                     { (gameState.mode === "solving" || gameState.mode === "scrambling") && gameStarted && (
                        <div className="mt-4">
                             <button className="w-full px-4 py-2.5 bg-orange-500 text-white rounded-md font-semibold shadow-sm hover:bg-orange-600 focus:ring-orange-400" onClick={() => handleSendCommand({ action: "stop_operation" })}>
                                Stop Current Action
                            </button>
                        </div>
                     )}
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                    <div className="flex flex-col items-center p-3 bg-slate-100 rounded-lg">
                        <div className="mb-2 text-center font-semibold text-slate-700">Camera Feed (Local Preview)</div>
                        <div className="relative w-[320px] h-[240px] rounded-md overflow-hidden border-2 border-slate-300 bg-slate-900 flex items-center justify-center shadow-inner">
                            {appliedCameraSettings.useIpCamera ? (
                                <img 
                                    ref={ipCamImgRef} 
                                    key={appliedCameraSettings.ipCameraAddress} // Forces re-render if URL changes
                                    src={appliedCameraSettings.ipCameraAddress} 
                                    alt="IP Camera Preview" width={320} height={240} className="object-contain" crossOrigin="anonymous" 
                                    onLoad={() => { console.log("IP Cam image loaded."); setIsCameraLoading(false);}} 
                                    onError={() => { console.error("IP Cam image error."); setGameState(prev => ({...prev, error_message: "IP Cam Preview Error."})); setIsCameraLoading(false);}}
                                />
                            ) : (
                                <video ref={videoRef} autoPlay playsInline muted width={320} height={240} className="object-contain" />
                            )}
                            {isCameraLoading && <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-60 text-white text-lg font-semibold">Loading Preview...</div>}
                        </div>
                        <canvas ref={canvasRef} className="hidden" /> {/* Keep hidden for frame grabbing */}
                    </div>
                    <div className="flex flex-col items-center p-3 bg-slate-100 rounded-lg">
                        <div className="mb-2 text-center font-semibold text-slate-700">Processed View (from Backend)</div>
                        <div className="relative w-[320px] h-[240px] rounded-md overflow-hidden border-2 border-slate-300 bg-slate-900 flex items-center justify-center shadow-inner">
                            {processedFrame ? (
                                <img src={processedFrame} width={320} height={240} alt="Processed Frame" className="object-contain"/>
                            ) : (
                                <span className="text-slate-500 text-center p-2">{gameStarted ? "Waiting for backend frame..." : "Game stopped."}</span>
                            )}
                        </div>
                    </div>
                </div>

                <div className="p-5 bg-slate-50 rounded-lg border border-slate-200 shadow">
                    <h3 className="text-xl font-semibold text-slate-700 mb-3">Game Status & Messages</h3>
                    <div className="text-sm space-y-1.5 text-slate-600">
                        <p>Mode: <span className="font-semibold text-slate-800">{gameState.mode}</span></p>
                        <p>Status: <span className="font-semibold text-slate-800">{gameState.status_message || "N/A"}</span></p>
                        {gameState.error_message && <p className="text-red-600 font-semibold">Error: {gameState.error_message}</p>}
                        {(gameState.mode === "solving" || gameState.mode === "scrambling") && gameState.total_solve_moves > 0 && (
                            <p>Progress: <span className="font-semibold text-slate-800">{gameState.solve_move_index}/{gameState.total_solve_moves} moves</span></p>
                        )}
                         {gameState.mode === "scanning" && (
                            <p>Scan Progress: <span className="font-semibold text-slate-800">{gameState.scan_index}/12</span></p>
                        )}
                        {gameState.solution_preview && <p>Solution Preview: <span className="font-mono text-xs text-slate-700 bg-slate-200 px-1 py-0.5 rounded">{gameState.solution_preview}</span></p>}
                    </div>
                </div>
            </div>
        </div>
    );
}