// app/page.jsx
"use client";
import React, { useState, useRef, useEffect, useCallback } from "react";
import { initializeVideoSource } from "@/utils/cameraUtils";

// Helper constant
const COLOR_NAMES = ["W", "R", "G", "Y", "O", "B"];
const MIN_FRAME_INTERVAL = 30; // Increase interval between frames to 100ms (10 FPS)
const JPEG_QUALITY = 1; // Increase JPEG quality for better image

export default function RubiksSolverPage() {
    const gameId = "rubiks";
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const wsRef = useRef(null);
    const ipCamImgRef = useRef(null);

    const [isClient, setIsClient] = useState(false);
    const [status, setStatus] = useState("Connecting...");
    const [output, setOutput] = useState(null);
    const [processedFrame, setProcessedFrame] = useState(null);
    const [showSettings, setShowSettings] = useState(false);
    const [cameraSettings, setCameraSettings] = useState({
        useIpCamera: false,
        ipCameraAddress: "",
    });
    const [appliedCameraSettings, setAppliedCameraSettings] = useState({
        useIpCamera: false,
        ipCameraAddress: "",
    });

    // Game state
    const [gameState, setGameState] = useState({
        mode: "idle",
        calibrationStep: 0,
        scanIndex: 0,
        solveIndex: 0,
        totalMoves: 0,
        statusMessage: "",
        errorMessage: null,
        serialConnected: false
    });
    const [gameStarted, setGameStarted] = useState(false);

    // Loading indicators
    const [isCameraLoading, setIsCameraLoading] = useState(false);
    const [isBackendLoading, setIsBackendLoading] = useState(false);

    const sendNextFrameRef = useRef(true);
    const lastSentRef = useRef(0);

    // Set isClient to true when component mounts
    useEffect(() => {
        setIsClient(true);
    }, []);

    useEffect(() => {
        // Only run this effect on the client side
        if (!isClient || !gameStarted) return;

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

        ws.onopen = () => {
            setStatus("Connected");
            // Send initial config
            ws.send(JSON.stringify({
                serial_port: "COM7",
                serial_baudrate: 9600
            }));
        };
        ws.onclose = () => setStatus("Disconnected");
        ws.onerror = () => setStatus("Error");

        ws.onmessage = (event) => {
            setIsBackendLoading(false);
            sendNextFrameRef.current = true;
            try {
                const data = JSON.parse(event.data);
                setOutput(data);
                setGameState(prev => ({
                    ...prev,
                    mode: data.mode || prev.mode,
                    calibrationStep: data.calibration_step !== undefined ? data.calibration_step : prev.calibrationStep,
                    scanIndex: data.scan_index !== undefined ? data.scan_index : prev.scanIndex,
                    solveIndex: data.solve_move_index || prev.solveIndex,
                    totalMoves: data.total_solve_moves || prev.totalMoves,
                    statusMessage: data.status_message || prev.statusMessage,
                    errorMessage: data.error_message || null,
                    serialConnected: data.serial_connected || false
                }));
                if (data.processed_frame) {
                    setProcessedFrame(`data:image/jpeg;base64,${data.processed_frame}`);
                }
            } catch {
                setOutput(event.data);
            }
        };

        function sendFrame() {
            if (stopped) return;
            const now = Date.now();
            if (!sendNextFrameRef.current) {
                requestAnimationFrame(sendFrame);
                return;
            }
            if (now - lastSentRef.current < MIN_FRAME_INTERVAL) {
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
                const ctx = canvas.getContext("2d", { alpha: false });
                try {
                    // Clear canvas and draw new frame
                    ctx.fillStyle = 'black';
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
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
                        JPEG_QUALITY
                    );
                } catch (e) {}
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
    }, [isClient, gameStarted, appliedCameraSettings]);

    // Only render client-side content after mounting
    if (!isClient) {
        return <div>Loading...</div>;
    }

    const handleCameraSettingsChange = (newSettings) => {
        setCameraSettings(newSettings);
    };

    const handleModeChange = (mode) => {
        if (wsRef.current && wsRef.current.readyState === 1) {
            wsRef.current.send(JSON.stringify({ mode }));
        }
    };

    const handleAction = (action) => {
        if (wsRef.current && wsRef.current.readyState === 1) {
            wsRef.current.send(JSON.stringify({ action }));
        }
    };

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

    return (
        <div className="flex flex-col items-center gap-4 p-4 min-h-screen bg-gradient-to-br from-gray-100 to-gray-300">
            <SettingsModal />
            <div className="w-full max-w-4xl bg-white rounded-xl shadow-lg p-6 mt-4">
                <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-4">
                    <h1 className="text-2xl font-bold mb-2 md:mb-0">Rubik's Cube Solver</h1>
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
                        <span
                            className={`px-2 py-1 rounded text-xs ${
                                gameState.serialConnected
                                    ? "bg-green-100 text-green-700"
                                    : "bg-red-100 text-red-700"
                            }`}
                        >
                            Arduino: {gameState.serialConnected ? "Connected" : "Disconnected"}
                        </span>
                        <button
                            onClick={() => setShowSettings(true)}
                            className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600"
                        >
                            Camera Settings
                        </button>
                    </div>
                </div>

                <div className="w-full max-w-lg bg-gray-50 rounded-xl shadow p-4 mb-4">
                    <h2 className="text-lg font-semibold mb-2">Game Controls</h2>
                    <div className="grid grid-cols-2 gap-2">
                        <button
                            className={`px-4 py-2 rounded ${
                                !gameStarted
                                    ? "bg-green-600 text-white"
                                    : "bg-gray-400 text-gray-700"
                            }`}
                            onClick={() => setGameStarted(true)}
                            disabled={gameStarted}
                        >
                            {gameStarted ? "Game Running..." : "Start Game"}
                        </button>
                        <button
                            className="px-4 py-2 bg-yellow-600 text-white rounded"
                            onClick={() => handleModeChange("calibrating")}
                            disabled={!gameStarted}
                        >
                            Calibrate Colors
                        </button>
                        <button
                            className="px-4 py-2 bg-blue-600 text-white rounded"
                            onClick={() => handleModeChange("scanning")}
                            disabled={!gameStarted}
                        >
                            Start Scanning
                        </button>
                        <button
                            className="px-4 py-2 bg-purple-600 text-white rounded"
                            onClick={() => handleModeChange("scrambling")}
                            disabled={!gameStarted}
                        >
                            Scramble Cube
                        </button>
                    </div>
                    {gameState.mode === "calibrating" && (
                        <div className="mt-2">
                            <button
                                className="px-4 py-2 bg-green-600 text-white rounded w-full"
                                onClick={() => handleAction("calibrate")}
                            >
                                Capture Color {COLOR_NAMES[gameState.calibrationStep]}
                            </button>
                        </div>
                    )}
                    {gameState.mode === "scanning" && (
                        <div className="mt-2">
                            <button
                                className="px-4 py-2 bg-green-600 text-white rounded w-full"
                                onClick={() => handleAction("scan")}
                            >
                                Capture Scan {gameState.scanIndex + 1}/12
                            </button>
                        </div>
                    )}
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
                    <div className="flex flex-col items-center">
                        <div className="mb-2 text-center font-medium">Camera Feed</div>
                        <div className="relative w-[320px] h-[240px] rounded-lg overflow-hidden border-2 border-gray-300 bg-black flex items-center justify-center">
                            {appliedCameraSettings.useIpCamera ? (
                                <img
                                    ref={ipCamImgRef}
                                    src={appliedCameraSettings.ipCameraAddress}
                                    alt="IP Camera"
                                    width={320}
                                    height={240}
                                    className="object-contain"
                                    crossOrigin="anonymous"
                                    onLoad={() => setIsCameraLoading(false)}
                                    onError={() => setStatus("IP Camera Error")}
                                />
                            ) : (
                                <video
                                    ref={videoRef}
                                    autoPlay
                                    playsInline
                                    width={320}
                                    height={240}
                                    className="object-contain"
                                    style={{ background: "#222" }}
                                />
                            )}
                            {isCameraLoading && (
                                <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-60 text-white text-lg">
                                    Loading...
                                </div>
                            )}
                        </div>
                        <canvas
                            ref={canvasRef}
                            width={320}
                            height={240}
                            style={{ display: "none" }}
                        />
                    </div>
                    <div className="flex flex-col items-center">
                        <div className="mb-2 text-center font-medium">Processed View</div>
                        <div className="relative w-[320px] h-[240px] rounded-lg overflow-hidden border-2 border-gray-300 bg-black flex items-center justify-center">
                            {processedFrame ? (
                                <img
                                    src={processedFrame}
                                    width={320}
                                    height={240}
                                    alt="Processed Frame"
                                    className="object-contain"
                                />
                            ) : (
                                <span className="text-gray-400">No frame</span>
                            )}
                            {isBackendLoading && (
                                <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-60 text-white text-lg">
                                    Processing...
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                <div className="mt-6 p-4 bg-gray-50 rounded-lg border">
                    <h3 className="text-lg font-medium mb-2">Game Status</h3>
                    <div className="text-sm">
                        <p className="mb-1">Mode: {gameState.mode}</p>
                        <p className="mb-1">Status: {gameState.statusMessage}</p>
                        {gameState.errorMessage && (
                            <p className="text-red-600 mb-1">Error: {gameState.errorMessage}</p>
                        )}
                        {gameState.mode === "solving" || gameState.mode === "scrambling" ? (
                            <p className="mb-1">
                                Progress: {gameState.solveIndex}/{gameState.totalMoves} moves
                            </p>
                        ) : null}
                    </div>
                </div>
            </div>
        </div>
    );
}