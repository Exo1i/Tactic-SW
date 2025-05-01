// app/page.jsx
"use client";
import React, { useState, useRef, useEffect, useCallback } from "react"; // Import useCallback

// Helper constant
const COLOR_NAMES = ["W", "R", "G", "Y", "O", "B"];

export default function RubiksSolverPage() {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const wsRef = useRef(null);
    const ipCamImgRef = useRef(null);

    const [status, setStatus] = useState({
        mode: "connecting",
        status_message: "Connecting to backend...",
        error_message: null,
        solution: null,
        serial_connected: false,
        calibration_step: null, // Add all potential fields from SolverStatus interface
        current_color: null,
        scan_index: null,
        solve_move_index: null,
        total_solve_moves: null,
    });
    const [wsConnectionStatus, setWsConnectionStatus] = useState("Connecting...");
    const [processedFrame, setProcessedFrame] = useState(null);
    const [showSettings, setShowSettings] = useState(false);
    const [appliedCameraSettings, setAppliedCameraSettings] = useState({
        useIpCamera: false,
        ipCameraUrl: "http://192.168.1.9:8080/video",
    });
    const [modalCameraSettings, setModalCameraSettings] = useState({
        useIpCamera: false,
        ipCameraUrl: "http://192.168.1.9:8080/video",
    });

    const [isCameraLoading, setIsCameraLoading] = useState(true);
    const [isBackendProcessing, setIsBackendProcessing] = useState(false);
    const [isActionLoading, setIsActionLoading] = useState(false);

    const sendNextFrameRef = useRef(true);
    const lastSentRef = useRef(0);
    const minFrameInterval = 100;

    // --- Camera Initialization ---
    const initCamera = useCallback(async () => {
        console.log("initCamera called with settings:", appliedCameraSettings);
        setIsCameraLoading(true);
        setStatus(prev => ({ ...prev, status_message: "Initializing camera...", error_message: null }));

        if (videoRef.current && videoRef.current.srcObject) {
            console.log("Stopping previous camera stream.");
            (videoRef.current.srcObject).getTracks().forEach(track => track.stop());
            videoRef.current.srcObject = null;
        }
        if (ipCamImgRef.current) {
            ipCamImgRef.current.src = '';
        }

        if (!appliedCameraSettings.useIpCamera) {
            console.log("Attempting to use device camera...");
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                console.error("getUserMedia not supported!");
                setStatus(prev => ({...prev, mode:"error", status_message:"Camera Error", error_message:"Browser does not support camera access (getUserMedia)."}));
                setIsCameraLoading(false);
                return;
            }
            try {
                const constraints = {
                    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'environment' },
                    audio: false,
                };
                const stream = await navigator.mediaDevices.getUserMedia(constraints);
                console.log("getUserMedia success, stream ID:", stream.id);
                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                    console.log("Stream assigned to video element.");
                } else {
                    console.error("videoRef is null when assigning stream.");
                    stream.getTracks().forEach(track => track.stop());
                    setIsCameraLoading(false);
                }
            } catch (error) {
                console.error("getUserMedia error:", error);
                let errorMsg = "Failed to access camera.";
                if (error.name === "NotAllowedError") {
                    errorMsg = "Camera permission denied. Please allow access in browser settings.";
                } else if (error.name === "NotFoundError") {
                    errorMsg = "No suitable camera found.";
                } else if (error.name === "NotReadableError") {
                    errorMsg = "Camera is already in use or hardware error.";
                }
                setStatus(prev => ({ ...prev, mode: "error", status_message: "Camera Error", error_message: errorMsg }));
                setIsCameraLoading(false);
            }
        } else {
            console.log("Using IP Camera:", appliedCameraSettings.ipCameraUrl);
            if (!appliedCameraSettings.ipCameraUrl) {
                console.error("IP Camera URL is empty.");
                setStatus(prev => ({...prev, mode:"error", status_message:"Camera Error", error_message:"IP Camera URL cannot be empty."}));
                setIsCameraLoading(false);
                return;
            }
            setStatus(prev => ({ ...prev, status_message: "Loading IP Camera feed..." }));
        }
    }, [appliedCameraSettings]);

    // --- WebSocket Connection & Frame Sending ---
    useEffect(() => {
        let stopped = false;
        let reconnectTimeout = null;
        let frameRequestId = null;

        const connectWebSocket = () => {
            if (stopped || (wsRef.current && wsRef.current.readyState === WebSocket.OPEN)) return;

            setWsConnectionStatus("Connecting...");
            const wsProtocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
            const wsUrl = `${wsProtocol}${window.location.hostname}:8000/ws/rubiks`;
            console.log("Connecting WebSocket to:", wsUrl);
            const ws = new WebSocket(wsUrl);
            wsRef.current = ws;

            ws.onopen = () => {
                console.log("WebSocket Connected");
                setWsConnectionStatus("Connected");
                if (reconnectTimeout) clearTimeout(reconnectTimeout);
            };

            ws.onclose = (event) => {
                console.log("WebSocket Disconnected:", event.code, event.reason);
                setWsConnectionStatus("Disconnected");
                wsRef.current = null;
                setStatus(prev => ({ ...prev, mode: "connecting", status_message: "Connection lost. Reconnecting...", serial_connected: false })); // Assume serial lost too
                if (!stopped) {
                    if (reconnectTimeout) clearTimeout(reconnectTimeout);
                    reconnectTimeout = setTimeout(connectWebSocket, 5000);
                }
            };

            ws.onerror = (error) => {
                console.error("WebSocket Error:", error);
                setWsConnectionStatus("Error");
            };

            ws.onmessage = (event) => {
                setIsBackendProcessing(false);
                setIsActionLoading(false); // Reset action loading on any message
                sendNextFrameRef.current = true;

                try {
                    const data = JSON.parse(event.data);
                    setStatus(prev => ({
                        ...prev,
                        ...data,
                        error_message: data.mode === 'error' ? data.error_message : null,
                        solution: data.solution !== undefined ? data.solution : prev.solution
                    }));
                    if (data.processed_frame) {
                        setProcessedFrame(`data:image/jpeg;base64,${data.processed_frame}`);
                    }
                } catch (e) {
                    console.error("Failed to parse WebSocket message:", e, "Data:", event.data);
                    setStatus(prev => ({
                        ...prev,
                        mode: "error",
                        error_message: "Received invalid data from backend."
                    }));
                }
            };
        };

        const sendFrame = () => {
            if (stopped) return;
            frameRequestId = requestAnimationFrame(sendFrame);

            const now = Date.now();
            if (!sendNextFrameRef.current ||
                wsRef.current?.readyState !== WebSocket.OPEN ||
                now - lastSentRef.current < minFrameInterval ||
                isCameraLoading) {
                return;
            }

            if (status.mode === "error") return;

            const ws = wsRef.current;
            const canvas = canvasRef.current;
            let sourceEl = null;

            if (appliedCameraSettings.useIpCamera) {
                const img = ipCamImgRef.current;
                if (img && img.complete && img.naturalWidth > 0) {
                    sourceEl = img;
                }
            } else {
                if (videoRef.current && videoRef.current.readyState >= videoRef.current.HAVE_ENOUGH_DATA) {
                    sourceEl = videoRef.current;
                }
            }

            if (!sourceEl || !canvas || !ws) return;

            try {
                const ctx = canvas.getContext("2d");
                if (!ctx) return;

                const targetWidth = 320;
                const targetHeight = 240;
                if (canvas.width !== targetWidth) canvas.width = targetWidth;
                if (canvas.height !== targetHeight) canvas.height = targetHeight;

                ctx.drawImage(sourceEl, 0, 0, targetWidth, targetHeight);

                canvas.toBlob((blob) => {
                    if (blob && ws.readyState === WebSocket.OPEN) {
                        setIsBackendProcessing(true);
                        sendNextFrameRef.current = false;
                        lastSentRef.current = now;

                        blob.arrayBuffer().then((buffer) => {
                            if (ws.readyState === WebSocket.OPEN) {
                                ws.send(buffer);
                            } else {
                                sendNextFrameRef.current = true; setIsBackendProcessing(false);
                            }
                        }).catch(err => {
                            console.error("Error getting ArrayBuffer:", err);
                            sendNextFrameRef.current = true; setIsBackendProcessing(false);
                        });
                    } else {
                        sendNextFrameRef.current = true;
                        setIsBackendProcessing(false);
                    }
                }, "image/jpeg", 0.7);

            } catch (e) {
                console.warn("Error processing frame:", e);
                sendNextFrameRef.current = true;
                setIsBackendProcessing(false);
            }
        };

        initCamera();
        connectWebSocket();
        frameRequestId = requestAnimationFrame(sendFrame);

        return () => {
            stopped = true;
            console.log("Running cleanup for RubiksSolverPage useEffect");
            if (frameRequestId) cancelAnimationFrame(frameRequestId);
            if (reconnectTimeout) clearTimeout(reconnectTimeout);
            if (wsRef.current) {
                console.log("Closing WebSocket connection.");
                wsRef.current.close();
                wsRef.current = null;
            }
            if (videoRef.current && videoRef.current.srcObject) {
                console.log("Stopping device camera stream.");
                (videoRef.current.srcObject).getTracks().forEach(track => track.stop());
                videoRef.current.srcObject = null;
            }
            if (ipCamImgRef.current) {
                ipCamImgRef.current.src = '';
            }
            console.log("Cleanup complete.");
        };
    }, [appliedCameraSettings, initCamera]); // Add initCamera as dependency

    // --- API Call Helper ---
    const callApi = useCallback(async (endpoint, method = 'POST', body = null) => {
        if (isActionLoading) {
            console.warn("Action already in progress, skipping call to", endpoint);
            return;
        }
        setIsActionLoading(true);
        try {
            const options = { method };
            if (body) {
                options.headers = { 'Content-Type': 'application/json' };
                options.body = JSON.stringify(body);
            }
            const apiUrl = `http://${window.location.hostname}:8000${endpoint}`;
            console.log(`Calling API: ${method} ${apiUrl}`, body || '');
            const response = await fetch(apiUrl, options);
            const data = await response.json();

            if (!response.ok) {
                console.error(`API Error ${response.status}:`, data);
                throw new Error(data.detail || `API Error ${response.status}`);
            }
            console.log(`API Success ${endpoint}:`, data);

            if (endpoint === '/start_scramble' || endpoint === '/start_solve' || endpoint === '/start_calibration') {
                setStatus(prev => ({...prev, status_message: data.message || "Request sent..."}));
            }
            return data;
        } catch (error) {
            console.error(`API call to ${endpoint} failed:`, error);
            setStatus(prev => ({
                ...prev,
                mode: "error",
                error_message: error.message || "API request failed."
            }));
            // Reset loading on API error *only if WS doesn't handle it*
            // Relying on WS is better, but add a safety timeout
            setTimeout(() => {
                if(isActionLoading){ setIsActionLoading(false) }
            } , 2000); // Reset after 2s if WS hasn't already
            return null;
        }
        // Removed finally block - rely on WS message to reset isActionLoading
    }, [isActionLoading]); // Depend on isActionLoading


    // --- Button Click Handlers ---
    const handleStartCalibration = () => callApi('/start_calibration');
    const handleCaptureColor = () => callApi('/capture_calibration_color');
    const handleSaveCalibration = () => callApi('/save_calibration');
    const handleResetCalibration = () => callApi('/reset_calibration');
    const handleStartSolve = () => callApi('/start_solve');
    const handleStopAndReset = () => callApi('/stop_and_reset');
    const handleStartScramble = () => callApi('/start_scramble');

    // --- Apply Settings from Modal ---
    const applySettings = () => {
        console.log("Applying new camera settings:", modalCameraSettings);
        setAppliedCameraSettings(modalCameraSettings);
        setShowSettings(false);
    };

    // --- Determine Button Disabled States ---
    const isEffectivelyBusy = status.mode !== 'idle' && status.mode !== 'error' && status.mode !== 'connecting';
    const actionDisabled = isEffectivelyBusy || isActionLoading || isCameraLoading;

    return (
        <div className="flex flex-col items-center gap-4 p-4 min-h-screen bg-gray-100 font-sans">
            {/* Settings Modal */}
            {showSettings && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-40 backdrop-blur-sm">
                    <div className="bg-white p-6 rounded-lg shadow-xl w-full max-w-md relative">
                        <button
                            className="absolute top-2 right-2 text-gray-500 hover:text-gray-800 text-2xl leading-none font-bold"
                            onClick={() => setShowSettings(false)}
                            aria-label="Close Settings"
                        >×</button>
                        <h3 className="text-lg font-medium mb-4 text-gray-800">Camera Settings</h3>
                        <div className="flex items-center mb-3">
                            <input
                                type="checkbox"
                                id="useIpCameraModal"
                                checked={modalCameraSettings.useIpCamera}
                                onChange={(e) => setModalCameraSettings({ ...modalCameraSettings, useIpCamera: e.target.checked })}
                                className="mr-2 h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                            />
                            <label htmlFor="useIpCameraModal" className="text-sm font-medium text-gray-700">
                                Use IP Camera
                            </label>
                        </div>
                        {modalCameraSettings.useIpCamera && (
                            <div className="mb-4">
                                <label htmlFor="ipCameraUrlModal" className="block mb-1 text-sm font-medium text-gray-700">
                                    IP Camera URL:
                                </label>
                                <input
                                    type="text"
                                    id="ipCameraUrlModal"
                                    value={modalCameraSettings.ipCameraUrl}
                                    onChange={(e) => setModalCameraSettings({ ...modalCameraSettings, ipCameraUrl: e.target.value })}
                                    placeholder="http://camera-ip:port/stream"
                                    className="w-full p-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                                />
                                <small className="text-xs text-gray-500 mt-1 block">
                                    E.g., http://192.168.1.100:8080/video (Must support MJPEG & CORS)
                                </small>
                            </div>
                        )}
                        <button
                            onClick={applySettings}
                            className="w-full px-4 py-2 bg-green-600 text-white font-medium rounded-md shadow-sm hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500"
                        >
                            Apply Settings
                        </button>
                    </div>
                </div>
            )}

            {/* Main Content Area */}
            <div className="w-full max-w-5xl bg-white rounded-xl shadow-lg p-6 mt-4">
                {/* Header */}
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-4 border-b border-gray-200 pb-4">
                    <h1 className="text-2xl font-bold text-gray-800 mb-2 sm:mb-0">Rubik's Cube Solver</h1>
                    <div className="flex items-center gap-3">
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                            wsConnectionStatus === "Connected" ? "bg-green-100 text-green-800"
                                : wsConnectionStatus === "Disconnected" ? "bg-red-100 text-red-800"
                                    : wsConnectionStatus === "Connecting..." ? "bg-yellow-100 text-yellow-800 animate-pulse"
                                        : "bg-gray-100 text-gray-800"
                        }`}>
                            WS: {wsConnectionStatus}
                        </span>
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                            status.serial_connected ? "bg-blue-100 text-blue-800" : "bg-red-100 text-red-800"
                        }`}>
                            {status.serial_connected ? "Serial OK" : "Serial disconnected"}
                         </span>
                        <button
                            onClick={() => { setModalCameraSettings(appliedCameraSettings); setShowSettings(true); }}
                            className="px-3 py-1 bg-blue-500 text-white rounded-md hover:bg-blue-600 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                            aria-label="Open Camera Settings"
                        >
                            Settings
                        </button>
                    </div>
                </div>

                {/* Status Display */}
                <div className="mb-5 p-3 bg-gray-50 rounded-lg border border-gray-200 shadow-sm">
                    <p className="text-sm font-medium text-gray-700">
                        Status: <span className={`font-semibold capitalize ${status.mode === 'error' ? 'text-red-600' : 'text-gray-900'}`}>{status.mode}</span>
                        {(status.mode === 'solving' || status.mode === 'scrambling') && status.total_solve_moves > 0 && (
                            <span className="ml-2 text-xs font-normal text-gray-500">
                                 ({status.solve_move_index || 0} / {status.total_solve_moves})
                             </span>
                        )}
                        {status.mode === 'scanning' && status.scan_index != null && (
                            <span className="ml-2 text-xs font-normal text-gray-500">
                                 (Scan {status.scan_index + 1} / 12)
                             </span>
                        )}
                    </p>
                    <p className="text-sm mt-1 text-gray-600 min-h-[1.25rem]">
                        {status.status_message || " "}
                    </p>
                    {status.mode === 'error' && status.error_message && (
                        <p className="text-sm mt-1 text-red-700 font-medium bg-red-50 p-2 rounded border border-red-200">
                            Error: {status.error_message}
                        </p>
                    )}
                    {status.solution && (
                        <div className="mt-2 text-sm text-blue-800 font-mono bg-blue-50 p-2 rounded border border-blue-200 max-h-24 overflow-y-auto">
                            <span className="font-semibold">Solution Found:</span>
                            <p className="whitespace-pre-wrap break-words text-xs leading-relaxed">{status.solution}</p>
                        </div>
                    )}
                </div>

                {/* Camera Feeds */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                    {/* Live Camera Feed */}
                    <div className="flex flex-col items-center">
                        <div className="mb-1 text-center text-sm font-medium text-gray-600">Live Camera Input</div>
                        <div className="relative w-[320px] h-[240px] rounded-lg overflow-hidden border-2 border-gray-300 bg-black flex items-center justify-center shadow-inner">
                            {appliedCameraSettings.useIpCamera && (
                                <img
                                    ref={ipCamImgRef}
                                    id="ip-camera-img-display"
                                    src={appliedCameraSettings.ipCameraUrl}
                                    alt="IP Camera Feed"
                                    width={320}
                                    height={240}
                                    className="object-contain"
                                    crossOrigin="anonymous"
                                    onLoad={() => { console.log("IP Cam loaded"); setIsCameraLoading(false); }}
                                    onError={(e) => {
                                        console.error("IP camera display error:", e);
                                        setIsCameraLoading(false);
                                        if (status.mode !== 'error') {
                                            setStatus(prev => ({
                                                ...prev,
                                                status_message: "IP Camera Error (Display)",
                                                error_message: "Failed to load IP Cam image. Check URL/Network/CORS."
                                            }));
                                        }
                                    }}
                                />
                            )}
                            {!appliedCameraSettings.useIpCamera && (
                                <video
                                    ref={videoRef}
                                    autoPlay playsInline muted
                                    width={320} height={240}
                                    className="object-contain"
                                    style={{ background: "#222" }}
                                    onCanPlay={() => { console.log("Device Cam ready"); setIsCameraLoading(false); }}
                                    onError={(e) => {
                                        console.error("Video element display error:", e);
                                        setIsCameraLoading(false);
                                    }}
                                />
                            )}
                            {isCameraLoading && (
                                <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-70 text-white text-lg">
                                    <div className="flex flex-col items-center">
                                        <svg className="animate-spin h-8 w-8 text-white mb-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                        </svg>
                                        <span>Loading Camera...</span>
                                    </div>
                                </div>
                            )}
                            {!isCameraLoading && status.mode === "error" && status.error_message?.toLowerCase().includes("camera") && (
                                <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-75 text-white text-sm p-2">
                                    <div className="p-3 text-center rounded bg-red-800 bg-opacity-90 shadow">
                                        <span className="font-bold block text-base">Camera Error</span>
                                        <span className="text-xs mt-1 block">{status.error_message}</span>
                                    </div>
                                </div>
                            )}
                            <canvas ref={canvasRef} width={320} height={240} className="hidden" />
                        </div>
                    </div>

                    {/* Processed Feed */}
                    <div className="flex flex-col items-center">
                        <div className="mb-1 text-center text-sm font-medium text-gray-600">Processed View (from Backend)</div>
                        <div className="relative w-[320px] h-[240px] rounded-lg overflow-hidden border-2 border-gray-300 bg-black flex items-center justify-center shadow-inner">
                            {processedFrame ? (
                                <img src={processedFrame} width={320} height={240} alt="Processed Frame" className="object-contain" />
                            ) : (
                                <span className="text-gray-400 italic">Waiting for backend...</span>
                            )}
                            {isBackendProcessing && (
                                <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-60 text-white text-xs p-1 animate-pulse">
                                    Backend Processing...
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                {/* Control Buttons */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {/* Calibration Controls */}
                    <div className="border border-gray-200 p-4 rounded-lg bg-gray-50 shadow-sm">
                        <h3 className="font-semibold text-base mb-3 text-center text-gray-700">Calibration</h3>
                        <div className="flex flex-col gap-2">
                            <button
                                onClick={handleStartCalibration}
                                disabled={actionDisabled}
                                className="btn bg-indigo-500 hover:bg-indigo-600 disabled:bg-gray-400"
                            > Start Calibration </button>
                            {status.mode === 'calibrating' && (
                                <>
                                    <button
                                        onClick={handleCaptureColor}
                                        disabled={isActionLoading || (status.calibration_step == null || status.calibration_step >= COLOR_NAMES.length)}
                                        className="btn bg-teal-500 hover:bg-teal-600 disabled:bg-gray-400"
                                    > Capture '{status.current_color || '?'}' </button>
                                    <button
                                        onClick={handleSaveCalibration}
                                        disabled={isActionLoading || (status.calibration_step == null || status.calibration_step < COLOR_NAMES.length)}
                                        className="btn bg-green-500 hover:bg-green-600 disabled:bg-gray-400"
                                    > Save Calibration </button>
                                </>
                            )}
                            <button
                                onClick={handleResetCalibration}
                                disabled={isActionLoading}
                                className="btn bg-yellow-500 hover:bg-yellow-600 disabled:bg-gray-400 mt-2"
                            > Reset Calibration </button>
                        </div>
                    </div>

                    {/* Solving Controls */}
                    <div className="border border-gray-200 p-4 rounded-lg bg-gray-50 shadow-sm">
                        <h3 className="font-semibold text-base mb-3 text-center text-gray-700">Solver</h3>
                        <div className="flex flex-col gap-2">
                            <button
                                onClick={handleStartSolve}
                                disabled={actionDisabled || !status.serial_connected}
                                title={!status.serial_connected ? "Serial port disconnected" : ""}
                                className={`btn bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 ${!status.serial_connected ? 'cursor-not-allowed' : ''}`}
                            > Solve Cube </button>
                            <button
                                onClick={handleStopAndReset}
                                disabled={status.mode === 'idle' || status.mode === 'connecting' || isActionLoading}
                                className="btn bg-orange-500 hover:bg-orange-600 disabled:bg-gray-400 mt-2"
                            > Stop & Reset </button>
                        </div>
                    </div>

                    {/* Scramble Controls */}
                    <div className="border border-gray-200 p-4 rounded-lg bg-gray-50 shadow-sm">
                        <h3 className="font-semibold text-base mb-3 text-center text-gray-700">Scramble</h3>
                        <div className="flex flex-col gap-2">
                            <button
                                onClick={handleStartScramble}
                                disabled={actionDisabled || !status.serial_connected}
                                title={!status.serial_connected ? "Serial port disconnected" : ""}
                                className={`btn bg-red-500 hover:bg-red-600 disabled:bg-gray-400 ${!status.serial_connected ? 'cursor-not-allowed' : ''}`}
                            > Scramble Cube </button>
                        </div>
                    </div>
                </div>

                {/* Action Loading Global Indicator */}
                {isActionLoading && (
                    <div className="mt-4 text-center text-sm text-gray-600 animate-pulse font-medium">
                        Waiting for backend response...
                    </div>
                )}
            </div>
            {/* Shared Button Styles */}
            <style jsx global>{`
                .btn {
                    padding: 0.5rem 1rem;
                    color: white;
                    border-radius: 0.375rem; /* rounded-md */
                    font-weight: 500; /* font-medium */
                    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05); /* shadow-sm */
                    transition: background-color 0.2s;
                    text-align: center;
                }
                .btn:disabled {
                    cursor: not-allowed;
                    opacity: 0.7;
                }
                .btn:focus {
                    outline: none;
                    /* Add a subtle focus ring */
                    box-shadow: 0 0 0 2px rgba(96, 165, 250, 0.5); /* Example focus ring (Tailwind blue-300 equivalent) */
                 }
             `}</style>
        </div>
    );
}