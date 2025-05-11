// app/page.jsx
"use client";
import React, { useState, useRef, useEffect, useCallback } from "react";

// Helper constant
const COLOR_NAMES = ["W", "R", "G", "Y", "O", "B"];

// --- SettingsModal Component (Moved Outside) ---
const SettingsModalContent = ({
    setShowSettings,
    serialPort, setSerialPort,
    cameraSettings, setCameraSettings,
    setAppliedCameraSettings
}) => {
    const handleApplySettings = () => {
        const newAppliedSettings = { // rotateAngle is no longer part of cameraSettings for apply
            ...cameraSettings,
            frameRate: Math.max(1, Math.min(30, parseInt(cameraSettings.frameRate, 10) || 10)),
            jpegQuality: Math.max(0.1, Math.min(1.0, parseFloat(cameraSettings.jpegQuality) || 0.75)),
            previewWidth: Math.max(160, Math.min(1280, parseInt(cameraSettings.previewWidth, 10) || 320)),
            previewHeight: Math.max(120, Math.min(720, parseInt(cameraSettings.previewHeight, 10) || 240)),
            // rotateAngle: parseInt(cameraSettings.rotateAngle, 10) || 0, // REMOVED
        };
        setCameraSettings(newAppliedSettings); 
        setAppliedCameraSettings(newAppliedSettings);
        setShowSettings(false);
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-60 p-4 overflow-y-auto">
            <div className="bg-white p-6 rounded-lg shadow-xl w-full max-w-lg relative max-h-[90vh] overflow-y-auto">
                <button onClick={() => setShowSettings(false)} className="absolute top-3 right-3 text-gray-400 hover:text-gray-600 text-2xl font-bold">×</button>
                <h3 className="text-xl font-semibold text-gray-800 mb-5">Settings</h3>
                
                <div className="mb-4">
                    <label htmlFor="serialPortModal" className="block mb-1.5 text-sm font-medium text-gray-700">Arduino Serial Port:</label>
                    <input 
                        type="text" 
                        id="serialPortModal" 
                        value={serialPort} 
                        onChange={(e) => setSerialPort(e.target.value)} 
                        placeholder="e.g., COM7 or /dev/ttyUSB0" 
                        className="w-full p-2.5 border border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500"
                    />
                </div>

                <div className="border-t pt-4 mt-4 mb-4">
                    <p className="text-md font-medium text-gray-700 mb-2">Camera Source</p>
                    <div className="flex items-center mb-3">
                        <input 
                            type="checkbox" 
                            id="useIpCameraModal" 
                            checked={cameraSettings.useIpCamera} 
                            onChange={(e) => setCameraSettings(prev => ({ ...prev, useIpCamera: e.target.checked, ipCameraAddress: !e.target.checked ? "" : (prev.ipCameraAddress || "http://192.168.1.9:8080/video")}))} 
                            className="mr-2 h-4 w-4 text-indigo-600 border-gray-300 rounded focus:ring-indigo-500"
                        />
                        <label htmlFor="useIpCameraModal" className="text-sm font-medium text-gray-700">Use IP Camera</label>
                    </div>
                    {cameraSettings.useIpCamera && (
                        <div className="mb-3">
                            <label htmlFor="ipCameraAddressModal" className="block mb-1.5 text-sm font-medium text-gray-700">IP Camera URL:</label>
                            <input 
                                type="text" 
                                id="ipCameraAddressModal" 
                                value={cameraSettings.ipCameraAddress} 
                                onChange={(e) => setCameraSettings(prev => ({ ...prev, ipCameraAddress: e.target.value }))} 
                                placeholder="http://camera-ip:port/video" 
                                className="w-full p-2.5 border border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500"
                            />
                        </div>
                    )}
                </div>

                <div className="border-t pt-4 mt-4 mb-5">
                     <p className="text-md font-medium text-gray-700 mb-3">Frame & Quality Settings</p>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                        <div>
                            <label htmlFor="frameRateModal" className="block mb-1.5 text-sm font-medium text-gray-700">Frame Rate (FPS): {cameraSettings.frameRate}</label>
                            <input type="range" id="frameRateModal" min="1" max="30" step="1" value={cameraSettings.frameRate} onChange={(e) => setCameraSettings(prev => ({ ...prev, frameRate: parseInt(e.target.value, 10) }))} className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-indigo-500"/>
                        </div>
                        <div>
                            <label htmlFor="jpegQualityModal" className="block mb-1.5 text-sm font-medium text-gray-700">JPEG Quality: {Number(cameraSettings.jpegQuality).toFixed(2)}</label>
                            <input type="range" id="jpegQualityModal" min="0.1" max="1.0" step="0.05" value={cameraSettings.jpegQuality} onChange={(e) => setCameraSettings(prev => ({ ...prev, jpegQuality: parseFloat(e.target.value) }))} className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-indigo-500"/>
                        </div>
                        <div>
                            <label htmlFor="previewWidthModal" className="block mb-1.5 text-sm font-medium text-gray-700">Preview Width (px):</label>
                            <input type="number" id="previewWidthModal" value={cameraSettings.previewWidth} onChange={(e) => setCameraSettings(prev => ({ ...prev, previewWidth: parseInt(e.target.value, 10) || prev.previewWidth }))} className="w-full p-2.5 border border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500"/>
                        </div>
                        <div>
                            <label htmlFor="previewHeightModal" className="block mb-1.5 text-sm font-medium text-gray-700">Preview Height (px):</label>
                            <input type="number" id="previewHeightModal" value={cameraSettings.previewHeight} onChange={(e) => setCameraSettings(prev => ({ ...prev, previewHeight: parseInt(e.target.value, 10) || prev.previewHeight }))} className="w-full p-2.5 border border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500"/>
                        </div>
                         {/* REMOVED Rotation Angle from here */}
                    </div>
                </div>
                
                <button
                    onClick={handleApplySettings}
                    className="w-full px-4 py-2.5 bg-indigo-600 text-white font-semibold rounded-md shadow-sm hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                >Apply & Restart Connection</button>
            </div>
        </div>
    );
};

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
    const [cameraSettings, setCameraSettings] = useState({ // For modal settings before applying
        useIpCamera: true,
        ipCameraAddress: "http://192.168.1.9:8080/video",
        frameRate: 10,      
        jpegQuality: 0.75,  
        previewWidth: 320,  
        previewHeight: 240, 
        // rotateAngle: 0, // REMOVED from here
    });
    const [appliedCameraSettings, setAppliedCameraSettings] = useState({ // For actual operational settings
        useIpCamera: true,
        ipCameraAddress: "http://192.168.1.9:8080/video",
        frameRate: 10,
        jpegQuality: 0.75,
        previewWidth: 320,
        previewHeight: 240,
        // rotateAngle: 0, // REMOVED from here
    });

    const [cameraAdjustments, setCameraAdjustments] = useState({
        zoom: 1,
        crop: { x: 0, y: 0, width: 1, height: 1 },
        rotateAngle: 0, // ADDED rotation here
    });
    const cameraAdjustmentsRef = useRef(cameraAdjustments);

    const [cropOverlayStyle, setCropOverlayStyle] = useState({});

    const [gameState, setGameState] = useState({
        mode: "idle", calibration_step: 0, scan_index: 0, solve_move_index: 0, total_solve_moves: 0,
        status_message: "Initialize game.", error_message: null, serial_connected: false,
        current_color_calibrating: null, solution_preview: null,
    });
    const [gameStarted, setGameStarted] = useState(false);
    const [isCameraLoading, setIsCameraLoading] = useState(false);

    const frameSenderHandle = useRef(null);
    const lastFrameSentTime = useRef(0);

    useEffect(() => { setIsClient(true); }, []);
    useEffect(() => { cameraAdjustmentsRef.current = cameraAdjustments; }, [cameraAdjustments]);

    useEffect(() => {
        if (!isClient || !gameStarted) {
            if (wsRef.current) { wsRef.current.close(); wsRef.current = null; }
            if (frameSenderHandle.current) { cancelAnimationFrame(frameSenderHandle.current); frameSenderHandle.current = null; }
            if (videoRef.current && videoRef.current.srcObject) { videoRef.current.srcObject.getTracks().forEach((t) => t.stop()); videoRef.current.srcObject = null; }
            if (ipCamImgRef.current) { ipCamImgRef.current.src = ""; }
            if (gameStarted === false && status !== "Idle" && !status.includes("Error") && !status.includes("Disconnected")) { setStatus("Disconnected (Game Stopped)"); }
            return;
        }

        setIsCameraLoading(true);
        let localStream = null;

        const currentPreviewWidth = appliedCameraSettings.previewWidth;
        const currentPreviewHeight = appliedCameraSettings.previewHeight;
        const currentMinFrameInterval = 1000 / appliedCameraSettings.frameRate;
        const currentJpegQuality = appliedCameraSettings.jpegQuality;
        // currentRotateAngle will be read from cameraAdjustmentsRef inside sendFrame

        const setupLocalCamera = async () => {
            if (videoRef.current && videoRef.current.srcObject) { videoRef.current.srcObject.getTracks().forEach((t) => t.stop()); videoRef.current.srcObject = null; }
            if (ipCamImgRef.current && !appliedCameraSettings.useIpCamera) { ipCamImgRef.current.src = ""; }
            if (appliedCameraSettings.useIpCamera) {
                if (!appliedCameraSettings.ipCameraAddress) { setGameState(prev => ({ ...prev, error_message: "IP Camera address is not set." })); setIsCameraLoading(false); }
            } else {
                try {
                    localStream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: currentPreviewWidth }, height: { ideal: currentPreviewHeight } } });
                    if (videoRef.current) {
                        videoRef.current.srcObject = localStream;
                        videoRef.current.onloadedmetadata = () => setIsCameraLoading(false);
                        videoRef.current.onerror = () => { setGameState(prev => ({ ...prev, error_message: "Local camera error." })); setIsCameraLoading(false); }
                    } else { setIsCameraLoading(false); }
                } catch (err) { setGameState(prev => ({ ...prev, error_message: "Local camera access failed." })); setIsCameraLoading(false); }
            }
        };
        setupLocalCamera();

        const ws = new WebSocket(`ws://localhost:8000/ws/${gameId}`);
        wsRef.current = ws;
        setStatus("Connecting to WebSocket...");

        ws.onopen = () => {
            setStatus("WebSocket Connected");
            const initialConfig = { serial_port: serialPort, serial_baudrate: 9600, video_source: appliedCameraSettings.useIpCamera ? appliedCameraSettings.ipCameraAddress : "0" };
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
                if (now - lastFrameSentTime.current < currentMinFrameInterval) {
                    frameSenderHandle.current = requestAnimationFrame(sendFrame);
                    return;
                }
                lastFrameSentTime.current = now;

                const canvas = canvasRef.current;
                let sourceElement = appliedCameraSettings.useIpCamera ? ipCamImgRef.current : videoRef.current;
                const currentAdjusts = cameraAdjustmentsRef.current; // Use ref for adjustments
                const currentRotateAngle = currentAdjusts.rotateAngle;

                if (sourceElement && canvas &&
                    ( (sourceElement.tagName === "VIDEO" && sourceElement.readyState >= 2) ||
                      (sourceElement.tagName === "IMG" && sourceElement.complete && sourceElement.naturalWidth > 0 && sourceElement.naturalHeight > 0) )) {
                    const sWidthOrig = sourceElement.videoWidth || sourceElement.naturalWidth;
                    const sHeightOrig = sourceElement.videoHeight || sourceElement.naturalHeight;

                    if (sWidthOrig && sHeightOrig) {
                        let cropX_abs = currentAdjusts.crop.x * sWidthOrig;
                        let cropY_abs = currentAdjusts.crop.y * sHeightOrig;
                        let cropW_abs = currentAdjusts.crop.width * sWidthOrig;
                        let cropH_abs = currentAdjusts.crop.height * sHeightOrig;

                        let zoomedSWidth = cropW_abs / currentAdjusts.zoom;
                        let zoomedSHeight = cropH_abs / currentAdjusts.zoom;
                        let zoomedSx = cropX_abs + (cropW_abs - zoomedSWidth) / 2;
                        let zoomedSy = cropY_abs + (cropH_abs - zoomedSHeight) / 2;
                        
                        const isSidewaysRotation = currentRotateAngle === 90 || currentRotateAngle === 270;
                        canvas.width = isSidewaysRotation ? currentPreviewHeight : currentPreviewWidth;
                        canvas.height = isSidewaysRotation ? currentPreviewWidth : currentPreviewHeight;

                        const ctx = canvas.getContext("2d", { alpha: false });
                        try {
                            ctx.save();
                            ctx.translate(canvas.width / 2, canvas.height / 2);
                            ctx.rotate((currentRotateAngle * Math.PI) / 180);
                            
                            // Draw into a rectangle matching currentPreviewWidth/Height before rotation
                            ctx.drawImage(
                                sourceElement, 
                                zoomedSx, zoomedSy, zoomedSWidth, zoomedSHeight,
                                -currentPreviewWidth / 2, -currentPreviewHeight / 2, 
                                currentPreviewWidth, currentPreviewHeight
                            );
                            ctx.restore();

                            canvas.toBlob(
                                (blob) => { if (blob && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) { wsRef.current.send(blob); }},
                                "image/jpeg", currentJpegQuality
                            );
                        } catch (e) {
                            console.error("Error drawing source to canvas or sending frame:", e);
                            if(ctx) ctx.restore();
                        }
                    }
                }
                frameSenderHandle.current = requestAnimationFrame(sendFrame);
            };
            frameSenderHandle.current = requestAnimationFrame(sendFrame);
        };

        ws.onmessage = (event) => { 
            try {
                const data = JSON.parse(event.data);
                setGameState(prev => ({
                    ...prev, mode: data.mode !== undefined ? data.mode : prev.mode,
                    calibration_step: data.calibration_step !== undefined ? data.calibration_step : prev.calibration_step,
                    scan_index: data.scan_index !== undefined ? data.scan_index : prev.scan_index,
                    solve_move_index: data.solve_move_index !== undefined ? data.solve_move_index : prev.solve_move_index,
                    total_solve_moves: data.total_solve_moves !== undefined ? data.total_solve_moves : prev.total_solve_moves,
                    status_message: data.status_message !== undefined ? data.status_message : prev.status_message,
                    error_message: data.error_message !== undefined ? data.error_message : null,
                    serial_connected: data.serial_connected !== undefined ? data.serial_connected : false,
                    current_color_calibrating: data.current_color_calibrating !== undefined ? data.current_color_calibrating : prev.current_color_calibrating,
                    solution_preview: data.solution_preview !== undefined ? data.solution_preview : prev.solution_preview,
                }));
                if (data.processed_frame) { setProcessedFrame(`data:image/jpeg;base64,${data.processed_frame}`); }
            } catch (e) { setGameState(prev => ({ ...prev, error_message: "Received invalid data from backend."})); }
        };
        ws.onclose = (event) => { 
            setStatus(`WebSocket Closed (Code: ${event.code})`);
            if (frameSenderHandle.current) { cancelAnimationFrame(frameSenderHandle.current); frameSenderHandle.current = null; }
        };
        ws.onerror = (err_event) => { 
            setStatus("WebSocket Error");
            if (frameSenderHandle.current) { cancelAnimationFrame(frameSenderHandle.current); frameSenderHandle.current = null; }
        };

        return () => { 
            if (frameSenderHandle.current) { cancelAnimationFrame(frameSenderHandle.current); frameSenderHandle.current = null; }
            if (wsRef.current) { wsRef.current.onopen = null; wsRef.current.onmessage = null; wsRef.current.onerror = null; wsRef.current.onclose = null; wsRef.current.close(); wsRef.current = null; }
            if (localStream) { localStream.getTracks().forEach((t) => t.stop()); }
            if (videoRef.current && videoRef.current.srcObject) { videoRef.current.srcObject.getTracks().forEach(t => t.stop()); videoRef.current.srcObject = null; }
            if (ipCamImgRef.current) { ipCamImgRef.current.src = ""; }
        };
    }, [ // REMOVED appliedCameraSettings.rotateAngle from dependencies
        isClient, gameStarted, 
        appliedCameraSettings.useIpCamera, appliedCameraSettings.ipCameraAddress, 
        appliedCameraSettings.frameRate, appliedCameraSettings.jpegQuality,
        appliedCameraSettings.previewWidth, appliedCameraSettings.previewHeight,
        serialPort
    ]);

    useEffect(() => { // Crop overlay effect
        if (!gameStarted && !isCameraLoading) { setCropOverlayStyle({ display: 'none' }); return; }
        const sourceElement = appliedCameraSettings.useIpCamera ? ipCamImgRef.current : videoRef.current;
        if (!sourceElement || isCameraLoading || !((sourceElement.tagName === "VIDEO" && sourceElement.videoWidth > 0) || (sourceElement.tagName === "IMG" && sourceElement.naturalWidth > 0))) { setCropOverlayStyle({ display: 'none' }); return; }
        const videoWidth = sourceElement.videoWidth || sourceElement.naturalWidth;
        const videoHeight = sourceElement.videoHeight || sourceElement.naturalHeight;
        if (!videoWidth || !videoHeight) { setCropOverlayStyle({ display: 'none' }); return; }
    
        let cropX_abs = cameraAdjustments.crop.x * videoWidth;
        let cropY_abs = cameraAdjustments.crop.y * videoHeight;
        let cropW_abs = cameraAdjustments.crop.width * videoWidth;
        let cropH_abs = cameraAdjustments.crop.height * videoHeight;
        let finalSx_orig = cropX_abs + (cropW_abs - cropW_abs / cameraAdjustments.zoom) / 2;
        let finalSy_orig = cropY_abs + (cropH_abs - cropH_abs / cameraAdjustments.zoom) / 2;
        let finalSWidth_orig = cropW_abs / cameraAdjustments.zoom;
        let finalSHeight_orig = cropH_abs / cameraAdjustments.zoom;
        
        const sourceAspectRatio = videoWidth / videoHeight;
        const displayPreviewWidth = appliedCameraSettings.previewWidth;
        const displayPreviewHeight = appliedCameraSettings.previewHeight;
        const previewBoxAspectRatio = displayPreviewWidth / displayPreviewHeight;

        let renderedW, renderedH, offsetX_in_box, offsetY_in_box;
        if (sourceAspectRatio > previewBoxAspectRatio) { 
            renderedW = displayPreviewWidth; renderedH = displayPreviewWidth / sourceAspectRatio;
            offsetX_in_box = 0; offsetY_in_box = (displayPreviewHeight - renderedH) / 2;
        } else { 
            renderedH = displayPreviewHeight; renderedW = displayPreviewHeight * sourceAspectRatio;
            offsetY_in_box = 0; offsetX_in_box = (displayPreviewWidth - renderedW) / 2;
        }
        const scaleToRendered = renderedW / videoWidth; 
        const overlayX_on_rendered = finalSx_orig * scaleToRendered;
        const overlayY_on_rendered = finalSy_orig * scaleToRendered;
        const overlayW_on_rendered = finalSWidth_orig * scaleToRendered;
        const overlayH_on_rendered = finalSHeight_orig * scaleToRendered;
    
        setCropOverlayStyle({
            position: 'absolute', left: `${offsetX_in_box + overlayX_on_rendered}px`,
            top: `${offsetY_in_box + overlayY_on_rendered}px`, width: `${overlayW_on_rendered}px`,
            height: `${overlayH_on_rendered}px`, border: '2px dashed rgba(255, 0, 0, 0.7)',
            boxSizing: 'border-box', display: 'block', pointerEvents: 'none',
        });
    }, [
        cameraAdjustments, // This will re-run if cameraAdjustments.rotateAngle changes too
        appliedCameraSettings.useIpCamera, appliedCameraSettings.ipCameraAddress, 
        appliedCameraSettings.previewWidth, appliedCameraSettings.previewHeight,
        isCameraLoading, gameStarted
    ]);

    const handleSendCommand = (commandPayload) => { 
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify(commandPayload));
        } else {
            setGameState(prev => ({ ...prev, error_message: "WebSocket not connected. Cannot send command."}));
        }
    };
    
    const handleCropChange = useCallback((key, rawValue) => { 
        setCameraAdjustments(prev => {
            const newAdj = { ...prev }; let newCrop = { ...newAdj.crop };
            let value = parseFloat(rawValue);
            if (isNaN(value)) value = (key === 'width' || key === 'height') ? 0.1 : 0;
            if (key === 'x') { value = Math.max(0, Math.min(value, 1 - newCrop.width)); newCrop.x = value; }
            else if (key === 'y') { value = Math.max(0, Math.min(value, 1 - newCrop.height)); newCrop.y = value; }
            else if (key === 'width') { value = Math.max(0.1, Math.min(value, 1 - newCrop.x)); newCrop.width = value; }
            else if (key === 'height') { value = Math.max(0.1, Math.min(value, 1 - newCrop.y)); newCrop.height = value; }
            newAdj.crop = newCrop; return newAdj;
        });
    }, []);

    if (!isClient) return <div className="flex justify-center items-center min-h-screen"><p className="text-xl">Loading Client...</p></div>;
    
    // Determine processed view dimensions based on current rotation in cameraAdjustments
    const isProcessedViewSideways = cameraAdjustments.rotateAngle === 90 || cameraAdjustments.rotateAngle === 270;
    const processedViewStyle = {
        width: `${isProcessedViewSideways ? appliedCameraSettings.previewHeight : appliedCameraSettings.previewWidth}px`,
        height: `${isProcessedViewSideways ? appliedCameraSettings.previewWidth : appliedCameraSettings.previewHeight}px`,
    };

    return (
        <div className="flex flex-col items-center gap-4 p-4 min-h-screen bg-gradient-to-br from-slate-200 to-slate-400 font-sans">
            {showSettings && (
                <SettingsModalContent
                    setShowSettings={setShowSettings}
                    serialPort={serialPort} setSerialPort={setSerialPort}
                    cameraSettings={cameraSettings} setCameraSettings={setCameraSettings}
                    setAppliedCameraSettings={setAppliedCameraSettings}
                />
            )}
            <div className="w-full max-w-5xl bg-white rounded-xl shadow-2xl p-6 md:p-8 mt-5">
                {/* Header */}
                <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-6 pb-4 border-b border-gray-200">
                    <h1 className="text-3xl font-bold text-slate-800 mb-3 md:mb-0">Rubik's Cube Solver</h1>
                    <div className="flex flex-wrap items-center gap-3">
                        <span title={status} className={`px-3 py-1.5 rounded-full text-xs font-semibold truncate max-w-[150px] md:max-w-xs ${status.includes("Connected") ? "bg-green-100 text-green-800" : status.includes("Error") || status.includes("Disconnected") || status.includes("Closed") ? "bg-red-100 text-red-800" : "bg-yellow-100 text-yellow-800"}`}>
                            {status}
                        </span>
                         <span className={`px-3 py-1.5 rounded-full text-xs font-semibold ${gameState.serial_connected ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"}`}>
                            ESP32: {gameState.serial_connected ? "Ok" : "Off"}
                        </span>
                        <button onClick={() => setShowSettings(true)} className="px-4 py-2 bg-indigo-500 text-white text-sm font-medium rounded-md shadow-sm hover:bg-indigo-600 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2">Settings</button>
                    </div>
                </div>

                {/* Controls */}
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
                
                {/* Camera Previews */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                    {/* Local Preview */}
                    <div className="flex flex-col items-center p-3 bg-slate-100 rounded-lg">
                        <div className="mb-2 text-center font-semibold text-slate-700">Camera Feed (Local Preview)</div>
                        <div 
                            style={{ 
                                width: `${appliedCameraSettings.previewWidth}px`, 
                                height: `${appliedCameraSettings.previewHeight}px` 
                            }}
                            className="relative rounded-md overflow-hidden border-2 border-slate-300 bg-slate-900 flex items-center justify-center shadow-inner"
                        >
                            {appliedCameraSettings.useIpCamera ? (
                                <img 
                                    ref={ipCamImgRef} 
                                    key={appliedCameraSettings.ipCameraAddress} 
                                    src={(gameStarted && appliedCameraSettings.ipCameraAddress) ? appliedCameraSettings.ipCameraAddress : undefined}
                                    alt="IP Camera Preview" 
                                    className="object-contain max-w-full max-h-full"
                                    crossOrigin="anonymous" 
                                    onLoad={() => { setIsCameraLoading(false);}} 
                                    onError={() => { 
                                        if (gameStarted && appliedCameraSettings.ipCameraAddress) { 
                                            setGameState(prev => ({...prev, error_message: "IP Cam Preview Error."})); 
                                        }
                                        setIsCameraLoading(false);
                                    }}
                                />
                            ) : (
                                <video 
                                    ref={videoRef} autoPlay playsInline muted 
                                    width={appliedCameraSettings.previewWidth}
                                    height={appliedCameraSettings.previewHeight}
                                    className="object-contain max-w-full max-h-full"
                                />
                            )}
                            {isCameraLoading && gameStarted && <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-60 text-white text-lg font-semibold">Loading Preview...</div>}
                            <div style={cropOverlayStyle}></div>
                        </div>
                        <canvas ref={canvasRef} className="hidden" />

                        {/* Zoom, Crop & Rotate Controls */}
                        <div className="mt-4 w-full max-w-xs space-y-3 p-2 bg-white/50 rounded-md">
                            <div>
                                <label htmlFor="zoom" className="block text-xs font-medium text-gray-700">Zoom: {Number(cameraAdjustments.zoom).toFixed(1)}x</label>
                                <input type="range" id="zoom" min="1" max="5" step="0.1" value={cameraAdjustments.zoom} onChange={(e) => setCameraAdjustments(prev => ({ ...prev, zoom: parseFloat(e.target.value) }))} className="w-full h-2 bg-gray-300 rounded-lg appearance-none cursor-pointer accent-indigo-600" disabled={!gameStarted} />
                            </div>
                            <div className="grid grid-cols-2 gap-x-3 gap-y-1">
                                <div>
                                    <label htmlFor="cropX" className="block text-xs font-medium text-gray-700">Crop X ({Number(cameraAdjustments.crop.x * 100).toFixed(0)}%)</label>
                                    <input type="range" id="cropX" min="0" max="1" step="0.01" value={cameraAdjustments.crop.x} onChange={e => handleCropChange('x', e.target.value)} className="w-full h-2 bg-gray-300 rounded-lg appearance-none cursor-pointer accent-indigo-600" disabled={!gameStarted}/>
                                </div>
                                <div>
                                    <label htmlFor="cropY" className="block text-xs font-medium text-gray-700">Crop Y ({Number(cameraAdjustments.crop.y * 100).toFixed(0)}%)</label>
                                    <input type="range" id="cropY" min="0" max="1" step="0.01" value={cameraAdjustments.crop.y} onChange={e => handleCropChange('y', e.target.value)} className="w-full h-2 bg-gray-300 rounded-lg appearance-none cursor-pointer accent-indigo-600" disabled={!gameStarted}/>
                                </div>
                                <div>
                                    <label htmlFor="cropW" className="block text-xs font-medium text-gray-700">Crop W ({Number(cameraAdjustments.crop.width * 100).toFixed(0)}%)</label>
                                    <input type="range" id="cropW" min="0.1" max="1" step="0.01" value={cameraAdjustments.crop.width} onChange={e => handleCropChange('width', e.target.value)} className="w-full h-2 bg-gray-300 rounded-lg appearance-none cursor-pointer accent-indigo-600" disabled={!gameStarted}/>
                                </div>
                                <div>
                                    <label htmlFor="cropH" className="block text-xs font-medium text-gray-700">Crop H ({Number(cameraAdjustments.crop.height * 100).toFixed(0)}%)</label>
                                    <input type="range" id="cropH" min="0.1" max="1" step="0.01" value={cameraAdjustments.crop.height} onChange={e => handleCropChange('height', e.target.value)} className="w-full h-2 bg-gray-300 rounded-lg appearance-none cursor-pointer accent-indigo-600" disabled={!gameStarted}/>
                                </div>
                            </div>
                             <div>
                                <label htmlFor="rotateAngle" className="block text-xs font-medium text-gray-700">Rotation: {cameraAdjustments.rotateAngle}°</label>
                                <input 
                                    type="range" 
                                    id="rotateAngle" 
                                    min="0" 
                                    max="360" 
                                    step="1" 
                                    value={cameraAdjustments.rotateAngle} 
                                    onChange={(e) => setCameraAdjustments(prev => ({...prev, rotateAngle: parseInt(e.target.value, 10)}))}
                                    className="w-full h-2 bg-gray-300 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                                    disabled={!gameStarted}
                                />
                            </div>
                            <button 
                                onClick={() => setCameraAdjustments({ zoom: 1, crop: { x: 0, y: 0, width: 1, height: 1 }, rotateAngle: 0 })} 
                                className="w-full mt-2 px-3 py-1.5 bg-slate-300 text-slate-700 text-xs rounded hover:bg-slate-400 disabled:opacity-50" 
                                disabled={!gameStarted}
                            >
                                Reset Adjustments
                            </button>
                        </div>
                    </div>

                    {/* Processed View */}
                    <div className="flex flex-col items-center p-3 bg-slate-100 rounded-lg">
                        <div className="mb-2 text-center font-semibold text-slate-700">Processed View (from Backend)</div>
                        <div 
                            style={processedViewStyle} // Use dynamically calculated style
                            className="relative rounded-md overflow-hidden border-2 border-slate-300 bg-slate-900 flex items-center justify-center shadow-inner"
                        >
                            {processedFrame ? (
                                <img 
                                    src={processedFrame} 
                                    alt="Processed Frame" 
                                    className="object-contain max-w-full max-h-full"
                                />
                            ) : (
                                <span className="text-slate-500 text-center p-2">{gameStarted ? "Waiting for backend frame..." : "Game stopped."}</span>
                            )}
                        </div>
                    </div>
                </div>

                {/* Game Status */}
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