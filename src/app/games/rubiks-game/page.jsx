"use client";
import React, { useCallback, useEffect, useRef, useState } from "react";

// Helper constant
const COLOR_NAMES = ["W", "R", "G", "Y", "O", "B"];

// DEBUG Counter for frames sent
let frameSentCounter = 0;

export default function RubiksSolverPage() {
  console.log("[Render] RubiksSolverPage component rendering"); // DEBUG
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
    calibration_step: null,
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
  const loadingTimeoutRef = useRef(null);

  const [isCameraLoading, setIsCameraLoading] = useState(true);
  const [isBackendProcessing, setIsBackendProcessing] = useState(false);
  const [isActionLoading, setIsActionLoading] = useState(false);

  const sendNextFrameRef = useRef(true);
  const lastSentRef = useRef(0);
  const minFrameInterval = 100; // ms, approx 10 FPS target

  // --- Camera Initialization ---
  const initCamera = useCallback(async () => {
    console.log("%c[initCamera] Start", "color: blue; font-weight: bold;", appliedCameraSettings);
    setIsCameraLoading(true);
    setStatus((prev) => ({ ...prev, status_message: "Initializing camera...", error_message: null }));

    if (videoRef.current && videoRef.current.srcObject) {
      console.log("[initCamera] Stopping previous device stream.");
      videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    if (ipCamImgRef.current) {
      console.log("[initCamera] Clearing previous IP Cam src.");
      ipCamImgRef.current.src = "";
    }

    if (!appliedCameraSettings.useIpCamera) {
      console.log("[initCamera] Attempting device camera...");
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        console.error("[initCamera] ERROR: getUserMedia not supported!");
        setStatus((prev) => ({
          ...prev,
          mode: "error",
          status_message: "Camera Error",
          error_message: "Browser does not support camera access (getUserMedia).",
        }));
        setIsCameraLoading(false);
        return;
      }
      try {
        const constraints = {
          video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "environment" },
          audio: false,
        };
        console.log("[initCamera] Requesting media device with constraints:", constraints);
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        console.log("[initCamera] getUserMedia SUCCESS, stream ID:", stream.id);
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          console.log("[initCamera] Stream assigned to video element.");
        } else {
          console.error("[initCamera] ERROR: videoRef is null when assigning stream.");
          stream.getTracks().forEach((track) => track.stop());
          setIsCameraLoading(false);
        }
      } catch (error) {
        console.error("[initCamera] getUserMedia ERROR:", error.name, error.message, error);
        let errorMsg = `Failed to access camera: ${error.name}.`;
        if (error.name === "NotAllowedError") errorMsg = "Camera permission denied. Please allow access.";
        else if (error.name === "NotFoundError") errorMsg = "No suitable camera found.";
        else if (error.name === "NotReadableError") errorMsg = "Camera is already in use or hardware error.";
        else if (error.name === "OverconstrainedError") errorMsg = `Camera does not support requested resolution/facingMode. ${error.message}`;
        else if (error.name === "AbortError") errorMsg = "Camera request aborted.";
        setStatus((prev) => ({ ...prev, mode: "error", status_message: "Camera Error", error_message: errorMsg }));
        setIsCameraLoading(false);
      }
    } else {
      console.log("[initCamera] Using IP Camera:", appliedCameraSettings.ipCameraUrl);
      if (!appliedCameraSettings.ipCameraUrl) {
        console.error("[initCamera] ERROR: IP Camera URL is empty.");
        setStatus((prev) => ({
          ...prev,
          mode: "error",
          status_message: "Camera Error",
          error_message: "IP Camera URL cannot be empty.",
        }));
        setIsCameraLoading(false);
        return;
      }
      setStatus((prev) => ({ ...prev, status_message: "Loading IP Camera feed..." }));
      if (ipCamImgRef.current) {
        console.log("[initCamera] Setting IP Cam src:", appliedCameraSettings.ipCameraUrl);
        ipCamImgRef.current.src = appliedCameraSettings.ipCameraUrl;
      } else {
        console.error("[initCamera] ERROR: ipCamImgRef is null when setting IP Cam src.");
        setIsCameraLoading(false);
      }
    }
    console.log("%c[initCamera] End", "color: blue; font-weight: bold;");
  }, [appliedCameraSettings]);

  // --- WebSocket Connection & Frame Sending ---
  useEffect(() => {
    console.log("%c[useEffect] Component Mounted / appliedCameraSettings changed", "color: green; font-weight: bold;");
    let stopped = false;
    let reconnectTimeout = null;
    let frameRequestId = null;

    const connectWebSocket = () => {
      console.log("[connectWebSocket] Attempting connection...");
      if (stopped) {
        console.log("[connectWebSocket] Aborted: component stopped.");
        return;
      }
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        console.log("[connectWebSocket] Aborted: Already connected.");
        return;
      }

      setWsConnectionStatus("Connecting...");
      const wsProtocol = window.location.protocol === "https:" ? "wss://" : "ws://";
      const wsUrl = `${wsProtocol}${window.location.hostname}:8000/ws/rubiks`;
      console.log("[connectWebSocket] Creating WebSocket to:", wsUrl);
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log("%c[WS] Connected", "color: green;");
        setWsConnectionStatus("Connected");
        if (reconnectTimeout) clearTimeout(reconnectTimeout);
        sendNextFrameRef.current = true;
      };

      ws.onclose = (event) => {
        console.warn(`%c[WS] Disconnected: Code=${event.code}, Reason='${event.reason}'`, "color: orange;");
        setWsConnectionStatus("Disconnected");
        wsRef.current = null;
        setStatus((prev) => ({
          ...prev,
          mode: "connecting",
          status_message: "Connection lost. Reconnecting...",
          serial_connected: false,
        }));
        if (!stopped) {
          if (reconnectTimeout) clearTimeout(reconnectTimeout);
          console.log("[WS] Scheduling reconnect in 5s...");
          reconnectTimeout = setTimeout(connectWebSocket, 5000);
        }
      };

      ws.onerror = (error) => {
        console.error("[WS] Error:", error);
        setWsConnectionStatus("Error");
      };

      ws.onmessage = (event) => {
        if (loadingTimeoutRef.current) {
          clearTimeout(loadingTimeoutRef.current);
          loadingTimeoutRef.current = null;
        }
        setIsBackendProcessing(false);
        setIsActionLoading(false);
        sendNextFrameRef.current = true;
        console.log("[WS] Received message, set sendNextFrameRef=true, isBackendProcessing=false, isActionLoading=false");

        try {
          const data = JSON.parse(event.data);
          console.log("[WS] Parsed data:", data);

          setStatus((prev) => {
            const nextState = {
              ...prev,
              mode: data.mode ?? prev.mode,
              status_message: data.status_message ?? prev.status_message,
              error_message: data.error_message ?? null,
              solution: data.solution ?? prev.solution,
              serial_connected: data.serial_connected ?? prev.serial_connected,
              calibration_step: data.calibration_step,
              current_color: data.current_color,
              scan_index: data.scan_index,
              solve_move_index: data.solve_move_index ?? prev.solve_move_index,
              total_solve_moves: data.total_solve_moves ?? prev.total_solve_moves,
            };
            console.log("[WS] Updated status:", nextState);
            return nextState;
          });

          if (data.processed_frame) {
            console.log("[WS] Received processed frame, updating processedFrame.");
            const frameSrc = `data:image/jpeg;base64,${data.processed_frame}`;
            setProcessedFrame(frameSrc);
          } else {
            console.log("[WS] No processed frame in message.");
          }
        } catch (e) {
          console.error("[WS] Failed to parse message:", e, "Raw data:", event.data);
          setStatus((prev) => ({ ...prev, mode: "error", error_message: "Error processing backend update." }));
        }
      };
    };

    const sendFrame = () => {
      if (stopped) {
        console.log("[sendFrame] Aborted: component stopped.");
        cancelAnimationFrame(frameRequestId);
        return;
      }

      const ws = wsRef.current;
      const now = Date.now();

      let skipReason = null;
      if (!sendNextFrameRef.current) skipReason = "sendNextFrameRef is false (waiting for backend response)";
      else if (!ws || ws.readyState !== WebSocket.OPEN) skipReason = `WebSocket not ready (State: ${ws?.readyState ?? "null"})`;
      else if (now - lastSentRef.current < minFrameInterval) skipReason = "Frame interval too short";
      else if (status.mode === "error") skipReason = "Status is 'error'";

      if (skipReason) {
        console.log(`[sendFrame] Skipping frame: ${skipReason}`);
        frameRequestId = requestAnimationFrame(sendFrame);
        return;
      }

      const canvas = canvasRef.current;
      let sourceEl = null;
      let sourceReady = false;

      if (appliedCameraSettings.useIpCamera) {
        const img = ipCamImgRef.current;
        if (img && img.complete && img.naturalWidth > 0) {
          sourceEl = img;
          sourceReady = true;
          console.log("[sendFrame] Using IP Cam image as source (Ready).");
        }
      } else {
        const video = videoRef.current;
        if (video && video.readyState >= video.HAVE_ENOUGH_DATA) {
          sourceEl = video;
          sourceReady = true;
          console.log("[sendFrame] Using video element as source (Ready).");
        }
      }

      if (!sourceReady) {
        frameRequestId = requestAnimationFrame(sendFrame);
        return;
      }
      if (!sourceEl) {
        console.warn("[sendFrame] No valid source element found.");
        frameRequestId = requestAnimationFrame(sendFrame);
        return;
      }
      if (!canvas) {
        console.error("[sendFrame] Canvas ref is null.");
        frameRequestId = requestAnimationFrame(sendFrame);
        return;
      }

      try {
        const ctx = canvas.getContext("2d");
        if (!ctx) {
          console.error("[sendFrame] Failed to get canvas context.");
          frameRequestId = requestAnimationFrame(sendFrame);
          return;
        }

        const targetWidth = 320;
        const targetHeight = 240;
        if (canvas.width !== targetWidth) canvas.width = targetWidth;
        if (canvas.height !== targetHeight) canvas.height = targetHeight;

        ctx.drawImage(sourceEl, 0, 0, targetWidth, targetHeight);

        canvas.toBlob(
          (blob) => {
            if (blob && ws.readyState === WebSocket.OPEN) {
              setIsBackendProcessing(true);
              sendNextFrameRef.current = false;
              lastSentRef.current = now;

              blob.arrayBuffer()
                .then((buffer) => {
                  if (ws.readyState === WebSocket.OPEN) {
                    frameSentCounter++;
                    console.log(`%c[sendFrame #${frameSentCounter}] Sending buffer (Size: ${buffer.byteLength})`, "color: purple;");
                    ws.send(buffer);
                  } else {
                    console.warn("[sendFrame] WebSocket closed before buffer could be sent.");
                    sendNextFrameRef.current = true;
                    setIsBackendProcessing(false);
                  }
                })
                .catch((err) => {
                  console.error("[sendFrame] Error getting ArrayBuffer:", err);
                  sendNextFrameRef.current = true;
                  setIsBackendProcessing(false);
                });
            } else {
              console.warn("[sendFrame] Blob creation failed or WebSocket closed.");
              sendNextFrameRef.current = true;
              setIsBackendProcessing(false);
            }
          },
          "image/jpeg",
          0.7
        );
      } catch (e) {
        console.warn("[sendFrame] Error processing/drawing frame:", e);
        sendNextFrameRef.current = true;
        setIsBackendProcessing(false);
      }

      frameRequestId = requestAnimationFrame(sendFrame);
    };

    initCamera();
    connectWebSocket();
    frameRequestId = requestAnimationFrame(sendFrame);

    return () => {
      stopped = true;
      console.log("%c[useEffect Cleanup] Running cleanup...", "color: red; font-weight: bold;");
      if (frameRequestId) {
        console.log("[useEffect Cleanup] Cancelling animation frame request:", frameRequestId);
        cancelAnimationFrame(frameRequestId);
      }
      if (reconnectTimeout) {
        console.log("[useEffect Cleanup] Clearing reconnect timeout.");
        clearTimeout(reconnectTimeout);
      }
      if (loadingTimeoutRef.current) {
        console.log("[useEffect Cleanup] Clearing API loading timeout.");
        clearTimeout(loadingTimeoutRef.current);
      }
      if (wsRef.current) {
        console.log("[useEffect Cleanup] Closing WebSocket connection.");
        wsRef.current.close();
        wsRef.current = null;
      }
      if (videoRef.current && videoRef.current.srcObject) {
        console.log("[useEffect Cleanup] Stopping device camera stream.");
        videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
        videoRef.current.srcObject = null;
      }
      if (ipCamImgRef.current) {
        console.log("[useEffect Cleanup] Clearing IP Cam src.");
        ipCamImgRef.current.src = "";
      }
      console.log("%c[useEffect Cleanup] Cleanup complete.", "color: red; font-weight: bold;");
    };
  }, [appliedCameraSettings, initCamera]);

  // --- API Call Helper ---
  const callApi = useCallback(
    async (endpoint, method = "POST", body = null) => {
      console.log(`[callApi] Attempting call to ${endpoint}`);
      if (isActionLoading) {
        console.warn(`[callApi] Action already in progress, skipping call to ${endpoint}`);
        return;
      }

      if (loadingTimeoutRef.current) {
        console.warn("[callApi] Clearing previous safety timeout.");
        clearTimeout(loadingTimeoutRef.current);
      }

      console.log(`[callApi] Setting isActionLoading = true for ${endpoint}`);
      setIsActionLoading(true);

      loadingTimeoutRef.current = setTimeout(() => {
        console.warn(`[callApi] SAFETY TIMEOUT triggered for ${endpoint} - resetting loading state`);
        setIsActionLoading(false);
        setIsBackendProcessing(false);
        sendNextFrameRef.current = true; // Allow frame sending
        loadingTimeoutRef.current = null;
      }, 5000);

      try {
        const options = { method };
        if (body) {
          options.headers = { "Content-Type": "application/json" };
          options.body = JSON.stringify(body);
        }
        const apiUrl = `http://${window.location.hostname}:8000${endpoint}`;
        console.log(`[callApi] Calling API: ${method} ${apiUrl}`, body || "");
        const response = await fetch(apiUrl, options);
        console.log(`[callApi] Response status for ${endpoint}: ${response.status}`);
        const data = await response.json();

        if (!response.ok) {
          console.error(`[callApi] API Error ${response.status} for ${endpoint}:`, data);
          throw new Error(data.detail || `API Error ${response.status}`);
        }
        console.log(`[callApi] API Success ${endpoint}:`, data);

        if (loadingTimeoutRef.current) {
          clearTimeout(loadingTimeoutRef.current);
          loadingTimeoutRef.current = null;
          console.log("[callApi] Cleared safety timeout on success.");
        }

        // Update status with API response
        if (data && typeof data === "object") {
          setStatus((prev) => ({
            ...prev,
            ...data,
            status_message: data.status_message || prev.status_message,
          }));
          console.log("[callApi] Updated status with API response:", data);
        }

        // Force a frame send to get updated processed view
        console.log("[callApi] Forcing frame send to update processed view.");
        sendNextFrameRef.current = true;
        setIsBackendProcessing(false);

        return data;
      } catch (error) {
        console.error(`[callApi] API call to ${endpoint} FAILED:`, error);

        if (loadingTimeoutRef.current) {
          clearTimeout(loadingTimeoutRef.current);
          loadingTimeoutRef.current = null;
          console.log("[callApi] Cleared safety timeout on error.");
        }

        setIsActionLoading(false);
        setIsBackendProcessing(false);
        sendNextFrameRef.current = true;
        console.log("[callApi] Reset isActionLoading=false, isBackendProcessing=false, sendNextFrameRef=true due to error.");

        setStatus((prev) => ({
          ...prev,
          mode: "error",
          error_message: error.message || "API request failed.",
        }));
        return null;
      }
    },
    [isActionLoading]
  );

  // --- Button Click Handlers ---
  const handleStartCalibration = () => {
    console.log("[Click] Start Calibration");
    callApi("/start_calibration");
  };
  const handleCaptureColor = () => {
    console.log("[Click] Capture Color");
    callApi("/capture_calibration_color");
  };
  const handleSaveCalibration = () => {
    console.log("[Click] Save Calibration");
    callApi("/save_calibration");
  };
  const handleResetCalibration = () => {
    console.log("[Click] Reset Calibration");
    callApi("/reset_calibration");
  };
  const handleStartSolve = () => {
    console.log("[Click] Start Solve");
    callApi("/start_solve");
  };
  const handleStopAndReset = () => {
    console.log("[Click] Stop & Reset");
    callApi("/stop_and_reset");
  };
  const handleStartScramble = () => {
    console.log("[Click] Start Scramble");
    callApi("/start_scramble");
  };

  // --- Apply Settings from Modal ---
  const applySettings = () => {
    console.log("[Settings] Applying new camera settings:", modalCameraSettings);
    setAppliedCameraSettings(modalCameraSettings);
    setShowSettings(false);
  };

  // --- Determine Button Disabled States ---
  const isEffectivelyBusy = status.mode !== "idle" && status.mode !== "error" && status.mode !== "connecting";
  const actionDisabled = isEffectivelyBusy || isActionLoading || isCameraLoading;

  // --- Render ---
  return (
    <div className="flex flex-col items-center gap-4 p-4 min-h-screen bg-gray-100 font-sans">
      {showSettings && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-40 backdrop-blur-sm">
          <div className="bg-white p-6 rounded-lg shadow-xl w-full max-w-md relative">
            <button
              className="absolute top-2 right-2 text-gray-500 hover:text-gray-800 text-2xl leading-none font-bold"
              onClick={() => setShowSettings(false)}
              aria-label="Close Settings"
            >
              ×
            </button>
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
                <small className="text-xs text-gray-500 mt-1 block">E.g., http://192.168.1.100:8080/video (MJPEG)</small>
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

      <div className="w-full max-w-5xl bg-white rounded-xl shadow-lg p-6 mt-4">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-4 border-b border-gray-200 pb-4">
          <h1 className="text-2xl font-bold text-gray-800 mb-2 sm:mb-0">Rubik's Cube Solver</h1>
          <div className="flex items-center gap-3">
            <span
              className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                wsConnectionStatus === "Connected"
                  ? "bg-green-100 text-green-800"
                  : wsConnectionStatus === "Disconnected"
                  ? "bg-red-100 text-red-800"
                  : wsConnectionStatus === "Connecting..."
                  ? "bg-yellow-100 text-yellow-800 animate-pulse"
                  : "bg-gray-100 text-gray-800"
              }`}
            >
              WS: {wsConnectionStatus}
            </span>
            <span
              className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                status.serial_connected ? "bg-blue-100 text-blue-800" : "bg-red-100 text-red-800"
              }`}
            >
              {status.serial_connected ? "Serial OK" : "Serial disconnected"}
            </span>
            <button
              onClick={() => {
                console.log("[Click] Open Settings");
                setModalCameraSettings(appliedCameraSettings);
                setShowSettings(true);
              }}
              className="px-3 py-1 bg-blue-500 text-white rounded-md hover:bg-blue-600 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              aria-label="Open Camera Settings"
            >
              Settings
            </button>
          </div>
        </div>

        <div className="mb-5 p-3 bg-gray-50 rounded-lg border border-gray-200 shadow-sm">
          <p className="text-sm font-medium text-gray-700">
            Status: <span className={`font-semibold capitalize ${status.mode === "error" ? "text-red-600" : "text-gray-900"}`}>{status.mode}</span>
            {(status.mode === "solving" || status.mode === "scrambling") && status.total_solve_moves > 0 && (
              <span className="ml-2 text-xs font-normal text-gray-500">
                ({status.solve_move_index || 0} / {status.total_solve_moves})
              </span>
            )}
            {status.mode === "scanning" && status.scan_index != null && (
              <span className="ml-2 text-xs font-normal text-gray-500">
                (Scan {status.scan_index + 1} / 12)
              </span>
            )}
          </p>
          <p className="text-sm mt-1 text-gray-600 min-h-[1.25rem]">{status.status_message || " "}</p>
          {status.mode === "error" && status.error_message && (
            <p className="text-sm mt-1 text-red-700 font-medium bg-red-50 p-2 rounded border border-red-200">
              Error: {status.error_message}
            </p>
          )}
          {status.solution && (
            <div className="mt-2 text-sm text-blue-800 font-mono bg-blue-50 p-2 rounded border border-blue-200 max-h-24 overflow-y-auto">
              <span className="font-semibold">Solution Found:</span>{" "}
              <p className="whitespace-pre-wrap break-words text-xs leading-relaxed">{status.solution}</p>
            </div>
          )}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div className="flex flex-col items-center">
            <div className="mb-1 text-center text-sm font-medium text-gray-600">Live Camera Input</div>
            <div className="relative w-[320px] h-[240px] rounded-lg overflow-hidden border-2 border-gray-300 bg-black flex items-center justify-center shadow-inner">
              {appliedCameraSettings.useIpCamera && (
                <img
                  ref={ipCamImgRef}
                  id="ip-camera-img-display"
                  alt="IP Camera Feed"
                  width={320}
                  height={240}
                  className="object-contain"
                  crossOrigin="anonymous"
                  onLoad={() => {
                    console.log("%c[IP Cam] onLoad event triggered.", "color: green;");
                    setIsCameraLoading(false);
                  }}
                  onError={(e) => {
                    console.error("[IP Cam] onError event triggered:", e);
                    setIsCameraLoading(false);
                    if (status.mode !== "error")
                      setStatus((prev) => ({
                        ...prev,
                        status_message: "IP Cam Load Error",
                        error_message: "Failed to load IP Cam image. Check URL/Network/CORS.",
                      }));
                  }}
                />
              )}
              {!appliedCameraSettings.useIpCamera && (
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  width={320}
                  height={240}
                  className="object-contain"
                  style={{ background: "#222" }}
                  onCanPlay={() => {
                    console.log("%c[Device Cam] onCanPlay event triggered.", "color: green;");
                    setIsCameraLoading(false);
                  }}
                  onError={(e) => {
                    console.error("[Device Cam] onError event triggered:", e);
                    setIsCameraLoading(false);
                  }}
                  onLoadedData={() => console.log("[Device Cam] onLoadedData")}
                  onPlaying={() => console.log("[Device Cam] onPlaying")}
                />
              )}
              {isCameraLoading && (
                <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-70 text-white text-lg">
                  <div className="flex flex-col items-center">
                    <svg className="animate-spin h-8 w-8 text-white mb-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      ></path>
                    </svg>
                    <span>Loading Camera...</span>
                  </div>
                </div>
              )}
              {!isCameraLoading && status.mode === "error" && status.error_message?.toLowerCase().includes("camera") && (
                <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-75 text-white text-sm p-2">
                  <div className="p-3 text-center rounded bg-red-800 bg-opacity-90 shadow">
                    <span className="font-bold block text-base">Camera Error</span>{" "}
                    <span className="text-xs mt-1 block">{status.error_message}</span>
                  </div>
                </div>
              )}
              <canvas ref={canvasRef} width={320} height={240} className="hidden" />
            </div>
          </div>

          <div className="flex flex-col items-center">
            <div className="mb-1 text-center text-sm font-medium text-gray-600">Processed View (Backend)</div>
            <div className="relative w-[320px] h-[240px] rounded-lg overflow-hidden border-2 border-gray-300 bg-black flex items-center justify-center shadow-inner">
              {processedFrame ? (
                <img src={processedFrame} width={320} height={240} alt="Processed Frame" className="object-contain" />
              ) : status.mode === "error" && status.error_message ? (
                <span className="text-red-500 font-semibold text-center px-2">{status.error_message}</span>
              ) : (
                <span className="text-gray-400 italic">
                  {wsConnectionStatus !== "Connected"
                    ? "Waiting for backend connection..."
                    : isBackendProcessing
                    ? "Processing frame..."
                    : "Waiting for backend..."}
                </span>
              )}
              {isBackendProcessing && (
                <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-60 text-white text-xs p-1 animate-pulse">
                  Backend Processing...
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="border border-gray-200 p-4 rounded-lg bg-gray-50 shadow-sm">
            <h3 className="font-semibold text-base mb-3 text-center text-gray-700">Calibration</h3>
            <div className="flex flex-col gap-2">
              <button onClick={handleStartCalibration} disabled={actionDisabled} className="btn bg-indigo-500 hover:bg-indigo-600 disabled:bg-gray-400">
                Start Calibration
              </button>
              {status.mode === "calibrating" && (
                <>
                  <button
                    onClick={handleCaptureColor}
                    disabled={isActionLoading || (status.calibration_step == null || status.calibration_step >= COLOR_NAMES.length)}
                    className="btn bg-teal-500 hover:bg-teal-600 disabled:bg-gray-400"
                  >
                    Capture '{status.current_color || "?"}'
                  </button>
                  <button
                    onClick={handleSaveCalibration}
                    disabled={isActionLoading || (status.calibration_step == null || status.calibration_step < COLOR_NAMES.length)}
                    className="btn bg-green-500 hover:bg-green-600 disabled:bg-gray-400"
                  >
                    Save Calibration
                  </button>
                </>
              )}
              <button onClick={handleResetCalibration} disabled={isActionLoading} className="btn bg-yellow-500 hover:bg-yellow-600 disabled:bg-gray-400 mt-2">
                Reset Calibration
              </button>
            </div>
          </div>
          <div className="border border-gray-200 p-4 rounded-lg bg-gray-50 shadow-sm">
            <h3 className="font-semibold text-base mb-3 text-center text-gray-700">Solver</h3>
            <div className="flex flex-col gap-2">
              <button
                onClick={handleStartSolve}
                disabled={actionDisabled || !status.serial_connected}
                title={!status.serial_connected ? "Serial disconnected" : ""}
                className={`btn bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 ${!status.serial_connected ? "cursor-not-allowed" : ""}`}
              >
                Solve Cube
              </button>
              <button
                onClick={handleStopAndReset}
                disabled={status.mode === "idle" || status.mode === "connecting" || isActionLoading}
                className="btn bg-orange-500 hover:bg-orange-600 disabled:bg-gray-400 mt-2"
              >
                Stop & Reset
              </button>
            </div>
          </div>
          <div className="border border-gray-200 p-4 rounded-lg bg-gray-50 shadow-sm">
            <h3 className="font-semibold text-base mb-3 text-center text-gray-700">Scramble</h3>
            <div className="flex flex-col gap-2">
              <button
                onClick={handleStartScramble}
                disabled={actionDisabled || !status.serial_connected}
                title={!status.serial_connected ? "Serial disconnected" : ""}
                className={`btn bg-red-500 hover:bg-red-600 disabled:bg-gray-400 ${!status.serial_connected ? "cursor-not-allowed" : ""}`}
              >
                Scramble Cube
              </button>
            </div>
          </div>
        </div>
      </div>

      <style jsx global>{`
        .btn {
          padding: 0.5rem 1rem;
          color: white;
          border-radius: 0.375rem;
          font-weight: 500;
          box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
          transition: background-color 0.2s;
          text-align: center;
        }
        .btn:disabled {
          cursor: not-allowed;
          opacity: 0.7;
        }
        .btn:focus {
          outline: none;
          box-shadow: 0 0 0 2px rgba(96, 165, 250, 0.5);
        }
      `}</style>
    </div>
  );
}