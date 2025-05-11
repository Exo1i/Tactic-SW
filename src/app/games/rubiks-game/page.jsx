"use client";
import React, { useCallback, useEffect, useRef, useState } from "react";

// Helper constant
const COLOR_NAMES = ["W", "R", "G", "Y", "O", "B"];

// **DEBUG** Counter for frames sent
let frameSentCounter = 0;

export default function RubiksSolverPage() {
  console.log("[Render] RubiksSolverPage component rendering"); // **DEBUG**
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
    ipCameraUrl: "http://192.168.1.9:8080/video", // **DEBUG** Default IP for testing
  });
  const [modalCameraSettings, setModalCameraSettings] = useState({
    useIpCamera: false,
    ipCameraUrl: "http://192.168.1.9:8080/video", // **DEBUG** Default IP for testing
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
    console.log(
      "%c[initCamera] Start",
      "color: blue; font-weight: bold;",
      appliedCameraSettings
    ); // **DEBUG**
    setIsCameraLoading(true);
    setStatus((prev) => ({
      ...prev,
      status_message: "Initializing camera...",
      error_message: null,
    }));

    // Stop previous stream if exists
    if (videoRef.current && videoRef.current.srcObject) {
      console.log("[initCamera] Stopping previous device stream."); // **DEBUG**
      videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    // Clear IP Cam image source
    if (ipCamImgRef.current) {
      console.log("[initCamera] Clearing previous IP Cam src."); // **DEBUG**
      ipCamImgRef.current.src = "";
    }

    if (!appliedCameraSettings.useIpCamera) {
      console.log("[initCamera] Attempting device camera..."); // **DEBUG**
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        console.error("[initCamera] ERROR: getUserMedia not supported!"); // **DEBUG**
        setStatus((prev) => ({
          ...prev,
          mode: "error",
          status_message: "Camera Error",
          error_message:
            "Browser does not support camera access (getUserMedia).",
        }));
        setIsCameraLoading(false);
        return;
      }
      try {
        const constraints = {
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: "environment",
          },
          audio: false,
        };
        console.log(
          "[initCamera] Requesting media device with constraints:",
          constraints
        ); // **DEBUG**
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        console.log("[initCamera] getUserMedia SUCCESS, stream ID:", stream.id); // **DEBUG**
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          console.log("[initCamera] Stream assigned to video element."); // **DEBUG**
          // onCanPlay handler will set isCameraLoading to false
        } else {
          console.error(
            "[initCamera] ERROR: videoRef is null when assigning stream."
          ); // **DEBUG**
          stream.getTracks().forEach((track) => track.stop());
          setIsCameraLoading(false); // Cannot proceed
        }
      } catch (error) {
        console.error(
          "[initCamera] getUserMedia ERROR:",
          error.name,
          error.message,
          error
        ); // **DEBUG** Enhanced logging
        let errorMsg = `Failed to access camera: ${error.name}.`;
        if (error.name === "NotAllowedError")
          errorMsg = "Camera permission denied. Please allow access.";
        else if (error.name === "NotFoundError")
          errorMsg = "No suitable camera found.";
        else if (error.name === "NotReadableError")
          errorMsg = "Camera is already in use or hardware error.";
        else if (error.name === "OverconstrainedError")
          errorMsg = `Camera does not support requested resolution/facingMode. ${error.message}`;
        else if (error.name === "AbortError")
          errorMsg = "Camera request aborted.";
        setStatus((prev) => ({
          ...prev,
          mode: "error",
          status_message: "Camera Error",
          error_message: errorMsg,
        }));
        setIsCameraLoading(false);
      }
    } else {
      console.log(
        "[initCamera] Using IP Camera:",
        appliedCameraSettings.ipCameraUrl
      ); // **DEBUG**
      if (!appliedCameraSettings.ipCameraUrl) {
        console.error("[initCamera] ERROR: IP Camera URL is empty."); // **DEBUG**
        setStatus((prev) => ({
          ...prev,
          mode: "error",
          status_message: "Camera Error",
          error_message: "IP Camera URL cannot be empty.",
        }));
        setIsCameraLoading(false);
        return;
      }
      setStatus((prev) => ({
        ...prev,
        status_message: "Loading IP Camera feed...",
      }));
      // The img element's onLoad handler will set isCameraLoading to false
      if (ipCamImgRef.current) {
        console.log(
          "[initCamera] Setting IP Cam src:",
          appliedCameraSettings.ipCameraUrl
        );
        ipCamImgRef.current.src = appliedCameraSettings.ipCameraUrl; // Trigger loading
      } else {
        console.error(
          "[initCamera] ERROR: ipCamImgRef is null when setting IP Cam src."
        );
        setIsCameraLoading(false);
      }
    }
    console.log("%c[initCamera] End", "color: blue; font-weight: bold;"); // **DEBUG**
  }, [appliedCameraSettings]); // Keep dependency

  // --- WebSocket Connection & Frame Sending ---
  useEffect(() => {
    console.log(
      "%c[useEffect] Component Mounted / appliedCameraSettings changed",
      "color: green; font-weight: bold;"
    ); // **DEBUG**
    let stopped = false;
    let reconnectTimeout = null;
    let frameRequestId = null;

    const connectWebSocket = () => {
      console.log("[connectWebSocket] Attempting connection..."); // **DEBUG**
      if (stopped) {
        console.log("[connectWebSocket] Aborted: component stopped.");
        return;
      } // **DEBUG**
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        console.log("[connectWebSocket] Aborted: Already connected.");
        return;
      } // **DEBUG**

      setWsConnectionStatus("Connecting...");
      const wsProtocol =
        window.location.protocol === "https:" ? "wss://" : "ws://";
      const wsUrl = `${wsProtocol}${window.location.hostname}:8000/ws/rubiks`;
      console.log("[connectWebSocket] Creating WebSocket to:", wsUrl); // **DEBUG**
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws; // Store immediately

      ws.onopen = () => {
        console.log("%c[WS] Connected", "color: green;"); // **DEBUG**
        setWsConnectionStatus("Connected");
        if (reconnectTimeout) clearTimeout(reconnectTimeout);
        sendNextFrameRef.current = true; // Ensure sending is enabled on connect
      };

      ws.onclose = (event) => {
        console.warn(
          `%c[WS] Disconnected: Code=${event.code}, Reason='${event.reason}'`,
          "color: orange;"
        ); // **DEBUG**
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
          console.log("[WS] Scheduling reconnect in 5s..."); // **DEBUG**
          reconnectTimeout = setTimeout(connectWebSocket, 5000);
        }
      };

      ws.onerror = (error) => {
        console.error("[WS] Error:", error); // **DEBUG**
        setWsConnectionStatus("Error");
        // Don't schedule reconnect on error immediately, onclose will handle it if connection drops
      };

      ws.onmessage = (event) => {
        // **DEBUG** Log raw message data if needed
        // console.log("[WS] Raw message received:", event.data);
        if (loadingTimeoutRef.current) {
          clearTimeout(loadingTimeoutRef.current);
          loadingTimeoutRef.current = null;
        }
        setIsBackendProcessing(false); // Assume backend finished processing this frame
        setIsActionLoading(false); // Reset action loading on ANY message from backend
        sendNextFrameRef.current = true; // Allow sending the next frame
        // console.log("[WS] Set sendNextFrameRef=true, isBackendProcessing=false"); // **DEBUG**

        try {
          const data = JSON.parse(event.data);
          // console.log("[WS] Parsed data:", data); // **DEBUG** Log parsed data

          setStatus((prev) => {
            // **DEBUG** Log previous and next state
            // console.log("[WS onmessage] Updating status. Prev:", prev, "New data:", data);
            const nextState = {
              ...prev,
              mode: data.mode ?? prev.mode,
              status_message: data.status_message ?? prev.status_message,
              error_message: data.error_message ?? null, // Explicitly handle null/undefined
              solution: data.solution ?? prev.solution, // Keep previous if not provided
              serial_connected: data.serial_connected ?? prev.serial_connected,
              calibration_step: data.calibration_step, // Can be null
              current_color: data.current_color, // Can be null
              scan_index: data.scan_index, // Can be null
              solve_move_index: data.solve_move_index ?? prev.solve_move_index, // Keep prev if null/undefined
              total_solve_moves:
                data.total_solve_moves ?? prev.total_solve_moves,
            };
            // console.log("[WS onmessage] Next state:", nextState);
            return nextState;
          });

          if (data.processed_frame) {
            // console.log("[WS] Received processed frame data."); // **DEBUG**
            const frameSrc = `data:image/jpeg;base64,${data.processed_frame}`;
            setProcessedFrame(frameSrc);
          } else {
            // console.log("[WS] No processed frame in this message."); // **DEBUG**
            // Optionally clear processed frame if none sent? setProcessedFrame(null);
          }
        } catch (e) {
          console.error(
            "[WS] Failed to parse message:",
            e,
            "Raw data:",
            event.data
          ); // **DEBUG** Log raw data on parse failure
          setStatus((prev) => ({
            ...prev,
            mode: "error",
            error_message: "Error processing backend update.",
          }));
        }
      };
    };
    const sendFrame = () => {
      // console.log(`[sendFrame ${frameRequestId}] Loop check`); // DEBUG Very verbose
      if (stopped) {
        console.log("[sendFrame] Aborted: component stopped.");
        cancelAnimationFrame(frameRequestId);
        return;
      }

      const ws = wsRef.current;
      const now = Date.now();

      // --- Condition checks for sending (Initial checks) ---
      let skipReason = null;
      if (!sendNextFrameRef.current)
        skipReason = "sendNextFrameRef is false (waiting for backend response)";
      else if (!ws || ws.readyState !== WebSocket.OPEN)
        skipReason = `WebSocket not ready (State: ${ws?.readyState ?? "null"})`;
      else if (now - lastSentRef.current < minFrameInterval)
        skipReason = "Frame interval too short";
      // Removed isCameraLoading check from here
      else if (status.mode === "error") skipReason = "Status is 'error'";

      if (skipReason) {
        // console.log(`[sendFrame] Skipping send (Initial checks): ${skipReason}`); // DEBUG Verbose
        frameRequestId = requestAnimationFrame(sendFrame); // Schedule next check
        return;
      }
      // console.log("[sendFrame] Initial conditions met."); // DEBUG

      const canvas = canvasRef.current;
      let sourceEl = null;
      let sourceReady = false; // Flag to check if the selected source is ready

      // Determine source element AND check its readiness
      if (appliedCameraSettings.useIpCamera) {
        const img = ipCamImgRef.current;
        if (img && img.complete && img.naturalWidth > 0) {
          sourceEl = img;
          sourceReady = true; // IP Cam image is loaded
          // console.log("[sendFrame] Using IP Cam image as source (Ready)."); // DEBUG
        } else {
          // console.log("[sendFrame] IP Cam image ref not ready/loaded yet."); // DEBUG
        }
      } else {
        const video = videoRef.current;
        if (video && video.readyState >= video.HAVE_ENOUGH_DATA) {
          // Use readyState directly
          sourceEl = video;
          sourceReady = true; // Video has enough data to play/capture
          // console.log("[sendFrame] Using video element as source (Ready)."); // DEBUG
        } else {
          // console.log(`[sendFrame] Video element not ready yet (State: ${video?.readyState}).`); // DEBUG
        }
      }

      // --- Skip if source is not ready or required elements missing ---
      if (!sourceReady) {
        // console.log("[sendFrame] Skipping send: Source element not ready."); // DEBUG
        frameRequestId = requestAnimationFrame(sendFrame); // Check again next frame
        return;
      }
      if (!sourceEl) {
        console.warn(
          "[sendFrame] No valid source element found (this shouldn't happen if sourceReady is true)."
        );
        frameRequestId = requestAnimationFrame(sendFrame);
        return;
      }
      if (!canvas) {
        console.error("[sendFrame] Canvas ref is null.");
        frameRequestId = requestAnimationFrame(sendFrame);
        return;
      }
      // No need to re-check WS state here, done in initial checks

      // console.log("[sendFrame] Source Ready & Conditions Met. Proceeding to draw and send."); // DEBUG - Should see this now!

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

        // console.log("[sendFrame] Drawing source to canvas..."); // DEBUG
        ctx.drawImage(sourceEl, 0, 0, targetWidth, targetHeight);

        // console.log("[sendFrame] Calling canvas.toBlob..."); // DEBUG
        canvas.toBlob(
          (blob) => {
            // console.log("[sendFrame] toBlob callback executed."); // DEBUG
            if (blob && ws.readyState === WebSocket.OPEN) {
              // console.log(`[sendFrame] Blob created (Size: ${blob.size}). Setting flags & getting ArrayBuffer...`); // DEBUG
              setIsBackendProcessing(true);
              sendNextFrameRef.current = false;
              lastSentRef.current = Date.now(); // Update last sent time *before* sending

              blob
                .arrayBuffer()
                .then((buffer) => {
                  // console.log("[sendFrame] ArrayBuffer created. Checking WebSocket state again..."); // DEBUG
                  if (ws.readyState === WebSocket.OPEN) {
                    frameSentCounter++; // DEBUG
                    console.log(
                      `%c[sendFrame #${frameSentCounter}] Sending buffer (Size: ${buffer.byteLength}) via WebSocket...`,
                      "color: purple;"
                    ); // DEBUG Highlight send
                    ws.send(buffer);
                  } else {
                    console.warn(
                      "[sendFrame] WebSocket closed before buffer could be sent."
                    ); // DEBUG
                    sendNextFrameRef.current = true;
                    setIsBackendProcessing(false);
                  }
                })
                .catch((err) => {
                  console.error("[sendFrame] Error getting ArrayBuffer:", err); // DEBUG
                  sendNextFrameRef.current = true;
                  setIsBackendProcessing(false);
                });
            } else {
              console.warn(
                "[sendFrame] Blob creation failed or WebSocket closed in toBlob callback."
              ); // DEBUG
              sendNextFrameRef.current = true; // Allow next attempt
              setIsBackendProcessing(false);
              if (!blob)
                console.warn("[sendFrame] Blob is null in toBlob callback.");
              if (ws.readyState !== WebSocket.OPEN)
                console.warn(
                  `[sendFrame] WebSocket not open in toBlob callback (State: ${ws.readyState}).`
                );
            }
          },
          "image/jpeg",
          0.7
        );
      } catch (e) {
        console.warn("[sendFrame] Error processing/drawing frame:", e); // DEBUG
        sendNextFrameRef.current = true; // Allow next attempt
        setIsBackendProcessing(false);
      }

      // Schedule the next frame check AFTER attempting the current one
      frameRequestId = requestAnimationFrame(sendFrame);
    }; // End of sendFrame function
    // --- useEffect Initialization ---
    console.log("[useEffect] Initializing Camera..."); // **DEBUG**
    initCamera(); // Trigger camera init
    console.log("[useEffect] Connecting WebSocket..."); // **DEBUG**
    connectWebSocket(); // Start WebSocket connection attempt
    console.log("[useEffect] Starting sendFrame loop..."); // **DEBUG**
    frameRequestId = requestAnimationFrame(sendFrame); // Start the frame sending loop

    // --- useEffect Cleanup ---
    return () => {
      stopped = true;
      console.log(
        "%c[useEffect Cleanup] Running cleanup...",
        "color: red; font-weight: bold;"
      ); // **DEBUG**
      if (frameRequestId) {
        console.log(
          "[useEffect Cleanup] Cancelling animation frame request:",
          frameRequestId
        ); // **DEBUG**
        cancelAnimationFrame(frameRequestId);
        frameRequestId = null;
      }
      if (reconnectTimeout) {
        console.log("[useEffect Cleanup] Clearing reconnect timeout."); // **DEBUG**
        clearTimeout(reconnectTimeout);
      }
      if (loadingTimeoutRef.current) {
        console.log("[useEffect Cleanup] Clearing API loading timeout."); // **DEBUG**
        clearTimeout(loadingTimeoutRef.current);
      }
      if (wsRef.current) {
        console.log("[useEffect Cleanup] Closing WebSocket connection."); // **DEBUG**
        wsRef.current.close();
        wsRef.current = null;
      }
      if (videoRef.current && videoRef.current.srcObject) {
        console.log("[useEffect Cleanup] Stopping device camera stream."); // **DEBUG**
        videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
        videoRef.current.srcObject = null;
      }
      if (ipCamImgRef.current) {
        console.log("[useEffect Cleanup] Clearing IP Cam src."); // **DEBUG**
        ipCamImgRef.current.src = "";
      }
      console.log(
        "%c[useEffect Cleanup] Cleanup complete.",
        "color: red; font-weight: bold;"
      ); // **DEBUG**
    };
  }, [appliedCameraSettings, initCamera]); // React to changes in settings

  // --- API Call Helper ---
  const callApi = useCallback(
    async (endpoint, method = "POST", body = null) => {
      console.log(`[callApi] Attempting call to ${endpoint}`); // **DEBUG**
      if (isActionLoading) {
        console.warn(
          `[callApi] Action already in progress, skipping call to ${endpoint}`
        ); // **DEBUG**
        return;
      }

      if (loadingTimeoutRef.current) {
        console.warn("[callApi] Clearing previous safety timeout."); // **DEBUG**
        clearTimeout(loadingTimeoutRef.current);
      }

      console.log(`[callApi] Setting isActionLoading = true for ${endpoint}`); // **DEBUG**
      setIsActionLoading(true); // Set loading state

      // Set a safety timeout
      loadingTimeoutRef.current = setTimeout(() => {
        console.warn(
          `[callApi] SAFETY TIMEOUT triggered for ${endpoint} - resetting loading state`
        ); // **DEBUG**
        setIsActionLoading(false);
        setIsBackendProcessing(false); // Also reset this just in case
        loadingTimeoutRef.current = null;
      }, 5000); // 5 seconds

      try {
        const options = { method };
        if (body) {
          options.headers = { "Content-Type": "application/json" };
          options.body = JSON.stringify(body);
        }
        const apiUrl = `http://${window.location.hostname}:8000${endpoint}`;
        console.log(`[callApi] Calling API: ${method} ${apiUrl}`, body || ""); // **DEBUG**
        const response = await fetch(apiUrl, options);
        console.log(
          `[callApi] Response status for ${endpoint}: ${response.status}`
        ); // **DEBUG**
        const data = await response.json(); // Attempt to parse JSON regardless of status

        if (!response.ok) {
          console.error(
            `[callApi] API Error ${response.status} for ${endpoint}:`,
            data
          ); // **DEBUG** Log error data
          throw new Error(data.detail || `API Error ${response.status}`);
        }
        console.log(`[callApi] API Success ${endpoint}:`, data); // **DEBUG** Log success data

        // Clear safety timeout on successful response
        if (loadingTimeoutRef.current) {
          clearTimeout(loadingTimeoutRef.current);
          loadingTimeoutRef.current = null;
        } // **DEBUG** Clear timeout log added implicitly

        // Update status only for specific start actions
        if (
          ["/start_scramble", "/start_solve", "/start_calibration"].includes(
            endpoint
          )
        ) {
          setStatus((prev) => ({
            ...prev,
            status_message: data.message || "Request sent...",
          }));
        }
        // NOTE: isActionLoading will be reset by the next WebSocket message received
        // We don't reset it here to prevent rapid button clicks if WS response is delayed.
        return data;
      } catch (error) {
        console.error(`[callApi] API call to ${endpoint} FAILED:`, error); // **DEBUG**

        // Clear the safety timeout on error
        if (loadingTimeoutRef.current) {
          clearTimeout(loadingTimeoutRef.current);
          loadingTimeoutRef.current = null;
          console.log("[callApi] Cleared safety timeout on error.");
        } // **DEBUG**

        setIsActionLoading(false); // Reset loading state immediately on error
        console.log(`[callApi] Reset isActionLoading = false due to error.`); // **DEBUG**
        setStatus((prev) => ({
          ...prev,
          mode: "error",
          error_message: error.message || "API request failed.",
        }));
        return null;
      }
      // **DEBUG** NOTE: 'finally' block is not strictly needed here as isActionLoading is reset by WS message or error handling.
    },
    [isActionLoading]
  ); // Dependency added

  // --- Button Click Handlers ---
  // **DEBUG** Add logs to handlers
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
    // Save IP camera URL to localStorage
    if (typeof modalCameraSettings.ipCameraUrl === "string") {
      localStorage.setItem("ipCameraAddress", modalCameraSettings.ipCameraUrl);
    }
    console.log(
      "[Settings] Applying new camera settings:",
      modalCameraSettings
    ); // **DEBUG**
    setAppliedCameraSettings(modalCameraSettings); // This will trigger the useEffect hook
    setShowSettings(false);
  };

  // --- Determine Button Disabled States ---
  const isEffectivelyBusy =
    status.mode !== "idle" &&
    status.mode !== "error" &&
    status.mode !== "connecting";
  // **DEBUG** Log why buttons might be disabled
  // console.log(`[Button State] isEffectivelyBusy=${isEffectivelyBusy} (Mode: ${status.mode}), isActionLoading=${isActionLoading}, isCameraLoading=${isCameraLoading}`);
  const actionDisabled =
    isEffectivelyBusy || isActionLoading || isCameraLoading;

  // --- Render ---
  // console.log("[Render] Rendering JSX..."); // **DEBUG**
  return (
    <div className="flex flex-col items-center gap-4 p-4 min-h-screen bg-gray-100 font-sans">
      {/* Settings Modal */}
      {showSettings && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-40 backdrop-blur-sm">
          <div className="bg-white p-6 rounded-lg shadow-xl w-full max-w-md relative">
            {/* ... (Modal Content - no debug logs needed here unless layout breaks) ... */}
            <button
              className="absolute top-2 right-2 text-gray-500 hover:text-gray-800 text-2xl leading-none font-bold"
              onClick={() => setShowSettings(false)}
              aria-label="Close Settings"
            >
              ×
            </button>
            <h3 className="text-lg font-medium mb-4 text-gray-800">
              Camera Settings
            </h3>
            <div className="flex items-center mb-3">
              <input
                type="checkbox"
                id="useIpCameraModal"
                checked={modalCameraSettings.useIpCamera}
                onChange={(e) =>
                  setModalCameraSettings({
                    ...modalCameraSettings,
                    useIpCamera: e.target.checked,
                  })
                }
                className="mr-2 h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <label
                htmlFor="useIpCameraModal"
                className="text-sm font-medium text-gray-700"
              >
                Use IP Camera
              </label>
            </div>
            {modalCameraSettings.useIpCamera && (
              <div className="mb-4">
                <label
                  htmlFor="ipCameraUrlModal"
                  className="block mb-1 text-sm font-medium text-gray-700"
                >
                  IP Camera URL:
                </label>
                <input
                  type="text"
                  id="ipCameraUrlModal"
                  value={modalCameraSettings.ipCameraUrl}
                  onChange={(e) =>
                    setModalCameraSettings({
                      ...modalCameraSettings,
                      ipCameraUrl: e.target.value,
                    })
                  }
                  placeholder="http://camera-ip:port/stream"
                  className="w-full p-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      applySettings();
                    }
                  }}
                />
                <small className="text-xs text-gray-500 mt-1 block">
                  E.g., http://192.168.1.100:8080/video (MJPEG)
                </small>
              </div>
            )}
            <button
              onClick={applySettings}
              className="w-full px-4 py-2 bg-green-600 text-white font-medium rounded-md shadow-sm hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500"
            >
              {" "}
              Apply Settings
            </button>
          </div>
        </div>
      )}

      {/* Main Content Area */}
      <div className="w-full max-w-5xl bg-white rounded-xl shadow-lg p-6 mt-4">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-4 border-b border-gray-200 pb-4">
          <h1 className="text-2xl font-bold text-gray-800 mb-2 sm:mb-0">
            Rubik's Cube Solver
          </h1>
          <div className="flex items-center gap-3">
            {/* ... (Status indicators - display only) ... */}
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
              {" "}
              WS: {wsConnectionStatus}{" "}
            </span>
            <span
              className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                status.serial_connected
                  ? "bg-blue-100 text-blue-800"
                  : "bg-red-100 text-red-800"
              }`}
            >
              {" "}
              {status.serial_connected
                ? "Serial OK"
                : "Serial disconnected"}{" "}
            </span>
            <button
              onClick={() => {
                console.log("[Click] Open Settings");
                setModalCameraSettings(appliedCameraSettings);
                setShowSettings(true);
              }} // **DEBUG**
              className="px-3 py-1 bg-blue-500 text-white rounded-md hover:bg-blue-600 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              aria-label="Open Camera Settings"
            >
              {" "}
              Settings
            </button>
          </div>
        </div>

        {/* Status Display */}
        {/* ... (Status text display - relies on state, no extra logs needed here) ... */}
        <div className="mb-5 p-3 bg-gray-50 rounded-lg border border-gray-200 shadow-sm">
          <p className="text-sm font-medium text-gray-700">
            Status:{" "}
            <span
              className={`font-semibold capitalize ${
                status.mode === "error" ? "text-red-600" : "text-gray-900"
              }`}
            >
              {status.mode}
            </span>
            {/* ... (Conditional progress indicators) ... */}
            {(status.mode === "solving" || status.mode === "scrambling") &&
              status.total_solve_moves > 0 && (
                <span className="ml-2 text-xs font-normal text-gray-500">
                  {" "}
                  ({status.solve_move_index || 0} / {status.total_solve_moves}){" "}
                </span>
              )}
            {status.mode === "scanning" && status.scan_index != null && (
              <span className="ml-2 text-xs font-normal text-gray-500">
                {" "}
                (Scan {status.scan_index + 1} / 12){" "}
              </span>
            )}
          </p>
          <p className="text-sm mt-1 text-gray-600 min-h-[1.25rem]">
            {" "}
            {status.status_message || " "}{" "}
          </p>
          {status.mode === "error" && status.error_message && (
            <p className="text-sm mt-1 text-red-700 font-medium bg-red-50 p-2 rounded border border-red-200">
              {" "}
              Error: {status.error_message}{" "}
            </p>
          )}
          {status.solution && (
            <div className="mt-2 text-sm text-blue-800 font-mono bg-blue-50 p-2 rounded border border-blue-200 max-h-24 overflow-y-auto">
              <span className="font-semibold">Solution Found:</span>{" "}
              <p className="whitespace-pre-wrap break-words text-xs leading-relaxed">
                {status.solution}
              </p>
            </div>
          )}
        </div>

        {/* Camera Feeds */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          {/* Live Camera Feed */}
          <div className="flex flex-col items-center">
            <div className="mb-1 text-center text-sm font-medium text-gray-600">
              Live Camera Input
            </div>
            <div className="relative w-[320px] h-[240px] rounded-lg overflow-hidden border-2 border-gray-300 bg-black flex items-center justify-center shadow-inner">
              {appliedCameraSettings.useIpCamera && (
                <img
                  ref={ipCamImgRef}
                  id="ip-camera-img-display"
                  // src is set in initCamera
                  alt="IP Camera Feed"
                  width={320}
                  height={240}
                  className="object-contain"
                  crossOrigin="anonymous"
                  onLoad={() => {
                    console.log(
                      "%c[IP Cam] onLoad event triggered.",
                      "color: green;"
                    );
                    setIsCameraLoading(false);
                  }} // **DEBUG**
                  onError={(e) => {
                    console.error("[IP Cam] onError event triggered:", e);
                    setIsCameraLoading(false);
                    if (status.mode !== "error")
                      setStatus((prev) => ({
                        ...prev,
                        status_message: "IP Cam Load Error",
                        error_message:
                          "Failed to load IP Cam image. Check URL/Network/CORS.",
                      }));
                  }} // **DEBUG**
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
                    console.log(
                      "%c[Device Cam] onCanPlay event triggered.",
                      "color: green;"
                    );
                    setIsCameraLoading(false);
                  }} // **DEBUG**
                  onError={(e) => {
                    console.error("[Device Cam] onError event triggered:", e);
                    setIsCameraLoading(false);
                  }} // **DEBUG**
                  onLoadedData={() => console.log("[Device Cam] onLoadedData")} // **DEBUG** Optional
                  onPlaying={() => console.log("[Device Cam] onPlaying")} // **DEBUG** Optional
                />
              )}
              {/* ... (Loading/Error Overlays - display only) ... */}
              {isCameraLoading && (
                <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-70 text-white text-lg">
                  <div className="flex flex-col items-center">
                    <svg
                      className="animate-spin h-8 w-8 text-white mb-2"
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      ></circle>
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
              {!isCameraLoading &&
                status.mode === "error" &&
                status.error_message?.toLowerCase().includes("camera") && (
                  <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-75 text-white text-sm p-2">
                    <div className="p-3 text-center rounded bg-red-800 bg-opacity-90 shadow">
                      <span className="font-bold block text-base">
                        Camera Error
                      </span>{" "}
                      <span className="text-xs mt-1 block">
                        {status.error_message}
                      </span>
                    </div>
                  </div>
                )}
              <canvas
                ref={canvasRef}
                width={320}
                height={240}
                className="hidden"
              />
              {/* Keep hidden canvas */}
            </div>
          </div>

          {/* Processed Feed */}
          <div className="flex flex-col items-center">
            {/* ... (Processed view display - relies on state) ... */}
            <div className="mb-1 text-center text-sm font-medium text-gray-600">
              Processed View (Backend)
            </div>
            <div className="relative w-[320px] h-[240px] rounded-lg overflow-hidden border-2 border-gray-300 bg-black flex items-center justify-center shadow-inner">
              {processedFrame ? (
                <img
                  src={processedFrame}
                  width={320}
                  height={240}
                  alt="Processed Frame"
                  className="object-contain"
                />
              ) : (
                <span className="text-gray-400 italic">
                  Waiting for backend...
                </span>
              )}
              {isBackendProcessing && (
                <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-60 text-white text-xs p-1 animate-pulse">
                  {" "}
                  Backend Processing...{" "}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Control Buttons */}
        {/* ... (Buttons - check disabled states visually or log `actionDisabled`) ... */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Calibration */}
          <div className="border border-gray-200 p-4 rounded-lg bg-gray-50 shadow-sm">
            <h3 className="font-semibold text-base mb-3 text-center text-gray-700">
              Calibration
            </h3>
            <div className="flex flex-col gap-2">
              <button
                onClick={handleStartCalibration}
                disabled={actionDisabled}
                className="btn bg-indigo-500 hover:bg-indigo-600 disabled:bg-gray-400"
              >
                {" "}
                Start Calibration
              </button>
              {status.mode === "calibrating" && (
                <>
                  <button
                    onClick={handleCaptureColor}
                    disabled={
                      isActionLoading ||
                      status.calibration_step == null ||
                      status.calibration_step >= COLOR_NAMES.length
                    }
                    className="btn bg-teal-500 hover:bg-teal-600 disabled:bg-gray-400"
                  >
                    {" "}
                    Capture '{status.current_color || "?"}'
                  </button>
                  <button
                    onClick={handleSaveCalibration}
                    disabled={
                      isActionLoading ||
                      status.calibration_step == null ||
                      status.calibration_step < COLOR_NAMES.length
                    }
                    className="btn bg-green-500 hover:bg-green-600 disabled:bg-gray-400"
                  >
                    {" "}
                    Save Calibration
                  </button>
                </>
              )}
              <button
                onClick={handleResetCalibration}
                disabled={isActionLoading}
                className="btn bg-yellow-500 hover:bg-yellow-600 disabled:bg-gray-400 mt-2"
              >
                {" "}
                Reset Calibration
              </button>
            </div>
          </div>
          {/* Solver */}
          <div className="border border-gray-200 p-4 rounded-lg bg-gray-50 shadow-sm">
            <h3 className="font-semibold text-base mb-3 text-center text-gray-700">
              Solver
            </h3>
            <div className="flex flex-col gap-2">
              <button
                onClick={handleStartSolve}
                disabled={actionDisabled || !status.serial_connected}
                title={!status.serial_connected ? "Serial disconnected" : ""}
                className={`btn bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 ${
                  !status.serial_connected ? "cursor-not-allowed" : ""
                }`}
              >
                {" "}
                Solve Cube
              </button>
              <button
                onClick={handleStopAndReset}
                disabled={
                  status.mode === "idle" ||
                  status.mode === "connecting" ||
                  isActionLoading
                }
                className="btn bg-orange-500 hover:bg-orange-600 disabled:bg-gray-400 mt-2"
              >
                {" "}
                Stop & Reset
              </button>
            </div>
          </div>
          {/* Scramble */}
          <div className="border border-gray-200 p-4 rounded-lg bg-gray-50 shadow-sm">
            <h3 className="font-semibold text-base mb-3 text-center text-gray-700">
              Scramble
            </h3>
            <div className="flex flex-col gap-2">
              <button
                onClick={handleStartScramble}
                disabled={actionDisabled || !status.serial_connected}
                title={!status.serial_connected ? "Serial disconnected" : ""}
                className={`btn bg-red-500 hover:bg-red-600 disabled:bg-gray-400 ${
                  !status.serial_connected ? "cursor-not-allowed" : ""
                }`}
              >
                {" "}
                Scramble Cube
              </button>
            </div>
          </div>
        </div>

        {/* Action Loading Global Indicator */}
        {isActionLoading && (
          <div className="mt-4 text-center text-sm text-gray-600 animate-pulse font-medium">
            Waiting for backend response...{" "}
            {/* **DEBUG** This should appear when buttons are clicked */}
          </div>
        )}
      </div>
      {/* Shared Button Styles */}
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
