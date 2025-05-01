// utils/cameraUtils.js

/**
 * Initializes a video source for the specified video element
 * @param {HTMLVideoElement} videoElement - The video element to initialize
 * @param {Object} settings - Camera settings object
 * @returns {Promise<boolean>} - Returns true if initialization was successful
 */
export async function initializeVideoSource(videoElement, settings = {}) {
    if (!videoElement) {
        console.error("No video element provided to initializeVideoSource");
        return false;
    }

    // Stop any existing stream
    if (videoElement.srcObject) {
        try {
            const tracks = videoElement.srcObject.getTracks();
            tracks.forEach(track => track.stop());
            videoElement.srcObject = null;
        } catch (err) {
            console.warn("Error stopping existing tracks:", err);
            // Continue anyway
        }
    }

    // Use IP Camera logic is moved to the main component
    // This function now only handles device cameras

    try {
        // First try with ideal constraints: HD and prefer front camera for ease of use
        const constraints = {
            audio: false,
            video: {
                width: {ideal: 1280},
                height: {ideal: 720},
                // facingMode: {ideal: "environment"} // Prefer back camera for cube scanning
            }
        };

        console.log("Requesting camera with constraints:", constraints);
        const stream = await navigator.mediaDevices.getUserMedia(constraints);

        // Check if we got a valid stream with video tracks
        if (!stream || stream.getVideoTracks().length === 0) {
            throw new Error("No video tracks in stream");
        }

        videoElement.srcObject = stream;

        // Return success but wait for oncanplay event in component
        // for actual UI update
        return true;

    } catch (firstError) {
        console.warn("Failed to get camera with ideal constraints:", firstError);

        try {
            // Fallback to basic constraints
            console.log("Trying fallback camera constraints");
            const fallbackConstraints = {
                audio: false,
                video: true  // Just request any video
            };

            const fallbackStream = await navigator.mediaDevices.getUserMedia(fallbackConstraints);

            if (!fallbackStream || fallbackStream.getVideoTracks().length === 0) {
                throw new Error("No video tracks in fallback stream");
            }

            videoElement.srcObject = fallbackStream;
            return true;

        } catch (fallbackError) {
            console.error("Camera access failed with fallback constraints:", fallbackError);

            // Provide descriptive error based on error type
            if (fallbackError.name === 'NotAllowedError' || fallbackError.name === 'PermissionDeniedError') {
                console.error("Camera permission denied by user");
                throw new Error("Camera permission denied. Please allow camera access in your browser settings.");
            } else if (fallbackError.name === 'NotFoundError') {
                console.error("No camera devices found");
                throw new Error("No camera found on this device.");
            } else if (fallbackError.name === 'NotReadableError' || fallbackError.name === 'AbortError') {
                console.error("Camera in use or not readable");
                throw new Error("Camera is in use by another application or not available.");
            } else {
                console.error("Camera access failed:", fallbackError.message);
                throw new Error(`Camera error: ${fallbackError.message}`);
            }
        }
    }
}

/**
 * Initialize device camera
 * @param {HTMLVideoElement} videoElement
 * @returns {Promise<boolean>}
 */
async function initDeviceCamera(videoElement) {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {facingMode: {ideal: "environment"}},
        });
        console.log("[initDeviceCamera] getUserMedia success. Stream ID:", stream?.id); // Log stream
        videoElement.srcObject = stream;
        // *** ADD THIS LOG ***
        console.log("[initDeviceCamera] Assigned srcObject. Value just before return:", videoElement.srcObject ? `Stream ID ${videoElement.srcObject.id}` : 'null');
        return true;
    } catch (error) {
        console.error("[initDeviceCamera] Error accessing device camera:", error.name, error.message); // Add name/message
        return false;
    }
}