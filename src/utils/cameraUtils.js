/**
 * Safely attempts to initialize a video element with a stream source
 * @param {HTMLVideoElement} videoElement - The video element to initialize
 * @param {Object} cameraSettings - Camera settings including IP camera options
 * @returns {Promise<boolean>} - Success status
 */
export async function initializeVideoSource(videoElement, cameraSettings) {
  if (!videoElement) return false;

  try {
    // Stop any existing stream
    if (videoElement.srcObject) {
      const tracks = videoElement.srcObject.getTracks();
      tracks.forEach((track) => track.stop());
      videoElement.srcObject = null;
    }

    // Clear any existing src attribute
    videoElement.src = "";

    // Use IP camera if enabled and address is provided
    if (cameraSettings.useIpCamera && cameraSettings.ipCameraAddress) {
      try {
        // For some IP cameras, we need to use img tags or special players
        // Here we use a simple approach that works for many basic HTTP streams
        videoElement.crossOrigin = "anonymous";
        videoElement.src = cameraSettings.ipCameraAddress;

        // Set a timeout to avoid UI freeze
        const playPromise = videoElement.play();
        if (playPromise !== undefined) {
          const timeoutPromise = new Promise((_, reject) => {
            setTimeout(
              () => reject(new Error("Timeout connecting to IP camera")),
              5000
            );
          });

          await Promise.race([playPromise, timeoutPromise]);
        }
        return true;
      } catch (ipError) {
        console.error("Failed to connect to IP camera:", ipError);
        // Fall back to device camera
        return initDeviceCamera(videoElement);
      }
    } else {
      // Use device camera
      return initDeviceCamera(videoElement);
    }
  } catch (error) {
    console.error("Error initializing video source:", error);
    return false;
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
      video: { facingMode: { ideal: "environment" } },
    });
    videoElement.srcObject = stream;
    return true;
  } catch (error) {
    console.error("Error accessing device camera:", error);
    return false;
  }
}
