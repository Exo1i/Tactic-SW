import React, { useState, useEffect } from "react";

function CameraSettings({ onCameraChange }) {
  const [ipCameraAddress, setIpCameraAddress] = useState("");
  const [useIpCamera, setUseIpCamera] = useState(false);

  // Load camera settings from localStorage on initial render
  useEffect(() => {
    const savedIpCamera = localStorage.getItem("ipCameraAddress");
    const savedUseIpCamera = localStorage.getItem("useIpCamera") === "true";

    if (savedIpCamera) setIpCameraAddress(savedIpCamera);
    if (savedUseIpCamera) setUseIpCamera(savedUseIpCamera);

    // Notify parent component of initial settings
    if (savedUseIpCamera && savedIpCamera) {
      onCameraChange({
        useIpCamera: savedUseIpCamera,
        ipCameraAddress: savedIpCamera,
      });
    }
  }, [onCameraChange]);

  const handleIpCameraChange = (e) => {
    const value = e.target.value;
    setIpCameraAddress(value);
    localStorage.setItem("ipCameraAddress", value);

    if (useIpCamera) {
      onCameraChange({ useIpCamera, ipCameraAddress: value });
    }
  };

  // Modify the handleUseIpCameraChange to be more careful about activating the IP camera
  const handleUseIpCameraChange = (e) => {
    const checked = e.target.checked;
    setUseIpCamera(checked);
    localStorage.setItem("useIpCamera", checked.toString());

    // Only send camera change when we have a valid address or we're disabling
    if (checked && ipCameraAddress && ipCameraAddress.trim() !== "") {
      onCameraChange({ useIpCamera: checked, ipCameraAddress });
    } else if (!checked) {
      onCameraChange({ useIpCamera: false, ipCameraAddress: null });
    }
  };

  // Add a dedicated apply button to avoid immediate freezes
  const handleApplySettings = () => {
    if (ipCameraAddress && ipCameraAddress.trim() !== "") {
      onCameraChange({
        useIpCamera: true,
        ipCameraAddress,
      });
    }
  };

  return (
    <div className="flex flex-col p-3 bg-gray-100 rounded-md shadow-sm mb-4">
      <div className="flex items-center mb-2">
        <input
          type="checkbox"
          id="use-ip-camera"
          className="mr-2"
          checked={useIpCamera}
          onChange={handleUseIpCameraChange}
        />
        <label htmlFor="use-ip-camera">Use IP Camera</label>
      </div>

      {useIpCamera && (
        <div className="flex flex-col">
          <label htmlFor="ip-camera-address" className="text-sm mb-1">
            IP Camera URL:
          </label>
          <input
            type="text"
            id="ip-camera-address"
            value={ipCameraAddress}
            onChange={handleIpCameraChange}
            placeholder="e.g. http://192.168.1.100:8080/video"
            className="px-2 py-1 border rounded"
          />
          <button
            onClick={handleApplySettings}
            className="mt-2 bg-blue-500 text-white py-1 px-3 rounded hover:bg-blue-600"
          >
            Apply Camera
          </button>
          <p className="text-xs text-gray-500 mt-1">
            Format: http://IP:PORT/video or
            rtsp://username:password@IP:PORT/path
          </p>
        </div>
      )}
    </div>
  );
}

export default CameraSettings;
