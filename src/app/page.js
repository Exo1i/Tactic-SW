"use client";
import { useRouter } from "next/navigation";
import { useState, useEffect, useCallback } from "react";

const games = [
  { name: "Shell Game", id: "shell-game" },
  { name: "Tic Tac Toe", id: "tic-tac-toe" },
  { name: "Rubik's Game", id: "rubiks-game"},
  { name: "Memory Matching", id: "memory-matching"},
  { name: "Shooter Gamer", id: "shooting-game"},
  { name: "Game 5", id: "game-5" },
];

export default function Home() {
  const router = useRouter();
  const [showSettingsModal, setShowSettingsModal] = useState(false);
  const [globalSettings, setGlobalSettings] = useState({
    webcam_ip: "",
    serial_config: {
      type: "usb",
      path: "",
      baudrate: 9600,
      host: "",
      port: 8888,
    },
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchGlobalSettings = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch("http://localhost:8000/api/settings");
      if (!response.ok) {
        throw new Error(`Failed to fetch settings: ${response.statusText}`);
      }
      const data = await response.json();
      setGlobalSettings(data);
    } catch (err) {
      setError(err.message);
      console.error("Error fetching global settings:", err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchGlobalSettings();
  }, [fetchGlobalSettings]);

  const handleSaveSettings = async (settingsToSave) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch("http://localhost:8000/api/settings", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(settingsToSave),
      });
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: "Failed to save settings" }));
        throw new Error(errorData.detail || `HTTP error ${response.status}`);
      }
      const data = await response.json();
      setGlobalSettings(data.current_settings || settingsToSave); // Use returned settings if available
      setShowSettingsModal(false);
    } catch (err) {
      setError(err.message);
      console.error("Error saving global settings:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const SettingsModal = () => {
    const [modalData, setModalData] = useState(JSON.parse(JSON.stringify(globalSettings))); // Deep copy

    useEffect(() => {
        // Update modalData if globalSettings change (e.g., after initial fetch or save)
        setModalData(JSON.parse(JSON.stringify(globalSettings)));
    }, [globalSettings]);


    const handleChange = (e) => {
      const { name, value, type } = e.target;
      if (name.startsWith("serial_config.")) {
        const key = name.split(".")[1];
        setModalData((prev) => ({
          ...prev,
          serial_config: {
            ...prev.serial_config,
            [key]: type === 'number' ? parseInt(value, 10) || 0 : value,
          },
        }));
      } else {
        setModalData((prev) => ({
          ...prev,
          [name]: value,
        }));
      }
    };
    
    const handleSerialTypeChange = (e) => {
        const newType = e.target.value;
        setModalData(prev => ({
            ...prev,
            serial_config: {
                ...prev.serial_config,
                type: newType,
                // Optionally reset other fields when type changes, or keep them
            }
        }));
    };

    const onSave = () => {
      // Ensure baudrate and port are numbers
      const settingsToSubmit = {
        ...modalData,
        serial_config: {
          ...modalData.serial_config,
          baudrate: parseInt(modalData.serial_config.baudrate, 10) || 9600,
          port: parseInt(modalData.serial_config.port, 10) || 8888,
        }
      };
      handleSaveSettings(settingsToSubmit);
    };

    if (!showSettingsModal) return null;

    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
        <div className="bg-white p-6 rounded-lg shadow-xl w-full max-w-lg">
          <h2 className="text-xl font-semibold mb-4">Global Settings</h2>
          
          {error && <p className="text-red-500 text-sm mb-2">Error: {error}</p>}

          <div className="mb-4">
            <label htmlFor="webcam_ip" className="block text-sm font-medium text-gray-700">Webcam IP Address:</label>
            <input
              type="text"
              name="webcam_ip"
              id="webcam_ip"
              value={modalData.webcam_ip}
              onChange={handleChange}
              className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
              placeholder="e.g., http://192.168.1.100:8080/video"
            />
          </div>

          <div className="mb-4">
            <label htmlFor="serial_type" className="block text-sm font-medium text-gray-700">Serial Connection Type:</label>
            <select
              name="serial_config.type"
              id="serial_type"
              value={modalData.serial_config.type}
              onChange={handleSerialTypeChange}
              className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
            >
              <option value="usb">USB</option>
              <option value="wifi">WiFi</option>
            </select>
          </div>

          {modalData.serial_config.type === "usb" && (
            <>
              <div className="mb-4">
                <label htmlFor="serial_path" className="block text-sm font-medium text-gray-700">Serial Path (USB):</label>
                <input
                  type="text"
                  name="serial_config.path"
                  id="serial_path"
                  value={modalData.serial_config.path}
                  onChange={handleChange}
                  className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                  placeholder="e.g., COM3 or /dev/ttyUSB0"
                />
              </div>
              <div className="mb-4">
                <label htmlFor="serial_baudrate" className="block text-sm font-medium text-gray-700">Baud Rate (USB):</label>
                <input
                  type="number"
                  name="serial_config.baudrate"
                  id="serial_baudrate"
                  value={modalData.serial_config.baudrate}
                  onChange={handleChange}
                  className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                  placeholder="e.g., 9600"
                />
              </div>
            </>
          )}

          {modalData.serial_config.type === "wifi" && (
            <>
              <div className="mb-4">
                <label htmlFor="serial_host" className="block text-sm font-medium text-gray-700">Serial Host (WiFi):</label>
                <input
                  type="text"
                  name="serial_config.host"
                  id="serial_host"
                  value={modalData.serial_config.host}
                  onChange={handleChange}
                  className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                  placeholder="e.g., 192.168.1.101"
                />
              </div>
              <div className="mb-4">
                <label htmlFor="serial_port" className="block text-sm font-medium text-gray-700">Serial Port (WiFi):</label>
                <input
                  type="number"
                  name="serial_config.port"
                  id="serial_port"
                  value={modalData.serial_config.port}
                  onChange={handleChange}
                  className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                  placeholder="e.g., 8888"
                />
              </div>
            </>
          )}

          <div className="flex justify-end gap-3 mt-6">
            <button
              onClick={() => setShowSettingsModal(false)}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-200 rounded-md hover:bg-gray-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500"
              disabled={isLoading}
            >
              Cancel
            </button>
            <button
              onClick={onSave}
              className="px-4 py-2 text-sm font-medium text-white bg-indigo-600 rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
              disabled={isLoading}
            >
              {isLoading ? "Saving..." : "Save Settings"}
            </button>
          </div>
        </div>
      </div>
    );
  };


  return (
    <div className="flex flex-col items-center justify-center min-h-screen gap-8 p-4">
      <SettingsModal />
      <div className="absolute top-4 right-4">
        <button
          onClick={() => setShowSettingsModal(true)}
          className="p-2 bg-gray-200 text-gray-700 rounded-full hover:bg-gray-300 focus:outline-none focus:ring-2 focus:ring-gray-500"
          title="Global Settings"
          disabled={isLoading && !showSettingsModal} // Disable if loading initial settings
        >
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6">
            <path strokeLinecap="round" strokeLinejoin="round" d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.646.87.074.04.147.083.22.127.324.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 011.37.49l1.296 2.247a1.125 1.125 0 01-.26 1.431l-1.003.827c-.293.24-.438.613-.431.992a6.759 6.759 0 010 1.255c-.007.378.138.75.43.99l1.005.828c.424.35.534.954.26 1.43l-1.298 2.247a1.125 1.125 0 01-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.57 6.57 0 01-.22.128c-.333.183-.583.495-.646.869l-.213 1.28c-.09.543-.56.941-1.11.941h-2.594c-.55 0-1.02-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.646-.87a6.52 6.52 0 01-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 01-1.369-.49l-1.297-2.247a1.125 1.125 0 01.26-1.431l1.004-.827c.292-.24.437-.613.43-.992a6.759 6.759 0 010-1.255c.007-.378-.137-.75-.43-.99l-1.004-.828a1.125 1.125 0 01-.26-1.43l1.297-2.247a1.125 1.125 0 011.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.087.22-.128.332-.183.582-.495.646-.869l.214-1.281z" />
            <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
        </button>
      </div>
      <h1 className="text-3xl font-bold mb-4">Choose a Game</h1>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {games.map((game) => (
          <button
            key={game.id}
            className="px-6 py-4 bg-blue-600 text-white rounded shadow hover:bg-blue-700 transition"
            onClick={() => router.push(`/games/${game.id}`)}
          >
            {game.name}
          </button>
        ))}
      </div>
    </div>
  );
}
