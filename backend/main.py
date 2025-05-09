
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import importlib
import sys
import os
import json
import inspect
import serial # Added for serial connection

# Import memory matching backend for color/yolo WebSocket endpoints
from games.memory_matching_backend import MemoryMatching
from games.rubiks_cube_game import RubiksCubeGame

app = FastAPI()

# Allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure backend/games/tic-tac-toe is in sys.path for dynamic imports (for 'utils', 'alphabeta', etc.)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TICTACTOE_DIR = os.path.join(BASE_DIR, "games", "tic-tac-toe")
TARGET_SHOOTER_DIR = os.path.join(BASE_DIR, "games") # For target_shooter_game if it has local imports
MEMORY_DIR = os.path.join(BASE_DIR, "games") # For memory matching backend module path
if TICTACTOE_DIR not in sys.path:
    sys.path.insert(0, TICTACTOE_DIR)
if TARGET_SHOOTER_DIR not in sys.path: # Add games directory itself for broader imports
    sys.path.insert(0, TARGET_SHOOTER_DIR)
if MEMORY_DIR not in sys.path:
    sys.path.insert(0, MEMORY_DIR)


# Global application settings and hardware resources
app_settings = {
    "webcam_ip": "http://192.168.49.1:4747/video",  # Default Webcam IP
    "serial_config": {
        "type": "usb",  # "usb" or "wifi"
        "path": "COM7",  # For USB: e.g., COM3 (Windows), /dev/ttyUSB0 (Linux)
        "baudrate": 9600,  # For USB
        "host": "192.168.1.100",  # For WiFi
        "port": 8888,  # For WiFi
    }
}
serial_connection_instance = None

def initialize_serial_connection():
    global serial_connection_instance, app_settings
    if serial_connection_instance and serial_connection_instance.is_open:
        try:
            serial_connection_instance.close()
        except Exception as e:
            print(f"Error closing existing serial connection: {e}")
    serial_connection_instance = None

    config = app_settings["serial_config"]
    print(f"Attempting to initialize serial connection with config: {config}")
    try:
        if config["type"] == "usb":
            serial_connection_instance = serial.Serial(
                config["path"],
                config["baudrate"],
                timeout=1
            )
            print(f"USB Serial connection established to {config['path']} at {config['baudrate']}")
        elif config["type"] == "wifi":
            # PySerial can connect to socket URLs for TCP-to-Serial bridges
            socket_url = f"socket://{config['host']}:{config['port']}"
            serial_connection_instance = serial.serial_for_url(socket_url, timeout=1)
            print(f"WiFi Serial connection established to {config['host']}:{config['port']}")
        else:
            print(f"Unsupported serial type: {config['type']}")
            return False
        
        if serial_connection_instance.is_open:
            print("Serial connection successfully opened.")
            # Optional: Send a handshake or test command here if your devices expect one
            # For example, wait for Arduino to reset if it's a direct USB connection
            if config["type"] == "usb":
                import time
                time.sleep(2) # Wait for Arduino to reset
            return True
        else:
            print("Serial port opened but test failed (not is_open).")
            serial_connection_instance = None # Ensure it's None if not truly open
            return False

    except serial.SerialException as e:
        print(f"Serial Error ({config['type']}): {e}")
        serial_connection_instance = None
        return False
    except Exception as e:
        print(f"Unexpected error initializing serial ({config['type']}): {e}")
        serial_connection_instance = None
        return False

@app.on_event("startup")
async def startup_event():
    initialize_serial_connection()
    # Initialize Memory Matching's YOLO model path if it's used globally (optional, or handle in its GameSession)
    # memory_matching_backend.YOLO_MODEL_PATH is now a constant, can be referenced.
    # If memory_matching_backend needs its own startup actions (like loading global model),
    # they would need to be callable functions. For now, GameSession will handle its own model.

@app.get("/api/settings")
async def get_settings():
    return app_settings

@app.post("/api/settings")
async def update_settings(new_settings: dict):
    global app_settings
    # Basic validation/merging
    if "webcam_ip" in new_settings:
        app_settings["webcam_ip"] = new_settings["webcam_ip"]
    if "serial_config" in new_settings:
        # Ensure all required sub-keys are present or use defaults
        current_serial_config = app_settings["serial_config"]
        updated_serial_config = new_settings["serial_config"]
        
        app_settings["serial_config"]["type"] = updated_serial_config.get("type", current_serial_config["type"])
        app_settings["serial_config"]["path"] = updated_serial_config.get("path", current_serial_config["path"])
        # Ensure baudrate is int
        try:
            app_settings["serial_config"]["baudrate"] = int(updated_serial_config.get("baudrate", current_serial_config["baudrate"]))
        except ValueError:
            app_settings["serial_config"]["baudrate"] = current_serial_config["baudrate"] # fallback
            
        app_settings["serial_config"]["host"] = updated_serial_config.get("host", current_serial_config["host"])
         # Ensure port is int
        try:
            app_settings["serial_config"]["port"] = int(updated_serial_config.get("port", current_serial_config["port"]))
        except ValueError:
            app_settings["serial_config"]["port"] = current_serial_config["port"] # fallback

    print(f"Updated settings: {app_settings}")
    initialize_serial_connection()  # Re-initialize with new settings
    return {"message": "Settings updated", "current_settings": app_settings}

# Map game_id to module path
GAME_MODULES = {
    "shell-game": "games.shellGame",
    "tic-tac-toe": "games.tic-tac-toe.tictactoe",
    "rubiks": "games.rubiks_cube_game",
    "target-shooter": "games.target_shooter_game",
    "game-2": "games.game2",
    "game-3": "games.game3",
    "game-4": "games.game4",
    "game-5": "games.game5",
    "color": "games.memory_matching_backend", # Points to the module
    "yolo": "games.memory_matching_backend",  # Points to the module
}

@app.websocket("/ws/{game_id}")
async def websocket_endpoint(websocket: WebSocket, game_id: str):
    await websocket.accept()
    game_session = None
    module_path = GAME_MODULES.get(game_id)

    if not module_path:
        await websocket.send_json({"status": "error", "message": "Unknown game"})
        await websocket.close()
        return

    try:
        game_module = importlib.import_module(module_path)
        
        if game_id == "rubiks":
            # Wait for initial config message (optional)
            first_message = await websocket.receive()
            config = None
            first_frame_bytes = None
            if "text" in first_message:
                try: 
                    config = json.loads(first_message["text"])
                except Exception:
                    config = None
            elif "bytes" in first_message:
                first_frame_bytes = first_message["bytes"]
            if config is not None:
                game_session = RubiksCubeGame(config)
            else:
                game_session = RubiksCubeGame()
        elif game_id == "target-shooter":
            game_launch_config = {
                "serial_instance": serial_connection_instance,
                "webcam_ip": app_settings["webcam_ip"],
                "model_path": "games/TargetDetection/runs/detect/train/weights/best.pt" 
            }
            first_message_data = await websocket.receive() # Expects initial config
            client_initial_command_data = None
            if "text" in first_message_data:
                try:
                    parsed_text = json.loads(first_message_data["text"])
                    if isinstance(parsed_text, dict):
                        client_initial_command_data = parsed_text
                        game_launch_config.update(client_initial_command_data)
                except Exception as e:
                    await websocket.send_json({"status": "error", "message": f"Invalid initial JSON config: {str(e)}"})
                    await websocket.close(); return
            else:
                await websocket.send_json({"status": "error", "message": "Expected initial JSON config as text."})
                await websocket.close(); return

            try:
                game_session = game_module.GameSession(game_launch_config)
            except Exception as e:
                await websocket.send_json({"status": "error", "message": f"Error initializing game session: {str(e)}"})
                await websocket.close(); return
            
            if client_initial_command_data and client_initial_command_data.get("action") == "initial_config":
                 game_session.process_command(client_initial_command_data) # Allow GameSession to react

            await game_session.manage_game_loop(websocket)
            return # TargetShooter manages its own loop

        elif game_id == "color" or game_id == "yolo":
            # Construct launch config for Memory Matching
            # Note: memory_matching_backend.YOLO_MODEL_PATH is a constant in that module
            # yolo_model_relative_path = MemoryMatching.YOLO_MODEL_PATH 
            # yolo_model_abs_path = os.path.join(BASE_DIR, "games", yolo_model_relative_path.lstrip("./"))

            game_launch_config = {
                "serial_instance": serial_connection_instance,
                "webcam_ip": app_settings["webcam_ip"],
                "mode": game_id,  # "color" or "yolo"
                # "yolo_model_path": yolo_model_abs_path,
                # Pass serial port config in case MM needs to re-init (though ideally uses serial_instance directly)
                "serial_port_config": app_settings["serial_config"] 
            }
            
            # Memory Matching might not need/expect an initial client message for config.
            # If it did, you'd handle it here similar to target-shooter.
            # For now, assume it initializes with server-side config.

            try:
                game_session = game_module.MemoryMatching(game_launch_config)
                print(f"MemoryMatching ({game_id}): GameSession instantiated.")
            except Exception as e:
                print(f"MemoryMatching ({game_id}): Error instantiating GameSession: {e}")
                await websocket.send_json({"status": "error", "message": f"Error initializing game session: {str(e)}"})
                await websocket.close()
                return

            await game_session.manage_game_loop(websocket) # This method needs to exist in MM's GameSession
            return # MemoryMatching GameSession will manage its own loop and interactions

        else: # Default handling for other games (e.g., shell-game, tic-tac-toe)
            first_message = await websocket.receive()
            config = None
            first_frame_bytes = None
            if "text" in first_message:
                try: config = json.loads(first_message["text"])
                except Exception: config = None # Or treat as command
            elif "bytes" in first_message:
                first_frame_bytes = first_message["bytes"]
            
            if config is not None:
                game_session = game_module.GameSession(config)
            else:
                game_session = game_module.GameSession()
            
            # If first message was a frame, process it before entering loop (for non-MM, non-TS games)
            if first_frame_bytes is not None:
                try:
                    result = await maybe_await(game_session.process_frame, first_frame_bytes)
                    await websocket.send_json(result)
                except Exception as e:
                    await websocket.send_json({"status": "error", "message": f"Error processing initial frame: {str(e)}"})
                    await websocket.close(); return
    
    except Exception as e:
        await websocket.send_json({"status": "error", "message": f"Failed to load or initialize game: {str(e)}"})
        await websocket.close()
        return

    # General game loop for games that don't manage their own WebSocket interaction (e.g. Rubik's, simple frame processors)
    # TargetShooter and MemoryMatching (color/yolo) will have returned by now.
    try:
        while True:
            data = await websocket.receive()
            result = None
            try:
                if "bytes" in data: # Frame data
                    if hasattr(game_session, "process_frame"):
                        result = await maybe_await(game_session.process_frame, data["bytes"])
                    else: result = {"status": "error", "message": "Game does not support frame processing"}
                elif "text" in data: # Command data (JSON or simple string)
                    command_content = data["text"]
                    parsed_command = None
                    try: parsed_command = json.loads(command_content)
                    except json.JSONDecodeError: # Not JSON, could be simple string command
                        if hasattr(game_session, "process_command_string"): # Optional method for simple strings
                           result = await maybe_await(game_session.process_command_string, command_content)
                        else: # Default to trying process_command with the raw string
                           parsed_command = command_content 
                    
                    if result is None: # If not handled by process_command_string or if it was JSON
                        if hasattr(game_session, "process_command"):
                             # Special handling for Rubik's specific text commands if parsed_command is dict
                            if game_id == "rubiks" and isinstance(parsed_command, dict):
                                if "mode" in parsed_command:
                                    if parsed_command["mode"] == "calibrating": game_session.mode = "calibrating"; game_session.calibration_step = 0
                                    elif parsed_command["mode"] == "scanning": game_session.start_scanning()
                                    elif parsed_command["mode"] == "scrambling": game_session.scramble_cube()
                                    elif parsed_command["mode"] == "idle": game_session.stop()
                                elif "action" in parsed_command:
                                    if parsed_command["action"] == "calibrate": game_session.calibrate_color()
                                    elif parsed_command["action"] == "scan": game_session.capture_scan()
                                    elif parsed_command["action"] == "stop": game_session.stop()
                                result = game_session.get_state()
                            else: # General command processing
                                result = await maybe_await(game_session.process_command, parsed_command)
                        else: result = {"status": "error", "message": "Game does not support command processing"}
                else:
                    result = {"status": "error", "message": "Invalid message format"}
                
                if result: await websocket.send_json(result)

            except Exception as e:
                error_message = f"Error during game interaction: {str(e)}"
                print(error_message)
                try: await websocket.send_json({"status": "error", "message": error_message})
                except Exception: pass # Ignore if send fails (e.g., socket closed)
                # Depending on severity, you might close and break
                # For now, continue to allow more interactions unless WebSocketDisconnect is raised
    
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for game_id: {game_id}")
        if game_session:
            if hasattr(game_session, "stop") and callable(game_session.stop):
                try: game_session.stop()
                except Exception as e: print(f"Error calling stop for {game_id}: {e}")
            if hasattr(game_session, 'cleanup') and callable(game_session.cleanup):
                try: game_session.cleanup()
                except Exception as e: print(f"Error calling cleanup for {game_id}: {e}")
    finally:
        print(f"Closing WebSocket connection for game_id: {game_id}")
        # Ensure game session resources are released if not already by disconnect exception block
        if game_session and not isinstance(game_session, (type(None))): # Check if game_session was instantiated
            if hasattr(game_session, "stop") and callable(game_session.stop) and not WebSocketDisconnect: # if not already called
                try: game_session.stop()
                except Exception as e: print(f"Error calling stop in finally for {game_id}: {e}")
            if hasattr(game_session, 'cleanup') and callable(game_session.cleanup) and not WebSocketDisconnect: # if not already called
                try: game_session.cleanup()
                except Exception as e: print(f"Error calling cleanup in finally for {game_id}: {e}")


# Helper to support both sync and async process_frame/process_command
async def maybe_await(func, *args, **kwargs):
    res = func(*args, **kwargs)
    if inspect.isawaitable(res):
        return await res
    return res

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
