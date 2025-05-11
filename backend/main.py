from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, WebSocketException, status as http_status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import importlib
import sys
import os
import json
import inspect
import asyncio
from typing import Optional, Dict, Any

# Import memory matching backend for color/yolo WebSocket endpoints
from games.rubiks_cube_game import RubiksCubeGame
from fastapi.responses import StreamingResponse
# from games.target_shooter_game import GameSession 
from utils.esp32_client import esp32_client
from games.memory_matching_backend import MemoryMatching

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure backend/games/tic-tac-toe is in sys.path for dynamic imports (for 'utils', 'alphabeta', etc.)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TICTACTOE_DIR = os.path.join(BASE_DIR, "games", "tic-tac-toe")
TARGET_SHOOTER_DIR = os.path.join(BASE_DIR, "games")  # For target_shooter_game if it has local imports
MEMORY_DIR = os.path.join(BASE_DIR, "games")  # For memory matching backend module path
RUBIKS_DIR = os.path.join(BASE_DIR, "games")  # For rubiks_cube_game if it has local imports

if TICTACTOE_DIR not in sys.path:
    sys.path.insert(0, TICTACTOE_DIR)
if TARGET_SHOOTER_DIR not in sys.path:  # Add games directory itself for broader imports
    sys.path.insert(0, TARGET_SHOOTER_DIR)
if MEMORY_DIR not in sys.path:
    sys.path.insert(0, MEMORY_DIR)
if RUBIKS_DIR not in sys.path:
    sys.path.insert(0, RUBIKS_DIR)

# Map game_id to module path
GAME_MODULES = {
    "shell-game": "games.shellGame",
    "tic-tac-toe": "games.tic-tac-toe.tictactoe",
    "rubiks": "games.rubiks_cube_game",
    "target-shooter": "games.target_shooter_game",
    "color": "games.memory_matching_backend",  # Points to the module
    "yolo": "games.memory_matching_backend",  # Points to the module
}

# --- Global singleton for Target Shooter session (for demo/dev only) ---
target_shooter_session = None
shell_game_session = None

# Store memory matching game instances - these will be lazily instantiated
memory_game_instances = {
    "color": None,  # Will be MemoryMatching instance
    "yolo": None    # Will be MemoryMatching instance
}

# Track active game sessions by websocket
ACTIVE_GAME_SESSIONS: Dict[WebSocket, Any] = {}

@app.get("/stream/target-shooter")
async def stream_target_shooter(request: Request):
    if not target_shooter_session:
        # Optionally, you could auto-create a session here, but better to require WS first
        return StreamingResponse(
            iter([b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"]),  # empty stream
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    generator = target_shooter_session.get_stream_generator()
    return StreamingResponse(generator, media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/stream/shell-game")
async def stream_shell_game(request: Request):
    global shell_game_session
    if not shell_game_session:
        # Optionally, you could auto-create a session here, but better to require WS first
        return StreamingResponse(
            iter([b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"]),  # empty stream
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    generator = shell_game_session.get_stream_generator()
    return StreamingResponse(generator, media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/shell-game/debug")
async def shell_game_debug():
    global shell_game_session
    if not shell_game_session:
        return {"status": "error", "message": "Shell game not running"}
    return shell_game_session.get_latest_debug_state()

@app.websocket("/ws/{game_id}")
async def websocket_endpoint(websocket: WebSocket, game_id: str):
    global target_shooter_session, shell_game_session, memory_game_instances
    await websocket.accept()
    game_session = None
    module_path = GAME_MODULES.get(game_id)

    if not module_path:
        await websocket.send_json({"status": "error", "message": "Unknown game"})
        await websocket.close(code=http_status.WS_1003_UNSUPPORTED_DATA)
        return

    try:
        # Special handling for memory matching games (color/yolo)
        if game_id in ["color", "yolo"]:
            # Check if we need to create a new instance
            if memory_game_instances[game_id] is None:
                memory_game_instances[game_id] = MemoryMatching({"mode": game_id}, esp32_client=esp32_client)
            
            # Check if the game is already running
            if memory_game_instances[game_id].running:
                await websocket.send_json({
                    "type": "error", 
                    "payload": f"{game_id.capitalize()} game is busy. Please try again later."
                })
                await websocket.close()
                return
            
            # Set ESP32 client on websocket for the game
            setattr(websocket, "esp32_client", esp32_client)
            
            # Run the game - this will handle the entire lifecycle
            await memory_game_instances[game_id].run_game(websocket)
            return  # Important! Game handles its own lifecycle so we return here
        
        elif game_id == "rubiks":
            try:
                initial_config = None
                first_message = await websocket.receive()
                
                if "text" in first_message:
                    try: 
                        initial_config_text = first_message["text"]
                        if initial_config_text: # Ensure text is not empty
                           initial_config = json.loads(initial_config_text)
                           print(f"Rubik's initial config from client: {initial_config}")
                        else: 
                            print("Rubik's: Received empty text message for initial config.")
                            initial_config = {} # Default empty config
                    except json.JSONDecodeError:
                        print(f"! WARN: Failed to decode initial JSON for Rubik's, using defaults. Received: {first_message['text']}")
                        initial_config = {} # Default empty config on decode error
                    except Exception as e:
                        print(f"! ERROR processing initial Rubik's config: {e}")
                        initial_config = {}
                elif "bytes" in first_message:
                    # If first message is bytes, it's likely a frame, proceed with default config
                    print("Rubik's: First message was bytes (frame), using default config.")
                    initial_config = {} 
                
                # Pass the global esp32_client instance to RubiksCubeGame
                game_session = RubiksCubeGame(config=initial_config, esp32_client=esp32_client)
                game_session.set_websocket(websocket) # Pass WebSocket if game needs to send directly (optional)
                
                ACTIVE_GAME_SESSIONS[websocket] = game_session
                
                initial_state = game_session.get_state()
                await websocket.send_json(initial_state)
                print("Sent initial state to Rubik's client.")
                
                # If first message was bytes, process it now after setup
                if "bytes" in first_message and initial_config == {}: # Ensure it was a frame for default config
                    first_frame_bytes = first_message["bytes"]
                    print("Rubik's: Processing first frame received during handshake.")
                    # process_frame is now async
                    result = await game_session.process_frame(first_frame_bytes)
                    if result:
                        await websocket.send_json(result)

            except json.JSONDecodeError: # This might be redundant if handled above
                print("! ERROR: Failed to decode initial JSON config for Rubik's game (outer catch).")
                await websocket.close(code=http_status.WS_1008_POLICY_VIOLATION)
                return
            except Exception as e:
                print(f"! ERROR: Initializing Rubik's game session: {e}")
                await websocket.send_json({"status": "error", "message": f"Failed to initialize Rubik's game: {str(e)}"})
                await websocket.close(code=http_status.WS_1011_INTERNAL_ERROR)
                return
        
        else: # Original code for other game types
            game_module = importlib.import_module(module_path)
            
            if game_id == "shell-game":
                # Save singleton for streaming
                # Pass esp32_client to ShellGame
                shell_game_session = game_module.ShellGame(esp32_client=esp32_client)
                game_session = shell_game_session
                # Start livefeed background task for WS
                livefeed_task = asyncio.create_task(shell_game_session.send_livefeed_ws(websocket))
            else:
                first_message = await websocket.receive()
                config = None
                first_frame_bytes = None
                
                if "text" in first_message:
                    try: config = json.loads(first_message["text"])
                    except Exception: config = None 
                elif "bytes" in first_message:
                    first_frame_bytes = first_message["bytes"]
                
                # Ensure esp32_client is passed to GameSession if it accepts it
                game_session_args = []
                game_session_kwargs = {}
                if config is not None: game_session_args.append(config)
                
                # Check if GameSession constructor accepts esp32_client
                sig = inspect.signature(game_module.GameSession.__init__)
                if 'esp32_client' in sig.parameters:
                    game_session_kwargs['esp32_client'] = esp32_client

                try:
                    game_session = game_module.GameSession(*game_session_args, **game_session_kwargs)
                except TypeError as te: # Fallback if signature check was not exhaustive
                    print(f"Warning: TypeError instantiating {game_id}, trying without esp32_client if it was added: {te}")
                    if 'esp32_client' in game_session_kwargs:
                        del game_session_kwargs['esp32_client']
                        game_session = game_module.GameSession(*game_session_args, **game_session_kwargs)
                    else: # Original error was not due to esp32_client
                        raise te 
                        
                ACTIVE_GAME_SESSIONS[websocket] = game_session

        # Special handling for Target Shooter: store singleton for streaming
        if game_id == "target-shooter":
            target_shooter_session = game_session

    except Exception as e:
        await websocket.send_json({"status": "error", "message": f"Failed to load or initialize game: {str(e)}"})
        await websocket.close(code=http_status.WS_1011_INTERNAL_ERROR)
        return

    # General game loop for games that don't manage their own WebSocket interaction (e.g. Rubik's, simple frame processors)
    # TargetShooter and MemoryMatching (color/yolo) will have returned by now.
    try:
        while True:
            data = await websocket.receive()
            if "type" in data and data["type"] == "websocket.disconnect":
                print(f"Client initiated disconnect for {game_id}.")
                break
                
            result = None
            try:
                if game_id == "rubiks" and "text" in data and data["text"] is not None:
                    try:
                        command_data = json.loads(data["text"])
                        print(f"--> CMD from Rubik's client: {command_data}")
                        
                        if "mode" in command_data:
                            new_mode = command_data["mode"]
                            if game_session.mode in ["solving", "scrambling"] and new_mode not in ["idle", "error"]:
                                game_session.error_message = f"Busy ({game_session.mode}). Cannot change mode now."
                            else:
                                game_session.mode = new_mode
                                game_session.error_message = None
                                if new_mode == "calibrating":
                                    game_session.start_calibration_mode() # Use method to set up
                                elif new_mode == "scanning":
                                    game_session.start_scanning_mode() # Use method to set up
                                elif new_mode == "scrambling": # This should be an action
                                    await game_session.scramble_cube() # Await async call
                                elif new_mode == "idle":
                                    # game_session.stop() # stop_current_operation handles more
                                    await game_session.stop_current_operation() # Await async call
                                    game_session.status_message = "Mode: Idle."
                        
                        elif "action" in command_data:
                            action = command_data["action"]
                            if action == "calibrate_color" or action == "calibrate":
                                if game_session.mode == "calibrating":
                                    game_session.capture_calibration_color() # Sync method
                                else:
                                    game_session.error_message = "Not in calibration mode to capture color."
                            elif action == "scramble_cube":
                                if game_session.mode == "idle" or game_session.mode == "error": # Allow scramble from idle/error
                                    await game_session.scramble_cube() # Await async call
                                else:
                                    game_session.error_message = "Can only scramble from idle/error mode."
                            elif action == "stop_operation" or action == "stop":
                                await game_session.stop_current_operation() # Await async call
                            elif action == "scan": # Assuming "scan" implies capture one face if in a specific mode
                                # This action 'scan' might need more context or a specific method in RubiksCubeGame
                                # For now, let's assume it's related to a manual step if that's ever implemented
                                # Or perhaps it's meant to trigger a single scan capture, which is not typical for auto-scan
                                print(f"Rubik's: Received 'scan' action. Current logic is auto-scan. Action ignored or TBD.")
                                # if hasattr(game_session, 'capture_single_scan_face'):
                                #    await game_session.capture_single_scan_face()
                        
                        result = game_session.get_state()
                        
                    except json.JSONDecodeError:
                        print("! ERROR: Invalid JSON command received for Rubik's.")
                        result = {"error_message": "Invalid command (not JSON).", "mode": game_session.mode if game_session else "unknown"}
                    
                elif "bytes" in data:  # Frame data
                    if game_id == "rubiks":
                        # process_frame is now async, use direct await
                        result = await game_session.process_frame(data["bytes"])
                    elif hasattr(game_session, "process_frame"):
                        result = await maybe_await(game_session.process_frame, data["bytes"])
                    else:
                        result = {"status": "error", "message": "Game does not support frame processing"}
                
                elif "text" in data:  # Command data for other games
                    if game_id != "rubiks":
                        command_content = data["text"]
                        parsed_command = None
                        try: parsed_command = json.loads(command_content)
                        except json.JSONDecodeError:  # Not JSON, could be simple string command
                            if hasattr(game_session, "process_command_string"):  # Optional method for simple strings
                               result = await maybe_await(game_session.process_command_string, command_content)
                            else:  # Default to trying process_command with the raw string
                               parsed_command = command_content 
                        
                        if result is None:  # If not handled by process_command_string or if it was JSON
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
                                result = await maybe_await(game_session.process_command, parsed_command)
                            else:  # Game does not support command processing
                                result = {"status": "error", "message": "Game does not support command processing"}
                else:
                    result = {"status": "error", "message": "Invalid message format"}
                
                if result: await websocket.send_json(result)

            except WebSocketException as e:
                print(f"! WebSocketException for {game_id}: {e.code} - {e.reason}")
                break
            except Exception as e:
                error_message = f"Error during game interaction ({game_id}): {type(e).__name__} - {str(e)}"
                print(error_message)
                import traceback
                traceback.print_exc() # For more detailed server logs
                try: 
                    await websocket.send_json({"status": "error", "message": error_message})
                except Exception: 
                    pass 
            
            await asyncio.sleep(0.005)
    
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for game_id: {game_id}")
    except WebSocketException as e: # Redundant if caught inside loop, but good for general WS issues
        print(f"! WebSocketException for {game_id} (outer): {e.code} - {e.reason}")
    except Exception as e:
        print(f"! UNEXPECTED ERROR in WebSocket loop for {game_id}: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()
        if game_session:
            try:
                error_state_info = game_session.get_state() if hasattr(game_session, "get_state") else {}
                await websocket.send_json({**error_state_info, "status": "error", "message": f"Critical Server error: {e}"})
            except Exception:
                pass
    finally:
        if websocket in ACTIVE_GAME_SESSIONS:
            session_to_clean = ACTIVE_GAME_SESSIONS.pop(websocket)
        
        if game_session:
            # For Rubik's game, stop_current_operation is now async.
            # Other games might have sync 'stop' or 'cleanup'.
            if game_id == "rubiks" and hasattr(game_session, 'stop_current_operation') and callable(game_session.stop_current_operation):
                try: await game_session.stop_current_operation()
                except Exception as e: print(f"Error calling async stop_current_operation for {game_id}: {e}")
            elif hasattr(game_session, 'stop') and callable(game_session.stop):
                try: game_session.stop()
                except Exception as e: print(f"Error calling stop for {game_id}: {e}")
            if hasattr(game_session, 'cleanup') and callable(game_session.cleanup):
                try: game_session.cleanup() # Assuming cleanup is sync
                except Exception as e: print(f"Error calling cleanup for {game_id}: {e}")
        
        if game_id == "shell-game" and 'livefeed_task' in locals() and livefeed_task:
            try: livefeed_task.cancel()
            except Exception: pass
                
        print(f"Cleaned up game session for {game_id}")

# Helper to support both sync and async process_frame/process_command
async def maybe_await(func, *args, **kwargs):
    res = func(*args, **kwargs)
    if inspect.isawaitable(res):
        return await res
    return res

# --- ESP32 client connect on startup ---
@app.on_event("startup")
async def startup_event():
    # Try to connect to ESP32 on startup
    connected = await esp32_client.connect()
    if not connected:
        print("Warning: ESP32 client failed to connect on startup. Will retry on demand.")

if __name__ == "__main__":
    print("Starting Game Backend...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
