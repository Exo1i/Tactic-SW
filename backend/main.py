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
        # Create and return an empty stream indicator
        return StreamingResponse(
            iter([b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"]),  # empty stream
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    
    # Get a fresh generator instance for continuous streaming
    generator = shell_game_session.get_stream_generator()
    
    # Use StreamingResponse with headers that encourage continuous streaming
    return StreamingResponse(
        generator,
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        }
    )

@app.get("/shell-game/debug")
async def shell_game_debug():
    global shell_game_session
    if not shell_game_session:
        return {"status": "error", "message": "Shell game not running"}
    return shell_game_session.get_latest_debug_state()

@app.websocket("/ws/{game_id}")
async def websocket_endpoint(websocket: WebSocket, game_id: str):
    """
    Main WebSocket endpoint handling all game communications.
    
    Games that handle their own WebSocket loop (early return):
    - Memory matching games ('color'/'yolo')
    
    Games managed by this function's WebSocket loop:
    - Rubik's Cube game ('rubiks')
    - Shell Game ('shell-game')
    - Tic-Tac-Toe ('tic-tac-toe')
    - Target Shooter ('target-shooter')
    """
    global target_shooter_session, shell_game_session, memory_game_instances
    await websocket.accept()
    game_session = None
    module_path = GAME_MODULES.get(game_id)

    if not module_path:
        await websocket.send_json({"status": "error", "message": "Unknown game"})
        await websocket.close(code=http_status.WS_1003_UNSUPPORTED_DATA)
        return

    try:
        # --- GAME INITIALIZATION LOGIC ---
        
        # Memory matching games (color/yolo) - handle their own WebSocket loop
        if game_id in ["color", "yolo"]:
            if memory_game_instances[game_id] is None:
                memory_game_instances[game_id] = MemoryMatching({"mode": game_id}, esp32_client=esp32_client)
            
            if memory_game_instances[game_id].running:
                await websocket.send_json({
                    "type": "error", 
                    "payload": f"{game_id.capitalize()} game is busy. Please try again later."
                })
                await websocket.close()
                return
            
            setattr(websocket, "esp32_client", esp32_client)
            await memory_game_instances[game_id].run_game(websocket)
            return  # Early return - game handles its own WebSocket loop
        
        # Rubik's cube game
        elif game_id == "rubiks":
            try:
                initial_config = None
                first_message = await websocket.receive()
                
                if "text" in first_message:
                    try: 
                        initial_config_text = first_message["text"]
                        if initial_config_text:
                           initial_config = json.loads(initial_config_text)
                           print(f"Rubik's initial config from client: {initial_config}")
                        else: 
                            initial_config = {}
                    except json.JSONDecodeError:
                        print(f"! WARN: Failed to decode initial JSON for Rubik's. Received: {first_message['text']}")
                        initial_config = {}
                    except Exception as e:
                        print(f"! ERROR processing initial Rubik's config: {e}")
                        initial_config = {}
                elif "bytes" in first_message:
                    print("Rubik's: First message was bytes (frame), using default config.")
                    initial_config = {} 
                
                game_session = RubiksCubeGame(config=initial_config, esp32_client=esp32_client)
                game_session.set_websocket(websocket)
                
                ACTIVE_GAME_SESSIONS[websocket] = game_session
                
                await websocket.send_json(game_session.get_state())
                print("Sent initial state to Rubik's client.")
                
                if "bytes" in first_message and initial_config == {}:
                    first_frame_bytes = first_message["bytes"]
                    result = await game_session.process_frame(first_frame_bytes)
                    if result:
                        await websocket.send_json(result)
            except json.JSONDecodeError:
                print("! ERROR: Failed to decode initial JSON config for Rubik's game.")
                await websocket.close(code=http_status.WS_1008_POLICY_VIOLATION)
                return
            except Exception as e:
                print(f"! ERROR: Initializing Rubik's game session: {e}")
                await websocket.send_json({"status": "error", "message": f"Failed to initialize Rubik's game: {str(e)}"})
                await websocket.close(code=http_status.WS_1011_INTERNAL_ERROR)
                return
        
        # Shell game
        elif game_id == "shell-game":
            game_module = importlib.import_module(module_path)
            shell_game_session = game_module.ShellGame(esp32_client=esp32_client)
            game_session = shell_game_session
            ACTIVE_GAME_SESSIONS[websocket] = game_session
            # Start livefeed task and store in the session object
            shell_game_session.livefeed_task = asyncio.create_task(shell_game_session.send_livefeed_ws(websocket))
            
            # Send initial status to the client
            await websocket.send_json({"status": "connected", "message": "Shell game started"})
        
        # Tic-tac-toe game
        elif game_id == "tic-tac-toe":
            game_module = importlib.import_module(module_path)
            first_message = await websocket.receive()
            config = None
            
            if "text" in first_message:
                try: config = json.loads(first_message["text"])
                except Exception: config = None 
            
            # Initialize game with or without config and ESP32 client
            try:
                if config:
                    game_session = game_module.GameSession(config, esp32_client=esp32_client)
                else:
                    game_session = game_module.GameSession(esp32_client=esp32_client)
            except TypeError:
                # Fallback if ESP32 client isn't accepted
                if config:
                    game_session = game_module.GameSession(config)
                else:
                    game_session = game_module.GameSession()
            
            ACTIVE_GAME_SESSIONS[websocket] = game_session
        
        # Target shooter game
        elif game_id == "target-shooter":
            game_module = importlib.import_module(module_path)
            first_message = await websocket.receive()
            config = None
            
            if "text" in first_message:
                try: config = json.loads(first_message["text"])
                except Exception: config = None 
            
            # Initialize with ESP32 client if supported
            try:
                if config:
                    game_session = game_module.GameSession(config, esp32_client=esp32_client)
                else:
                    game_session = game_module.GameSession(esp32_client=esp32_client)
            except TypeError:
                if config:
                    game_session = game_module.GameSession(config)
                else:
                    game_session = game_module.GameSession()
                    
            ACTIVE_GAME_SESSIONS[websocket] = game_session
            target_shooter_session = game_session
        
        # Unknown game type
        else:
            await websocket.send_json({"status": "error", "message": f"Game type '{game_id}' initialization not implemented"})
            await websocket.close(code=http_status.WS_1003_UNSUPPORTED_DATA)
            return

    except Exception as e:
        await websocket.send_json({"status": "error", "message": f"Failed to load or initialize game: {str(e)}"})
        await websocket.close(code=http_status.WS_1011_INTERNAL_ERROR)
        return

    # --- MAIN WEBSOCKET LOOP ---
    # This loop handles communication for games that don't manage their own WebSocket interaction
    try:
        # For games like shell-game, this loop needs to continue running
        while True:
            # Handle shell game differently - no need to wait for client frames
            if game_id == "shell-game" and shell_game_session:
                # Process a frame automatically without waiting for client input
                try:
                    # Check if WebSocket is still connected before processing
                    if websocket.client_state.name != "CONNECTED":
                        print(f"WebSocket state is {websocket.client_state.name}, breaking from shell game loop")
                        break
                        
                    result = await maybe_await(shell_game_session.process_frame, None)
                    
                    # Double-check connection state before sending (connection may have closed during processing)
                    if result and websocket.client_state.name == "CONNECTED":
                        try:
                            await websocket.send_json(result)
                        except RuntimeError as re:
                            if "websocket connection is closed" in str(re).lower():
                                print("WebSocket closed during send, breaking from shell game loop")
                                break
                            raise  # Re-raise if it's a different RuntimeError")
                    
                    # Add small delay to control processing rate
                    await asyncio.sleep(0.05)  # 20 FPS
                    
                except WebSocketDisconnect:
                    print("Shell game WebSocket disconnected gracefully")
                    break
                except Exception as e:
                    error_str = str(e).lower()
                    # Handle specific error codes and connection issues
                    if "1005" in error_str or "1000" in error_str or "1001" in error_str or "1012" in error_str :
                        print(f"Shell game WebSocket protocol error: {e} - closing connection")
                        break
                    elif any(msg in error_str for msg in ["connection", "closed", "disconnect"]):
                        print(f"Connection appears to be closed: {e}")
                        break
                    
                    # For other errors, log and continue with a delay
                    print(f"Error in shell game frame processing: {e}")
                    await asyncio.sleep(0.5)  # Shorter delay on error for responsiveness
                    
                continue  # Skip to next iteration - don't try to receive from client
            
            # For other games, receive message from client
            data = await websocket.receive()
            
            # Process frames for games that handle them separately (not memory-matching)
            if game_id == "shell-game" and shell_game_session:
                if "bytes" in data:
                    # Process frame from client
                    result = await maybe_await(shell_game_session.process_frame, data["bytes"])
                    await websocket.send_json(result)
                elif "text" in data:
                    # Process commands from client
                    command_data = json.loads(data["text"])
                    if "action" in command_data:
                        result = await maybe_await(shell_game_session.process_command, command_data)
                        await websocket.send_json(result)
            
            # Handle Rubik's specific text commands
            elif game_id == "rubiks" and "text" in data and data["text"] is not None:
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
                                game_session.start_calibration_mode()
                            elif new_mode == "scanning":
                                game_session.start_scanning_mode()
                            elif new_mode == "scrambling":
                                await game_session.scramble_cube()
                            elif new_mode == "idle":
                                await game_session.stop_current_operation()
                                game_session.status_message = "Mode: Idle."
                    
                    elif "action" in command_data:
                        action = command_data["action"]
                        if action == "calibrate_color" or action == "calibrate":
                            if game_session.mode == "calibrating":
                                game_session.capture_calibration_color()
                            else:
                                game_session.error_message = "Not in calibration mode to capture color."
                        elif action == "scramble_cube":
                            if game_session.mode == "idle" or game_session.mode == "error":
                                await game_session.scramble_cube()
                            else:
                                game_session.error_message = "Can only scramble from idle/error mode."
                        elif action == "stop_operation" or action == "stop":
                            await game_session.stop_current_operation()
                        elif action == "scan":
                            print(f"Rubik's: Received 'scan' action. Current logic is auto-scan. Action ignored.")
                    
                    result = game_session.get_state()
                    
                except json.JSONDecodeError:
                    print("! ERROR: Invalid JSON command received for Rubik's.")
                    result = {"error_message": "Invalid command (not JSON).", "mode": game_session.mode if game_session else "unknown"}
                
            # Handle frame data (video frames)
            elif "bytes" in data:
                if game_id == "rubiks":
                    result = await game_session.process_frame(data["bytes"])
                elif hasattr(game_session, "process_frame"):
                    result = await maybe_await(game_session.process_frame, data["bytes"])
                else:
                    result = {"status": "error", "message": "Game does not support frame processing"}
            
            # Handle text commands for non-Rubik's games
            elif "text" in data and game_id != "rubiks":
                command_content = data["text"]
                parsed_command = None
                try: 
                    parsed_command = json.loads(command_content)
                except json.JSONDecodeError:
                    if hasattr(game_session, "process_command_string"):
                       result = await maybe_await(game_session.process_command_string, command_content)
                    else:
                       parsed_command = command_content 
                
                if result is None:
                    if hasattr(game_session, "process_command"):
                        result = await maybe_await(game_session.process_command, parsed_command)
                    else:
                        result = {"status": "error", "message": "Game does not support command processing"}
            else:
                result = {"status": "error", "message": "Invalid message format"}
            
            if result: 
                await websocket.send_json(result)

            # Small sleep to prevent CPU hogging
            await asyncio.sleep(0.005)
    
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for game_id: {game_id}")
    except WebSocketException as e:
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
        # --- CLEANUP SECTION ---
        # Remove from active sessions
        if websocket in ACTIVE_GAME_SESSIONS:
            ACTIVE_GAME_SESSIONS.pop(websocket)
        
        # Clean up global game sessions if this was the active one
        if game_id == "target-shooter" and target_shooter_session is game_session:
            target_shooter_session = None
            print("Cleared global target_shooter_session")
            
        if game_id == "shell-game" and shell_game_session is game_session:
            shell_game_session = None
            print("Cleared global shell_game_session")
            
        if game_id in ["color", "yolo"] and memory_game_instances.get(game_id) is not None:
            if hasattr(memory_game_instances[game_id], "running") and memory_game_instances[game_id].running:
                try:
                    memory_game_instances[game_id].running = False
                    print(f"Marked memory game {game_id} as not running")
                except Exception as e:
                    print(f"Error stopping memory game {game_id}: {e}")
        
        if game_session:
            # Rubik's game has async cleanup
            if game_id == "rubiks" and hasattr(game_session, 'stop_current_operation'):
                try: 
                    await game_session.stop_current_operation()
                except Exception as e: 
                    print(f"Error stopping Rubik's game: {e}")
            # Other games have sync stop methods
            elif hasattr(game_session, 'stop'):
                try: 
                    game_session.stop()
                except Exception as e: 
                    print(f"Error stopping game {game_id}: {e}")
            
            # General cleanup method if available
            if hasattr(game_session, 'cleanup'):
                try: 
                    game_session.cleanup()
                except Exception as e: 
                    print(f"Error cleaning up game {game_id}: {e}")
        
        # Clean up shell game livefeed task
        if game_id == "shell-game" and shell_game_session and hasattr(shell_game_session, 'livefeed_task'):
            try:
                shell_game_session.livefeed_task.cancel()
                await asyncio.sleep(0.1)  # Short delay to allow task cancellation to take effect
            except Exception as e:
                print(f"Error cancelling livefeed task: {e}")
                
        # Close the websocket if it's still open
        try:
            if not websocket.client_state.name.startswith("DISCONNECTED"):
                await websocket.close()
        except Exception:
            pass
                
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
