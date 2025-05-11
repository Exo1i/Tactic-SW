from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import importlib
import sys
import os
import json
import inspect
import asyncio
from starlette.websockets import WebSocketState
import cv2
import numpy as np

# Import memory matching backend for color/yolo WebSocket endpoints
from games.rubiks_cube_game import RubiksCubeGame
from fastapi.responses import StreamingResponse
from games.target_shooter_game import GameSession  # Import GameSession for streaming
from utils.esp32_client import esp32_client
from games.memory_matching_backend import MemoryMatching  # Import MemoryMatching game class

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
TARGET_SHOOTER_DIR = os.path.join(BASE_DIR, "games")  # For target_shooter_game if it has local imports
MEMORY_DIR = os.path.join(BASE_DIR, "games")  # For memory matching backend module path
if TICTACTOE_DIR not in sys.path:
    sys.path.insert(0, TICTACTOE_DIR)
if TARGET_SHOOTER_DIR not in sys.path:  # Add games directory itself for broader imports
    sys.path.insert(0, TARGET_SHOOTER_DIR)
if MEMORY_DIR not in sys.path:
    sys.path.insert(0, MEMORY_DIR)

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
shell_game_session = None  # Add global singleton for shell game
target_shooter_session = None

# Store memory matching game instances - these will be lazily instantiated
memory_game_instances = {
    "color": None,  # Will be MemoryMatching instance
    "yolo": None    # Will be MemoryMatching instance
}

@app.get("/stream/target-shooter")
async def stream_target_shooter(request: Request):
    global target_shooter_session
    
    print(f"Stream requested from {request.client.host if request.client else 'Unknown'}. Session exists: {target_shooter_session is not None}")
    
    if not target_shooter_session:
        print("No target shooter session available for streaming. Sending placeholder stream.")
        
        async def empty_stream_generator():
            boundary_str = "frame"
            boundary = b"--" + boundary_str.encode()
            boundary_end = b"--" + boundary_str.encode() + b"--\r\n"
            content_type_header = b"Content-Type: image/jpeg\r\n"
            crlf = b"\r\n"

            img = np.zeros((480, 640, 3), dtype=np.uint8) 
            cv2.putText(img, "No active game session", (100, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 100), 2)
            cv2.putText(img, "Start a game to see the feed.", (100, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            ret_val, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            
            jpg_bytes_empty = b''
            if ret_val:
                jpg_bytes_empty = buffer.tobytes()
            else:
                print("Empty stream generator: Failed to encode placeholder frame.")

            for _ in range(3): 
                yield boundary + crlf
                yield content_type_header
                yield f"Content-Length: {len(jpg_bytes_empty)}\r\n".encode()
                yield crlf
                yield jpg_bytes_empty
                yield crlf
                await asyncio.sleep(0.5) 
            yield boundary_end
        
        return StreamingResponse(
            empty_stream_generator(),
            media_type=f"multipart/x-mixed-replace; boundary=frame", # Boundary here must match the one used in generator
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
                "Connection": "close", # Close after sending placeholders
                "Access-Control-Allow-Origin": "*" 
            }
        )
    
    print(f"Returning target shooter stream generator for client {request.client.host if request.client else 'Unknown'}")
    generator = target_shooter_session.get_stream_generator()
    
    return StreamingResponse(
        generator, 
        media_type=f"multipart/x-mixed-replace; boundary=frame", # Boundary here must match
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "keep-alive", # Keep alive for the actual game stream
            "Access-Control-Allow-Origin": "*"
        }
    )

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
        await websocket.close()
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
        
        # Original code for other game types
        game_module = importlib.import_module(module_path)
        
        if game_id == "shell-game":
            # Save singleton for streaming
            # Pass esp32_client to ShellGame
            shell_game_session = game_module.ShellGame(esp32_client=esp32_client)
            game_session = shell_game_session
            # Start livefeed background task for WS
            livefeed_task = asyncio.create_task(shell_game_session.send_livefeed_ws(websocket))
        elif game_id == "rubiks":
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
                # Pass ESP32 client to the game
                game_session = RubiksCubeGame(config, esp32_client=esp32_client)
            else:
                game_session = RubiksCubeGame(esp32_client=esp32_client)
        else:
            # Default: try to receive config as first message, else treat as frame
            first_message = await websocket.receive()
            config = None
            first_frame_bytes = None
            if "text" in first_message:
                try: config = json.loads(first_message["text"])
                except Exception: config = None  # Or treat as command
            elif "bytes" in first_message:
                first_frame_bytes = first_message["bytes"]
            
            if config is not None:
                # Pass ESP32 client to the game if the class accepts it
                try:
                    game_session = game_module.GameSession(config, esp32_client=esp32_client)
                except TypeError:
                    game_session = game_module.GameSession(config)
            else:
                try:
                    game_session = game_module.GameSession(esp32_client=esp32_client)
                except TypeError:
                    game_session = game_module.GameSession()

        # Special handling for Target Shooter: store singleton for streaming
        if game_id == "target-shooter":
            # First store the session globally so streaming endpoint can access it
            print(f"Setting global target_shooter_session before game loop")
            target_shooter_session = game_session
            
            # Then start the game loop
            if hasattr(game_session, "manage_game_loop"):
                print(f"Starting manage_game_loop for {game_id}")
                try:
                    await game_session.manage_game_loop(websocket)
                    print(f"Game loop ended normally for {game_id}")
                    return
                except Exception as e:
                    print(f"Error in game loop for {game_id}: {e}")

    except Exception as e:
        await websocket.send_json({"status": "error", "message": f"Failed to load or initialize game: {str(e)}"})
        await websocket.close()
        return

    # General game loop for games that don't manage their own WebSocket interaction (e.g. Rubik's, simple frame processors)
    # TargetShooter and MemoryMatching (color/yolo) will have returned by now.
    try:
        while True:
            if websocket.application_state == WebSocketState.DISCONNECTED:
                print(f"WebSocket disconnected for game_id: {game_id}")
                break

            data = await websocket.receive()
            result = None
            try:
                if "bytes" in data:  # Frame data
                    if hasattr(game_session, "process_frame"):
                        result = await maybe_await(game_session.process_frame, data["bytes"])
                    else: result = {"status": "error", "message": "Game does not support frame processing"}
                elif "text" in data:  # Command data (JSON or simple string)
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
                                result = game_session.get_state()
                            else:  # General command processing
                                result = await maybe_await(game_session.process_command, parsed_command)
                        else: result = {"status": "error", "message": "Game does not support command processing"}
                else:
                    result = {"status": "error", "message": "Invalid message format"}
                
                if result: await websocket.send_json(result)

            except Exception as e:
                error_message = f"Error during game interaction: {str(e)}"
                print(error_message)
                try: await websocket.send_json({"status": "error", "message": error_message})
                except Exception: pass  # Ignore if send fails (e.g., socket closed)
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
        # Cancel livefeed task if running
        if game_id == "shell-game":
            try:
                livefeed_task.cancel()
            except Exception:
                pass
    finally:
        print(f"Closing WebSocket connection for game_id: {game_id}")
        # Ensure game session resources are released if not already by disconnect exception block
        if game_session and not isinstance(game_session, (type(None))):  # Check if game_session was instantiated
            if hasattr(game_session, "stop") and callable(game_session.stop) and not WebSocketDisconnect:  # if not already called
                try: game_session.stop()
                except Exception as e: print(f"Error calling stop in finally for {game_id}: {e}")
            if hasattr(game_session, 'cleanup') and callable(game_session.cleanup) and not WebSocketDisconnect:  # if not already called
                try: game_session.cleanup()
                except Exception as e: print(f"Error calling cleanup in finally for {game_id}: {e}")
        if game_id == "shell-game":
            try:
                livefeed_task.cancel()
            except Exception:
                pass

# Helper to support both sync and async process_frame/process_command
async def maybe_await(func, *args, **kwargs):
    res = func(*args, **kwargs)
    if inspect.isawaitable(res):
        return await res
    return res

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
