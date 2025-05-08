from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import importlib
import sys
import os
import json
import inspect

# Import memory matching backend for color/yolo WebSocket endpoints
from games import memory_matching_backend
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
if TICTACTOE_DIR not in sys.path:
    sys.path.insert(0, TICTACTOE_DIR)

# Map game_id to module path
GAME_MODULES = {
    "shell-game": "games.shellGame",
    "tic-tac-toe": "games.tic-tac-toe.tictactoe",
    "rubiks": "games.rubiks_cube_game",
    "game-2": "games.game2",   # Placeholder, implement games/game2.py
    "game-3": "games.game3",   # Placeholder, implement games/game3.py
    "game-4": "games.game4",   # Placeholder, implement games/game4.py
    "game-5": "games.game5",   # Placeholder, implement games/game5.py
    "memory-matching": "games.memory_matching_backend",
    "shooter-game": "games.Shooter.TargetDetection.temp",  # <-- Add this line
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
        # Special handling for Rubik's Cube
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
        else:
            # Default: try to receive config as first message, else treat as frame
            first_message = await websocket.receive()
            config = None
            first_frame_bytes = None
            if "text" in first_message:
                try:
                    config = json.loads(first_message["text"])
                    
                    # Special handling for shooter-game init message
                    if game_id == "shooter-game" and config.get("action") == "start":
                        print(f"Shooter Game: Received start command with config: {config}")
                        # The config will be passed to GameSession constructor
                        
                except Exception as e:
                    print(f"Error parsing config: {e}")
                    config = None
            elif "bytes" in first_message:
                first_frame_bytes = first_message["bytes"]
            
            if config is not None:
                game_session = game_module.GameSession(config)
            else:
                game_session = game_module.GameSession()
    except Exception as e:
        await websocket.send_json({"status": "error", "message": f"Failed to load game: {e}"})
        await websocket.close()
        return

    # Special handling for memory-matching: run full game logic
    if game_id == "memory-matching":
        await game_session.run_game(websocket)
        return

    try:
        # If first message was a frame, process it before entering loop
        if game_id == "rubiks" and 'first_frame_bytes' in locals() and first_frame_bytes is not None:
            result = game_session.process_frame(first_frame_bytes)
            await websocket.send_json(result)
        elif 'first_frame_bytes' in locals() and first_frame_bytes is not None:
            try:
                result = await maybe_await(game_session.process_frame, first_frame_bytes)
                await websocket.send_json(result)
            except Exception as e:
                # Optionally log the error
                await websocket.close()
                return  # <-- Add return here to prevent further sends
        while True:
            data = await websocket.receive()
            try:
                if "bytes" in data:
                    # For games that process video frames (like Rubik's cube)
                    if game_id == "rubiks":
                        result = game_session.process_frame(data["bytes"])
                    elif game_id == "shooter-game":
                        # Shooter-game returns raw bytes, convert to base64 for frontend
                        result = await maybe_await(game_session.process_frame, data["bytes"])
                        # Convert frame bytes to base64 for frontend compatibility
                        import base64
                        frame_b64 = None
                        if result.get("frame"):
                            frame_b64 = base64.b64encode(result["frame"]).decode("utf-8")
                        await websocket.send_json({
                            "status": result.get("status", "ok"),
                            "processed_frame": frame_b64,
                            "balloons_info": result.get("balloons_info", []),
                        })
                        continue  # Already sent, skip default send
                    else:
                        result = await maybe_await(game_session.process_frame, data["bytes"])
                elif "text" in data:
                    config = {}
                    try:
                        config = json.loads(data["text"])
                        
                        # Special handling for shooter-game commands
                        if game_id == "shooter-game":
                            if config.get("action") == "stop":
                                print(f"Shooter Game: Received stop command")
                                if hasattr(game_session, "stop"):
                                    game_session.stop()
                                await websocket.send_json({"status": "stopped"})
                                await websocket.close()
                                break
                            elif config.get("action") == "emergency":
                                print(f"Shooter Game: Received EMERGENCY command")
                                if hasattr(game_session, "stop"):
                                    game_session.stop()
                                await websocket.send_json({"status": "emergency_stopped"})
                                await websocket.close()
                                break
                                
                    except Exception as e:
                        print(f"Error parsing message: {e}")
                        pass
                        
                    if game_id == "rubiks":
                        # Handle Rubik's cube specific commands
                        if "mode" in config:
                            if config["mode"] == "calibrating":
                                game_session.mode = "calibrating"
                                game_session.calibration_step = 0
                            elif config["mode"] == "scanning":
                                game_session.start_scanning()
                            elif config["mode"] == "scrambling":
                                game_session.scramble_cube()
                            elif config["mode"] == "idle":
                                game_session.stop()
                        elif "action" in config:
                            if config["action"] == "calibrate":
                                game_session.calibrate_color()
                            elif config["action"] == "scan":
                                game_session.capture_scan()
                            elif config["action"] == "stop":
                                game_session.stop()
                        result = game_session.get_state()
                    else:
                        result = await maybe_await(game_session.process_command, config)
                else:
                    result = {"status": "error", "message": "Invalid message format"}
                await websocket.send_json(result)
            except Exception as e:
                # Only try to send error if websocket is open
                try:
                    await websocket.send_json({"status": "error", "message": str(e)})
                except Exception:
                    pass
                # Optionally close the websocket and break the loop
                try:
                    await websocket.close()
                except Exception:
                    pass
                break  # <-- Exit the receive loop after closing
    except WebSocketDisconnect:
        if game_session:
            # Release camera connection for shell-game and similar games
            if hasattr(game_session, "stop"):
                game_session.stop()
            # Optionally call cleanup if it exists
            if hasattr(game_session, 'cleanup'):
                game_session.cleanup()

# Helper to support both sync and async process_frame
async def maybe_await(func, *args, **kwargs):
    res = func(*args, **kwargs)
    if inspect.isawaitable(res):
        return await res
    return res

# Add endpoints for memory matching game (color and yolo)
@app.websocket("/ws/memory-matching/color")
async def ws_memory_matching_color(websocket: WebSocket):
    await memory_matching_backend.stream_game(websocket, "color")

@app.websocket("/ws/memory-matching/yolo")
async def ws_memory_matching_yolo(websocket: WebSocket):
    await memory_matching_backend.stream_game(websocket, "yolo")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
