from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import importlib
import sys
import os
import json

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
}

@app.websocket("/ws/{game_id}")
async def websocket_endpoint(websocket: WebSocket, game_id: str):
    await websocket.accept()
    game_session = None
    # Dynamically import the game module and create a session
    module_path = GAME_MODULES.get(game_id)
    if module_path:
        try:
            game_module = importlib.import_module(module_path)
            # For tic-tac-toe, allow passing arguments via query params or initial message
            if game_id == "tic-tac-toe":
                # Wait for initial config message
                config = await websocket.receive_json()
                # Example: {"model": "...", "zoom": 2.0, "check_interval": 5.0}
                game_session = game_module.GameSession(config)
            elif game_id == "rubiks":
                # For Rubik's cube, create a RubiksCubeGame instance
                config = await websocket.receive_json()
                game_session = game_module.RubiksCubeGame(config)
            else:
                game_session = game_module.GameSession()
        except Exception as e:
            await websocket.send_json({"status": "error", "message": f"Failed to load game: {e}"})
            await websocket.close()
            return
    else:
        await websocket.send_json({"status": "error", "message": "Unknown game"})
        await websocket.close()
        return

    try:
        while True:
            data = await websocket.receive()
            try:
                if "bytes" in data:
                    # For games that process video frames (like Rubik's cube)
                    result = game_session.process_frame(data["bytes"])
                elif "text" in data:
                    # For games that process text commands
                    config = {}
                    try:
                        config = json.loads(data["text"])
                    except:
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
                        result = game_session.process_command(config)
                else:
                    result = {"status": "error", "message": "Invalid message format"}
                
                await websocket.send_json(result)
            except Exception as e:
                result = {"status": "error", "message": str(e)}
                await websocket.send_json(result)
    except WebSocketDisconnect:
        if game_session and hasattr(game_session, 'cleanup'):
            game_session.cleanup()

# Add endpoints for memory matching game (color and yolo)
@app.websocket("/ws/memory-matching/color")
async def ws_memory_matching_color(websocket: WebSocket):
    await memory_matching_backend.stream_game(websocket, "color")

@app.websocket("/ws/memory-matching/yolo")
async def ws_memory_matching_yolo(websocket: WebSocket):
    await memory_matching_backend.stream_game(websocket, "yolo")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
