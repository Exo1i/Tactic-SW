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
    "game-2": "games.game2",   # Placeholder, implement games/game2.py
    "game-3": "games.game3",   # Placeholder, implement games/game3.py
    "game-4": "games.game4",   # Placeholder, implement games/game4.py
    "game-5": "games.game5",   # Placeholder, implement games/game5.py
    "memory-matching": "games.memory_matching_backend",  # <-- Add this line
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
        # Try to receive a config message (JSON) as first message, else treat as frame
        first_message = await websocket.receive()
        config = None
        first_frame_bytes = None
        if first_message["type"] == "websocket.receive":
            if "text" in first_message:
                try:
                    config = json.loads(first_message["text"])
                except Exception:
                    config = None
            elif "bytes" in first_message:
                first_frame_bytes = first_message["bytes"]
        # Pass config to GameSession if present
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
        if first_frame_bytes is not None:
            result = await maybe_await(game_session.process_frame, first_frame_bytes)
            await websocket.send_json(result)
        while True:
            data = await websocket.receive_bytes()
            result = await maybe_await(game_session.process_frame, data)
            await websocket.send_json(result)
    except WebSocketDisconnect:
        pass

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
