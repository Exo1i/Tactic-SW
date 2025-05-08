# File: main.py (Modified)
from typing import Dict, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import json
import logging # For better logging

# Assuming the new game logic is in games/rubiks_cube_game_reimplemented.py
from games.rubiks_cube_game import RubiksCubeGame  # <-- USE THE NEW CLASS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

# --- Global Game Session Holder ---
# This is a simple way for HTTP routes to access the WebSocket's game session.
# For multiple concurrent users, this needs a more robust session management system.
# For a single-user scenario (one browser tab interacting), this can work.
rubiks_game_session_holder: Dict[str, Optional[RubiksCubeGame]] = {"session": None}


@app.websocket("/ws/rubiks") # Only one endpoint for Rubik's for now
async def websocket_rubiks_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted for Rubik's Cube.")
    
    game_session = RubiksCubeGame() # Instantiate our new game class
    rubiks_game_session_holder["session"] = game_session # Store for HTTP routes

    try:
        # Send initial state
        await websocket.send_json(game_session.get_state())

        while True:
            data = await websocket.receive()
            if "bytes" in data:
                frame_bytes = data["bytes"]
                # logger.debug(f"Received frame bytes, size: {len(frame_bytes)}")
                result_state = game_session.process_frame(frame_bytes)
                await websocket.send_json(result_state)
            elif "text" in data:
                # Text data from WebSocket is not used by current frontend for Rubik's commands
                # Commands come via HTTP. If you plan to send commands via WS text:
                try:
                    command_data = json.loads(data["text"])
                    logger.info(f"Received text command via WebSocket: {command_data}")
                    # Example: handle_ws_command(game_session, command_data)
                    # For now, just send back current state if WS text is received.
                    await websocket.send_json(game_session.get_state())
                except json.JSONDecodeError:
                    logger.warning("Received invalid JSON text via WebSocket.")
                    await websocket.send_json({"error": "Invalid JSON command"})
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for Rubik's Cube.")
    except Exception as e:
        logger.error(f"Error in Rubik's WebSocket handler: {e}", exc_info=True)
        try: # Attempt to send error to client before closing
            await websocket.send_json({"mode": "error", "error_message": str(e)})
        except: pass # If sending fails, nothing more to do
    finally:
        if game_session:
            game_session.cleanup()
        rubiks_game_session_holder["session"] = None # Clear the session
        logger.info("Rubik's Cube game session cleaned up.")

# --- HTTP Endpoints for Commands (as per frontend) ---
# These need to access the game_session created by the WebSocket.

def get_active_game_session() -> RubiksCubeGame:
    session = rubiks_game_session_holder.get("session")
    if not session:
        raise HTTPException(status_code=409, detail="No active Rubik's Cube game session. Connect via WebSocket first.")
    return session

@app.post("/start_calibration")
async def api_start_calibration():
    game = get_active_game_session()
    game.start_calibration()
    return game.get_state() # Return new state

@app.post("/capture_calibration_color")
async def api_capture_color():
    game = get_active_game_session()
    game.capture_calibration_color()
    return game.get_state()

@app.post("/save_calibration")
async def api_save_calibration():
    game = get_active_game_session()
    game.save_calibration()
    return game.get_state()

@app.post("/reset_calibration")
async def api_reset_calibration():
    game = get_active_game_session()
    game.reset_calibration()
    return game.get_state()

@app.post("/start_solve") # Frontend uses this to initiate scanning then solving
async def api_start_solve():
    game = get_active_game_session()
    game.start_solve() # This now sets mode to "scanning"
    return game.get_state()

@app.post("/stop_and_reset")
async def api_stop_reset():
    game = get_active_game_session()
    game.stop_and_reset()
    return game.get_state()

@app.post("/start_scramble")
async def api_start_scramble():
    game = get_active_game_session()
    game.start_scramble()
    # Scrambling can take time, state might change during/after.
    # _send_arduino_command might set mode to idle after scramble.
    return game.get_state() # Return state immediately after initiating

# --- (Your other game endpoints like memory-matching can remain if needed) ---
# from games import memory_matching_backend
# @app.websocket("/ws/memory-matching/color") ...
# @app.websocket("/ws/memory-matching/yolo") ...

if __name__ == "__main__":
    # Make sure 'games' directory is in PYTHONPATH or accessible
    # For Uvicorn: uvicorn main:app --reload --host 0.0.0.0 --port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)