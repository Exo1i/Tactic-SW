# main.py (FastAPI backend)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, WebSocketException, status as http_status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sys
import os
import json
import asyncio
from typing import Optional, Dict, Any # For type hinting

# Ensure backend/games is in sys.path if RubiksCubeGame is in games/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GAMES_DIR = os.path.join(BASE_DIR, "games")
if GAMES_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR) 

from games.rubiks_cube_game import RubiksCubeGame 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ACTIVE_GAME_SESSIONS: Dict[WebSocket, RubiksCubeGame] = {}

@app.websocket("/ws/{game_id}")
async def websocket_endpoint(websocket: WebSocket, game_id: str):
    await websocket.accept()
    game_session: Optional[RubiksCubeGame] = None

    if game_id == "rubiks":
        try:
            initial_config_raw = await websocket.receive_text()
            initial_config = json.loads(initial_config_raw)
            print(f"Rubik's initial config from client: {initial_config}")

            game_session = RubiksCubeGame(initial_config)
            ACTIVE_GAME_SESSIONS[websocket] = game_session
            
            initial_state = game_session.get_state()
            # For the very first message, send a null frame. The game loop will send actual frames.
            initial_payload = {**initial_state, "processed_frame": None} 
            await websocket.send_json(initial_payload)
            print("Sent initial state to client.")

        except json.JSONDecodeError:
            print("! ERROR: Failed to decode initial JSON config for Rubik's game.")
            await websocket.close(code=http_status.WS_1008_POLICY_VIOLATION)
            return
        except WebSocketDisconnect:
            print("WebSocket disconnected during Rubik's game initialization.")
            return # Cleanup will be handled in finally
        except Exception as e:
            print(f"! ERROR: Initializing Rubik's game session: {e}")
            await websocket.close(code=http_status.WS_1011_INTERNAL_ERROR)
            return # Cleanup in finally
    else:
        print(f"! ERROR: Unknown game_id: {game_id}")
        await websocket.close(code=http_status.WS_1003_UNSUPPORTED_DATA)
        return

    # Main loop to receive messages (frames or commands)
    try:
        while True:
            message = await websocket.receive() 

            if message["type"] == "websocket.disconnect":
                print(f"Client initiated disconnect for {game_id}.")
                break 

            if game_id == "rubiks" and game_session: # Ensure game_session exists
                response_payload = None
                if "text" in message and message["text"] is not None:
                    try:
                        command_data = json.loads(message["text"])
                        print(f"--> CMD from client: {command_data}")
                        
                        # Handle specific commands to update game_session state
                        if "mode" in command_data:
                            new_mode = command_data["mode"]
                            if game_session.mode in ["solving", "scrambling"] and new_mode not in ["idle", "error"]:
                                game_session.error_message = f"Busy ({game_session.mode}). Cannot change mode now."
                            else:
                                game_session.mode = new_mode
                                game_session.error_message = None 
                                if game_session.mode == "calibrating":
                                    game_session.calibration_step = 0
                                    game_session.status_message = f"Calibrate: Show {game_session.COLOR_NAMES_CALIBRATION[0]}"
                                elif game_session.mode == "scanning":
                                    game_session.start_scanning_mode() 
                                elif game_session.mode == "idle":
                                    game_session.status_message = "Mode: Idle."
                        
                        elif "action" in command_data:
                            action = command_data["action"]
                            if action == "calibrate_color":
                                if game_session.mode == "calibrating":
                                    game_session.capture_calibration_color()
                                else: game_session.error_message = "Not in calibration mode to capture color."
                            elif action == "scramble_cube":
                                if game_session.mode == "idle":
                                     game_session.scramble_cube()
                                else: game_session.error_message = "Can only scramble from idle mode."
                            elif action == "stop_operation":
                                game_session.stop_current_operation()
                        
                        # After command, send updated state (no new frame from this path)
                        response_payload = game_session.get_state()
                        response_payload["processed_frame"] = None # Indicate no new visual data from command

                    except json.JSONDecodeError:
                        print("! ERROR: Invalid JSON command received.")
                        response_payload = {"error_message": "Invalid command (not JSON).", "mode": game_session.mode, "processed_frame": None}
                    except Exception as e:
                        print(f"! ERROR: Processing command: {e}")
                        game_session.error_message = str(e)
                        response_payload = game_session.get_state()
                        response_payload["processed_frame"] = None

                elif "bytes" in message and message["bytes"] is not None:
                    frame_data = message["bytes"]
                    # print(f"--> FRAME from client, len: {len(frame_data)}") # DEBUG: Usually too verbose
                    response_payload = game_session.process_frame(frame_data) 
                
                if response_payload:
                    # print(f"<-- RSP to client: mode={response_payload.get('mode')}, frame_present={response_payload.get('processed_frame') is not None}") # DEBUG
                    await websocket.send_json(response_payload)
            
            await asyncio.sleep(0.005) 

    except WebSocketDisconnect:
        print(f"WebSocket disconnected for {game_id} (during main loop).")
    except WebSocketException as e: 
        print(f"! WebSocketException for {game_id}: {e.code} - {e.reason}")
    except Exception as e:
        print(f"! UNEXPECTED ERROR in WebSocket loop for {game_id}: {type(e).__name__} - {e}")
        if game_session: # Try to inform client if connection is still partly alive
            try:
                error_state_info = game_session.get_state()
                await websocket.send_json({**error_state_info, "error_message": f"Critical Server error: {e}", "mode": "error", "processed_frame": None})
            except Exception as send_err:
                print(f"  (Could not send error to client: {send_err})")
    finally:
        if websocket in ACTIVE_GAME_SESSIONS:
            session_to_clean = ACTIVE_GAME_SESSIONS.pop(websocket)
            if hasattr(session_to_clean, 'cleanup'):
                session_to_clean.cleanup()
            print(f"Cleaned up game session for {game_id} on disconnect/error.")
        else:
            print(f"Session for game {game_id} already cleaned up or not found for this websocket.")

if __name__ == "__main__":
    print("Starting Rubik's Cube Solver Backend...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")