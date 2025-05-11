import asyncio
import base64
import json
import logging
import random
# import threading # No longer needed for serial_lock
import time
from typing import Dict, Any, Optional, List, Tuple
import os # Added for path checking

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from starlette.websockets import WebSocketState
from starlette.exceptions import HTTPException # Added for catch_all

# Import ESP32 WebSocket Client - This makes the global esp32_client available
from utils.esp32_client import esp32_client as global_esp32_client_instance

# --- Configuration ---
# SERIAL_PORT and BAUD_RATE are no longer needed
CAMERA_URL = 'http://192.168.2.19:4747/video' # Primary Camera URL

YOLO_MODEL_PATH = "./yolov5s.pt" # Or your specific model path

# --- Constants ---
GRID_ROWS = 2
GRID_COLS = 4
CARD_COUNT = GRID_ROWS * GRID_COLS
FLIP_DELAY_SECONDS = 0.5 # Delay in game logic loops

# --- Arm Control Values ---
arm_values = [[110, 40, 125], [87, 65, 120], [87, 110, 120], [110, 140, 125],
              [150, 55, 155], [130, 80, 140], [130, 105, 140], [150, 125, 155]]
arm_home = [180, 90, 0]
arm_temp1 = [90, 10, 120]
arm_temp2 = [90, 170, 120]
arm_trash = [140, 0, 140]

# --- Arm Sync Operation Constants ---
ARM_SYNC_STEP_DELAY = 0.3 # Reduced sleep between successful steps

# --- Retry Constants ---
ARM_MAX_RETRIES = 10 # Max attempts for a single arm command before failing sequence
ARM_RETRY_DELAY_SECONDS = 1.0 # Wait time between arm command retries
DETECTION_MAX_RETRIES = 1000 # Max attempts for detection before failing detection step
DETECTION_RETRY_DELAY_SECONDS = 0.75 # Wait time between detection retries
DETECTION_PERMANENT_FAIL_STATE = "PERMA_FAIL" # Special state if detection *really* fails

# --- YOLO Specific Constants ---
YOLO_TARGET_LABELS = ['orange', 'apple', 'cat', 'car', 'umbrella', 'banana', 'fire hydrant', 'person']
YOLO_FRAME_WIDTH = 640
YOLO_FRAME_HEIGHT = 480

# --- Board/Color Detection Specific Constants ---
COLOR_DEFINITIONS = { "red": (0, 0, 255), "yellow": (0, 255, 255), "green": (0, 255, 0), "blue": (255, 0, 0) }
COLOR_RANGES = [
    {'name': 'red', 'bgr': (0, 0, 255), 'lower': [(0, 120, 70), (170, 120, 70)], 'upper': [(10, 255, 255), (179, 255, 255)]},
    {'name': 'yellow', 'bgr': (0, 255, 255), 'lower': [(20, 100, 100)], 'upper': [(35, 255, 255)]},
    {'name': 'green', 'bgr': (0, 255, 0), 'lower': [(40, 40, 40)], 'upper': [(85, 255, 255)]},
    {'name': 'blue', 'bgr': (255, 0, 0), 'lower': [(95, 100, 50)], 'upper': [(130, 255, 255)]},
    {'name': 'black', 'bgr': (0, 0, 0), 'lower': [(0, 0, 0)], 'upper': [(180, 255, 50)]}
]
COLOR_CELL_THRESHOLD = 500
COLOR_BOARD_DETECT_WIDTH = 400
COLOR_BOARD_DETECT_HEIGHT = 200

# --- Globals ---
# Removed serial object and lock
latest_frame: Optional[np.ndarray] = None
latest_transformed_frame: Optional[np.ndarray] = None
frame_lock = asyncio.Lock()
active_games: Dict[str, Dict[str, Any]] = {"color": {}, "yolo": {}}
game_locks: Dict[str, asyncio.Lock] = { "color": asyncio.Lock(), "yolo": asyncio.Lock() }
yolo_model_global = None

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- ESP32 Command Functions ---
async def _send_switch_command(esp32_ws_client): # Renamed param for clarity
    """Send a switch command to ESP32 to activate ARM mode"""
    if esp32_ws_client is None:
        logging.error("[ESP32] No ESP32 client available, skipping switch command")
        return False
        
    try:
        logging.info("[ESP32] Sending switch command to activate ARM mode")
        # Assuming esp32_ws_client is an instance of ESP32Client from esp32_client.py
        await esp32_ws_client.send_json({
            "action": "switch",
            "game": "ARM" # Or a game-specific identifier if needed by ESP32
        })
        return True
    except Exception as e:
        logging.error(f"[ESP32] Error sending switch command: {e}")
        return False

async def send_arm_command_async(esp32_ws_client, degree1: int, degree2: int, degree3: int, magnet: int, movement: int) -> bool:
    """
    Asynchronous function to send an arm command via ESP32 WebSocket.
    Includes retries on failure. Returns True on success, False on failure.
    """
    if esp32_ws_client is None:
        logging.error("[ESP32] No ESP32 client available, skipping command")
        return False
        
    if not (0 <= degree1 <= 180 and 0 <= degree2 <= 180 and 0 <= degree3 <= 180):
        logging.error(f"[ESP32] Invalid servo degrees: ({degree1}, {degree2}, {degree3}). Must be 0-180.")
        return False
    if magnet not in [0, 1]:
        logging.error(f"[ESP32] Invalid magnet value: {magnet}. Must be 0 or 1.")
        return False

    command = f"{degree1},{degree2},{degree3},{magnet},{movement}"
    command_strip = command.strip()

    # ARM_MAX_RETRIES and ARM_RETRY_DELAY_SECONDS are already global constants

    # attempt = 0
    # while attempt < ARM_MAX_RETRIES:
    #     attempt += 1
    #     logging.info(f"[ESP32] Sending command (Attempt {attempt}/{ARM_MAX_RETRIES}): {command_strip}")

    #     try:
    #         # Assuming esp32_ws_client is an instance of ESP32Client from esp32_client.py
    #         success = await esp32_ws_client.send_json({
    #             "action": "command",
    #             "command": command
    #         }) # send_json in ESP32Client already handles JSON dumping
            
    #         if success:
    #             logging.info(f"[ESP32] Command '{command_strip}' sent successfully on attempt {attempt}.")
    #             # The ESP32Client's send_command/send_json might already include a delay.
    #             # If not, and one is strictly needed *after ESP32 processing*, it's more complex.
    #             # The current ESP32Client has a 2s delay *after sending*.
    #             # await asyncio.sleep(2.0) # This might be redundant if ESP32Client handles it
    #             return True
    #         else:
    #             logging.error(f"[ESP32] Command '{command_strip}' failed to send on attempt {attempt} (client reported failure).")
    #     except Exception as e:
    #         logging.error(f"[ESP32] Error during command attempt {attempt} for '{command_strip}': {e}")
        
    #     if attempt < ARM_MAX_RETRIES:
    #         logging.info(f"[ESP32] Waiting {ARM_RETRY_DELAY_SECONDS}s before retry...")
    #         await asyncio.sleep(ARM_RETRY_DELAY_SECONDS)

    # logging.error(f"[ESP32] Command '{command_strip}' FAILED after {ARM_MAX_RETRIES} attempts.")
    return 1

# --- Asynchronous Arm Movement Logic using ESP32 WebSocket ---
async def from_to_async(esp32_ws_client, src: str, dest: str, card_id: int) -> bool:
    logging.info(f"Executing ASYNC movement sequence: card {card_id} from {src} to {dest}")
    success = True

    if src not in ["card", "temp1", "temp2", "home"] or dest not in ["card", "temp1", "temp2", "trash", "home"]:
        logging.error(f"Invalid src ('{src}') or dest ('{dest}') location.")
        return False
    if src == "card" or dest == "card":
        if not (0 <= card_id < len(arm_values)):
            logging.error(f"Invalid card_id {card_id} for arm_values length {len(arm_values)}")
            return False

    try:
        # ARM_SYNC_STEP_DELAY is a global constant
        if src == "card" and dest == "temp1":
            logging.debug("Seq: card -> temp1")
            if not await send_arm_command_async(esp32_ws_client, arm_values[card_id][0], arm_values[card_id][1], arm_values[card_id][2], 1, 0): success = False
            if success: await asyncio.sleep(ARM_SYNC_STEP_DELAY)
            if success and not await send_arm_command_async(esp32_ws_client, arm_home[0], arm_home[1], arm_home[2], 1, 1): success = False
            if success: await asyncio.sleep(ARM_SYNC_STEP_DELAY)
            if success and not await send_arm_command_async(esp32_ws_client, arm_temp1[0], arm_temp1[1], arm_temp1[2], 0, 0): success = False
            if success: await asyncio.sleep(ARM_SYNC_STEP_DELAY)
            if success and not await send_arm_command_async(esp32_ws_client, arm_home[0], arm_home[1], arm_home[2], 0, 1): success = False
        # ... (all other elif conditions for movement sequences, ensuring esp32_ws_client is passed) ...
        elif src == "card" and dest == "temp2":
            logging.debug("Seq: card -> temp2")
            if not await send_arm_command_async(esp32_ws_client, arm_values[card_id][0], arm_values[card_id][1], arm_values[card_id][2], 1, 0): success = False
            if success: await asyncio.sleep(ARM_SYNC_STEP_DELAY)
            if success and not await send_arm_command_async(esp32_ws_client, arm_home[0], arm_home[1], arm_home[2], 1, 1): success = False
            if success: await asyncio.sleep(ARM_SYNC_STEP_DELAY)
            if success and not await send_arm_command_async(esp32_ws_client, arm_temp2[0], arm_temp2[1], arm_temp2[2], 0, 0): success = False
            if success: await asyncio.sleep(ARM_SYNC_STEP_DELAY)
            if success and not await send_arm_command_async(esp32_ws_client, arm_home[0], arm_home[1], arm_home[2], 0, 1): success = False

        elif src == "card" and dest == "trash":
            logging.debug("Seq: card -> trash")
            if not await send_arm_command_async(esp32_ws_client, arm_values[card_id][0], arm_values[card_id][1], arm_values[card_id][2], 1, 0): success = False
            if success: await asyncio.sleep(ARM_SYNC_STEP_DELAY)
            if success and not await send_arm_command_async(esp32_ws_client, arm_home[0], arm_home[1], arm_home[2], 1, 1): success = False
            if success: await asyncio.sleep(ARM_SYNC_STEP_DELAY)
            if success and not await send_arm_command_async(esp32_ws_client, arm_trash[0], arm_trash[1], arm_trash[2], 0, 0): success = False
            if success: await asyncio.sleep(ARM_SYNC_STEP_DELAY)
            if success and not await send_arm_command_async(esp32_ws_client, arm_home[0], arm_home[1], arm_home[2], 0, 1): success = False

        elif src == "temp1" and dest == "trash":
            logging.debug("Seq: temp1 -> trash")
            if not await send_arm_command_async(esp32_ws_client, arm_temp1[0], arm_temp1[1], arm_temp1[2], 1, 0): success = False
            if success: await asyncio.sleep(ARM_SYNC_STEP_DELAY)
            if success and not await send_arm_command_async(esp32_ws_client, arm_home[0], arm_home[1], arm_home[2], 1, 1): success = False
            if success: await asyncio.sleep(ARM_SYNC_STEP_DELAY)
            if success and not await send_arm_command_async(esp32_ws_client, arm_trash[0], arm_trash[1], arm_trash[2], 0, 0): success = False
            if success: await asyncio.sleep(ARM_SYNC_STEP_DELAY)
            if success and not await send_arm_command_async(esp32_ws_client, arm_home[0], arm_home[1], arm_home[2], 0, 1): success = False

        elif src == "temp2" and dest == "trash":
            logging.debug("Seq: temp2 -> trash")
            if not await send_arm_command_async(esp32_ws_client, arm_temp2[0], arm_temp2[1], arm_temp2[2], 1, 0): success = False
            if success: await asyncio.sleep(ARM_SYNC_STEP_DELAY)
            if success and not await send_arm_command_async(esp32_ws_client, arm_home[0], arm_home[1], arm_home[2], 1, 1): success = False
            if success: await asyncio.sleep(ARM_SYNC_STEP_DELAY)
            if success and not await send_arm_command_async(esp32_ws_client, arm_trash[0], arm_trash[1], arm_trash[2], 0, 0): success = False
            if success: await asyncio.sleep(ARM_SYNC_STEP_DELAY)
            if success and not await send_arm_command_async(esp32_ws_client, arm_home[0], arm_home[1], arm_home[2], 0, 1): success = False

        elif src == "temp1" and dest == "card":
            logging.debug("Seq: temp1 -> card")
            if not await send_arm_command_async(esp32_ws_client, arm_temp1[0], arm_temp1[1], arm_temp1[2], 1, 0): success = False
            if success: await asyncio.sleep(ARM_SYNC_STEP_DELAY)
            if success and not await send_arm_command_async(esp32_ws_client, arm_home[0], arm_home[1], arm_home[2], 1, 1): success = False
            if success: await asyncio.sleep(ARM_SYNC_STEP_DELAY)
            if success and not await send_arm_command_async(esp32_ws_client, arm_values[card_id][0], arm_values[card_id][1], arm_values[card_id][2], 0, 0): success = False
            if success: await asyncio.sleep(ARM_SYNC_STEP_DELAY)
            if success and not await send_arm_command_async(esp32_ws_client, arm_home[0], arm_home[1], arm_home[2], 0, 1): success = False

        elif src == "temp2" and dest == "card":
            logging.debug("Seq: temp2 -> card")
            if not await send_arm_command_async(esp32_ws_client, arm_temp2[0], arm_temp2[1], arm_temp2[2], 1, 0): success = False
            if success: await asyncio.sleep(ARM_SYNC_STEP_DELAY)
            if success and not await send_arm_command_async(esp32_ws_client, arm_home[0], arm_home[1], arm_home[2], 1, 1): success = False
            if success: await asyncio.sleep(ARM_SYNC_STEP_DELAY)
            if success and not await send_arm_command_async(esp32_ws_client, arm_values[card_id][0], arm_values[card_id][1], arm_values[card_id][2], 0, 0): success = False
            if success: await asyncio.sleep(ARM_SYNC_STEP_DELAY)
            if success and not await send_arm_command_async(esp32_ws_client, arm_home[0], arm_home[1], arm_home[2], 0, 1): success = False

        elif src == "home" and dest == "home":
            logging.debug("Seq: home -> home")
            if not await send_arm_command_async(esp32_ws_client, arm_home[0], arm_home[1], arm_home[2], 0, 1): success = False
        else:
            logging.error(f"Invalid/unhandled src/dest combination: {src} -> {dest}")
            success = False

        if not success:
            logging.error(f"ASYNC movement sequence FAILED: A command failed for card {card_id} ({src} -> {dest})")
            if not (src == "home" and dest == "home"):
                logging.warning("Attempting to return arm home after sequence failure.")
                await send_arm_command_async(esp32_ws_client, arm_home[0], arm_home[1], arm_home[2], 0, 1)
            return False

        logging.info(f"ASYNC movement sequence COMPLETED successfully: card {card_id} ({src} -> {dest})")
        return True

    except Exception as e:
        logging.error(f"Unexpected error during ASYNC sequence ({src} -> {dest}, card {id}): {e}", exc_info=True)
        logging.warning("Attempting to return arm home after unexpected sequence error.")
        await send_arm_command_async(esp32_ws_client, arm_home[0], arm_home[1], arm_home[2], 0, 1) # Ensure client is passed
        return False


async def from_to(websocket: WebSocket, src: str, dest: str, card_id: int) -> bool:
    action_name = f"{src}_to_{dest}"
    logging.info(f"Initiating arm movement: card {card_id} [{action_name}]")
    move_successful = False

    game_version = "color" # Default
    # Determine current game_version more reliably if possible, or assume based on active_games structure
    # This part of logic finding the client is important
    current_esp32_client = None
    current_game_needs_switch = False

    # Try to get the client specific to this WebSocket's game context first
    # The game runners (run_yolo_game, run_color_game) are expected to store
    # the client in active_games[game_version]["esp32_client"]
    # and switch_command_sent in active_games[game_version]["switch_command_sent"]

    # Determine which game this websocket is associated with.
    # This might need refinement if a single websocket connection can switch game types.
    # For now, we'll try to infer or use a default.
    # A better way would be to pass game_version into from_to, or have from_to called
    # from within a context where game_version is known.
    # Let's assume the websocket object might have a game_version attribute set by the endpoint.
    
    # Attempt to find the game version associated with this WebSocket based on active_games
    # This is a bit indirect. If `from_to` is called from `run_yolo_game` or `run_color_game`,
    # those functions already know their `game_state_key`.
    # The current logic tries to find *any* active game's client.
    # This might be problematic if multiple game types can be active with different clients.
    # However, given the structure with game_locks, usually only one game of a type is active.

    # Try to get the game_version if it's an attribute of the websocket (set by websocket_endpoint)
    # Or, if from_to is called from a game runner, it should ideally pass its game_version
    # For now, let's assume the existing logic for finding an esp32_client will work if
    # the game runners correctly populate active_games.

    inferred_game_version = getattr(websocket, 'game_version_context', None) # Hypothetical attribute

    if inferred_game_version and inferred_game_version in active_games:
        game_data = active_games.get(inferred_game_version, {})
        current_esp32_client = game_data.get("esp32_client")
        if current_esp32_client:
            game_version = inferred_game_version # Found it
            current_game_needs_switch = not game_data.get("switch_command_sent", False)
    
    if not current_esp32_client: # Fallback to iterating active_games
        for gv_key in ["color", "yolo"]: # Prioritize
            if gv_key in active_games and active_games[gv_key].get("esp32_client"):
                current_esp32_client = active_games[gv_key].get("esp32_client")
                game_version = gv_key
                current_game_needs_switch = not active_games[gv_key].get("switch_command_sent", False)
                break
    
    if not current_esp32_client:
        logging.error(f"Cannot find ESP32 client in active games for {action_name}. Movement will fail.")
        move_successful = False
    else:
        try:
            if current_game_needs_switch:
                logging.info(f"Sending switch command for game {game_version} before arm movement.")
                switch_ok = await _send_switch_command(current_esp32_client)
                if switch_ok:
                    active_games[game_version]["switch_command_sent"] = True
                else:
                    logging.error(f"Switch command failed for {game_version}. Arm movement may fail or use wrong mode.")
                    # Decide if to proceed or fail here. For now, proceed with caution.

            move_successful = await from_to_async(current_esp32_client, src, dest, card_id)
        except Exception as e:
            logging.error(f"Error executing from_to_async for {action_name}: {e}", exc_info=True)
            move_successful = False
            logging.warning(f"Attempting safe return home after error during {action_name}.")
            try:
                await send_arm_command_async(current_esp32_client, arm_home[0], arm_home[1], arm_home[2], 0, 1)
            except Exception as home_e:
                logging.error(f"Failed to return arm home after error in {action_name}: {home_e}")

    try:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({
                "type": "arm_status",
                "payload": {"status": "finished", "success": move_successful, "action": action_name, "card_id": card_id}
            })
        else:
            logging.warning(f"WebSocket disconnected before arm status update for {action_name}.")
    except WebSocketDisconnect:
        logging.warning(f"WebSocket disconnected during arm status update send for {action_name}.")
    except Exception as ws_e:
        logging.error(f"Failed to send arm status update via WebSocket for {action_name}: {ws_e}")

    logging.info(f"Arm movement finished for {action_name} (card {id}). Success: {move_successful}")
    return move_successful

# --- FastAPI App ---
app = FastAPI(title="Memory Matching Game Backend")

# --- YOLO Model Loading ---
@app.on_event("startup")
async def load_yolo_model_on_startup():
    global yolo_model_global
    try:
        import importlib
        try: importlib.import_module('ultralytics')
        except ImportError:
            logging.error("Ultralytics library not found. YOLO mode unavailable.")
            yolo_model_global = None; return
        from ultralytics import YOLO
        if not os.path.exists(YOLO_MODEL_PATH):
            logging.error(f"YOLO model file not found: {YOLO_MODEL_PATH}. YOLO mode unavailable.")
            yolo_model_global = None; return
        logging.info(f"Loading YOLO model globally from: {YOLO_MODEL_PATH}")
        yolo_model_global = YOLO(YOLO_MODEL_PATH)
        logging.info("Warming up YOLO model...")
        dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
        try: yolo_model_global.predict(dummy_img, verbose=False)
        except Exception as wu_e: logging.error(f"YOLO warmup failed: {wu_e}");
        logging.info("Global YOLO model loaded and warmed up successfully.")
    except Exception as e:
        logging.error(f"Error loading global YOLO model: {e}", exc_info=True)
        yolo_model_global = None

# --- Board Detection / Transformation Helper Functions (No changes needed here) ---
def find_board_corners(frame: np.ndarray) -> Optional[np.ndarray]:
    if frame is None or frame.size == 0: return None
    try:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_white, upper_white = np.array([0, 0, 150]), np.array([180, 70, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        min_area_threshold = frame.shape[0] * frame.shape[1] * 0.05
        if contour_area < min_area_threshold: return None
        perimeter = cv2.arcLength(largest_contour, True)
        epsilon = 0.03 * perimeter
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        if len(approx) == 4:
            corners = np.array([p[0] for p in approx], dtype=np.float32)
            return sort_corners(corners)
        return None
    except Exception as e: logging.error(f"Error finding board corners: {e}"); return None

def sort_corners(corners: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = corners.sum(axis=1); rect[0] = corners[np.argmin(s)]; rect[2] = corners[np.argmax(s)]
    diff = np.diff(corners, axis=1); rect[1] = corners[np.argmin(diff)]; rect[3] = corners[np.argmax(diff)]
    return rect

def transform_board(frame: np.ndarray, corners: np.ndarray) -> Optional[np.ndarray]:
    if frame is None or corners is None: return None
    try:
        dst_points = np.array([
            [0, 0],
            [COLOR_BOARD_DETECT_WIDTH - 1, 0],
            [COLOR_BOARD_DETECT_WIDTH - 1, COLOR_BOARD_DETECT_HEIGHT - 1],
            [0, COLOR_BOARD_DETECT_HEIGHT - 1]
        ], dtype=np.float32)
        M = cv2.getPerspectiveTransform(corners, dst_points)
        warped = cv2.warpPerspective(frame, M, (COLOR_BOARD_DETECT_WIDTH, COLOR_BOARD_DETECT_HEIGHT))
        return warped
    except Exception as e: logging.error(f"Error during perspective transform: {e}"); return None

# --- Camera Capture (No changes needed regarding ESP32 client) ---
async def capture_frames(websocket: WebSocket, camera_source: str, game_version: str):
    global latest_frame, latest_transformed_frame
    cap = None
    logging.info(f"Starting frame capture task for {game_version} from {camera_source}")
    frame_count = 0; last_log_time = time.time()
    while game_version in active_games and active_games[game_version].get("running"):
        processed_frame_for_send, warped_board_for_send, corners = None, None, None
        try:
            if cap is None or not cap.isOpened():
                logging.info(f"Attempting to open camera source: {camera_source}")
                cap = cv2.VideoCapture(camera_source); await asyncio.sleep(1.5)
                if not cap.isOpened():
                    logging.error(f"Cannot open camera: {camera_source}. Retrying in 5s.")
                    cap = None
                    if websocket.client_state == WebSocketState.CONNECTED:
                        try: await websocket.send_json({"type": "error", "payload": f"Cannot open camera: {camera_source}"})
                        except Exception: pass
                    await asyncio.sleep(5); continue
                else:
                    logging.info(f"Camera {camera_source} opened successfully.")
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, YOLO_FRAME_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, YOLO_FRAME_HEIGHT)
                    logging.info(f"Camera properties set (attempted): {YOLO_FRAME_WIDTH}x{YOLO_FRAME_HEIGHT}")
            ret, frame = cap.read()
            if not ret or frame is None:
                logging.warning("Failed to grab frame. Releasing and retrying...")
                if cap is not None: cap.release(); cap = None
                async with frame_lock: latest_frame, latest_transformed_frame = None, None
                await asyncio.sleep(1); continue
            frame_count += 1; current_frame_copy = frame.copy()
            local_latest_transformed = None
            corners = find_board_corners(current_frame_copy)
            if corners is not None:
                warped = transform_board(current_frame_copy, corners)
                if warped is not None:
                    local_latest_transformed = warped; warped_board_for_send = warped
            async with frame_lock: latest_frame = current_frame_copy; latest_transformed_frame = local_latest_transformed
            processed_frame_for_send = current_frame_copy; timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(processed_frame_for_send, timestamp, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(processed_frame_for_send, f"Mode: {game_version.upper()}", (processed_frame_for_send.shape[1] - 150, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
            if corners is not None: cv2.polylines(processed_frame_for_send, [np.int32(corners)], isClosed=True, color=(0, 255, 255), thickness=2)
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, 75]
            ret_main, buffer_main = cv2.imencode('.jpg', processed_frame_for_send, encode_param)
            jpg_main_as_text = base64.b64encode(buffer_main).decode('utf-8') if ret_main else None
            jpg_transformed_as_text = None
            if warped_board_for_send is not None:
                ret_trans, buffer_trans = cv2.imencode('.jpg', warped_board_for_send, encode_param)
                if ret_trans: jpg_transformed_as_text = base64.b64encode(buffer_trans).decode('utf-8')
            if websocket.client_state == WebSocketState.CONNECTED:
                if jpg_main_as_text:
                    payload = {"frame": jpg_main_as_text}
                    if jpg_transformed_as_text: payload["transformed_frame"] = jpg_transformed_as_text
                    await websocket.send_json({"type": "frame_update", "payload": payload})
                else: logging.warning("Skipping send: main frame encoding failed.")
            else: logging.warning("WS closed in capture loop."); break
            current_time = time.time()
            if current_time - last_log_time >= 5.0:
                fps = frame_count / (current_time - last_log_time); logging.info(f"Camera FPS: {fps:.2f}"); frame_count = 0; last_log_time = current_time
            await asyncio.sleep(0.035)
        except WebSocketDisconnect: logging.info(f"WS disconnected gracefully in capture loop."); break
        except Exception as e: logging.error(f"Error in frame capture loop for {game_version}: {e}", exc_info=True); await asyncio.sleep(1)
    if cap is not None:
        try: cap.release(); logging.info(f"Camera released for {game_version}.")
        except Exception as e: logging.error(f"Error releasing camera for {game_version}: {e}")
    async with frame_lock: latest_frame, latest_transformed_frame = None, None
    logging.info(f"Frame capture task stopped for {game_version}.")

# --- YOLO/Color Detection Helper Functions (No changes needed regarding ESP32 client) ---
# ... (yolo_assign_color, detect_object_at_card, detect_color_at_card remain the same internally regarding detection logic)
# They correctly use async with frame_lock, and retry logic.

# MODIFIED detect_object_at_card with RETRIES and using TRANSFORMED FRAME
async def detect_object_at_card(card_id: int) -> Optional[str]:
    global latest_transformed_frame, yolo_model_global 
    if yolo_model_global is None: logging.error("YOLO model not loaded."); return None
    attempt = 0
    while attempt < DETECTION_MAX_RETRIES:
        attempt += 1; logging.info(f"YOLO Detection Attempt {attempt}/{DETECTION_MAX_RETRIES} for card {card_id}")
        warped_board = None
        async with frame_lock:
            if latest_transformed_frame is not None: warped_board = latest_transformed_frame.copy()
        if warped_board is None:
            logging.warning(f"Attempt {attempt}: No transformed board for YOLO on card {card_id}.")
            if attempt < DETECTION_MAX_RETRIES: await asyncio.sleep(DETECTION_RETRY_DELAY_SECONDS)
            continue
        try:
            row, col = card_id // GRID_COLS, card_id % GRID_COLS
            cell_width = COLOR_BOARD_DETECT_WIDTH // GRID_COLS; cell_height = COLOR_BOARD_DETECT_HEIGHT // GRID_ROWS
            x1, y1 = col * cell_width, row * cell_height; x2, y2 = x1 + cell_width, y1 + cell_height
            padding = 5
            roi_x1, roi_y1 = max(0, x1 + padding), max(0, y1 + padding)
            roi_x2, roi_y2 = min(COLOR_BOARD_DETECT_WIDTH - 1, x2 - padding), min(COLOR_BOARD_DETECT_HEIGHT - 1, y2 - padding)
            if roi_x1 >= roi_x2 or roi_y1 >= roi_y2:
                logging.warning(f"Attempt {attempt}: Invalid ROI for YOLO card {card_id}.");
                if attempt < DETECTION_MAX_RETRIES: await asyncio.sleep(DETECTION_RETRY_DELAY_SECONDS); continue
            card_roi = warped_board[roi_y1:roi_y2, roi_x1:roi_x2]
            if card_roi.size == 0:
                logging.warning(f"Attempt {attempt}: Empty ROI for YOLO card {card_id}.");
                if attempt < DETECTION_MAX_RETRIES: await asyncio.sleep(DETECTION_RETRY_DELAY_SECONDS); continue
            def sync_predict(model, roi):
                target_indices = [i for i, lbl in enumerate(model.names.values()) if lbl.lower() in YOLO_TARGET_LABELS]
                return model.predict(roi, conf=0.45, verbose=False, device='cpu', classes=target_indices if target_indices else None)
            try: results = await asyncio.to_thread(sync_predict, yolo_model_global, card_roi)
            except Exception as predict_err:
                logging.error(f"Attempt {attempt}: Error YOLO predict thread card {card_id}: {predict_err}", exc_info=True)
                if attempt < DETECTION_MAX_RETRIES: await asyncio.sleep(DETECTION_RETRY_DELAY_SECONDS); continue
            detected_object_label, highest_conf = None, 0.0
            if results:
                for result in results:
                    boxes, names = getattr(result, 'boxes', None), getattr(result, 'names', {})
                    if boxes:
                        for box in boxes:
                            cls_tensor, conf_tensor = getattr(box, 'cls', None), getattr(box, 'conf', None)
                            if cls_tensor is not None and conf_tensor is not None and cls_tensor.numel() > 0 and conf_tensor.numel() > 0:
                                try:
                                    label_index, score = int(cls_tensor[0].item()), conf_tensor[0].item()
                                    label = names.get(label_index, f"unknown_idx_{label_index}").lower()
                                    if label in YOLO_TARGET_LABELS and score > highest_conf:
                                        highest_conf, detected_object_label = score, label
                                except Exception as proc_err: logging.error(f"Attempt {attempt}: Error processing YOLO box card {card_id}: {proc_err}")
            if detected_object_label:
                logging.info(f"Success YOLO Detect attempt {attempt}: '{detected_object_label}' (conf: {highest_conf:.2f}) card {card_id}")
                return detected_object_label
            else: logging.warning(f"Attempt {attempt}: No target object on card {card_id}.")
        except cv2.error as cv_err: logging.error(f"Attempt {attempt}: OpenCV error YOLO card {card_id}: {cv_err}", exc_info=True)
        except Exception as e: logging.error(f"Attempt {attempt}: Unexpected error YOLO card {card_id}: {e}", exc_info=True)
        if attempt < DETECTION_MAX_RETRIES: await asyncio.sleep(DETECTION_RETRY_DELAY_SECONDS)
    logging.error(f"YOLO Detection FAILED permanently card {card_id} after {DETECTION_MAX_RETRIES} attempts.")
    return DETECTION_PERMANENT_FAIL_STATE

async def detect_color_at_card(card_id: int) -> Optional[str]:
    global latest_transformed_frame
    attempt = 0
    while attempt < DETECTION_MAX_RETRIES:
        attempt += 1; logging.info(f"Color Detection Attempt {attempt}/{DETECTION_MAX_RETRIES} for card {card_id}")
        warped_board = None
        async with frame_lock:
            if latest_transformed_frame is not None: warped_board = latest_transformed_frame.copy()
        if warped_board is None:
            logging.warning(f"Attempt {attempt}: No transformed board for Color card {card_id}.")
            if attempt < DETECTION_MAX_RETRIES: await asyncio.sleep(DETECTION_RETRY_DELAY_SECONDS)
            continue
        try:
            row, col = card_id // GRID_COLS, card_id % GRID_COLS
            cell_width = COLOR_BOARD_DETECT_WIDTH // GRID_COLS; cell_height = COLOR_BOARD_DETECT_HEIGHT // GRID_ROWS
            x1, y1 = col * cell_width, row * cell_height; x2, y2 = x1 + cell_width, y1 + cell_height
            padding = 5
            roi_x1, roi_y1 = max(0, x1 + padding), max(0, y1 + padding)
            roi_x2, roi_y2 = min(COLOR_BOARD_DETECT_WIDTH - 1, x2 - padding), min(COLOR_BOARD_DETECT_HEIGHT - 1, y2 - padding)
            if roi_x1 >= roi_x2 or roi_y1 >= roi_y2:
                logging.warning(f"Attempt {attempt}: Invalid ROI Color card {card_id}.");
                if attempt < DETECTION_MAX_RETRIES: await asyncio.sleep(DETECTION_RETRY_DELAY_SECONDS); continue
            cell_roi = warped_board[roi_y1:roi_y2, roi_x1:roi_x2]
            if cell_roi.size == 0:
                logging.warning(f"Attempt {attempt}: Empty ROI Color card {card_id}.");
                if attempt < DETECTION_MAX_RETRIES: await asyncio.sleep(DETECTION_RETRY_DELAY_SECONDS); continue
            hsv_roi = cv2.cvtColor(cell_roi, cv2.COLOR_BGR2HSV)
            detected_colors_count: Dict[str, int] = {}
            for color_def in COLOR_RANGES:
                color_name, total_mask = color_def['name'], np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
                for l_bound, u_bound in zip(color_def['lower'], color_def['upper']):
                    lower, upper = np.array(l_bound), np.array(u_bound)
                    total_mask = cv2.bitwise_or(total_mask, cv2.inRange(hsv_roi, lower, upper))
                pixel_count = cv2.countNonZero(total_mask)
                if pixel_count > 0: detected_colors_count[color_name] = pixel_count
            dominant_color, max_pixels, black_threshold = None, 0, COLOR_CELL_THRESHOLD * 0.7
            if "black" in detected_colors_count and detected_colors_count["black"] > black_threshold:
                logging.info(f"Attempt {attempt}: Card {card_id} detected as 'black'."); return "black"
            for color_name, pixel_count in detected_colors_count.items():
                if color_name != "black" and pixel_count >= COLOR_CELL_THRESHOLD and pixel_count > max_pixels:
                    max_pixels, dominant_color = pixel_count, color_name
            if dominant_color:
                logging.info(f"Success Color Detect attempt {attempt}: '{dominant_color}' (pixels: {max_pixels}) card {card_id}")
                return dominant_color
            else:
                relevant_counts = {k:v for k,v in detected_colors_count.items() if k!='black' and v > 10}
                logging.warning(f"Attempt {attempt}: No dominant face color card {card_id}. Counts(<Thr): {relevant_counts}")
        except cv2.error as cv_err: logging.error(f"Attempt {attempt}: OpenCV error color detection: {cv_err}", exc_info=True)
        except Exception as e: logging.error(f"Attempt {attempt}: Unexpected error color detection: {e}", exc_info=True)
        if attempt < DETECTION_MAX_RETRIES: await asyncio.sleep(DETECTION_RETRY_DELAY_SECONDS)
    logging.error(f"Color Detection FAILED permanently card {card_id} after {DETECTION_MAX_RETRIES} attempts.")
    return DETECTION_PERMANENT_FAIL_STATE


# --- Game Logic Runners ---
async def run_yolo_game(websocket: WebSocket):
    game_state_key = "yolo"
    logging.info(f"[{game_state_key.upper()}] Starting Game Logic...")
    # Ensure esp32_client is on the websocket object (set by websocket_endpoint or GameSession.run_game)
    current_esp32_client = getattr(websocket, "esp32_client", None)
    
    active_games[game_state_key] = {
        "running": True,
        "esp32_client": current_esp32_client, # Store the client for this game instance
        "switch_command_sent": False # Track if switch command has been sent for this game
    }
    game_state = active_games[game_state_key]

    if not current_esp32_client:
        logging.warning(f"[{game_state_key.upper()}] No ESP32 client provided to run_yolo_game, arm control will fail")
    else:
        logging.info(f"[{game_state_key.upper()}] Using ESP32 client: {current_esp32_client}")


    # if yolo_model_global is None:
        # logging.error(f"[{game_state_key.upper()}] YOLO Model not loaded. Cannot start."); game_state["running"] = False
        # if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "error", "payload": "YOLO model missing."})
        # return

    game_state.update({
        "card_states": {i: {"isFlippedBefore": False, "object": None, "isMatched": False} for i in range(CARD_COUNT)},
        "objects_found": {obj: [] for obj in YOLO_TARGET_LABELS}, "pairs_found": 0, "current_flipped_cards": [],
        "last_detect_fail_id": None,
    }) # running, esp32_client, switch_command_sent already set
    logging.info(f"[{game_state_key.upper()}] Initialized game state.")

    # ... (rest of run_yolo_game logic, ensure `from_to` is called with `websocket` which will help it find the client)
    # The `from_to` function will use `active_games[game_state_key]["esp32_client"]`
    # and `active_games[game_state_key]["switch_command_sent"]`
    
    # Make sure websocket.game_version_context is set for from_to to potentially use
    setattr(websocket, 'game_version_context', game_state_key)

    if websocket.client_state == WebSocketState.CONNECTED:
        try:
            await websocket.send_json({"type": "game_state", "payload": {k: game_state[k] for k in ["card_states", "pairs_found", "current_flipped_cards"]}})
            await websocket.send_json({"type": "message", "payload": "YOLO Game Started (Board Detect Mode). Initializing arm..."})
        except Exception as send_e: logging.error(f"[{game_state_key.upper()}] Send initial error: {send_e}"); game_state["running"] = False

    camera_task = None
    init_home_success = False
    if game_state.get("running"):
        camera_task = asyncio.create_task(capture_frames(websocket, CAMERA_URL, game_state_key))
        logging.info(f"[{game_state_key.upper()}] Sending initial home command...")
        init_home_success = await from_to(websocket, "home", "home", -1) # `from_to` will use the client from game_state
        if not init_home_success:
            logging.error(f"[{game_state_key.upper()}] Initial arm homing failed!")
            if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "error", "payload": "Arm init failed."})
            game_state["running"] = False
        elif websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "message", "payload": "Arm ready. Starting game."})
    
    try:
        # --- Main Game Loop ---
        while game_state.get("pairs_found", 0) < (CARD_COUNT // 2) and game_state.get("running", False):
            await asyncio.sleep(FLIP_DELAY_SECONDS)
            if websocket.client_state != WebSocketState.CONNECTED: logging.warning(f"[{game_state_key.upper()}] WS disconnected."); game_state["running"] = False; break

            current_flipped = game_state.get("current_flipped_cards", [])
            card_states = game_state.get("card_states", {})
            objects_found = game_state.get("objects_found", {})
            pairs_found = game_state.get("pairs_found", 0)
            logging.info(f"[{game_state_key.upper()}] Loop Start: Flipped={current_flipped}, Pairs={pairs_found}/{CARD_COUNT // 2}")

            async def update_frontend_state(extra_message: Optional[str] = None):
                payload = {"card_states": card_states, "pairs_found": game_state.get("pairs_found", 0), "current_flipped": current_flipped}
                if websocket.client_state == WebSocketState.CONNECTED:
                    try:
                        await websocket.send_json({"type": "game_state", "payload": payload})
                        if extra_message: await websocket.send_json({"type": "message", "payload": extra_message})
                    except Exception as send_e: logging.error(f"[{game_state_key.upper()}] Send state failed: {send_e}")
            def choose_random_card() -> Optional[int]:
                available = [i for i, s in card_states.items() if not s.get("isMatched") and i not in current_flipped and s.get("object") != DETECTION_PERMANENT_FAIL_STATE]
                if not available: return None
                never_flipped = [i for i in available if not card_states[i].get("isFlippedBefore")]
                if never_flipped:
                    chosen = random.choice(never_flipped)
                    if chosen == game_state.get("last_detect_fail_id") and len(never_flipped) > 1: chosen = random.choice([c for c in never_flipped if c != chosen])
                    game_state["last_detect_fail_id"] = None; return chosen
                previously_flipped = available
                if previously_flipped:
                    chosen = random.choice(previously_flipped)
                    if chosen == game_state.get("last_detect_fail_id") and len(previously_flipped) > 1: chosen = random.choice([c for c in previously_flipped if c != chosen])
                    game_state["last_detect_fail_id"] = None; return chosen
                return None
            def find_pair() -> Optional[Tuple[int, int]]:
                for obj, ids in objects_found.items():
                    if obj in [None, "DETECT_FAIL", DETECTION_PERMANENT_FAIL_STATE]: continue
                    valid = [i for i in ids if card_states.get(i,{}).get("isFlippedBefore") and not card_states.get(i,{}).get("isMatched") and i not in current_flipped]
                    if len(valid) >= 2: logging.info(f"[{game_state_key.upper()}] Found known pair for '{obj}': {valid[0]},{valid[1]}"); return valid[0], valid[1]
                return None
            def find_match(card_id_to_match: int) -> Optional[int]:
                obj = card_states.get(card_id_to_match,{}).get("object")
                if obj in [None, "DETECT_FAIL", DETECTION_PERMANENT_FAIL_STATE]: return None
                for other_id in objects_found.get(obj, []):
                    if other_id != card_id_to_match and card_states.get(other_id,{}).get("isFlippedBefore") and not card_states.get(other_id,{}).get("isMatched") and other_id not in current_flipped:
                        logging.info(f"[{game_state_key.upper()}] Found match for {card_id_to_match} ('{obj}'): {other_id}"); return other_id
                return None

            if len(current_flipped) == 0:
                known_pair = find_pair()
                if known_pair:
                    card1_id, card2_id = known_pair; obj = card_states[card1_id]['object']
                    logging.info(f"[{game_state_key.upper()}] State 0 -> Strategy: Remove known pair {card1_id}&{card2_id} ('{obj}')")
                    await update_frontend_state(f"Found pair: {obj}. Removing {card1_id}&{card2_id}.")
                    success1 = await from_to(websocket, "card", "trash", card1_id)
                    if not success1: logging.error(f"[{game_state_key.upper()}] Move fail {card1_id} to trash."); await update_frontend_state(f"Arm fail move {card1_id}."); continue
                    success2 = await from_to(websocket, "card", "trash", card2_id)
                    if not success2: logging.error(f"[{game_state_key.upper()}] Move fail {card2_id} to trash. {card1_id} already gone!"); card_states[card1_id]["isMatched"] = True; await update_frontend_state(f"Arm fail move {card2_id}."); continue
                    card_states[card1_id]["isMatched"] = True; card_states[card2_id]["isMatched"] = True; game_state["pairs_found"] = pairs_found + 1; logging.info(f"Pairs found: {game_state['pairs_found']}")
                    await update_frontend_state(); continue
                else:
                    card_to_flip = choose_random_card()
                    if card_to_flip is not None:
                        logging.info(f"[{game_state_key.upper()}] State 0 -> Strategy: Flip card {card_to_flip}.")
                        await update_frontend_state(f"Choosing card {card_to_flip}. Detecting...")
                        detected_obj = await detect_object_at_card(card_to_flip)
                        if detected_obj == DETECTION_PERMANENT_FAIL_STATE:
                            logging.error(f"[{game_state_key.upper()}] PERMANENT detection failure for card {card_to_flip}. Marking unusable.")
                            await update_frontend_state(f"Critical detection fail on {card_to_flip}. Skipping card.")
                            card_states[card_to_flip]["isFlippedBefore"] = True; card_states[card_to_flip]["object"] = DETECTION_PERMANENT_FAIL_STATE; card_states[card_to_flip]["isMatched"] = True
                            game_state["last_detect_fail_id"] = card_to_flip; await update_frontend_state()
                        elif detected_obj is not None:
                            logging.info(f"Detected '{detected_obj}'. Moving card {card_to_flip} to Temp1.")
                            await update_frontend_state(f"Detected '{detected_obj}'. Moving card {card_to_flip} to Temp1.")
                            success_move = await from_to(websocket, "card", "temp1", card_to_flip)
                            if success_move:
                                card_states[card_to_flip]["object"] = detected_obj; card_states[card_to_flip]["isFlippedBefore"] = True
                                if detected_obj not in objects_found: objects_found[detected_obj] = []
                                if card_to_flip not in objects_found[detected_obj]: objects_found[detected_obj].append(card_to_flip)
                                current_flipped.append(card_to_flip); logging.info(f"Card {card_to_flip} ('{detected_obj}') in Temp1. Flipped: {current_flipped}")
                            else: logging.error(f"[{game_state_key.upper()}] Arm fail: {card_to_flip} to Temp1."); await update_frontend_state(f"Arm fail move {card_to_flip}.")
                            await update_frontend_state()
                        else: logging.error(f"[{game_state_key.upper()}] detect_object_at_card returned unexpected None for card {card_to_flip}."); await update_frontend_state(f"Internal error during detection for {card_to_flip}."); game_state["last_detect_fail_id"] = card_to_flip
                    else: logging.info(f"[{game_state_key.upper()}] State 0: No known pair & no available card to flip. Waiting."); await asyncio.sleep(1)
            elif len(current_flipped) == 1:
                first_card_id = current_flipped[0]; first_object = card_states.get(first_card_id,{}).get("object", "UNKNOWN")
                logging.info(f"[{game_state_key.upper()}] State 1: Card {first_card_id} ('{first_object}') in Temp1.")
                if first_object in ["UNKNOWN", DETECTION_PERMANENT_FAIL_STATE]:
                    logging.warning(f"First card {first_card_id} has invalid state '{first_object}'. Returning."); await update_frontend_state(f"Problem with card {first_card_id}. Returning.")
                    await from_to(websocket, "temp1", "card", first_card_id); current_flipped.clear(); await update_frontend_state(); continue
                match_id = find_match(first_card_id)
                if match_id is not None:
                    logging.info(f"[{game_state_key.upper()}] State 1 -> Strategy: Found known match {match_id}. Removing pair.")
                    await update_frontend_state(f"Found match for '{first_object}': {match_id}. Removing.")
                    success1 = await from_to(websocket, "temp1", "trash", first_card_id)
                    if not success1: logging.error(f"Arm fail temp1->trash {first_card_id}. Returning card instead."); await update_frontend_state(f"Arm fail temp1->trash {first_card_id}. Returning card."); await from_to(websocket, "temp1", "card", first_card_id); current_flipped.clear(); await update_frontend_state(); continue
                    success2 = await from_to(websocket, "card", "trash", match_id)
                    if not success2: logging.error(f"Arm fail card->trash {match_id}. {first_card_id} already gone!"); card_states[first_card_id]["isMatched"]=True; await update_frontend_state(f"Arm fail move {match_id}."); current_flipped.clear(); await update_frontend_state(); continue
                    card_states[first_card_id]["isMatched"]=True; card_states[match_id]["isMatched"]=True; card_states[match_id]["isFlippedBefore"]=True; card_states[match_id]["object"]=first_object
                    game_state["pairs_found"] = pairs_found + 1; current_flipped.clear(); logging.info(f"Pairs found: {game_state['pairs_found']}")
                    await update_frontend_state(); continue
                else:
                    card_to_flip = choose_random_card()
                    if card_to_flip is not None:
                        logging.info(f"[{game_state_key.upper()}] State 1 -> Strategy: No known match. Flipping {card_to_flip}.")
                        await update_frontend_state(f"No match for '{first_object}'. Choosing {card_to_flip}...")
                        detected_obj = await detect_object_at_card(card_to_flip)
                        if detected_obj == DETECTION_PERMANENT_FAIL_STATE:
                            logging.error(f"[{game_state_key.upper()}] PERMANENT detection fail on second card {card_to_flip}.")
                            await update_frontend_state(f"Critical detect fail {card_to_flip}. Returning first card.")
                            card_states[card_to_flip]["isFlippedBefore"]=True; card_states[card_to_flip]["object"]=DETECTION_PERMANENT_FAIL_STATE; card_states[card_to_flip]["isMatched"]=True; game_state["last_detect_fail_id"] = card_to_flip
                            await from_to(websocket, "temp1", "card", first_card_id); current_flipped.clear()
                        elif detected_obj is not None:
                            logging.info(f"Detected '{detected_obj}'. Moving {card_to_flip} to Temp2.")
                            await update_frontend_state(f"Detected '{detected_obj}'. Moving {card_to_flip} to Temp2.")
                            success_move = await from_to(websocket, "card", "temp2", card_to_flip)
                            if success_move:
                                card_states[card_to_flip]["object"]=detected_obj; card_states[card_to_flip]["isFlippedBefore"]=True
                                if detected_obj not in objects_found: objects_found[detected_obj] = []
                                if card_to_flip not in objects_found[detected_obj]: objects_found[detected_obj].append(card_to_flip)
                                current_flipped.append(card_to_flip); logging.info(f"Card {card_to_flip} ('{detected_obj}') in Temp2. Flipped: {current_flipped}")
                            else: logging.error(f"Arm fail card->temp2 {card_to_flip}. Returning first card."); await update_frontend_state(f"Arm fail move {card_to_flip}. Returning first card."); await from_to(websocket, "temp1", "card", first_card_id); current_flipped.clear()
                        else: logging.error(f"Detect returned unexpected None for {card_to_flip}. Returning first."); await update_frontend_state(f"Internal error during detection for {card_to_flip}. Returning first card."); await from_to(websocket, "temp1", "card", first_card_id); current_flipped.clear(); game_state["last_detect_fail_id"] = card_to_flip
                        await update_frontend_state()
                    else: logging.warning(f"State 1: No second card available? Returning {first_card_id}."); await update_frontend_state(f"No second card found. Returning {first_card_id}."); await from_to(websocket, "temp1", "card", first_card_id); current_flipped.clear(); await update_frontend_state()
            elif len(current_flipped) == 2:
                card1_id, card2_id = current_flipped[0], current_flipped[1]
                obj1, obj2 = card_states.get(card1_id,{}).get("object"), card_states.get(card2_id,{}).get("object")
                logging.info(f"[{game_state_key.upper()}] State 2: Cards {card1_id} ('{obj1}') & {card2_id} ('{obj2}') in Temp1/2. Checking for match.")
                is_match = (obj1 is not None and obj1 != DETECTION_PERMANENT_FAIL_STATE and obj2 is not None and obj2 != DETECTION_PERMANENT_FAIL_STATE and obj1 == obj2)
                if is_match:
                    logging.info(f"MATCH FOUND: {card1_id}&{card2_id} ('{obj1}'). Removing from Temp1/2.")
                    await update_frontend_state(f"Match: {obj1}! Removing {card1_id}&{card2_id}.")
                    success1 = await from_to(websocket, "temp1", "trash", card1_id)
                    if not success1: logging.error(f"Arm fail temp1->trash {card1_id}. Returning both cards."); await update_frontend_state(f"Arm fail temp1->trash {card1_id}. Returning both cards."); await from_to(websocket, "temp1", "card", card1_id); await from_to(websocket, "temp2", "card", card2_id); current_flipped.clear(); await update_frontend_state(); continue
                    success2 = await from_to(websocket, "temp2", "trash", card2_id)
                    if not success2: logging.error(f"Arm fail temp2->trash {card2_id}. {card1_id} already gone!"); card_states[card1_id]["isMatched"]=True; await update_frontend_state(f"Arm fail temp2->trash {card2_id}. Return it."); await from_to(websocket, "temp2", "card", card2_id); current_flipped.clear(); await update_frontend_state(); continue
                    card_states[card1_id]["isMatched"]=True; card_states[card2_id]["isMatched"]=True; game_state["pairs_found"] = pairs_found + 1; current_flipped.clear(); logging.info(f"Pairs found: {game_state['pairs_found']}")
                    await update_frontend_state()
                else:
                    logging.info(f"NO MATCH ('{obj1}' vs '{obj2}'). Returning {card1_id} from Temp1 & {card2_id} from Temp2.")
                    await update_frontend_state(f"No match. Returning {card1_id} from Temp1 & {card2_id} from Temp2.")
                    success_ret1 = await from_to(websocket, "temp1", "card", card1_id)
                    success_ret2 = await from_to(websocket, "temp2", "card", card2_id)
                    if not (success_ret1 and success_ret2): logging.error(f"Failed returning one or both cards: {card1_id} (Success: {success_ret1}), {card2_id} (Success: {success_ret2})."); await update_frontend_state(f"Warning: Arm fail returning cards {card1_id}/{card2_id}.")
                    current_flipped.clear(); await update_frontend_state()
                    if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "cards_hidden", "payload": [card1_id, card2_id]})
            elif len(current_flipped) > 2:
                logging.error(f"Invalid State: {len(current_flipped)} cards flipped: {current_flipped}. Attempting recovery.")
                await update_frontend_state("Error: Invalid state detected. Returning cards.")
                if len(current_flipped)>0 and card_states.get(current_flipped[0], {}).get("object"): await from_to(websocket, "temp1", "card", current_flipped[0])
                if len(current_flipped)>1 and card_states.get(current_flipped[1], {}).get("object"): await from_to(websocket, "temp2", "card", current_flipped[1])
                current_flipped.clear(); await update_frontend_state()
                if game_state.get("pairs_found", 0) >= 4:
                    print(f"[{game_state_key.upper()}] Game Finished! All pairs found.")
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_json({"type": "game_over", "payload": "All pairs found!"})
                    game_state["running"] = False
                    await asyncio.sleep(1.0)
                    break
    except WebSocketDisconnect: logging.info(f"[{game_state_key.upper()}] WS disconnected."); game_state["running"] = False
    except asyncio.CancelledError: logging.info(f"[{game_state_key.upper()}] Task cancelled."); game_state["running"] = False
    except Exception as e:
        logging.error(f"[{game_state_key.upper()}] CRITICAL Loop Error: {e}", exc_info=True); game_state["running"] = False
        if websocket.client_state == WebSocketState.CONNECTED:
            try: await websocket.send_json({"type": "error", "payload": f"Game Error: {e}"})
            except Exception: pass
    finally:
        logging.info(f"[{game_state_key.upper()}] Cleaning up runner..."); game_state["running"] = False
        if camera_task and not camera_task.done():
            logging.info(f"[{game_state_key.upper()}] Cancelling camera task..."); camera_task.cancel()
            try: await camera_task
            except asyncio.CancelledError: logging.info(f"[{game_state_key.upper()}] Camera task cancelled.")
            except Exception as task_e: logging.error(f"[{game_state_key.upper()}] Camera task cancel/await error: {task_e}")
        if locals().get('init_home_success', True) and game_state.get("esp32_client"): # Ensure client exists for final home
            logging.info(f"[{game_state_key.upper()}] Ensuring arm home post-game."); await from_to(websocket, "home", "home", -1)
        else: logging.warning(f"[{game_state_key.upper()}] Skipping final home (initial homing failed or no client).")
        # Clear the game_version_context from websocket
        if hasattr(websocket, 'game_version_context'):
            delattr(websocket, 'game_version_context')
        logging.info(f"[{game_state_key.upper()}] Runner finished cleanup.")


async def run_color_game(websocket: WebSocket):
    game_state_key = "color"
    logging.info(f"[{game_state_key.upper()}] Starting Game Logic...")
    current_esp32_client = getattr(websocket, "esp32_client", None)

    active_games[game_state_key] = {
        "running": True,
        "esp32_client": current_esp32_client,
        "switch_command_sent": False
    }
    game_state = active_games[game_state_key]

    if not current_esp32_client:
        logging.warning(f"[{game_state_key.upper()}] No ESP32 client provided to run_color_game, arm control will fail")
    else:
        logging.info(f"[{game_state_key.upper()}] Using ESP32 client: {current_esp32_client}")
        
    # Make sure websocket.game_version_context is set for from_to to potentially use
    setattr(websocket, 'game_version_context', game_state_key)

    game_state.update({
        "card_states": {i: {"isFlippedBefore": False, "color": None, "isMatched": False} for i in range(CARD_COUNT)},
        "colors_found": {color: [] for color in COLOR_DEFINITIONS.keys()}, "pairs_found": 0, "current_flipped_cards": [],
        "last_detect_fail_id": None,
    })
    logging.info(f"[{game_state_key.upper()}] Initialized game state.")
    # ... (rest of run_color_game logic, similar to run_yolo_game setup and main loop) ...
    # The structure within run_color_game will mirror run_yolo_game for ESP32 client handling,
    # state updates, and calling `from_to`.

    if websocket.client_state == WebSocketState.CONNECTED:
        try:
            await websocket.send_json({"type": "game_state", "payload": {k: game_state[k] for k in ["card_states", "pairs_found", "current_flipped_cards"]}})
            await websocket.send_json({"type": "message", "payload": "Color Game Started. Initializing arm..."})
        except Exception as send_e: logging.error(f"[{game_state_key.upper()}] Send initial error: {send_e}"); game_state["running"] = False

    camera_task = None
    init_home_success = False
    if game_state.get("running"):
        camera_task = asyncio.create_task(capture_frames(websocket, CAMERA_URL, game_state_key))
        logging.info(f"[{game_state_key.upper()}] Sending initial home command...")
        init_home_success = await from_to(websocket, "home", "home", -1)
        if not init_home_success:
            logging.error(f"[{game_state_key.upper()}] Initial arm homing failed!")
            if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "error", "payload": "Arm init failed."})
            game_state["running"] = False
        elif websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "message", "payload": "Arm ready. Starting game."})
    
    try:
        # --- Main Game Loop ---
        while game_state.get("pairs_found", 0) < (CARD_COUNT // 2) and game_state.get("running", False):
            await asyncio.sleep(FLIP_DELAY_SECONDS)
            if websocket.client_state != WebSocketState.CONNECTED: logging.warning(f"[{game_state_key.upper()}] WS disconnected."); game_state["running"] = False; break

            current_flipped = game_state.get("current_flipped_cards", [])
            card_states = game_state.get("card_states", {})
            colors_found = game_state.get("colors_found", {})
            pairs_found = game_state.get("pairs_found", 0)
            logging.info(f"[{game_state_key.upper()}] Loop Start: Flipped={current_flipped}, Pairs={pairs_found}/{CARD_COUNT // 2}")

            async def update_frontend_state(extra_message: Optional[str] = None):
                payload = {"card_states": card_states, "pairs_found": game_state.get("pairs_found", 0), "current_flipped": current_flipped}
                if websocket.client_state == WebSocketState.CONNECTED:
                    try:
                        await websocket.send_json({"type": "game_state", "payload": payload})
                        if extra_message: await websocket.send_json({"type": "message", "payload": extra_message})
                    except Exception as send_e: logging.error(f"[{game_state_key.upper()}] Send state failed: {send_e}")
            def choose_random_card() -> Optional[int]: # Same logic as YOLO's
                available = [i for i, s in card_states.items() if not s.get("isMatched") and i not in current_flipped and s.get("color") != DETECTION_PERMANENT_FAIL_STATE]
                if not available: return None
                never_flipped = [i for i in available if not card_states[i].get("isFlippedBefore")]
                if never_flipped:
                    chosen = random.choice(never_flipped)
                    if chosen == game_state.get("last_detect_fail_id") and len(never_flipped) > 1: chosen = random.choice([c for c in never_flipped if c != chosen])
                    game_state["last_detect_fail_id"] = None; return chosen
                previously_flipped = available
                if previously_flipped:
                    chosen = random.choice(previously_flipped)
                    if chosen == game_state.get("last_detect_fail_id") and len(previously_flipped) > 1: chosen = random.choice([c for c in previously_flipped if c != chosen])
                    game_state["last_detect_fail_id"] = None; return chosen
                return None
            def find_pair() -> Optional[Tuple[int, int]]: # Uses 'color'
                for color, ids in colors_found.items():
                    if color in [None, "DETECT_FAIL", DETECTION_PERMANENT_FAIL_STATE, "black"]: continue
                    valid = [i for i in ids if card_states.get(i,{}).get("isFlippedBefore") and not card_states.get(i,{}).get("isMatched") and i not in current_flipped]
                    if len(valid) >= 2: logging.info(f"[{game_state_key.upper()}] Found known pair for '{color}': {valid[0]},{valid[1]}"); return valid[0], valid[1]
                return None
            def find_match(card_id_to_match: int) -> Optional[int]: # Uses 'color'
                color = card_states.get(card_id_to_match,{}).get("color")
                if color in [None, "DETECT_FAIL", DETECTION_PERMANENT_FAIL_STATE, "black"]: return None
                for other_id in colors_found.get(color, []):
                    if other_id != card_id_to_match and card_states.get(other_id,{}).get("isFlippedBefore") and not card_states.get(other_id,{}).get("isMatched") and other_id not in current_flipped:
                        logging.info(f"[{game_state_key.upper()}] Found match for {card_id_to_match} ('{color}'): {other_id}"); return other_id
                return None

            if len(current_flipped) == 0:
                known_pair = find_pair()
                if known_pair:
                    card1_id, card2_id = known_pair; color = card_states[card1_id]['color']
                    logging.info(f"[{game_state_key.upper()}] State 0 -> Strategy: Remove known pair {card1_id}&{card2_id} ('{color}')")
                    await update_frontend_state(f"Found pair: {color}. Removing {card1_id}&{card2_id}.")
                    success1 = await from_to(websocket, "card", "trash", card1_id)
                    if not success1: logging.error(f"Move fail {card1_id}"); await update_frontend_state(f"Arm fail {card1_id}."); continue
                    success2 = await from_to(websocket, "card", "trash", card2_id)
                    if not success2: logging.error(f"Move fail {card2_id}. {card1_id} gone!"); card_states[card1_id]["isMatched"] = True; await update_frontend_state(f"Arm fail {card2_id}."); continue
                    card_states[card1_id]["isMatched"]=True; card_states[card2_id]["isMatched"]=True; game_state["pairs_found"] = pairs_found + 1; logging.info(f"Pairs found: {game_state['pairs_found']}")
                    await update_frontend_state(); continue
                else:
                    card_to_flip = choose_random_card()
                    if card_to_flip is not None:
                        logging.info(f"[{game_state_key.upper()}] State 0 -> Strategy: Flip card {card_to_flip}.")
                        await update_frontend_state(f"Choosing card {card_to_flip}. Detecting color...")
                        detected_color = await detect_color_at_card(card_to_flip)
                        if detected_color == DETECTION_PERMANENT_FAIL_STATE:
                            logging.error(f"[{game_state_key.upper()}] PERMANENT color detect fail {card_to_flip}.")
                            await update_frontend_state(f"Critical color detect fail {card_to_flip}. Skipping.")
                            card_states[card_to_flip]["isFlippedBefore"]=True; card_states[card_to_flip]["color"]=DETECTION_PERMANENT_FAIL_STATE; card_states[card_to_flip]["isMatched"]=True; game_state["last_detect_fail_id"] = card_to_flip; await update_frontend_state()
                        elif detected_color == "black":
                            logging.warning(f"Detected black back on {card_to_flip}. Marking, not moving.")
                            await update_frontend_state(f"Detected back of card {card_to_flip}.")
                            card_states[card_to_flip]["isFlippedBefore"] = True; card_states[card_to_flip]["color"] = "black"; game_state["last_detect_fail_id"] = card_to_flip; await update_frontend_state()
                        elif detected_color is not None:
                            logging.info(f"Detected '{detected_color}'. Moving {card_to_flip} to Temp1.")
                            await update_frontend_state(f"Detected '{detected_color}'. Moving {card_to_flip} to Temp1.")
                            success_move = await from_to(websocket, "card", "temp1", card_to_flip)
                            if success_move:
                                card_states[card_to_flip]["color"]=detected_color; card_states[card_to_flip]["isFlippedBefore"]=True
                                if detected_color not in colors_found: colors_found[detected_color] = []
                                if card_to_flip not in colors_found[detected_color]: colors_found[detected_color].append(card_to_flip)
                                current_flipped.append(card_to_flip); logging.info(f"Card {card_to_flip} ('{detected_color}') in Temp1. Flipped: {current_flipped}")
                            else: logging.error(f"Arm fail card->temp1 {card_to_flip}."); await update_frontend_state(f"Arm fail {card_to_flip}.")
                            await update_frontend_state()
                        else: logging.error(f"Detect_color returned unexpected None for {card_to_flip}."); await update_frontend_state(f"Internal error detect {card_to_flip}."); game_state["last_detect_fail_id"] = card_to_flip
                    else: logging.info(f"State 0: No pair & no available card."); await asyncio.sleep(1)
            elif len(current_flipped) == 1:
                first_card_id = current_flipped[0]; first_color = card_states.get(first_card_id,{}).get("color", "UNKNOWN")
                logging.info(f"[{game_state_key.upper()}] State 1: Card {first_card_id} ('{first_color}') in Temp1.")
                if first_color in ["UNKNOWN", DETECTION_PERMANENT_FAIL_STATE, "black"]:
                    logging.warning(f"First card {first_card_id} has invalid state '{first_color}'. Returning."); await update_frontend_state(f"Problem with {first_card_id}. Returning.")
                    await from_to(websocket, "temp1", "card", first_card_id); current_flipped.clear(); await update_frontend_state(); continue
                match_id = find_match(first_card_id)
                if match_id is not None:
                    logging.info(f"State 1 -> Strategy: Found known match {match_id}. Removing pair.")
                    await update_frontend_state(f"Found match for '{first_color}': {match_id}. Removing.")
                    success1 = await from_to(websocket, "temp1", "trash", first_card_id)
                    if not success1: logging.error(f"Arm fail temp1->trash {first_card_id}."); await update_frontend_state(f"Arm fail {first_card_id}. Return."); await from_to(websocket, "temp1", "card", first_card_id); current_flipped.clear(); await update_frontend_state(); continue
                    success2 = await from_to(websocket, "card", "trash", match_id)
                    if not success2: logging.error(f"Arm fail card->trash {match_id}. {first_card_id} gone!"); card_states[first_card_id]["isMatched"] = True; await update_frontend_state(f"Arm fail {match_id}."); current_flipped.clear(); await update_frontend_state(); continue
                    card_states[first_card_id]["isMatched"]=True; card_states[match_id]["isMatched"]=True; card_states[match_id]["isFlippedBefore"]=True; card_states[match_id]["color"]=first_color
                    game_state["pairs_found"] = pairs_found + 1; current_flipped.clear(); logging.info(f"Pairs found: {game_state['pairs_found']}")
                    await update_frontend_state(); continue
                else:
                    card_to_flip = choose_random_card()
                    if card_to_flip is not None:
                        logging.info(f"State 1 -> Strategy: No known match. Flipping {card_to_flip}.")
                        await update_frontend_state(f"No match for '{first_color}'. Choosing {card_to_flip}...")
                        detected_color = await detect_color_at_card(card_to_flip)
                        if detected_color == DETECTION_PERMANENT_FAIL_STATE:
                            logging.error(f"PERMANENT color detect fail {card_to_flip}. Return first."); await update_frontend_state(f"Critical detect fail {card_to_flip}. Return first.")
                            card_states[card_to_flip]["isFlippedBefore"]=True; card_states[card_to_flip]["color"]=DETECTION_PERMANENT_FAIL_STATE; card_states[card_to_flip]["isMatched"]=True; game_state["last_detect_fail_id"] = card_to_flip
                            await from_to(websocket, "temp1", "card", first_card_id); current_flipped.clear()
                        elif detected_color == "black":
                            logging.warning(f"Detected black back on {card_to_flip}. Return first."); await update_frontend_state(f"Detected back of {card_to_flip}. Return first.")
                            card_states[card_to_flip]["isFlippedBefore"] = True; card_states[card_to_flip]["color"] = "black"; game_state["last_detect_fail_id"] = card_to_flip
                            await from_to(websocket, "temp1", "card", first_card_id); current_flipped.clear()
                        elif detected_color is not None:
                            logging.info(f"Detected '{detected_color}'. Moving {card_to_flip} to Temp2.")
                            await update_frontend_state(f"Detected '{detected_color}'. Moving {card_to_flip} to Temp2.")
                            success_move = await from_to(websocket, "card", "temp2", card_to_flip)
                            if success_move:
                                card_states[card_to_flip]["color"]=detected_color; card_states[card_to_flip]["isFlippedBefore"]=True
                                if detected_color not in colors_found: colors_found[detected_color] = []
                                if card_to_flip not in colors_found[detected_color]: colors_found[detected_color].append(card_to_flip)
                                current_flipped.append(card_to_flip); logging.info(f"Card {card_to_flip} ('{detected_color}') in Temp2. Flipped: {current_flipped}")
                            else: logging.error(f"Arm fail card->temp2 {card_to_flip}. Return first."); await update_frontend_state(f"Arm fail {card_to_flip}. Return first."); await from_to(websocket, "temp1", "card", first_card_id); current_flipped.clear()
                        else: logging.error(f"Detect_color returned None for {card_to_flip}. Return first."); await update_frontend_state(f"Internal error detect {card_to_flip}. Return first."); await from_to(websocket, "temp1", "card", first_card_id); current_flipped.clear(); game_state["last_detect_fail_id"] = card_to_flip
                        await update_frontend_state()
                    else: logging.warning(f"State 1: No second card available? Return {first_card_id}."); await update_frontend_state(f"No second card. Return {first_card_id}."); await from_to(websocket, "temp1", "card", first_card_id); current_flipped.clear(); await update_frontend_state()
            elif len(current_flipped) == 2:
                card1_id, card2_id = current_flipped[0], current_flipped[1]
                color1, color2 = card_states.get(card1_id,{}).get("color"), card_states.get(card2_id,{}).get("color")
                logging.info(f"[{game_state_key.upper()}] State 2: Cards {card1_id} ('{color1}') & {card2_id} ('{color2}') in Temp1/2. Checking.")
                is_match = (color1 is not None and color1 not in ["UNKNOWN", DETECTION_PERMANENT_FAIL_STATE, "black"] and color2 is not None and color2 not in ["UNKNOWN", DETECTION_PERMANENT_FAIL_STATE, "black"] and color1 == color2)
                if is_match:
                    logging.info(f"MATCH FOUND (Color): {card1_id}&{card2_id} ('{color1}'). Removing from Temp.")
                    await update_frontend_state(f"Match: {color1}! Removing {card1_id}&{card2_id}.")
                    success1 = await from_to(websocket, "temp1", "trash", card1_id)
                    if not success1: logging.error(f"Arm fail temp1->trash {card1_id}. Return both."); await update_frontend_state(f"Arm fail {card1_id}. Return both."); await from_to(websocket, "temp1", "card", card1_id); await from_to(websocket, "temp2", "card", card2_id); current_flipped.clear(); await update_frontend_state(); continue
                    success2 = await from_to(websocket, "temp2", "trash", card2_id)
                    if not success2: logging.error(f"Arm fail temp2->trash {card2_id}. {card1_id} gone!"); card_states[card1_id]["isMatched"]=True; await update_frontend_state(f"Arm fail {card2_id}. Return it."); await from_to(websocket, "temp2", "card", card2_id); current_flipped.clear(); await update_frontend_state(); continue
                    card_states[card1_id]["isMatched"]=True; card_states[card2_id]["isMatched"]=True; game_state["pairs_found"] = pairs_found + 1; current_flipped.clear(); logging.info(f"Pairs found: {game_state['pairs_found']}")
                    await update_frontend_state()
                else:
                    logging.info(f"NO MATCH (Color) ('{color1}' vs '{color2}'). Returning {card1_id}&{card2_id}.")
                    await update_frontend_state(f"No match. Returning {card1_id}&{card2_id}.")
                    success_ret1 = await from_to(websocket, "temp1", "card", card1_id)
                    success_ret2 = await from_to(websocket, "temp2", "card", card2_id)
                    if not (success_ret1 and success_ret2): logging.error(f"Failed return {card1_id} or {card2_id}."); await update_frontend_state(f"Warn: Arm fail return {card1_id}/{card2_id}.")
                    current_flipped.clear(); await update_frontend_state()
                    if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "cards_hidden", "payload": [card1_id, card2_id]})
            elif len(current_flipped) > 2:
                logging.error(f"Invalid State: {len(current_flipped)} flipped: {current_flipped}. Recovering.")
                await update_frontend_state("Error: Invalid state. Returning cards.")
                if len(current_flipped)>0 and card_states.get(current_flipped[0],{}).get("color"): await from_to(websocket, "temp1", "card", current_flipped[0])
                if len(current_flipped)>1 and card_states.get(current_flipped[1],{}).get("color"): await from_to(websocket, "temp2", "card", current_flipped[1])
                current_flipped.clear(); await update_frontend_state()

            if game_state.get("pairs_found", 0) >= CARD_COUNT // 2:
                logging.info(f"[{game_state_key.upper()}] Game Finished! All pairs found.")
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_json({"type": "game_over", "payload": "All pairs found!"})
                game_state["running"] = False
                await asyncio.sleep(1.0)
                break
    except WebSocketDisconnect: logging.info(f"[{game_state_key.upper()}] WS disconnected."); game_state["running"] = False
    except asyncio.CancelledError: logging.info(f"[{game_state_key.upper()}] Task cancelled."); game_state["running"] = False
    except Exception as e:
        logging.error(f"[{game_state_key.upper()}] CRITICAL Loop Error: {e}", exc_info=True); game_state["running"] = False
        if websocket.client_state == WebSocketState.CONNECTED:
            try: await websocket.send_json({"type": "error", "payload": f"Game Error: {e}"})
            except Exception: pass
    finally:
        logging.info(f"[{game_state_key.upper()}] Cleaning up runner..."); game_state["running"] = False
        if camera_task and not camera_task.done():
            logging.info(f"[{game_state_key.upper()}] Cancelling camera task..."); camera_task.cancel()
            try: await camera_task
            except asyncio.CancelledError: logging.info(f"[{game_state_key.upper()}] Camera task cancelled.")
            except Exception as task_e: logging.error(f"[{game_state_key.upper()}] Camera task cancel/await error: {task_e}")
        if locals().get('init_home_success', True) and game_state.get("esp32_client"):
            logging.info(f"[{game_state_key.upper()}] Ensuring arm home post-game."); await from_to(websocket, "home", "home", -1)
        else: logging.warning(f"[{game_state_key.upper()}] Skipping final home (init failed or no client).")
        if hasattr(websocket, 'game_version_context'):
            delattr(websocket, 'game_version_context')
        logging.info(f"[{game_state_key.upper()}] Runner finished cleanup.")


# --- GameSession Adapter Class ---
class MemoryMatching:
    def __init__(self, config=None, esp32_client=None): # Accepts esp32_client
        self.mode = "color"
        if config and isinstance(config, dict) and "mode" in config:
            if config["mode"] in ("color", "yolo"):
                self.mode = config["mode"]
        self._initialized = False
        self.esp32_client = esp32_client # Store the passed client
        # Track if game is running (used by main.py)
        self.running = False
        # Game state tracking
        self.game_state = {
            "running": False,
            "esp32_client": esp32_client,
            "switch_command_sent": False,
            "card_states": {},
            "colors_found" if self.mode == "color" else "objects_found": {},
            "pairs_found": 0,
            "current_flipped_cards": [],
            "last_detect_fail_id": None
        }
        # Add reference to game lock
        self.game_lock = asyncio.Lock()
        print(f"MemoryMatching initialized with mode: {self.mode}")
        
        if self.mode == "yolo":
            asyncio.create_task(self._ensure_init())
        
    async def _ensure_init(self):
        global yolo_model_global
        if self.mode == "yolo" and yolo_model_global is None:
            print("Loading YOLO model on startup...")
            await load_yolo_model_on_startup()
        self._initialized = True

    async def process_frame(self, frame_bytes):
        # (Simplified stateless processing, no arm interaction here)
        if not self._initialized: await self._ensure_init()
        results = {}
        try:
            np_arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            corners = find_board_corners(frame)
            warped = transform_board(frame, corners) if corners is not None else None
            if warped is None: return {"status": "error", "message": "Board not detected"}
            board_state = []
            for card_id in range(GRID_ROWS * GRID_COLS):
                if self.mode == "color":
                    row, col = card_id // GRID_COLS, card_id % GRID_COLS
                    cell_width = COLOR_BOARD_DETECT_WIDTH // GRID_COLS; cell_height = COLOR_BOARD_DETECT_HEIGHT // GRID_ROWS
                    x1, y1 = col * cell_width, row * cell_height; x2, y2 = x1 + cell_width, y1 + cell_height
                    padding = 5
                    roi_x1, roi_y1 = max(0, x1 + padding), max(0, y1 + padding)
                    roi_x2, roi_y2 = min(COLOR_BOARD_DETECT_WIDTH-1, x2-padding), min(COLOR_BOARD_DETECT_HEIGHT-1, y2-padding)
                    cell_roi = warped[roi_y1:roi_y2, roi_x1:roi_x2]
                    color = None
                    if cell_roi.size > 0:
                        hsv_roi = cv2.cvtColor(cell_roi, cv2.COLOR_BGR2HSV); detected_colors_count = {}
                        for color_def in COLOR_RANGES:
                            color_name, total_mask = color_def['name'], np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
                            for l_b, u_b in zip(color_def['lower'], color_def['upper']):
                                total_mask = cv2.bitwise_or(total_mask, cv2.inRange(hsv_roi, np.array(l_b), np.array(u_b)))
                            pixel_count = cv2.countNonZero(total_mask)
                            if pixel_count > 0: detected_colors_count[color_name] = pixel_count
                        dominant_color, max_pixels = None, 0
                        for cn, pc in detected_colors_count.items():
                            if cn != "black" and pc > max_pixels: max_pixels, dominant_color = pc, cn
                        color = dominant_color or "unknown"
                    board_state.append(color)
                else: # yolo mode
                    if yolo_model_global is None: board_state.append("no_model"); continue
                    row, col = card_id // GRID_COLS, card_id % GRID_COLS
                    cell_width = COLOR_BOARD_DETECT_WIDTH // GRID_COLS; cell_height = COLOR_BOARD_DETECT_HEIGHT // GRID_ROWS
                    x1, y1 = col * cell_width, row * cell_height; x2, y2 = x1 + cell_width, y1 + cell_height
                    padding = 5
                    roi_x1, roi_y1 = max(0, x1 + padding), max(0, y1 + padding)
                    roi_x2, roi_y2 = min(COLOR_BOARD_DETECT_WIDTH-1, x2-padding), min(COLOR_BOARD_DETECT_HEIGHT-1, y2-padding)
                    card_roi = warped[roi_y1:roi_y2, roi_x1:roi_x2]
                    label = None
                    if card_roi.size > 0:
                        try:
                            target_indices = [i for i,lbl in enumerate(yolo_model_global.names.values()) if lbl.lower() in YOLO_TARGET_LABELS]
                            results_yolo = yolo_model_global.predict(card_roi, conf=0.45, verbose=False, device='cpu', classes=target_indices if target_indices else None)
                            detected_object_label, highest_conf = None, 0.0
                            for result in results_yolo:
                                boxes, names = getattr(result, 'boxes', None), getattr(result, 'names', {})
                                if boxes:
                                    for box in boxes:
                                        cls_t, conf_t = getattr(box, 'cls', None), getattr(box, 'conf', None)
                                        if cls_t is not None and conf_t is not None and cls_t.numel()>0 and conf_t.numel()>0:
                                            lbl_idx, score = int(cls_t[0].item()), conf_t[0].item()
                                            lbl_name = names.get(lbl_idx, f"unk_{lbl_idx}").lower()
                                            if lbl_name in YOLO_TARGET_LABELS and score > highest_conf:
                                                highest_conf, detected_object_label = score, lbl_name
                            label = detected_object_label or "unknown"
                        except Exception: label = "error"
                    board_state.append(label)
            return {"status": "ok", "mode": self.mode, "board": board_state}
        except Exception as e: return {"status": "error", "message": str(e)}

    async def run_game(self, websocket: WebSocket):
        """Main entry point called by main.py's WebSocket endpoint"""
        if self.game_lock.locked():
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json({
                    "type": "error", 
                    "payload": f"{self.mode.capitalize()} game is busy. Please try again later."
                })
            return
            
        async with self.game_lock:
            self.running = True
            self.game_state["running"] = True
            self.game_state["esp32_client"] = self.esp32_client
            
            # Set context on websocket for from_to function
            setattr(websocket, 'game_version_context', self.mode)
            
            if self.mode == "yolo":
                await run_yolo_game(websocket)
            else: # Default to color
                await run_color_game(websocket)
            
            self.running = False
            self.game_state["running"] = False
    
    def stop(self):
        """Called when the WebSocket disconnects"""
        self.running = False
        if "running" in self.game_state:
            self.game_state["running"] = False
            
            
