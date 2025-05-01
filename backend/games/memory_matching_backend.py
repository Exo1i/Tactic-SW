import asyncio
import base64
import json
import logging
import random
import threading
import time
from typing import Dict, Any, Optional, List, Tuple
import os # Added for path checking

import cv2
import numpy as np
import serial
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from starlette.websockets import WebSocketState
from starlette.exceptions import HTTPException # Added for catch_all


# --- Configuration ---
SERIAL_PORT = '/dev/ttyUSB0'  # Adjust as needed (e.g., 'COM3' on Windows)
BAUD_RATE = 9600
CAMERA_URL = 'http://192.168.2.19:4747/mjpegfeed?640x480' # Primary Camera URL

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
# REMOVED: YOLO_GRID_TOP, YOLO_GRID_LEFT, YOLO_GRID_WIDTH, YOLO_GRID_HEIGHT - Now uses warped board
YOLO_FRAME_WIDTH = 640 # Still needed for camera capture setting
YOLO_FRAME_HEIGHT = 480 # Still needed for camera capture setting

# --- Board/Color Detection Specific Constants ---
COLOR_DEFINITIONS = { "red": (0, 0, 255), "yellow": (0, 255, 255), "green": (0, 255, 0), "blue": (255, 0, 0) }
COLOR_RANGES = [
    {'name': 'red', 'bgr': (0, 0, 255), 'lower': [(0, 120, 70), (170, 120, 70)], 'upper': [(10, 255, 255), (179, 255, 255)]},
    {'name': 'yellow', 'bgr': (0, 255, 255), 'lower': [(20, 100, 100)], 'upper': [(35, 255, 255)]},
    {'name': 'green', 'bgr': (0, 255, 0), 'lower': [(40, 40, 40)], 'upper': [(85, 255, 255)]},
    {'name': 'blue', 'bgr': (255, 0, 0), 'lower': [(95, 100, 50)], 'upper': [(130, 255, 255)]},
    {'name': 'black', 'bgr': (0, 0, 0), 'lower': [(0, 0, 0)], 'upper': [(180, 255, 50)]}
]
COLOR_CELL_THRESHOLD = 500 # For color detection
COLOR_BOARD_DETECT_WIDTH = 400 # Width of the warped board image
COLOR_BOARD_DETECT_HEIGHT = 200 # Height of the warped board image

# --- Globals ---
ser: Optional[serial.Serial] = None
serial_lock = threading.Lock()
latest_frame: Optional[np.ndarray] = None # Raw frame from camera
latest_transformed_frame: Optional[np.ndarray] = None # Warped board frame
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

# --- Serial Communication ---
def setup_serial() -> bool:
    """Initializes the serial connection."""
    global ser
    if ser and ser.is_open:
        logging.info("Serial port already open.")
        return True
    try:
        logging.info(f"Attempting to open serial port {SERIAL_PORT} at {BAUD_RATE} baud...")
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2) # Arduino boot/reset delay
        if not ser.is_open:
            raise serial.SerialException("Port opened but test failed (not is_open).")
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        time.sleep(0.1)
        if ser.in_waiting > 0:
            initial_data = ser.read(ser.in_waiting).decode(errors='replace')
            logging.info(f"Cleared initial serial data: {initial_data.strip()}")
        logging.info(f"Serial port {SERIAL_PORT} opened successfully!")
        return True
    except serial.SerialException as e:
        logging.error(f"Serial Error opening/configuring port {SERIAL_PORT}: {e}")
        ser = None
        return False
    except Exception as e:
        logging.error(f"Unexpected error opening serial port {SERIAL_PORT}: {e}")
        ser = None
        return False

# MODIFIED send_arm_command_sync with RETRIES
def send_arm_command_sync(degree1: int, degree2: int, degree3: int, magnet: int, movement: int) -> Optional[str]:
    """
    Synchronous function to send an arm command and wait for 'done'.
    Includes retries on failure. Returns response string or None on persistent error.
    """
    global ser
    if not ser or not ser.is_open:
        logging.error("Serial port not available or not open for sending command.")
        return None

    # Input validation
    if not (0 <= degree1 <= 180 and 0 <= degree2 <= 180 and 0 <= degree3 <= 180):
        logging.error(f"Invalid servo degrees: ({degree1}, {degree2}, {degree3}). Must be 0-180.")
        return None
    if magnet not in [0, 1]:
        logging.error(f"Invalid magnet value: {magnet}. Must be 0 or 1.")
        return None

    command = f"{degree1},{degree2},{degree3},{magnet},{movement}\n"
    command_bytes = command.encode('utf-8')
    command_strip = command.strip() # For logging

    attempt = 0
    while attempt < ARM_MAX_RETRIES:
        attempt += 1
        logging.info(f"Sending command (Attempt {attempt}/{ARM_MAX_RETRIES}): {command_strip}")

        with serial_lock: # Acquire lock for each attempt's send/receive cycle
            try:
                if not ser or not ser.is_open: # Re-check inside lock
                    logging.error(f"Serial port became unavailable before attempt {attempt}.")
                    # Break retry loop if serial is gone
                    attempt = ARM_MAX_RETRIES # Force loop exit
                    continue # Go to end of while loop check

                ser.reset_input_buffer() # Clear buffer before reading response
                ser.write(command_bytes)
                ser.flush() # Ensure data is sent

                # Wait for the "done" response
                response = b''
                start_time = time.time()
                timeout = 12.0 # Timeout FOR THIS ATTEMPT

                while time.time() - start_time < timeout:
                    if ser.in_waiting > 0:
                        chunk = ser.read(ser.in_waiting)
                        response += chunk
                        if b"done" in response.lower():
                            break
                    time.sleep(0.02) # Short sleep to avoid busy-waiting

                decoded_response = response.decode('utf-8', errors='replace').strip()
                logging.debug(f"Attempt {attempt} raw response: {response}")
                logging.info(f"Attempt {attempt} decoded response: '{decoded_response}'")

                if "done" in decoded_response.lower():
                    logging.info(f"Command '{command_strip}' successful on attempt {attempt}.")
                    return decoded_response # SUCCESS! Exit function.

                else:
                    logging.warning(f"Command '{command_strip}' attempt {attempt} failed: 'done' not received or timed out. Response: '{decoded_response}'")
                    # Failure on this attempt, loop will continue if attempts remain

            except serial.SerialException as e:
                logging.error(f"Serial communication error during attempt {attempt}: {e}")
                # Assume serial port is bad, try to close and signal persistent failure
                try: ser.close()
                except Exception: pass
                ser = None
                attempt = ARM_MAX_RETRIES # Force loop exit after serial error
            except Exception as e:
                logging.error(f"Unexpected error during serial command attempt {attempt}: {e}", exc_info=True)
                # Loop might continue if attempts remain, maybe it was temporary

        # If this attempt failed and more retries are allowed, wait before next attempt
        if attempt < ARM_MAX_RETRIES:
            logging.info(f"Waiting {ARM_RETRY_DELAY_SECONDS}s before retry...")
            time.sleep(ARM_RETRY_DELAY_SECONDS)

    # If loop finishes without success
    logging.error(f"Command '{command_strip}' FAILED after {ARM_MAX_RETRIES} attempts.")
    return None # PERSISTENT FAILURE

# --- Synchronous Arm Movement Logic (No change needed here, relies on robust send_arm_command_sync) ---
def from_to_sync(src: str, dest: str, card_id: int) -> bool:
    """
    Synchronous function implementing the specific arm movement sequence.
    Uses send_arm_command_sync (which now has retries). Returns True on success, False on failure.
    """
    logging.info(f"Executing SYNC movement sequence: card {card_id} from {src} to {dest}")
    success = True # Assume success initially

    # Input validation (same as before)
    if src not in ["card", "temp1", "temp2", "home"] or dest not in ["card", "temp1", "temp2", "trash", "home"]:
        logging.error(f"Invalid src ('{src}') or dest ('{dest}') location.")
        return False
    if src == "card" or dest == "card":
        if not (0 <= card_id < len(arm_values)):
            logging.error(f"Invalid card_id {card_id} for arm_values length {len(arm_values)}")
            return False

    try:
        # --- Movement Sequences ---
        # The logic is the same, but now each send_arm_command_sync call will retry internally.
        # If any command *persistently* fails (returns None), the success flag becomes False.
        if src == "card" and dest == "temp1":
            logging.debug("Seq: card -> temp1")
            if send_arm_command_sync(arm_values[card_id][0], arm_values[card_id][1], arm_values[card_id][2], 1, 0) is None: success = False # 1. Pick
            if success: time.sleep(ARM_SYNC_STEP_DELAY)
            if success and send_arm_command_sync(arm_home[0], arm_home[1], arm_home[2], 1, 1) is None: success = False # 2. Home w/ item
            if success: time.sleep(ARM_SYNC_STEP_DELAY)
            if success and send_arm_command_sync(arm_temp1[0], arm_temp1[1], arm_temp1[2], 0, 0) is None: success = False # 3. Drop Temp1
            if success: time.sleep(ARM_SYNC_STEP_DELAY)
            if success and send_arm_command_sync(arm_home[0], arm_home[1], arm_home[2], 0, 1) is None: success = False # 4. Home empty

        elif src == "card" and dest == "temp2":
            logging.debug("Seq: card -> temp2")
            if send_arm_command_sync(arm_values[card_id][0], arm_values[card_id][1], arm_values[card_id][2], 1, 0) is None: success = False
            if success: time.sleep(ARM_SYNC_STEP_DELAY)
            if success and send_arm_command_sync(arm_home[0],arm_home[1], arm_home[2], 1, 1) is None: success = False
            if success: time.sleep(ARM_SYNC_STEP_DELAY)
            if success and send_arm_command_sync(arm_temp2[0], arm_temp2[1], arm_temp2[2], 0, 0) is None: success = False
            if success: time.sleep(ARM_SYNC_STEP_DELAY)
            if success and send_arm_command_sync(arm_home[0], arm_home[1], arm_home[2], 0, 1) is None: success = False

        elif src == "card" and dest == "trash":
            logging.debug("Seq: card -> trash")
            if send_arm_command_sync(arm_values[card_id][0], arm_values[card_id][1], arm_values[card_id][2], 1, 0) is None: success = False
            if success: time.sleep(ARM_SYNC_STEP_DELAY)
            if success and send_arm_command_sync(arm_home[0], arm_home[1], arm_home[2], 1, 1) is None: success = False
            if success: time.sleep(ARM_SYNC_STEP_DELAY)
            if success and send_arm_command_sync(arm_trash[0], arm_trash[1], arm_trash[2], 0, 0) is None: success = False
            if success: time.sleep(ARM_SYNC_STEP_DELAY)
            if success and send_arm_command_sync(arm_home[0], arm_home[1], arm_home[2], 0, 1) is None: success = False

        elif src == "temp1" and dest == "trash":
            logging.debug("Seq: temp1 -> trash")
            if send_arm_command_sync(arm_temp1[0], arm_temp1[1], arm_temp1[2], 1, 0) is None: success = False
            if success: time.sleep(ARM_SYNC_STEP_DELAY)
            if success and send_arm_command_sync(arm_home[0], arm_home[1], arm_home[2], 1, 1) is None: success = False
            if success: time.sleep(ARM_SYNC_STEP_DELAY)
            if success and send_arm_command_sync(arm_trash[0], arm_trash[1], arm_trash[2], 0, 0) is None: success = False
            if success: time.sleep(ARM_SYNC_STEP_DELAY)
            if success and send_arm_command_sync(arm_home[0], arm_home[1], arm_home[2], 0, 1) is None: success = False

        elif src == "temp2" and dest == "trash":
            logging.debug("Seq: temp2 -> trash")
            if send_arm_command_sync(arm_temp2[0], arm_temp2[1], arm_temp2[2], 1, 0) is None: success = False
            if success: time.sleep(ARM_SYNC_STEP_DELAY)
            if success and send_arm_command_sync(arm_home[0], arm_home[1], arm_home[2], 1, 1) is None: success = False
            if success: time.sleep(ARM_SYNC_STEP_DELAY)
            if success and send_arm_command_sync(arm_trash[0], arm_trash[1], arm_trash[2], 0, 0) is None: success = False
            if success: time.sleep(ARM_SYNC_STEP_DELAY)
            if success and send_arm_command_sync(arm_home[0], arm_home[1], arm_home[2], 0, 1) is None: success = False

        elif src == "temp1" and dest == "card":
            logging.debug("Seq: temp1 -> card")
            if send_arm_command_sync(arm_temp1[0], arm_temp1[1], arm_temp1[2], 1, 0) is None: success = False
            if success: time.sleep(ARM_SYNC_STEP_DELAY)
            if success and send_arm_command_sync(arm_home[0], arm_home[1], arm_home[2], 1, 1) is None: success = False
            if success: time.sleep(ARM_SYNC_STEP_DELAY)
            if success and send_arm_command_sync(arm_values[card_id][0], arm_values[card_id][1], arm_values[card_id][2], 0, 0) is None: success = False
            if success: time.sleep(ARM_SYNC_STEP_DELAY)
            if success and send_arm_command_sync(arm_home[0], arm_home[1], arm_home[2], 0, 1) is None: success = False

        elif src == "temp2" and dest == "card":
            logging.debug("Seq: temp2 -> card")
            if send_arm_command_sync(arm_temp2[0], arm_temp2[1], arm_temp2[2], 1, 0) is None: success = False
            if success: time.sleep(ARM_SYNC_STEP_DELAY)
            if success and send_arm_command_sync(arm_home[0], arm_home[1], arm_home[2], 1, 1) is None: success = False
            if success: time.sleep(ARM_SYNC_STEP_DELAY)
            if success and send_arm_command_sync(arm_values[card_id][0], arm_values[card_id][1], arm_values[card_id][2], 0, 0) is None: success = False
            if success: time.sleep(ARM_SYNC_STEP_DELAY)
            if success and send_arm_command_sync(arm_home[0], arm_home[1], arm_home[2], 0, 1) is None: success = False

        elif src == "home" and dest == "home":
            logging.debug("Seq: home -> home")
            if send_arm_command_sync(arm_home[0], arm_home[1], arm_home[2], 0, 1) is None: success = False

        else:
            logging.error(f"Invalid/unhandled src/dest combination: {src} -> {dest}")
            success = False

        # --- Post-Sequence Handling ---
        if not success:
            logging.error(f"SYNC movement sequence FAILED: A command failed persistently for card {card_id} ({src} -> {dest})")
            # Attempt safe recovery if sequence failed mid-way
            if not (src == "home" and dest == "home"):
                logging.warning("Attempting to return arm home after sequence failure.")
                send_arm_command_sync(arm_home[0], arm_home[1], arm_home[2], 0, 1) # Try recovery
            return False

        logging.info(f"SYNC movement sequence COMPLETED successfully: card {card_id} ({src} -> {dest})")
        return True

    except Exception as e: # Catch unexpected errors during sequence logic
        logging.error(f"Unexpected error during SYNC sequence ({src} -> {dest}, card {card_id}): {e}", exc_info=True)
        logging.warning("Attempting to return arm home after unexpected sequence error.")
        send_arm_command_sync(arm_home[0], arm_home[1], arm_home[2], 0, 1) # Try recovery
        return False

# --- Asynchronous Wrapper for Arm Movement (No change needed) ---
async def from_to(websocket: WebSocket, src: str, dest: str, card_id: int) -> bool:
    """
    Asynchronous wrapper that runs from_to_sync in a thread and sends WebSocket status.
    """
    action_name = f"{src}_to_{dest}"
    logging.info(f"Initiating ASYNC wrapper for arm movement: card {card_id} [{action_name}]")
    move_successful = False
    try:
        move_successful = await asyncio.to_thread(from_to_sync, src, dest, card_id)
    except Exception as e:
        logging.error(f"Error calling/executing from_to_sync via asyncio.to_thread for {action_name}: {e}", exc_info=True)
        move_successful = False
        logging.warning("Attempting safe return home after thread execution error.")
        try:
            await asyncio.to_thread(send_arm_command_sync, arm_home[0], arm_home[1], arm_home[2], 0, 1)
        except Exception as home_e:
            logging.error(f"Failed to return arm home after thread error: {home_e}")

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

    logging.info(f"ASYNC wrapper finished for {action_name} (card {card_id}). Success: {move_successful}")
    return move_successful

# --- FastAPI App ---
app = FastAPI(title="Memory Matching Game Backend")

# --- YOLO Model Loading (No change needed) ---
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
        except Exception as wu_e: logging.error(f"YOLO warmup failed: {wu_e}"); # Continue anyway?
        logging.info("Global YOLO model loaded and warmed up successfully.")
    except Exception as e:
        logging.error(f"Error loading global YOLO model: {e}", exc_info=True)
        yolo_model_global = None

# --- Board Detection / Transformation Helper Functions (Used by both modes) ---
def find_board_corners(frame: np.ndarray) -> Optional[np.ndarray]: # No change needed
    if frame is None or frame.size == 0: return None
    try:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Adjust these white ranges if needed based on your board and lighting
        lower_white, upper_white = np.array([0, 0, 150]), np.array([180, 70, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        # Adjust min area threshold if necessary
        min_area_threshold = frame.shape[0] * frame.shape[1] * 0.05
        if contour_area < min_area_threshold: return None
        perimeter = cv2.arcLength(largest_contour, True)
        epsilon = 0.03 * perimeter # Adjust epsilon (0.02-0.04 typical) if corner detection is poor
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        if len(approx) == 4:
            corners = np.array([p[0] for p in approx], dtype=np.float32)
            # if not cv2.isContourConvex(approx): return None # Optional check for convexity
            return sort_corners(corners)
        return None
    except Exception as e: logging.error(f"Error finding board corners: {e}"); return None

def sort_corners(corners: np.ndarray) -> np.ndarray: # No change needed
    rect = np.zeros((4, 2), dtype="float32")
    s = corners.sum(axis=1); rect[0] = corners[np.argmin(s)]; rect[2] = corners[np.argmax(s)]
    diff = np.diff(corners, axis=1); rect[1] = corners[np.argmin(diff)]; rect[3] = corners[np.argmax(diff)]
    return rect

def transform_board(frame: np.ndarray, corners: np.ndarray) -> Optional[np.ndarray]: # No change needed
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

# --- Camera Capture (MODIFIED to always attempt board transform) ---
async def capture_frames(websocket: WebSocket, camera_source: str, game_version: str):
    global latest_frame, latest_transformed_frame
    cap = None
    logging.info(f"Starting frame capture task for {game_version} from {camera_source}")
    frame_count = 0
    last_log_time = time.time()

    while game_version in active_games and active_games[game_version].get("running"):
        processed_frame_for_send = None
        warped_board_for_send = None
        corners = None # Reset corners each iteration

        try:
            # --- Camera Opening Logic (Unchanged) ---
            if cap is None or not cap.isOpened():
                logging.info(f"Attempting to open camera source: {camera_source}")
                cap = cv2.VideoCapture(camera_source)
                await asyncio.sleep(1.5) # Give camera time to initialize
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
                if cap is not None: cap.release()
                cap = None
                async with frame_lock: latest_frame, latest_transformed_frame = None, None
                await asyncio.sleep(1); continue

            frame_count += 1
            current_frame_copy = frame.copy()

            # --- Board Detection and Transformation (ALWAYS attempt) ---
            local_latest_transformed = None # Use local var to avoid holding lock too long
            corners = find_board_corners(current_frame_copy)
            if corners is not None:
                warped = transform_board(current_frame_copy, corners)
                if warped is not None:
                    local_latest_transformed = warped
                    warped_board_for_send = warped # Keep a copy for sending
            # else: corners is None, warped not generated

            # --- Update Global Frames ---
            async with frame_lock:
                latest_frame = current_frame_copy
                latest_transformed_frame = local_latest_transformed # Update global warped frame

            # --- Prepare Frames for Sending ---
            processed_frame_for_send = current_frame_copy # Use the raw frame for main view
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(processed_frame_for_send, timestamp, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(processed_frame_for_send, f"Mode: {game_version.upper()}", (processed_frame_for_send.shape[1] - 150, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
            # Draw corners on the raw frame if found
            if corners is not None:
                cv2.polylines(processed_frame_for_send, [np.int32(corners)], isClosed=True, color=(0, 255, 255), thickness=2)

            # --- Encoding and Sending (Unchanged Logic) ---
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
                    # Send transformed frame if available (for both modes now)
                    if jpg_transformed_as_text: payload["transformed_frame"] = jpg_transformed_as_text
                    await websocket.send_json({"type": "frame_update", "payload": payload})
                else: logging.warning("Skipping send: main frame encoding failed.")
            else: logging.warning("WS closed in capture loop."); break

            # --- FPS Logging and Sleep (Unchanged) ---
            current_time = time.time()
            if current_time - last_log_time >= 5.0:
                fps = frame_count / (current_time - last_log_time)
                logging.info(f"Camera FPS: {fps:.2f}"); frame_count = 0; last_log_time = current_time
            await asyncio.sleep(0.035) # Adjust sleep based on desired capture rate

        except WebSocketDisconnect: logging.info(f"WS disconnected gracefully in capture loop."); break
        except Exception as e:
            logging.error(f"Error in frame capture loop for {game_version}: {e}", exc_info=True)
            await asyncio.sleep(1) # Avoid busy-looping on persistent errors

    # --- Cleanup (Unchanged) ---
    if cap is not None:
        try: cap.release(); logging.info(f"Camera released for {game_version}.")
        except Exception as e: logging.error(f"Error releasing camera for {game_version}: {e}")

    async with frame_lock: latest_frame, latest_transformed_frame = None, None
    logging.info(f"Frame capture task stopped for {game_version}.")


# --- YOLO Helper Functions ---
def yolo_assign_color(label: str) -> Tuple[int, int, int]: # Unused, kept for reference
    color_map = {'orange': (0, 165, 255), 'apple': (0, 0, 255), 'cat': (255, 0, 0), 'car': (0, 255, 255), 'umbrella': (0, 255, 0), 'banana': (0, 215, 255), 'fire hydrant': (0, 0, 139), 'person': (128, 0, 128)}
    return color_map.get(label.lower(), (200, 200, 200))

# MODIFIED detect_object_at_card with RETRIES and using TRANSFORMED FRAME
async def detect_object_at_card(card_id: int) -> Optional[str]:
    """
    Detects the highest confidence target object in a specific card region
    USING THE TRANSFORMED (WARPED) BOARD VIEW.
    Retries detection up to DETECTION_MAX_RETRIES times on failure (no board, no object).
    Returns object label string, DETECTION_PERMANENT_FAIL_STATE, or None if model missing.
    """
    global latest_transformed_frame, yolo_model_global # Use transformed frame now
    if yolo_model_global is None:
        logging.error("YOLO model not loaded. Cannot perform detection.")
        return None # Indicate model is missing

    attempt = 0
    while attempt < DETECTION_MAX_RETRIES:
        attempt += 1
        logging.info(f"YOLO Detection Attempt {attempt}/{DETECTION_MAX_RETRIES} for card {card_id}")

        # --- Get Warped Board Frame ---
        warped_board = None
        async with frame_lock: # Get latest WARPED frame
            if latest_transformed_frame is not None:
                warped_board = latest_transformed_frame.copy()

        if warped_board is None:
            logging.warning(f"Attempt {attempt}: No transformed board available for YOLO detection on card {card_id}.")
            if attempt < DETECTION_MAX_RETRIES:
                logging.info(f"Waiting {DETECTION_RETRY_DELAY_SECONDS}s for board detection...")
                await asyncio.sleep(DETECTION_RETRY_DELAY_SECONDS)
            continue # Retry getting board

        try:
            # --- ROI Calculation (on warped board) ---
            # Uses the same logic as detect_color_at_card
            row, col = card_id // GRID_COLS, card_id % GRID_COLS
            cell_width = COLOR_BOARD_DETECT_WIDTH // GRID_COLS
            cell_height = COLOR_BOARD_DETECT_HEIGHT // GRID_ROWS
            x1, y1 = col * cell_width, row * cell_height
            x2, y2 = x1 + cell_width, y1 + cell_height
            padding = 5 # Padding inside the cell
            roi_x1 = max(0, x1 + padding)
            roi_y1 = max(0, y1 + padding)
            roi_x2 = min(COLOR_BOARD_DETECT_WIDTH - 1, x2 - padding)
            roi_y2 = min(COLOR_BOARD_DETECT_HEIGHT - 1, y2 - padding)

            if roi_x1 >= roi_x2 or roi_y1 >= roi_y2:
                logging.warning(f"Attempt {attempt}: Invalid ROI calculated for YOLO card {card_id} on warped board. Skipping this attempt.")
                if attempt < DETECTION_MAX_RETRIES: await asyncio.sleep(DETECTION_RETRY_DELAY_SECONDS)
                continue # Retry

            card_roi = warped_board[roi_y1:roi_y2, roi_x1:roi_x2]
            if card_roi.size == 0:
                logging.warning(f"Attempt {attempt}: Empty ROI extracted for YOLO card {card_id} from warped board. Skipping this attempt.")
                if attempt < DETECTION_MAX_RETRIES: await asyncio.sleep(DETECTION_RETRY_DELAY_SECONDS)
                continue # Retry

            # --- Prediction (in thread) ---
            def sync_predict(model, roi):
                target_indices = [i for i, lbl in enumerate(model.names.values()) if lbl.lower() in YOLO_TARGET_LABELS]
                # Run prediction on the smaller ROI
                return model.predict(roi, conf=0.45, verbose=False, device='cpu', classes=target_indices if target_indices else None)

            try:
                # Run prediction in a separate thread to avoid blocking asyncio loop
                results = await asyncio.to_thread(sync_predict, yolo_model_global, card_roi)
            except Exception as predict_err:
                logging.error(f"Attempt {attempt}: Error during YOLO predict thread execution for card {card_id}: {predict_err}", exc_info=True)
                if attempt < DETECTION_MAX_RETRIES: await asyncio.sleep(DETECTION_RETRY_DELAY_SECONDS)
                continue # Retry

            # --- Process Results ---
            detected_object_label = None
            highest_conf = 0.0
            if results:
                for result in results: # Iterate through prediction results (usually just one for single image)
                    boxes = getattr(result, 'boxes', None) # Access the Boxes object
                    names = getattr(result, 'names', {})  # Class names dictionary
                    if boxes: # Check if boxes were found
                        for box in boxes: # Iterate through detected boxes
                            cls_tensor = getattr(box, 'cls', None) # Class index tensor
                            conf_tensor = getattr(box, 'conf', None) # Confidence score tensor
                            if cls_tensor is not None and conf_tensor is not None and cls_tensor.numel() > 0 and conf_tensor.numel() > 0:
                                try:
                                    label_index = int(cls_tensor[0].item()) # Get class index as int
                                    score = conf_tensor[0].item()       # Get confidence score as float
                                    label = names.get(label_index, f"unknown_idx_{label_index}").lower() # Get label name
                                    # Check if it's one of our targets and has highest confidence so far
                                    if label in YOLO_TARGET_LABELS and score > highest_conf:
                                        highest_conf = score
                                        detected_object_label = label
                                except Exception as proc_err:
                                    logging.error(f"Attempt {attempt}: Error processing YOLO box data for card {card_id}: {proc_err}")

            # --- Check if detection SUCCEEDED on this attempt ---
            if detected_object_label:
                logging.info(f"Successful YOLO Detection on attempt {attempt}: '{detected_object_label}' (conf: {highest_conf:.2f}) for card {card_id} (on warped board)")
                return detected_object_label # SUCCESS! Exit function.
            else:
                logging.warning(f"Attempt {attempt}: No target object detected meeting criteria on card {card_id} (warped board).")
                # Loop will continue if attempts remain

        except cv2.error as cv_err:
            logging.error(f"Attempt {attempt}: OpenCV error during YOLO detection processing for card {card_id}: {cv_err}", exc_info=True)
        except Exception as e: # Catch errors in ROI calc, etc.
            logging.error(f"Attempt {attempt}: Unexpected error during YOLO detection processing for card {card_id}: {e}", exc_info=True)
            # Loop will continue if attempts remain

        # If detection failed on this attempt and more retries allowed, wait
        if attempt < DETECTION_MAX_RETRIES:
            logging.info(f"Waiting {DETECTION_RETRY_DELAY_SECONDS}s before next YOLO detection attempt...")
            await asyncio.sleep(DETECTION_RETRY_DELAY_SECONDS)

    # If loop finishes without success
    logging.error(f"YOLO Detection FAILED permanently for card {card_id} after {DETECTION_MAX_RETRIES} attempts (using warped board).")
    return DETECTION_PERMANENT_FAIL_STATE # Indicate persistent failure

# --- Color Detection Helper Functions ---
# MODIFIED detect_color_at_card with RETRIES (Logic unchanged, just confirming retry structure)
async def detect_color_at_card(card_id: int) -> Optional[str]:
    """
    Detects dominant color in a specific card region using the warped board view.
    Retries detection up to DETECTION_MAX_RETRIES times on failure (no board, no color, black).
    Returns color name string, 'black', DETECTION_PERMANENT_FAIL_STATE, or None.
    """
    global latest_transformed_frame # Uses transformed frame
    attempt = 0
    while attempt < DETECTION_MAX_RETRIES:
        attempt += 1
        logging.info(f"Color Detection Attempt {attempt}/{DETECTION_MAX_RETRIES} for card {card_id}")

        warped_board = None
        async with frame_lock: # Get latest warped frame
            if latest_transformed_frame is not None:
                warped_board = latest_transformed_frame.copy()

        if warped_board is None:
            logging.warning(f"Attempt {attempt}: No transformed board available for Color detection on card {card_id}.")
            if attempt < DETECTION_MAX_RETRIES:
                logging.info(f"Waiting {DETECTION_RETRY_DELAY_SECONDS}s for board detection...")
                await asyncio.sleep(DETECTION_RETRY_DELAY_SECONDS)
            continue # Retry getting board

        try:
            # --- ROI Calculation (on warped board) ---
            row, col = card_id // GRID_COLS, card_id % GRID_COLS
            cell_width = COLOR_BOARD_DETECT_WIDTH // GRID_COLS
            cell_height = COLOR_BOARD_DETECT_HEIGHT // GRID_ROWS
            x1, y1 = col * cell_width, row * cell_height
            x2, y2 = x1 + cell_width, y1 + cell_height
            padding = 5
            roi_x1 = max(0, x1 + padding)
            roi_y1 = max(0, y1 + padding)
            roi_x2 = min(COLOR_BOARD_DETECT_WIDTH - 1, x2 - padding)
            roi_y2 = min(COLOR_BOARD_DETECT_HEIGHT - 1, y2 - padding)

            if roi_x1 >= roi_x2 or roi_y1 >= roi_y2:
                logging.warning(f"Attempt {attempt}: Invalid ROI for Color card {card_id}. Skipping this attempt.")
                if attempt < DETECTION_MAX_RETRIES: await asyncio.sleep(DETECTION_RETRY_DELAY_SECONDS)
                continue # Retry

            cell_roi = warped_board[roi_y1:roi_y2, roi_x1:roi_x2]
            if cell_roi.size == 0:
                logging.warning(f"Attempt {attempt}: Empty ROI for Color card {card_id}. Skipping this attempt.")
                if attempt < DETECTION_MAX_RETRIES: await asyncio.sleep(DETECTION_RETRY_DELAY_SECONDS)
                continue # Retry

            # --- Color Analysis ---
            hsv_roi = cv2.cvtColor(cell_roi, cv2.COLOR_BGR2HSV)
            detected_colors_count: Dict[str, int] = {}
            for color_def in COLOR_RANGES:
                color_name = color_def['name']
                total_mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
                # Handle multiple ranges (e.g., for red wrapping around hue)
                for l_bound, u_bound in zip(color_def['lower'], color_def['upper']):
                    lower = np.array(l_bound)
                    upper = np.array(u_bound)
                    mask_part = cv2.inRange(hsv_roi, lower, upper)
                    total_mask = cv2.bitwise_or(total_mask, mask_part)
                pixel_count = cv2.countNonZero(total_mask)
                if pixel_count > 0: detected_colors_count[color_name] = pixel_count

            # --- Determine Dominant Color ---
            dominant_color = None
            max_pixels = 0
            # Check black first with a slightly adjusted threshold if needed
            black_threshold = COLOR_CELL_THRESHOLD * 0.7

            if "black" in detected_colors_count and detected_colors_count["black"] > black_threshold:
                logging.info(f"Attempt {attempt}: Card {card_id} detected as 'black' (back).")
                # Treat black as a valid detection for THIS attempt. Game logic decides action.
                return "black" # Return black immediately

            # Find dominant non-black color meeting threshold
            for color_name, pixel_count in detected_colors_count.items():
                if color_name != "black" and pixel_count >= COLOR_CELL_THRESHOLD:
                    if pixel_count > max_pixels:
                        max_pixels = pixel_count
                        dominant_color = color_name

            # --- Check if detection SUCCEEDED (found a valid face color) ---
            if dominant_color:
                logging.info(f"Successful Color Detection on attempt {attempt}: '{dominant_color}' (pixels: {max_pixels}) for card {card_id}")
                return dominant_color # SUCCESS! Exit function.
            else:
                # Log potentially detected colors below threshold for debugging
                relevant_counts = {k:v for k,v in detected_colors_count.items() if k!='black' and v > 10}
                logging.warning(f"Attempt {attempt}: No dominant face color found meeting threshold on card {card_id}. Counts(<Thr): {relevant_counts}")
                # Loop will continue if attempts remain

        except cv2.error as cv_err:
            logging.error(f"Attempt {attempt}: OpenCV error during color detection: {cv_err}", exc_info=True)
        except Exception as e:
            logging.error(f"Attempt {attempt}: Unexpected error during color detection: {e}", exc_info=True)
            # Loop will continue if attempts remain

        # If detection failed (no board, no color, error) and more retries allowed, wait
        if attempt < DETECTION_MAX_RETRIES:
            logging.info(f"Waiting {DETECTION_RETRY_DELAY_SECONDS}s before next color detection attempt...")
            await asyncio.sleep(DETECTION_RETRY_DELAY_SECONDS)

    # If loop finishes without success (finding a dominant face color or black back)
    logging.error(f"Color Detection FAILED permanently for card {card_id} after {DETECTION_MAX_RETRIES} attempts.")
    return DETECTION_PERMANENT_FAIL_STATE # Indicate persistent failure


# --- Game Logic Runners (No major changes needed in structure, rely on detection results) ---

async def run_yolo_game(websocket: WebSocket):
    """Runs the YOLO version of the Memory Game with robust detection (using warped board)."""
    game_state_key = "yolo"
    logging.info(f"[{game_state_key.upper()}] Starting Game Logic...")
    game_state = active_games[game_state_key]

    if yolo_model_global is None: # Check model loaded
        logging.error(f"[{game_state_key.upper()}] YOLO Model not loaded. Cannot start."); game_state["running"] = False
        if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "error", "payload": "YOLO model missing."})
        return

    # Initialize state (same as before)
    game_state.update({
        "card_states": {i: {"isFlippedBefore": False, "object": None, "isMatched": False} for i in range(CARD_COUNT)},
        "objects_found": {obj: [] for obj in YOLO_TARGET_LABELS}, "pairs_found": 0, "current_flipped_cards": [],
        "running": True, "last_detect_fail_id": None,
    })
    logging.info(f"[{game_state_key.upper()}] Initialized game state.")

    if websocket.client_state == WebSocketState.CONNECTED: # Send initial state
        try:
            await websocket.send_json({"type": "game_state", "payload": {k: game_state[k] for k in ["card_states", "pairs_found", "current_flipped_cards"]}})
            await websocket.send_json({"type": "message", "payload": "YOLO Game Started (Board Detect Mode). Initializing arm..."})
        except Exception as send_e: logging.error(f"[{game_state_key.upper()}] Send initial error: {send_e}"); game_state["running"] = False

    camera_task = None
    init_home_success = False
    if game_state.get("running"): # Start camera and init arm if still running
        # Camera task now always tries board detection
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
            objects_found = game_state.get("objects_found", {})
            pairs_found = game_state.get("pairs_found", 0)
            logging.info(f"[{game_state_key.upper()}] Loop Start: Flipped={current_flipped}, Pairs={pairs_found}/{CARD_COUNT // 2}")

            # Helper functions (choose_random_card, find_pair, find_match) remain the same logic
            # --- Helper: Update Frontend ---
            async def update_frontend_state(extra_message: Optional[str] = None):
                payload = {"card_states": card_states, "pairs_found": game_state.get("pairs_found", 0), "current_flipped": current_flipped}
                if websocket.client_state == WebSocketState.CONNECTED:
                    try:
                        await websocket.send_json({"type": "game_state", "payload": payload})
                        if extra_message: await websocket.send_json({"type": "message", "payload": extra_message})
                    except Exception as send_e: logging.error(f"[{game_state_key.upper()}] Send state failed: {send_e}")
            # --- Helper: Choose Card ---
            def choose_random_card() -> Optional[int]:
                available = [i for i, s in card_states.items() if not s.get("isMatched") and i not in current_flipped and s.get("object") != DETECTION_PERMANENT_FAIL_STATE] # Exclude permanent fails
                if not available: return None
                never_flipped = [i for i in available if not card_states[i].get("isFlippedBefore")]
                if never_flipped:
                    chosen = random.choice(never_flipped)
                    # Avoid immediately retrying a card that just failed detection if others available
                    if chosen == game_state.get("last_detect_fail_id") and len(never_flipped) > 1:
                        chosen = random.choice([c for c in never_flipped if c != chosen])
                    game_state["last_detect_fail_id"] = None # Reset fail ID once a choice is made
                    return chosen
                # If all available cards have been flipped before
                previously_flipped = available
                if previously_flipped:
                    chosen = random.choice(previously_flipped)
                    if chosen == game_state.get("last_detect_fail_id") and len(previously_flipped) > 1:
                        chosen = random.choice([c for c in previously_flipped if c != chosen])
                    game_state["last_detect_fail_id"] = None
                    return chosen
                return None # Should not happen if available is not empty
            # --- Helper: Find Known Pair ---
            def find_pair() -> Optional[Tuple[int, int]]:
                for obj, ids in objects_found.items():
                    if obj in [None, "DETECT_FAIL", DETECTION_PERMANENT_FAIL_STATE]: continue # Ignore invalid/failed objects
                    # Find cards with this object that HAVE been flipped and are NOT matched/currently flipped
                    valid = [i for i in ids if card_states.get(i,{}).get("isFlippedBefore") and not card_states.get(i,{}).get("isMatched") and i not in current_flipped]
                    if len(valid) >= 2:
                        logging.info(f"[{game_state_key.upper()}] Found known pair for '{obj}': {valid[0]},{valid[1]}")
                        return valid[0], valid[1] # Return the first two found
                return None
            # --- Helper: Find Match for a given card ID ---
            def find_match(card_id_to_match: int) -> Optional[int]:
                obj = card_states.get(card_id_to_match,{}).get("object")
                if obj in [None, "DETECT_FAIL", DETECTION_PERMANENT_FAIL_STATE]: return None # Cannot match invalid object
                # Look for other cards with the same object
                for other_id in objects_found.get(obj, []):
                    # Ensure it's not the same card, has been flipped, not matched, not currently flipped
                    if other_id != card_id_to_match and \
                            card_states.get(other_id,{}).get("isFlippedBefore") and \
                            not card_states.get(other_id,{}).get("isMatched") and \
                            other_id not in current_flipped:
                        logging.info(f"[{game_state_key.upper()}] Found match for {card_id_to_match} ('{obj}'): {other_id}")
                        return other_id
                return None

            # --- ================== ---
            # --- Game State Machine ---
            # --- ================== ---

            # === STATE 0: No cards flipped ===
            if len(current_flipped) == 0:
                known_pair = find_pair()
                if known_pair:
                    # Strategy: Remove a known pair
                    card1_id, card2_id = known_pair; obj = card_states[card1_id]['object']
                    logging.info(f"[{game_state_key.upper()}] State 0 -> Strategy: Remove known pair {card1_id}&{card2_id} ('{obj}')")
                    await update_frontend_state(f"Found pair: {obj}. Removing {card1_id}&{card2_id}.")
                    # Move first card to trash
                    success1 = await from_to(websocket, "card", "trash", card1_id)
                    if not success1:
                        logging.error(f"[{game_state_key.upper()}] Move fail {card1_id} to trash.")
                        await update_frontend_state(f"Arm fail move {card1_id}.")
                        # Maybe add retry logic here or break? For now, continue to next loop iteration
                        continue
                    # Move second card to trash
                    success2 = await from_to(websocket, "card", "trash", card2_id)
                    if not success2:
                        logging.error(f"[{game_state_key.upper()}] Move fail {card2_id} to trash. {card1_id} already gone!")
                        # Mark first as matched anyway, update state, continue
                        card_states[card1_id]["isMatched"] = True
                        await update_frontend_state(f"Arm fail move {card2_id}.")
                        continue
                    # Both moved successfully
                    card_states[card1_id]["isMatched"] = True; card_states[card2_id]["isMatched"] = True
                    game_state["pairs_found"] = pairs_found + 1; logging.info(f"Pairs found: {game_state['pairs_found']}")
                    await update_frontend_state()
                    continue # Go to next game loop iteration
                else:
                    # Strategy: Flip a random card
                    card_to_flip = choose_random_card()
                    if card_to_flip is not None:
                        logging.info(f"[{game_state_key.upper()}] State 0 -> Strategy: Flip card {card_to_flip}.")
                        await update_frontend_state(f"Choosing card {card_to_flip}. Detecting...")
                        # --- Detection Call (Now Retries Internally, uses warped board) ---
                        detected_obj = await detect_object_at_card(card_to_flip)

                        if detected_obj == DETECTION_PERMANENT_FAIL_STATE:
                            logging.error(f"[{game_state_key.upper()}] PERMANENT detection failure for card {card_to_flip}. Marking unusable.")
                            await update_frontend_state(f"Critical detection fail on {card_to_flip}. Skipping card.")
                            card_states[card_to_flip]["isFlippedBefore"] = True # Mark as looked at
                            card_states[card_to_flip]["object"] = DETECTION_PERMANENT_FAIL_STATE
                            card_states[card_to_flip]["isMatched"] = True # Treat as unusable/matched to remove from play
                            game_state["last_detect_fail_id"] = card_to_flip # Record failure to avoid immediate retry
                            await update_frontend_state() # Update UI
                        elif detected_obj is not None: # Includes successful detection after retries
                            logging.info(f"Detected '{detected_obj}'. Moving card {card_to_flip} to Temp1.")
                            await update_frontend_state(f"Detected '{detected_obj}'. Moving card {card_to_flip} to Temp1.")
                            success_move = await from_to(websocket, "card", "temp1", card_to_flip)
                            if success_move:
                                # Update card state
                                card_states[card_to_flip]["object"] = detected_obj
                                card_states[card_to_flip]["isFlippedBefore"] = True
                                # Add to found objects list
                                if detected_obj not in objects_found: objects_found[detected_obj] = []
                                if card_to_flip not in objects_found[detected_obj]: objects_found[detected_obj].append(card_to_flip)
                                # Add to currently flipped list
                                current_flipped.append(card_to_flip)
                                logging.info(f"Card {card_to_flip} ('{detected_obj}') in Temp1. Flipped: {current_flipped}")
                            else:
                                logging.error(f"[{game_state_key.upper()}] Arm fail: {card_to_flip} to Temp1.")
                                await update_frontend_state(f"Arm fail move {card_to_flip}.")
                                # Card stays on board, state not updated, will be chosen again later maybe
                            await update_frontend_state()
                        else: # Should not happen if model loaded, detect returns string or PERMA_FAIL
                            logging.error(f"[{game_state_key.upper()}] detect_object_at_card returned unexpected None for card {card_to_flip}.")
                            await update_frontend_state(f"Internal error during detection for {card_to_flip}.")
                            game_state["last_detect_fail_id"] = card_to_flip # Record failure
                    else:
                        logging.info(f"[{game_state_key.upper()}] State 0: No known pair & no available card to flip. Waiting.")
                        await asyncio.sleep(1) # Prevent busy loop if game gets stuck

            # === STATE 1: One card flipped ===
            elif len(current_flipped) == 1:
                first_card_id = current_flipped[0]
                first_object = card_states.get(first_card_id,{}).get("object", "UNKNOWN")
                logging.info(f"[{game_state_key.upper()}] State 1: Card {first_card_id} ('{first_object}') in Temp1.")

                # Handle edge case: card in temp somehow failed detection previously
                if first_object in ["UNKNOWN", DETECTION_PERMANENT_FAIL_STATE]:
                    logging.warning(f"First card {first_card_id} has invalid state '{first_object}'. Returning.");
                    await update_frontend_state(f"Problem with card {first_card_id}. Returning.")
                    await from_to(websocket, "temp1", "card", first_card_id) # Attempt to return
                    current_flipped.clear(); await update_frontend_state(); continue

                # Strategy: Check if a match for the first card is known
                match_id = find_match(first_card_id)
                if match_id is not None:
                    logging.info(f"[{game_state_key.upper()}] State 1 -> Strategy: Found known match {match_id}. Removing pair.")
                    await update_frontend_state(f"Found match for '{first_object}': {match_id}. Removing.")
                    # Move first card (from temp) to trash
                    success1 = await from_to(websocket, "temp1", "trash", first_card_id)
                    if not success1:
                        logging.error(f"Arm fail temp1->trash {first_card_id}. Returning card instead.")
                        await update_frontend_state(f"Arm fail temp1->trash {first_card_id}. Returning card.")
                        await from_to(websocket, "temp1", "card", first_card_id) # Attempt return
                        current_flipped.clear(); await update_frontend_state(); continue
                    # Move second card (from board) to trash
                    success2 = await from_to(websocket, "card", "trash", match_id)
                    if not success2:
                        logging.error(f"Arm fail card->trash {match_id}. {first_card_id} already gone!")
                        card_states[first_card_id]["isMatched"]=True # Mark first as matched anyway
                        await update_frontend_state(f"Arm fail move {match_id}.")
                        current_flipped.clear(); await update_frontend_state(); continue
                    # Both moved successfully
                    card_states[first_card_id]["isMatched"]=True; card_states[match_id]["isMatched"]=True
                    # Ensure second card state reflects match discovery
                    card_states[match_id]["isFlippedBefore"]=True; card_states[match_id]["object"]=first_object
                    game_state["pairs_found"] = pairs_found + 1; current_flipped.clear(); logging.info(f"Pairs found: {game_state['pairs_found']}")
                    await update_frontend_state(); continue
                else:
                    # Strategy: No known match, flip a second random card
                    card_to_flip = choose_random_card()
                    if card_to_flip is not None:
                        logging.info(f"[{game_state_key.upper()}] State 1 -> Strategy: No known match. Flipping {card_to_flip}.")
                        await update_frontend_state(f"No match for '{first_object}'. Choosing {card_to_flip}...")
                        # --- Detection Call for second card ---
                        detected_obj = await detect_object_at_card(card_to_flip)

                        if detected_obj == DETECTION_PERMANENT_FAIL_STATE:
                            logging.error(f"[{game_state_key.upper()}] PERMANENT detection fail on second card {card_to_flip}.")
                            await update_frontend_state(f"Critical detect fail {card_to_flip}. Returning first card.")
                            # Mark second card unusable
                            card_states[card_to_flip]["isFlippedBefore"]=True; card_states[card_to_flip]["object"]=DETECTION_PERMANENT_FAIL_STATE; card_states[card_to_flip]["isMatched"]=True
                            game_state["last_detect_fail_id"] = card_to_flip
                            # Return the first card
                            await from_to(websocket, "temp1", "card", first_card_id)
                            current_flipped.clear()
                        elif detected_obj is not None: # Success
                            logging.info(f"Detected '{detected_obj}'. Moving {card_to_flip} to Temp2.")
                            await update_frontend_state(f"Detected '{detected_obj}'. Moving {card_to_flip} to Temp2.")
                            success_move = await from_to(websocket, "card", "temp2", card_to_flip)
                            if success_move:
                                # Update second card state
                                card_states[card_to_flip]["object"]=detected_obj; card_states[card_to_flip]["isFlippedBefore"]=True
                                # Add to found objects
                                if detected_obj not in objects_found: objects_found[detected_obj] = []
                                if card_to_flip not in objects_found[detected_obj]: objects_found[detected_obj].append(card_to_flip)
                                # Add to currently flipped list
                                current_flipped.append(card_to_flip); logging.info(f"Card {card_to_flip} ('{detected_obj}') in Temp2. Flipped: {current_flipped}")
                            else:
                                logging.error(f"Arm fail card->temp2 {card_to_flip}. Returning first card.")
                                await update_frontend_state(f"Arm fail move {card_to_flip}. Returning first card.")
                                await from_to(websocket, "temp1", "card", first_card_id); current_flipped.clear()
                        else: # Should not happen
                            logging.error(f"Detect returned unexpected None for {card_to_flip}. Returning first.")
                            await update_frontend_state(f"Internal error detect {card_to_flip}. Returning first card.")
                            await from_to(websocket, "temp1", "card", first_card_id); current_flipped.clear()
                            game_state["last_detect_fail_id"] = card_to_flip
                        await update_frontend_state()
                    else:
                        logging.warning(f"State 1: No second card available? Returning {first_card_id}.")
                        await update_frontend_state(f"No second card found. Returning {first_card_id}.")
                        await from_to(websocket, "temp1", "card", first_card_id); current_flipped.clear(); await update_frontend_state()

            # === STATE 2: Two cards flipped ===
            elif len(current_flipped) == 2:
                card1_id, card2_id = current_flipped[0], current_flipped[1]
                obj1 = card_states.get(card1_id,{}).get("object")
                obj2 = card_states.get(card2_id,{}).get("object")
                logging.info(f"[{game_state_key.upper()}] State 2: Cards {card1_id} ('{obj1}') & {card2_id} ('{obj2}') in Temp1/2. Checking for match.")

                # Check if both objects are valid and identical
                is_match = (obj1 is not None and obj1 != DETECTION_PERMANENT_FAIL_STATE and \
                            obj2 is not None and obj2 != DETECTION_PERMANENT_FAIL_STATE and \
                            obj1 == obj2)

                if is_match:
                    # Strategy: Match found, remove both from temp locations
                    logging.info(f"MATCH FOUND: {card1_id}&{card2_id} ('{obj1}'). Removing from Temp1/2.")
                    await update_frontend_state(f"Match: {obj1}! Removing {card1_id}&{card2_id}.")
                    # Move first from Temp1 to trash
                    success1 = await from_to(websocket, "temp1", "trash", card1_id)
                    if not success1:
                        logging.error(f"Arm fail temp1->trash {card1_id}. Returning both cards.")
                        await update_frontend_state(f"Arm fail temp1->trash {card1_id}. Return both.")
                        await from_to(websocket, "temp1", "card", card1_id) # Attempt return
                        await from_to(websocket, "temp2", "card", card2_id) # Attempt return
                        current_flipped.clear(); await update_frontend_state(); continue
                    # Move second from Temp2 to trash
                    success2 = await from_to(websocket, "temp2", "trash", card2_id)
                    if not success2:
                        logging.error(f"Arm fail temp2->trash {card2_id}. {card1_id} already gone!")
                        card_states[card1_id]["isMatched"]=True # Mark first as matched
                        await update_frontend_state(f"Arm fail temp2->trash {card2_id}. Return it.")
                        await from_to(websocket, "temp2", "card", card2_id) # Attempt return second card
                        current_flipped.clear(); await update_frontend_state(); continue
                    # Both removed successfully
                    card_states[card1_id]["isMatched"]=True; card_states[card2_id]["isMatched"]=True
                    game_state["pairs_found"] = pairs_found + 1; current_flipped.clear(); logging.info(f"Pairs found: {game_state['pairs_found']}")
                    await update_frontend_state()
                    # Optional: Send message that cards are being permanently hidden
                    # if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "cards_hidden", "payload": [card1_id, card2_id]})
                else:
                    # Strategy: No match, return both cards from temp locations
                    logging.info(f"NO MATCH ('{obj1}' vs '{obj2}'). Returning {card1_id} from Temp1 & {card2_id} from Temp2.")
                    await update_frontend_state(f"No match. Returning {card1_id}&{card2_id}.")
                    # Return first card
                    success_ret1 = await from_to(websocket, "temp1", "card", card1_id)
                    # Return second card
                    success_ret2 = await from_to(websocket, "temp2", "card", card2_id)
                    if not (success_ret1 and success_ret2):
                        # Log error but continue, game state might be inconsistent if arm failed
                        logging.error(f"Failed returning one or both cards: {card1_id} (Success: {success_ret1}), {card2_id} (Success: {success_ret2}).")
                        await update_frontend_state(f"Warning: Arm fail returning cards {card1_id}/{card2_id}.")
                    current_flipped.clear(); await update_frontend_state()
                    # Send message that cards are being hidden (returned)
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_json({"type": "cards_hidden", "payload": [card1_id, card2_id]})

            # === Invalid State ===
            elif len(current_flipped) > 2: # Should not happen, recovery attempt
                logging.error(f"Invalid State: {len(current_flipped)} cards flipped: {current_flipped}. Attempting recovery.")
                await update_frontend_state("Error: Invalid state detected. Returning cards.")
                # Try returning cards based on expected temp locations
                if len(current_flipped)>0 and card_states.get(current_flipped[0], {}).get("object"): await from_to(websocket, "temp1", "card", current_flipped[0])
                if len(current_flipped)>1 and card_states.get(current_flipped[1], {}).get("object"): await from_to(websocket, "temp2", "card", current_flipped[1])
                # Add more returns if somehow > 2 cards are tracked
                current_flipped.clear(); await update_frontend_state()

            # --- Check Game End ---
            if game_state.get("pairs_found", 0) >= CARD_COUNT // 2:
                logging.info(f"[{game_state_key.upper()}] Game Finished! All pairs found.")
                if websocket.client_state == WebSocketState.CONNECTED:
                    await update_frontend_state() # Send final state
                    await websocket.send_json({"type": "game_over", "payload": "Congratulations! All pairs found."})
                game_state["running"] = False # Signal loop to stop
                await asyncio.sleep(1.0) # Short pause before breaking
                break

    # --- Exception Handling & Cleanup (Structure remains the same) ---
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
        if locals().get('init_home_success', True): # Check if homing succeeded initially
            logging.info(f"[{game_state_key.upper()}] Ensuring arm home post-game."); await from_to(websocket, "home", "home", -1)
        else: logging.warning(f"[{game_state_key.upper()}] Skipping final home (initial homing failed).")
        logging.info(f"[{game_state_key.upper()}] Runner finished cleanup.")


async def run_color_game(websocket: WebSocket):
    """Runs the Color version of the Memory Game with robust detection."""
    game_state_key = "color"
    logging.info(f"[{game_state_key.upper()}] Starting Game Logic...")
    game_state = active_games[game_state_key]

    # Initialize state
    game_state.update({
        "card_states": {i: {"isFlippedBefore": False, "color": None, "isMatched": False} for i in range(CARD_COUNT)},
        "colors_found": {color: [] for color in COLOR_DEFINITIONS.keys()}, "pairs_found": 0, "current_flipped_cards": [],
        "running": True, "last_detect_fail_id": None,
    })
    logging.info(f"[{game_state_key.upper()}] Initialized game state.")

    if websocket.client_state == WebSocketState.CONNECTED:
        try:
            await websocket.send_json({"type": "game_state", "payload": {k: game_state[k] for k in ["card_states", "pairs_found", "current_flipped_cards"]}})
            await websocket.send_json({"type": "message", "payload": "Color Game Started. Initializing arm..."})
        except Exception as send_e: logging.error(f"[{game_state_key.upper()}] Send initial error: {send_e}"); game_state["running"] = False

    camera_task = None
    init_home_success = False
    if game_state.get("running"):
        # Camera task always attempts board detection
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

            # Helper functions (choose_random_card, find_pair, find_match) same logic as YOLO but use 'color' field
            # --- Helper: Update Frontend ---
            async def update_frontend_state(extra_message: Optional[str] = None):
                payload = {"card_states": card_states, "pairs_found": game_state.get("pairs_found", 0), "current_flipped": current_flipped}
                if websocket.client_state == WebSocketState.CONNECTED:
                    try:
                        await websocket.send_json({"type": "game_state", "payload": payload})
                        if extra_message: await websocket.send_json({"type": "message", "payload": extra_message})
                    except Exception as send_e: logging.error(f"[{game_state_key.upper()}] Send state failed: {send_e}")
            # --- Helper: Choose Card ---
            def choose_random_card() -> Optional[int]:
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
            # --- Helper: Find Known Pair ---
            def find_pair() -> Optional[Tuple[int, int]]:
                for color, ids in colors_found.items():
                    # Ignore invalid colors ('black' is also ignored here as it's the back)
                    if color in [None, "DETECT_FAIL", DETECTION_PERMANENT_FAIL_STATE, "black"]: continue
                    valid = [i for i in ids if card_states.get(i,{}).get("isFlippedBefore") and not card_states.get(i,{}).get("isMatched") and i not in current_flipped]
                    if len(valid) >= 2:
                        logging.info(f"[{game_state_key.upper()}] Found known pair for '{color}': {valid[0]},{valid[1]}")
                        return valid[0], valid[1]
                return None
            # --- Helper: Find Match ---
            def find_match(card_id_to_match: int) -> Optional[int]:
                color = card_states.get(card_id_to_match,{}).get("color")
                if color in [None, "DETECT_FAIL", DETECTION_PERMANENT_FAIL_STATE, "black"]: return None
                for other_id in colors_found.get(color, []):
                    if other_id != card_id_to_match and \
                            card_states.get(other_id,{}).get("isFlippedBefore") and \
                            not card_states.get(other_id,{}).get("isMatched") and \
                            other_id not in current_flipped:
                        logging.info(f"[{game_state_key.upper()}] Found match for {card_id_to_match} ('{color}'): {other_id}")
                        return other_id
                return None

            # --- ====================== ---
            # --- Color Game State Machine ---
            # --- ====================== ---

            # === STATE 0: No cards flipped ===
            if len(current_flipped) == 0:
                known_pair = find_pair()
                if known_pair:
                    # Strategy: Remove known pair
                    card1_id, card2_id = known_pair; color = card_states[card1_id]['color']
                    logging.info(f"[{game_state_key.upper()}] State 0 -> Strategy: Remove known pair {card1_id}&{card2_id} ('{color}')")
                    await update_frontend_state(f"Found pair: {color}. Removing {card1_id}&{card2_id}.")
                    success1 = await from_to(websocket, "card", "trash", card1_id)
                    if not success1: logging.error(f"Move fail {card1_id}"); await update_frontend_state(f"Arm fail {card1_id}."); continue
                    success2 = await from_to(websocket, "card", "trash", card2_id)
                    if not success2: logging.error(f"Move fail {card2_id}. {card1_id} gone!"); card_states[card1_id]["isMatched"] = True; await update_frontend_state(f"Arm fail {card2_id}."); continue
                    card_states[card1_id]["isMatched"]=True; card_states[card2_id]["isMatched"]=True
                    game_state["pairs_found"] = pairs_found + 1; logging.info(f"Pairs found: {game_state['pairs_found']}")
                    await update_frontend_state(); continue
                else:
                    # Strategy: Flip random card
                    card_to_flip = choose_random_card()
                    if card_to_flip is not None:
                        logging.info(f"[{game_state_key.upper()}] State 0 -> Strategy: Flip card {card_to_flip}.")
                        await update_frontend_state(f"Choosing card {card_to_flip}. Detecting color...")
                        # --- Detection Call (Retries Internally) ---
                        detected_color = await detect_color_at_card(card_to_flip)

                        if detected_color == DETECTION_PERMANENT_FAIL_STATE:
                            logging.error(f"[{game_state_key.upper()}] PERMANENT color detect fail {card_to_flip}.")
                            await update_frontend_state(f"Critical color detect fail {card_to_flip}. Skipping.")
                            card_states[card_to_flip]["isFlippedBefore"]=True; card_states[card_to_flip]["color"]=DETECTION_PERMANENT_FAIL_STATE; card_states[card_to_flip]["isMatched"]=True
                            game_state["last_detect_fail_id"] = card_to_flip
                            await update_frontend_state()
                        elif detected_color == "black":
                            # Detected the back, don't move, just record state
                            logging.warning(f"Detected black back on {card_to_flip}. Marking, not moving.")
                            await update_frontend_state(f"Detected back of card {card_to_flip}.")
                            card_states[card_to_flip]["isFlippedBefore"] = True
                            card_states[card_to_flip]["color"] = "black" # Record it saw black
                            game_state["last_detect_fail_id"] = card_to_flip # Treat black as a temporary fail for choice logic
                            await update_frontend_state()
                        elif detected_color is not None: # Success (found a face color)
                            logging.info(f"Detected '{detected_color}'. Moving {card_to_flip} to Temp1.")
                            await update_frontend_state(f"Detected '{detected_color}'. Moving {card_to_flip} to Temp1.")
                            success_move = await from_to(websocket, "card", "temp1", card_to_flip)
                            if success_move:
                                card_states[card_to_flip]["color"]=detected_color; card_states[card_to_flip]["isFlippedBefore"]=True
                                if detected_color not in colors_found: colors_found[detected_color] = []
                                if card_to_flip not in colors_found[detected_color]: colors_found[detected_color].append(card_to_flip)
                                current_flipped.append(card_to_flip); logging.info(f"Card {card_to_flip} ('{detected_color}') in Temp1. Flipped: {current_flipped}")
                            else:
                                logging.error(f"Arm fail card->temp1 {card_to_flip}."); await update_frontend_state(f"Arm fail {card_to_flip}.")
                            await update_frontend_state()
                        else: # Should not happen
                            logging.error(f"Detect_color returned unexpected None for {card_to_flip}.")
                            await update_frontend_state(f"Internal error detect {card_to_flip}.")
                            game_state["last_detect_fail_id"] = card_to_flip
                    else:
                        logging.info(f"State 0: No pair & no available card."); await asyncio.sleep(1)

            # === STATE 1: One card flipped ===
            elif len(current_flipped) == 1:
                first_card_id = current_flipped[0]; first_color = card_states.get(first_card_id,{}).get("color", "UNKNOWN")
                logging.info(f"[{game_state_key.upper()}] State 1: Card {first_card_id} ('{first_color}') in Temp1.")

                # Handle invalid state in temp
                if first_color in ["UNKNOWN", DETECTION_PERMANENT_FAIL_STATE, "black"]:
                    logging.warning(f"First card {first_card_id} has invalid state '{first_color}'. Returning."); await update_frontend_state(f"Problem with {first_card_id}. Returning.")
                    await from_to(websocket, "temp1", "card", first_card_id); current_flipped.clear(); await update_frontend_state(); continue

                # Strategy: Check for known match
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
                    # Strategy: Flip second random card
                    card_to_flip = choose_random_card()
                    if card_to_flip is not None:
                        logging.info(f"State 1 -> Strategy: No known match. Flipping {card_to_flip}.")
                        await update_frontend_state(f"No match for '{first_color}'. Choosing {card_to_flip}...")
                        # --- Detection Call for second card ---
                        detected_color = await detect_color_at_card(card_to_flip)

                        if detected_color == DETECTION_PERMANENT_FAIL_STATE:
                            logging.error(f"PERMANENT color detect fail {card_to_flip}. Return first."); await update_frontend_state(f"Critical detect fail {card_to_flip}. Return first.")
                            card_states[card_to_flip]["isFlippedBefore"]=True; card_states[card_to_flip]["color"]=DETECTION_PERMANENT_FAIL_STATE; card_states[card_to_flip]["isMatched"]=True
                            game_state["last_detect_fail_id"] = card_to_flip
                            await from_to(websocket, "temp1", "card", first_card_id); current_flipped.clear()
                        elif detected_color == "black":
                            # Detected back, return first card
                            logging.warning(f"Detected black back on {card_to_flip}. Return first."); await update_frontend_state(f"Detected back of {card_to_flip}. Return first.")
                            card_states[card_to_flip]["isFlippedBefore"] = True; card_states[card_to_flip]["color"] = "black"
                            game_state["last_detect_fail_id"] = card_to_flip
                            await from_to(websocket, "temp1", "card", first_card_id); current_flipped.clear()
                        elif detected_color is not None: # Success (found face color)
                            logging.info(f"Detected '{detected_color}'. Moving {card_to_flip} to Temp2.")
                            await update_frontend_state(f"Detected '{detected_color}'. Moving {card_to_flip} to Temp2.")
                            success_move = await from_to(websocket, "card", "temp2", card_to_flip)
                            if success_move:
                                card_states[card_to_flip]["color"]=detected_color; card_states[card_to_flip]["isFlippedBefore"]=True
                                if detected_color not in colors_found: colors_found[detected_color] = []
                                if card_to_flip not in colors_found[detected_color]: colors_found[detected_color].append(card_to_flip)
                                current_flipped.append(card_to_flip); logging.info(f"Card {card_to_flip} ('{detected_color}') in Temp2. Flipped: {current_flipped}")
                            else:
                                logging.error(f"Arm fail card->temp2 {card_to_flip}. Return first."); await update_frontend_state(f"Arm fail {card_to_flip}. Return first.")
                                await from_to(websocket, "temp1", "card", first_card_id); current_flipped.clear()
                        else: # Should not happen
                            logging.error(f"Detect_color returned None for {card_to_flip}. Return first."); await update_frontend_state(f"Internal error detect {card_to_flip}. Return first.")
                            await from_to(websocket, "temp1", "card", first_card_id); current_flipped.clear()
                            game_state["last_detect_fail_id"] = card_to_flip
                        await update_frontend_state()
                    else:
                        logging.warning(f"State 1: No second card available? Return {first_card_id}."); await update_frontend_state(f"No second card. Return {first_card_id}."); await from_to(websocket, "temp1", "card", first_card_id); current_flipped.clear(); await update_frontend_state()

            # === STATE 2: Two cards flipped ===
            elif len(current_flipped) == 2:
                card1_id, card2_id = current_flipped[0], current_flipped[1]
                color1 = card_states.get(card1_id,{}).get("color")
                color2 = card_states.get(card2_id,{}).get("color")
                logging.info(f"[{game_state_key.upper()}] State 2: Cards {card1_id} ('{color1}') & {card2_id} ('{color2}') in Temp1/2. Checking.")

                # Check if both are valid face colors and match
                is_match = (color1 is not None and color1 not in ["UNKNOWN", DETECTION_PERMANENT_FAIL_STATE, "black"] and \
                            color2 is not None and color2 not in ["UNKNOWN", DETECTION_PERMANENT_FAIL_STATE, "black"] and \
                            color1 == color2)

                if is_match:
                    # Strategy: Match, remove both from temp
                    logging.info(f"MATCH FOUND (Color): {card1_id}&{card2_id} ('{color1}'). Removing from Temp.")
                    await update_frontend_state(f"Match: {color1}! Removing {card1_id}&{card2_id}.")
                    success1 = await from_to(websocket, "temp1", "trash", card1_id)
                    if not success1: logging.error(f"Arm fail temp1->trash {card1_id}. Return both."); await update_frontend_state(f"Arm fail {card1_id}. Return both."); await from_to(websocket, "temp1", "card", card1_id); await from_to(websocket, "temp2", "card", card2_id); current_flipped.clear(); await update_frontend_state(); continue
                    success2 = await from_to(websocket, "temp2", "trash", card2_id)
                    if not success2: logging.error(f"Arm fail temp2->trash {card2_id}. {card1_id} gone!"); card_states[card1_id]["isMatched"]=True; await update_frontend_state(f"Arm fail {card2_id}. Return it."); await from_to(websocket, "temp2", "card", card2_id); current_flipped.clear(); await update_frontend_state(); continue
                    card_states[card1_id]["isMatched"]=True; card_states[card2_id]["isMatched"]=True
                    game_state["pairs_found"] = pairs_found + 1; current_flipped.clear(); logging.info(f"Pairs found: {game_state['pairs_found']}")
                    await update_frontend_state()
                    # Optional: if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "cards_hidden", "payload": [card1_id, card2_id]})
                else:
                    # Strategy: No match, return both
                    logging.info(f"NO MATCH (Color) ('{color1}' vs '{color2}'). Returning {card1_id}&{card2_id}.")
                    await update_frontend_state(f"No match. Returning {card1_id}&{card2_id}.")
                    success_ret1 = await from_to(websocket, "temp1", "card", card1_id)
                    success_ret2 = await from_to(websocket, "temp2", "card", card2_id)
                    if not (success_ret1 and success_ret2): logging.error(f"Failed return {card1_id} or {card2_id}."); await update_frontend_state(f"Warn: Arm fail return {card1_id}/{card2_id}.")
                    current_flipped.clear(); await update_frontend_state()
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_json({"type": "cards_hidden", "payload": [card1_id, card2_id]})

            # === Invalid State ===
            elif len(current_flipped) > 2:
                # Recovery attempt
                logging.error(f"Invalid State: {len(current_flipped)} flipped: {current_flipped}. Recovering.")
                await update_frontend_state("Error: Invalid state. Returning cards.")
                if len(current_flipped)>0 and card_states.get(current_flipped[0],{}).get("color"): await from_to(websocket, "temp1", "card", current_flipped[0])
                if len(current_flipped)>1 and card_states.get(current_flipped[1],{}).get("color"): await from_to(websocket, "temp2", "card", current_flipped[1])
                current_flipped.clear(); await update_frontend_state()

            # --- Check Game End ---
            if game_state.get("pairs_found", 0) >= CARD_COUNT // 2:
                logging.info(f"[{game_state_key.upper()}] Game Finished! All pairs found.")
                if websocket.client_state == WebSocketState.CONNECTED:
                    await update_frontend_state()
                    await websocket.send_json({"type": "game_over", "payload": "Congratulations! All pairs found."})
                game_state["running"] = False; await asyncio.sleep(1.0); break

    # --- Exception Handling & Cleanup (Structure remains the same) ---
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
        if locals().get('init_home_success', True):
            logging.info(f"[{game_state_key.upper()}] Ensuring arm home post-game."); await from_to(websocket, "home", "home", -1)
        else: logging.warning(f"[{game_state_key.upper()}] Skipping final home (init failed).")
        logging.info(f"[{game_state_key.upper()}] Runner finished cleanup.")


# --- WebSocket Endpoint (No change needed) ---
@app.websocket("/ws/{game_version}")
async def websocket_endpoint(websocket: WebSocket, game_version: str):
    client_ip = websocket.client.host if websocket.client else "unknown"
    await websocket.accept()
    logging.info(f"WebSocket connection accepted from {client_ip} for game version: '{game_version}'")

    # Validate game version
    if game_version not in ["color", "yolo"]:
        logging.error(f"Invalid game version '{game_version}' requested by {client_ip}.")
        await websocket.send_json({"type": "error", "payload": f"Invalid game version '{game_version}'. Use 'color' or 'yolo'."})
        await websocket.close(code=1008); return # 1008 = Policy Violation

    lock = game_locks[game_version]
    if lock.locked():
        logging.warning(f"{game_version.capitalize()} game already in progress. Rejecting connection from {client_ip}.")
        await websocket.send_json({"type": "error", "payload": f"{game_version.capitalize()} game is busy. Please try again later."})
        await websocket.close(code=1008); return

    acquired_lock, runner_task = False, None
    try:
        logging.info(f"Attempting to acquire lock for {game_version} game by {client_ip}...")
        async with lock:
            acquired_lock = True
            logging.info(f"Lock acquired for {game_version} game by {client_ip}.")

            # Setup serial connection *after* acquiring lock
            if not setup_serial():
                logging.error(f"Serial port setup failed for {game_version} game start ({client_ip}).")
                await websocket.send_json({"type": "error", "payload": "Failed to initialize serial connection to arm."})
                await websocket.close(code=1011); return # 1011 = Internal Server Error

            # Initialize game state and start runner task
            active_games[game_version] = {"running": True} # Mark as active
            if game_version == "yolo":
                runner_task = asyncio.create_task(run_yolo_game(websocket))
            elif game_version == "color":
                runner_task = asyncio.create_task(run_color_game(websocket))
            # else: # Should be caught by initial validation, but good practice
            #    raise ValueError(f"Internal Error: Invalid game version '{game_version}' reached runner creation")

            logging.info(f"Game runner task created for {game_version} requested by {client_ip}.")

            # Monitor loop: Keep connection open while runner is active
            while True:
                # Check if runner task finished
                if runner_task.done():
                    logging.info(f"{game_version} runner task finished for {client_ip}.")
                    try:
                        runner_task.result() # Raise exceptions from the task if any occurred
                    except asyncio.CancelledError:
                        logging.info(f"{game_version} runner task was cancelled for {client_ip}.")
                    except Exception as runner_exception:
                        logging.error(f"{game_version} runner task failed with exception: {runner_exception}", exc_info=True)
                        # Send error to client if possible
                        if websocket.client_state == WebSocketState.CONNECTED:
                            try: await websocket.send_json({"type": "error", "payload": f"Game ended due to server error."})
                            except Exception as send_err: logging.error(f"Failed to send runner error to {client_ip}: {send_err}")
                    break # Exit monitor loop

                # Check if client disconnected
                if websocket.client_state != WebSocketState.CONNECTED:
                    logging.info(f"WebSocket disconnected by client {client_ip}. Stopping runner for {game_version}.")
                    # Signal runner to stop and cancel task
                    if game_version in active_games: active_games[game_version]["running"] = False
                    if runner_task and not runner_task.done(): runner_task.cancel()
                    break # Exit monitor loop

                # Keepalive or handle client messages if needed (currently just waits)
                await asyncio.sleep(0.5)

            logging.info(f"Monitor loop finished for {game_version} by {client_ip}. Lock scope ending.")

    except WebSocketDisconnect:
        # Client disconnected abruptly before or during lock acquisition/runner start
        logging.info(f"WebSocket disconnected abruptly by {client_ip} for {game_version}.")
        # If runner task exists and isn't done, cancel it
        if runner_task and not runner_task.done():
            if game_version in active_games: active_games[game_version]["running"] = False
            runner_task.cancel()
    except Exception as e:
        # Catch-all for unexpected errors in the WebSocket endpoint itself
        logging.error(f"Unexpected error in WebSocket endpoint for {game_version} ({client_ip}): {e}", exc_info=True)
        # Try sending an error message if connected
        if websocket.client_state == WebSocketState.CONNECTED:
            try: await websocket.send_json({"type": "error", "payload": f"Server connection error: {e}"})
            except Exception: pass
        # Ensure runner task is cancelled if it exists
        if runner_task and not runner_task.done():
            if game_version in active_games: active_games[game_version]["running"] = False
            runner_task.cancel()
    finally:
        # --- Cleanup, runs regardless of how the endpoint exits ---
        logging.info(f"--- Cleaning up WebSocket endpoint for {game_version} ({client_ip}) ---")

        # Ensure the runner task is awaited/cancelled properly
        if runner_task:
            try:
                if not runner_task.done():
                    logging.warning(f"Runner task for {game_version} wasn't done, ensuring cancellation.")
                    runner_task.cancel()
                # Wait briefly for the task to finish cleanup, handle potential exceptions
                await asyncio.wait_for(runner_task, timeout=5.0)
            except asyncio.CancelledError:
                logging.info(f"Runner task {game_version} cleanup confirmed cancelled.")
            except asyncio.TimeoutError:
                logging.error(f"Timeout waiting for runner task {game_version} cleanup.")
            except Exception as task_clean_e:
                logging.error(f"Error during final await of runner task {game_version}: {task_clean_e}")

        # Clear the active game state
        if game_version in active_games:
            active_games[game_version] = {} # Reset state
            logging.info(f"Cleared active game state for {game_version}.")

        # Close serial port if it's open (important to free the resource)
        # Use a local variable to avoid potential race if another thread tries setup_serial
        current_ser = ser
        if current_ser and current_ser.is_open:
            logging.info(f"Closing serial port {SERIAL_PORT} after {game_version} game.")
            try:
                current_ser.close()
                # Only set global ser to None if we successfully closed *this* instance
                if ser == current_ser:
                    ser = None
                logging.info("Serial port closed successfully.")
            except Exception as close_e:
                logging.error(f"Error closing serial port: {close_e}")
                ser = None # Assume it's unusable now

        # Ensure WebSocket is closed
        if websocket.client_state != WebSocketState.DISCONNECTED:
            try:
                await websocket.close(code=1000) # 1000 = Normal Closure
                logging.info(f"WebSocket connection closed gracefully for {client_ip}.")
            except Exception as ws_close_e:
                logging.error(f"Error closing WebSocket for {client_ip}: {ws_close_e}")

        # Lock is released automatically by 'async with' context manager
        if acquired_lock:
            logging.info(f"Lock released for {game_version} game by {client_ip}.")

        logging.info(f"--- WebSocket Endpoint cleanup finished for {game_version} ({client_ip}) ---")


# --- Static Files and Root Endpoint (No change needed) ---
frontend_dir = "build"; static_dir = os.path.join(frontend_dir, "static")
if os.path.isdir(frontend_dir) and os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logging.info(f"Serving static files from: {static_dir}")
else: logging.warning(f"Static files directory not found: '{static_dir}'")

@app.get("/")
async def read_index():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        logging.error("index.html not found in build directory.")
        return {"error": "Frontend index.html not found. Ensure the frontend is built correctly in the 'build' directory."}

@app.get("/{full_path:path}")
async def catch_all(full_path: str):
    # Prevent serving API/WS routes as index.html
    if full_path.startswith(("static/", "ws/", "api/")):
        raise HTTPException(status_code=404, detail="Resource not found")
    # Serve index.html for other paths (React Router support)
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        # Only raise 500 if index is fundamentally missing
        logging.error("index.html not found in build directory for catch-all route.")
        raise HTTPException(status_code=500, detail="Frontend index.html is missing.")

# --- Main Execution (No change needed) ---
if __name__ == "__main__":
    import uvicorn
    # Check for frontend build directory existence
    if not os.path.exists(frontend_dir) or not os.path.exists(os.path.join(frontend_dir, "index.html")):
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! WARNING: Frontend build directory 'build' or 'build/index.html' not found!")
        print("!!!          Please build the frontend first (e.g., `npm run build` or `yarn build`)")
        print("!!!          and ensure the 'build' folder is in the same directory as this script.")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

    # Print startup info
    print("--- Starting Memory Matching Game Backend ---")
    print(f"Access the game via: http://<your-server-ip>:8000")
    print(f"Using Serial Port: {SERIAL_PORT}")
    print(f"Using Camera URL: {CAMERA_URL}")
    yolo_status = f"Found ({YOLO_MODEL_PATH})" if os.path.exists(YOLO_MODEL_PATH) else f"NOT FOUND ({YOLO_MODEL_PATH})"
    print(f"YOLO Model Status: {yolo_status}")
    print("---------------------------------------------")

    # Run the FastAPI server
    uvicorn.run(
        "memory_matching_backend:app",
        host="0.0.0.0", # Listen on all available network interfaces
        port=8000,
        reload=False, # Disable auto-reload for production/stable testing
        log_level="info" # Set uvicorn's base log level
    )

# --- END OF FILE memory_matching_backend.py ---