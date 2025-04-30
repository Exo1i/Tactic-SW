import asyncio
import base64
import json
import logging
import random
import threading
import time
from typing import Dict, Any, Optional, List, Tuple

import cv2
import numpy as np
import serial
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
# Corrected import for WebSocket state
from starlette.websockets import WebSocketState


# --- Configuration ---
SERIAL_PORT = 'COM5'  # Adjust as needed
BAUD_RATE = 9600
CAMERA_URL = 'http://192.168.2.19:4747/mjpegfeed?640x480' # Primary Camera URL
# CAMERA_URL_COLOR = 'http://192.168.144.44:4747/mjpegfeed?640x480' # Optional different URL for color version

YOLO_MODEL_PATH = "yolov5s.pt" # Or your specific model path

# --- Constants ---
GRID_ROWS = 2
GRID_COLS = 4
CARD_COUNT = GRID_ROWS * GRID_COLS
FLIP_DELAY_SECONDS = 1.0 # Delay in game logic loops

# --- Arm Control Values ---
arm_values = [[110, 40, 125], [87, 65, 120], [87, 110, 120], [110, 140, 125],
              [150, 55, 155], [130, 80, 140], [130, 105, 140], [150, 125, 155]]
arm_home = [180, 90, 0]
arm_temp1 = [90, 10, 120]
arm_temp2 = [90, 170, 120]
arm_trash = [140, 0, 140] # Used for matched pairs

# --- YOLO Specific Constants ---
YOLO_TARGET_LABELS = ['orange', 'apple', 'cat', 'car', 'umbrella', 'banana', 'fire hydrant', 'person']
# Fixed grid dimensions in the camera frame (YOLO version) - ADJUST THESE PIXEL VALUES
YOLO_GRID_TOP = 90
YOLO_GRID_LEFT = 30
YOLO_GRID_WIDTH = 550
YOLO_GRID_HEIGHT = 280
YOLO_FRAME_WIDTH = 640 # Expected frame width from camera
YOLO_FRAME_HEIGHT = 480 # Expected frame height

# --- Color Detection Specific Constants ---
COLOR_DEFINITIONS = { # BGR format for drawing, Name for state
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0)
}
COLOR_RANGES = [ # HSV Lower/Upper bounds
    {'name': 'red', 'bgr': (0, 0, 255), 'lower': [(0, 100, 100), (170, 100, 100)], 'upper': [(10, 255, 255), (179, 255, 255)]},
    {'name': 'yellow', 'bgr': (0, 255, 255), 'lower': [(20, 100, 100)], 'upper': [(35, 255, 255)]},
    {'name': 'green', 'bgr': (0, 255, 0), 'lower': [(40, 40, 40)], 'upper': [(80, 255, 255)]},
    {'name': 'blue', 'bgr': (255, 0, 0), 'lower': [(100, 100, 100)], 'upper': [(130, 255, 255)]},
    {'name': 'black', 'bgr': (0, 0, 0), 'lower': [(0, 0, 0)], 'upper': [(180, 255, 50)]} # Detect black to ignore unflipped cards
]
COLOR_CELL_THRESHOLD = 600  # Min colored pixels to detect color in a cell
COLOR_BOARD_DETECT_WIDTH = 400 # Width of the perspective-warped board image
COLOR_BOARD_DETECT_HEIGHT = 200 # Height (based on 2 rows for 400 width)

# --- Globals ---
ser: Optional[serial.Serial] = None
serial_lock = threading.Lock() # Lock for thread-safe serial access
latest_frame: Optional[np.ndarray] = None
frame_lock = asyncio.Lock()
active_games: Dict[str, Dict[str, Any]] = {"color": {}, "yolo": {}}
game_locks: Dict[str, asyncio.Lock] = {
    "color": asyncio.Lock(),
    "yolo": asyncio.Lock()
}
yolo_model_global = None # Global variable to hold the loaded YOLO model

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Serial Communication ---
def setup_serial() -> bool:
    global ser
    if ser and ser.is_open:
        logging.info("Serial port already open.")
        return True
    try:
        logging.info(f"Attempting to open serial port {SERIAL_PORT}...")
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
        # Add a short delay after opening, before first use
        time.sleep(2)
        if not ser.is_open:
             raise serial.SerialException("Port opened but not accessible")
        logging.info(f"Serial port {SERIAL_PORT} opened successfully!")
        return True
    except serial.SerialException as e:
        logging.error(f"Serial Error on port {SERIAL_PORT}: {e}")
        ser = None
        return False
    except Exception as e:
        logging.error(f"Unexpected error opening serial port {SERIAL_PORT}: {e}")
        ser = None
        return False

def send_arm_command_sync(degree1: int, degree2: int, degree3: int, magnet: int, movement: int) -> Optional[str]:
    """Synchronous version of send_arm_command for use with asyncio.to_thread"""
    global ser
    if not ser or not ser.is_open:
        logging.error("Serial port not available or not open.")
        return None

    with serial_lock: # Ensure only one thread accesses serial at a time
        try:
            command = f"{degree1},{degree2},{degree3},{magnet},{movement}\n"
            logging.info(f"Sending serial command: {command.strip()}")
            ser.write(command.encode('utf-8'))
            ser.flush() # Ensure data is sent

            # Improved read loop with timeout for "done" response
            response = b''
            start_time = time.time()
            timeout = 7.0 # Increased timeout for arm movement + response

            while time.time() - start_time < timeout:
                if ser.in_waiting > 0:
                    try:
                        # Read available bytes, decode incrementally
                        chunk = ser.read(ser.in_waiting)
                        response += chunk
                        # Check if "done" is in the decoded accumulated response
                        if b"done" in response.lower():
                            break
                    except Exception as read_e:
                         logging.error(f"Error reading from serial: {read_e}")
                         break
                # --- Use synchronous sleep in this thread ---
                time.sleep(0.05)
                # --- End Correction ---

            decoded_response = response.decode('utf-8', errors='replace').strip()
            logging.info(f"Serial response received: '{decoded_response}'")

            if "done" in decoded_response.lower():
                logging.info("Arm movement confirmed 'done'.")
                return decoded_response
            else:
                logging.warning(f"Arm movement 'done' message not received or timed out. Response: '{decoded_response}'")
                return None # Indicate potential failure or timeout

        except serial.SerialException as e:
            logging.error(f"Serial communication error: {e}")
            try:
                ser.close()
            except Exception: pass
            ser = None
            return None
        except Exception as e:
            logging.error(f"Unexpected error during serial communication: {e}")
            return None

async def send_arm_command(degree1: int, degree2: int, degree3: int, magnet: int, movement: int) -> Optional[str]:
    """Asynchronous wrapper for send_arm_command"""
    return await asyncio.to_thread(send_arm_command_sync, degree1, degree2, degree3, magnet, movement)

async def from_to(websocket: WebSocket, src: str, dest: str, card_id: int) -> bool:
    """Async function to handle arm movements and send updates"""
    logging.info(f"Initiating arm movement: card {card_id} from {src} to {dest}")
    move_successful = False
    start_pos: List[int] = []
    end_pos: List[int] = []
    home_pos: List[int] = arm_home

    try:
        # Determine start position based on source
        if src == "card":
            if 0 <= card_id < len(arm_values):
                start_pos = arm_values[card_id]
            else:
                 raise ValueError(f"Invalid card_id {card_id} for arm_values")
        elif src == "temp1":
            start_pos = arm_temp1
        elif src == "temp2":
            start_pos = arm_temp2
        else:
            raise ValueError(f"Invalid source location: {src}")

        # Determine end position based on destination
        if dest == "temp1":
            end_pos = arm_temp1
        elif dest == "temp2":
            end_pos = arm_temp2
        elif dest == "trash":
            end_pos = arm_trash
        elif dest == "card":
             if 0 <= card_id < len(arm_values):
                end_pos = arm_values[card_id]
             else:
                 raise ValueError(f"Invalid card_id {card_id} for arm_values")
        else:
            raise ValueError(f"Invalid destination location: {dest}")

        # --- Movement Sequence ---
        logging.info(f"Step 1: Moving to source {src} at {start_pos}...")
        resp = await send_arm_command(start_pos[0], start_pos[1], start_pos[2], 0, 0) # Magnet off
        if resp is None: raise Exception("Arm command failed: Move to source")
        await asyncio.sleep(0.5) # Settling time

        logging.info(f"Step 2: Picking up (Magnet ON) at {start_pos}...")
        resp = await send_arm_command(start_pos[0], start_pos[1], start_pos[2], 1, 0)
        if resp is None: raise Exception("Arm command failed: Pick up")
        await asyncio.sleep(0.5) # Ensure grip

        logging.info(f"Step 3: Moving home {home_pos} (with item)...")
        resp = await send_arm_command(home_pos[0], home_pos[1], home_pos[2], 1, 1)
        if resp is None: raise Exception("Arm command failed: Move home with item")
        await asyncio.sleep(0.5)

        logging.info(f"Step 4: Moving to destination {dest} at {end_pos} (with item)...")
        resp = await send_arm_command(end_pos[0], end_pos[1], end_pos[2], 1, 0)
        if resp is None: raise Exception("Arm command failed: Move to destination")
        await asyncio.sleep(0.5)

        logging.info(f"Step 5: Dropping item (Magnet OFF) at {end_pos}...")
        resp = await send_arm_command(end_pos[0], end_pos[1], end_pos[2], 0, 0)
        if resp is None: raise Exception("Arm command failed: Drop item")
        await asyncio.sleep(0.5) # Ensure release

        logging.info(f"Step 6: Moving home {home_pos} (empty)...")
        resp = await send_arm_command(home_pos[0], home_pos[1], home_pos[2], 0, 1)
        if resp is None: raise Exception("Arm command failed: Move home empty")
        await asyncio.sleep(0.5)

        move_successful = True
        logging.info(f"Arm movement sequence for card {card_id} ({src} to {dest}) completed successfully.")

    except ValueError as ve:
         logging.error(f"Configuration error during arm movement: {ve}")
         move_successful = False
    except Exception as e:
        logging.error(f"Error during arm movement sequence ({src} to {dest}, card {card_id}): {e}")
        move_successful = False
    finally:
        # Attempt to return home safely if anything failed
        if not move_successful:
            logging.warning("Movement sequence potentially failed or incomplete. Attempting safe return home.")
            try:
                await send_arm_command(home_pos[0], home_pos[1], home_pos[2], 0, 1) # Magnet off, move type 1
            except Exception as home_e:
                logging.error(f"Failed to return arm home after error: {home_e}")
        # Send status update regardless of success
        try:
             # Check websocket state before sending
             if websocket.client_state == WebSocketState.CONNECTED:
                 await websocket.send_json({
                     "type": "arm_status",
                     "payload": {"status": "finished", "success": move_successful, "action": f"{src}_to_{dest}", "card_id": card_id}
                 })
        except Exception as ws_e:
             logging.error(f"Failed to send arm status update via WebSocket: {ws_e}")

    return move_successful

# --- FastAPI App ---
app = FastAPI()

# --- YOLO Model Loading (Load once on startup) ---
@app.on_event("startup")
async def load_yolo_model():
    global yolo_model_global
    try:
        # Ensure ultralytics is installed before importing
        import importlib
        try:
            importlib.import_module('ultralytics')
        except ImportError:
             logging.error("Ultralytics library not found. Please install it: pip install ultralytics")
             yolo_model_global = None
             return

        from ultralytics import YOLO
        logging.info(f"Loading YOLO model globally from: {YOLO_MODEL_PATH}")
        # Check if model file exists
        import os
        if not os.path.exists(YOLO_MODEL_PATH):
             logging.error(f"YOLO model file not found at: {YOLO_MODEL_PATH}")
             yolo_model_global = None
             return

        yolo_model_global = YOLO(YOLO_MODEL_PATH)
        # Optional: Dummy prediction to warm up
        dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
        yolo_model_global.predict(dummy_img, verbose=False)
        logging.info("Global YOLO model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading global YOLO model: {e}", exc_info=True)
        yolo_model_global = None

# --- Camera Capture ---
async def capture_frames(websocket: WebSocket, camera_source: str, game_version: str):
    """Continuously captures frames, updates global frame, and sends via WebSocket"""
    global latest_frame
    cap = None
    logging.info(f"Starting frame capture task for {game_version} from {camera_source}")

    while game_version in active_games and active_games[game_version].get("running"):
        try:
            if cap is None or not cap.isOpened():
                logging.info(f"Opening camera source: {camera_source}")
                cap = cv2.VideoCapture(camera_source)
                # Allow some time for camera to initialize
                await asyncio.sleep(1.0)
                if not cap.isOpened():
                    logging.error(f"Cannot open camera source: {camera_source}")
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_json({"type": "error", "payload": f"Cannot open camera: {camera_source}"})
                    await asyncio.sleep(5)
                    continue
                else:
                    logging.info(f"Camera {camera_source} opened successfully.")
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, YOLO_FRAME_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, YOLO_FRAME_HEIGHT)
                    # cap.set(cv2.CAP_PROP_FPS, 30) # Setting FPS might not work reliably on all streams

            ret, frame = cap.read()
            if not ret or frame is None:
                logging.warning("Failed to grab frame from camera. Retrying...")
                if cap is not None:
                    cap.release()
                cap = None
                await asyncio.sleep(1)
                continue

            async with frame_lock:
                latest_frame = frame.copy()

            processed_frame = frame
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(processed_frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(processed_frame, f"Mode: {game_version.upper()}", (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ret:
                logging.warning("Failed to encode frame")
                continue

            jpg_as_text = base64.b64encode(buffer).decode('utf-8')

            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json({
                    "type": "frame",
                    "payload": jpg_as_text
                })
            else:
                logging.warning("WebSocket disconnected during frame send, stopping capture.")
                break # Exit if websocket is closed

            await asyncio.sleep(0.04) # ~25 FPS target

        except WebSocketDisconnect:
            logging.info(f"WebSocket disconnected during frame capture for {game_version}.")
            break
        except Exception as e:
            logging.error(f"Error during frame capture loop for {game_version}: {e}", exc_info=True)
            if cap is not None:
                 cap.release()
                 cap = None
            if websocket.client_state == WebSocketState.CONNECTED:
                 try:
                     await websocket.send_json({"type": "error", "payload": f"Camera capture error: {e}"})
                 except Exception: pass
            await asyncio.sleep(2)

    if cap is not None:
        cap.release()
        logging.info(f"Camera {camera_source} released for {game_version}.")
    async with frame_lock:
         latest_frame = None
    logging.info(f"Frame capture task stopped for {game_version}.")


# --- YOLO Helper Functions ---
def yolo_assign_color(label: str) -> Tuple[int, int, int]:
    """Assigns a BGR color based on the YOLO label."""
    color_map = { 'orange': (0, 165, 255), 'apple': (0, 0, 255), 'cat': (255, 0, 0),
                  'car': (255, 255, 0), 'umbrella': (0, 255, 0), 'banana': (0, 255, 255),
                  'fire hydrant': (0, 128, 255), 'person': (0,0,0) }
    return color_map.get(label.lower(), (255, 255, 255))

async def detect_object_at_card(card_id: int) -> Optional[str]:
    """Detects object in a specific card region using the global YOLO model."""
    global latest_frame, yolo_model_global
    if yolo_model_global is None:
        logging.error("YOLO model not loaded. Cannot perform detection.")
        return None

    frame_to_process = None
    async with frame_lock:
        if latest_frame is not None:
            frame_to_process = latest_frame.copy()

    if frame_to_process is None:
        logging.warning(f"No frame available for YOLO detection on card {card_id}")
        return None

    try:
        row = card_id // GRID_COLS
        col = card_id % GRID_COLS
        cell_width = YOLO_GRID_WIDTH // GRID_COLS
        cell_height = YOLO_GRID_HEIGHT // GRID_ROWS
        card_x1 = YOLO_GRID_LEFT + (col * cell_width)
        card_y1 = YOLO_GRID_TOP + (row * cell_height)
        card_x2 = card_x1 + cell_width
        card_y2 = card_y1 + cell_height
        padding = 5
        roi_x1 = max(0, card_x1 + padding)
        roi_y1 = max(0, card_y1 + padding)
        roi_x2 = min(frame_to_process.shape[1], card_x2 - padding)
        roi_y2 = min(frame_to_process.shape[0], card_y2 - padding)

        if roi_x1 >= roi_x2 or roi_y1 >= roi_y2:
             logging.warning(f"Invalid ROI calculated for card {card_id}. Skipping detection.")
             return None

        card_roi = frame_to_process[roi_y1:roi_y2, roi_x1:roi_x2]

        if card_roi.size == 0:
             logging.warning(f"Empty ROI for card {card_id}. Skipping detection.")
             return None

        # Run prediction in a separate thread to avoid blocking event loop if model is slow
        def sync_predict(model, roi):
             return model.predict(roi, conf=0.4, verbose=False)

        results = await asyncio.to_thread(sync_predict, yolo_model_global, card_roi)
        # results = yolo_model_global.predict(card_roi, conf=0.4, verbose=False) # Direct call if predict is fast enough

        detected_object_label = None
        highest_conf = 0.0
        if results: # Ensure results is not None or empty
             for result in results:
                 if hasattr(result, 'boxes'): # Check if result object has 'boxes'
                      for box in result.boxes:
                           if hasattr(box, 'cls') and hasattr(box, 'conf') and box.cls is not None and box.conf is not None and len(box.cls) > 0 and len(box.conf) > 0:
                                label_index = int(box.cls[0])
                                if label_index < len(result.names): # Check index validity
                                     label = result.names[label_index].lower()
                                     if label in YOLO_TARGET_LABELS:
                                          score = box.conf.item()
                                          if score > highest_conf:
                                               highest_conf = score
                                               detected_object_label = label
                                else:
                                     logging.warning(f"Invalid label index {label_index} in YOLO results.")
                           else:
                                logging.warning("YOLO box object missing expected attributes (cls/conf) or they are empty.")
                 else:
                     logging.warning("YOLO result object missing 'boxes' attribute.")

        if detected_object_label:
            logging.info(f"Detected '{detected_object_label}' (conf: {highest_conf:.2f}) on card {card_id}")
        else:
            logging.info(f"No target object detected on card {card_id}")
        return detected_object_label

    except Exception as e:
        logging.error(f"Error during YOLO detection for card {card_id}: {e}", exc_info=True)
        return None

# --- Color Detection Helper Functions ---
def find_board_corners(frame: np.ndarray) -> Optional[np.ndarray]:
    """Find the four corners of the game board based on white border."""
    if frame is None or frame.size == 0: return None
    try:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 50, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < 5000: return None
        epsilon = 0.03 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        if len(approx) == 4:
            corners = np.array([point[0] for point in approx], dtype=np.float32)
            return sort_corners(corners)
        else:
             logging.warning(f"Board contour approximation has {len(approx)} vertices, expected 4.")
             return None
    except Exception as e:
        logging.error(f"Error finding board corners: {e}", exc_info=True)
        return None


def sort_corners(corners: np.ndarray) -> np.ndarray:
    """Sorts corners: Top-Left, Top-Right, Bottom-Right, Bottom-Left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)]
    rect[2] = corners[np.argmax(s)]
    diff = np.diff(corners, axis=1)
    rect[1] = corners[np.argmin(diff)]
    rect[3] = corners[np.argmax(diff)]
    return rect

def transform_board(frame: np.ndarray, corners: np.ndarray) -> Optional[np.ndarray]:
    """Apply perspective transform to get top-down view of board."""
    if frame is None or corners is None: return None
    try:
        (tl, tr, br, bl) = corners
        dst_points = np.array([
            [0, 0],
            [COLOR_BOARD_DETECT_WIDTH - 1, 0],
            [COLOR_BOARD_DETECT_WIDTH - 1, COLOR_BOARD_DETECT_HEIGHT - 1],
            [0, COLOR_BOARD_DETECT_HEIGHT - 1]
        ], dtype=np.float32)
        M = cv2.getPerspectiveTransform(corners, dst_points)
        warped = cv2.warpPerspective(frame, M, (COLOR_BOARD_DETECT_WIDTH, COLOR_BOARD_DETECT_HEIGHT))
        return warped
    except Exception as e:
        logging.error(f"Error during perspective transform: {e}")
        return None

async def detect_color_at_card(card_id: int) -> Optional[str]:
    """Detects color in a specific card region using board detection and warping."""
    global latest_frame
    frame_to_process = None
    async with frame_lock:
        if latest_frame is not None:
             frame_to_process = latest_frame.copy()

    if frame_to_process is None:
        logging.warning(f"No frame available for Color detection on card {card_id}")
        return None

    try:
        corners = find_board_corners(frame_to_process)
        if corners is None:
            logging.warning("Board corners not detected. Cannot perform color detection.")
            return None

        warped_board = transform_board(frame_to_process, corners)
        if warped_board is None:
            logging.warning("Board transformation failed.")
            return None

        row = card_id // GRID_COLS
        col = card_id % GRID_COLS
        cell_width = COLOR_BOARD_DETECT_WIDTH // GRID_COLS
        cell_height = COLOR_BOARD_DETECT_HEIGHT // GRID_ROWS
        x1 = col * cell_width
        y1 = row * cell_height
        x2 = x1 + cell_width
        y2 = y1 + cell_height
        padding = 3
        roi_x1 = x1 + padding
        roi_y1 = y1 + padding
        roi_x2 = x2 - padding
        roi_y2 = y2 - padding

        if roi_x1 >= roi_x2 or roi_y1 >= roi_y2:
             logging.warning(f"Invalid ROI in warped image for card {card_id}.")
             return None

        cell_roi = warped_board[roi_y1:roi_y2, roi_x1:roi_x2]
        if cell_roi.size == 0:
            logging.warning(f"Empty ROI extracted for card {card_id}.")
            return None

        hsv_roi = cv2.cvtColor(cell_roi, cv2.COLOR_BGR2HSV)
        detected_colors_count: Dict[str, int] = {}

        for color_def in COLOR_RANGES:
            color_name = color_def['name']
            total_mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
            for l_bound, u_bound in zip(color_def['lower'], color_def['upper']):
                lower = np.array(l_bound, dtype=np.uint8)
                upper = np.array(u_bound, dtype=np.uint8)
                mask_part = cv2.inRange(hsv_roi, lower, upper)
                total_mask = cv2.bitwise_or(total_mask, mask_part)
            pixel_count = cv2.countNonZero(total_mask)
            if pixel_count > 0:
                detected_colors_count[color_name] = pixel_count

        dominant_color = None
        max_pixels = 0

        # Check for black first
        if "black" in detected_colors_count and detected_colors_count["black"] > COLOR_CELL_THRESHOLD * 0.7:
             logging.info(f"Card {card_id} detected as mostly black (unflipped).")
             return None # Treat black as undetected/unflipped

        # Find dominant among non-black colors meeting threshold
        for color_name, pixel_count in detected_colors_count.items():
             if color_name != "black" and pixel_count > max_pixels and pixel_count >= COLOR_CELL_THRESHOLD:
                  max_pixels = pixel_count
                  dominant_color = color_name

        if dominant_color:
            logging.info(f"Detected color '{dominant_color}' (pixels: {max_pixels}) on card {card_id}")
        else:
            logging.info(f"No dominant color found meeting threshold on card {card_id}. Detected counts: {detected_colors_count}")
        return dominant_color

    except Exception as e:
        logging.error(f"Error during Color detection for card {card_id}: {e}", exc_info=True)
        return None

# --- Game Logic Runners ---
async def run_yolo_game(websocket: WebSocket):
    """Runs the YOLO version of the Memory Game."""
    global yolo_model_global
    logging.info("Starting YOLO Game Logic...")
    if yolo_model_global is None:
        logging.error("YOLO Model not loaded, cannot start game.")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({"type": "error", "payload": "YOLO model failed to load on server startup."})
        return

    game_state = active_games["yolo"]
    game_state["card_states"] = {i: {"isFlippedBefore": False, "object": None, "isMatched": False} for i in range(CARD_COUNT)}
    game_state["objects_found"] = {obj: [] for obj in YOLO_TARGET_LABELS}
    game_state["pairs_found"] = 0
    game_state["current_flipped_cards"] = []
    game_state["running"] = True

    if websocket.client_state == WebSocketState.CONNECTED:
        await websocket.send_json({"type": "game_state", "payload": game_state["card_states"]})
        await websocket.send_json({"type": "message", "payload": "YOLO Game Started. Initializing arm..."})

    camera_task = asyncio.create_task(capture_frames(websocket, CAMERA_URL, "yolo"))

    try:
        await send_arm_command(arm_home[0], arm_home[1], arm_home[2], 0, 1)

        while game_state["pairs_found"] < (CARD_COUNT // 2) and game_state["running"]:
            await asyncio.sleep(FLIP_DELAY_SECONDS)

            if websocket.client_state != WebSocketState.CONNECTED:
                 logging.warning("WebSocket disconnected during game loop.")
                 game_state["running"] = False
                 break

            current_flipped = game_state["current_flipped_cards"]
            card_states = game_state["card_states"]
            objects_found = game_state["objects_found"]

            logging.info(f"YOLO Loop: Flipped={current_flipped}, Pairs={game_state['pairs_found']}")

            async def update_frontend_state():
                payload = {
                    "card_states": card_states,
                    "pairs_found": game_state["pairs_found"],
                    "current_flipped": current_flipped # Send currently flipped for potential UI highlight
                    }
                if websocket.client_state == WebSocketState.CONNECTED:
                    try: await websocket.send_json({"type": "game_state", "payload": payload})
                    except Exception as send_e: logging.error(f"Failed to send game state update: {send_e}")

            def choose_random_card():
                 # Prioritize cards that haven't been flipped at all
                 never_flipped = [i for i, s in card_states.items() if not s.get("isFlippedBefore") and not s.get("isMatched")]
                 if never_flipped: return random.choice(never_flipped)
                 # If all flipped once, pick one that isn't matched yet
                 unmatched = [i for i, s in card_states.items() if s.get("isFlippedBefore") and not s.get("isMatched") and i not in current_flipped]
                 if unmatched: return random.choice(unmatched)
                 return None # No cards left to choose

            def find_pair():
                 # Find pairs among cards that have been flipped but not yet matched
                 for obj, ids in objects_found.items():
                     valid_ids = [i for i in ids if card_states.get(i,{}).get("isFlippedBefore") and not card_states.get(i,{}).get("isMatched")]
                     if len(valid_ids) >= 2:
                         return valid_ids[0], valid_ids[1]
                 return None

            def find_match(card_id):
                obj = card_states.get(card_id,{}).get("object")
                if not obj: return None
                # Find another card ID for the same object, flipped but not matched, and not the card itself
                return next((other_id for other_id in objects_found.get(obj, [])
                             if other_id != card_id and card_states.get(other_id,{}).get("isFlippedBefore") and not card_states.get(other_id,{}).get("isMatched")),
                            None)

            # --- State Machine Logic ---
            if len(current_flipped) == 0:
                pair = find_pair()
                if pair:
                    card1_id, card2_id = pair
                    obj = card_states[card1_id]['object']
                    logging.info(f"Strategy: Found known matching pair {card1_id} & {card2_id} for object '{obj}'. Removing.")
                    if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "message", "payload": f"Found pair: {obj}. Removing cards {card1_id} & {card2_id}."})
                    # Assume cards are still on board (not in temp yet)
                    success1 = await from_to(websocket, "card", "trash", card1_id)
                    success2 = await from_to(websocket, "card", "trash", card2_id)
                    if success1 and success2:
                         card_states[card1_id]["isMatched"] = True
                         card_states[card2_id]["isMatched"] = True
                         # No need to remove from objects_found if we use isMatched flag
                         game_state["pairs_found"] += 1
                    else:
                         logging.error(f"Failed to physically remove pair {card1_id} & {card2_id}. State not updated.")
                         if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "error", "payload": f"Failed arm movement for pair {card1_id}/{card2_id}."})
                    await update_frontend_state()
                    continue
                else:
                    card_to_flip = choose_random_card()
                    if card_to_flip is not None:
                        logging.info(f"Strategy: No known pair. Flipping random card {card_to_flip}.")
                        if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "message", "payload": f"Choosing card {card_to_flip}. Detecting..."})
                        detected_obj = await detect_object_at_card(card_to_flip)
                        if detected_obj is not None: # Allow detection of 'None' maybe? Let's assume detection must yield a label
                            if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "message", "payload": f"Detected '{detected_obj}' on card {card_to_flip}. Moving to Temp1."})
                            success = await from_to(websocket, "card", "temp1", card_to_flip)
                            if success:
                                card_states[card_to_flip]["object"] = detected_obj
                                card_states[card_to_flip]["isFlippedBefore"] = True
                                if detected_obj in objects_found:
                                    if card_to_flip not in objects_found[detected_obj]:
                                         objects_found[detected_obj].append(card_to_flip)
                                else: logging.warning(f"Detected object '{detected_obj}' not in target list?")
                                current_flipped.append(card_to_flip)
                            else:
                                logging.error(f"Failed to move card {card_to_flip} to Temp1. State not updated.")
                                if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "error", "payload": f"Failed arm movement for card {card_to_flip}."})
                        else:
                             logging.warning(f"Failed to detect object on card {card_to_flip}. Cannot proceed.")
                             if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "message", "payload": f"Could not detect object on card {card_to_flip}. Choosing another."})
                             # Mark as flipped before so we don't choose it immediately again?
                             # card_states[card_to_flip]["isFlippedBefore"] = True # Controversial
                             # card_states[card_to_flip]["object"] = "DETECT_FAIL"
                        await update_frontend_state()
                    else:
                        logging.info("No more cards to choose, but game not over? Checking state.")
                        await asyncio.sleep(2)

            elif len(current_flipped) == 1:
                 first_card_id = current_flipped[0]
                 match_id = find_match(first_card_id)
                 if match_id is not None:
                     obj = card_states[first_card_id]['object']
                     logging.info(f"Strategy: Card {first_card_id} is flipped. Found known match: {match_id}. Removing pair.")
                     if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "message", "payload": f"Found match for '{obj}'. Removing cards {first_card_id} & {match_id}."})
                     success1 = await from_to(websocket, "temp1", "trash", first_card_id) # From temp1
                     success2 = await from_to(websocket, "card", "trash", match_id)    # From board
                     if success1 and success2:
                          card_states[first_card_id]["isMatched"] = True
                          card_states[match_id]["isMatched"] = True
                          card_states[match_id]["isFlippedBefore"] = True # Ensure match is marked flipped
                          game_state["pairs_found"] += 1
                          current_flipped.clear()
                     else:
                          logging.error(f"Failed to physically remove matched pair {first_card_id} & {match_id}. State not updated.")
                          if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "error", "payload": f"Failed arm movement for matched pair {first_card_id}/{match_id}."})
                          # Attempt to return first card?
                          await from_to(websocket, "temp1", "card", first_card_id)
                          current_flipped.clear()
                     await update_frontend_state()
                     continue
                 else:
                    card_to_flip = choose_random_card()
                    if card_to_flip is not None:
                         logging.info(f"Strategy: No known match for {first_card_id}. Flipping random card {card_to_flip}.")
                         if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "message", "payload": f"No known match. Choosing card {card_to_flip}. Detecting..."})
                         detected_obj = await detect_object_at_card(card_to_flip)
                         if detected_obj is not None:
                            if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "message", "payload": f"Detected '{detected_obj}' on card {card_to_flip}. Moving to Temp2."})
                            success = await from_to(websocket, "card", "temp2", card_to_flip)
                            if success:
                                card_states[card_to_flip]["object"] = detected_obj
                                card_states[card_to_flip]["isFlippedBefore"] = True
                                if detected_obj in objects_found:
                                     if card_to_flip not in objects_found[detected_obj]:
                                         objects_found[detected_obj].append(card_to_flip)
                                current_flipped.append(card_to_flip)
                            else:
                                logging.error(f"Failed to move card {card_to_flip} to Temp2. State not updated.")
                                if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "error", "payload": f"Failed arm movement for card {card_to_flip}."})
                         else:
                             logging.warning(f"Failed to detect object on card {card_to_flip}. Cannot flip second card.")
                             if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "message", "payload": f"Could not detect object on card {card_to_flip}. Will try again later."})
                             # card_states[card_to_flip]["isFlippedBefore"] = True
                             # card_states[card_to_flip]["object"] = "DETECT_FAIL"
                         await update_frontend_state()
                    else:
                        logging.info(f"One card flipped ({first_card_id}), but no other cards left? Returning first card.")
                        await from_to(websocket, "temp1", "card", first_card_id)
                        current_flipped.clear()
                        await update_frontend_state()

            elif len(current_flipped) == 2:
                card1_id, card2_id = current_flipped[0], current_flipped[1]
                obj1 = card_states.get(card1_id,{}).get("object")
                obj2 = card_states.get(card2_id,{}).get("object")
                logging.info(f"Strategy: Two cards flipped: {card1_id} ({obj1}) and {card2_id} ({obj2}). Checking match.")

                if obj1 is not None and obj1 == obj2: # Match found!
                    logging.info(f"Match confirmed between {card1_id} and {card2_id} ({obj1}). Removing pair.")
                    if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "message", "payload": f"Match found: {obj1}! Removing cards {card1_id} & {card2_id}."})
                    success1 = await from_to(websocket, "temp1", "trash", card1_id)
                    success2 = await from_to(websocket, "temp2", "trash", card2_id)
                    if success1 and success2:
                        card_states[card1_id]["isMatched"] = True
                        card_states[card2_id]["isMatched"] = True
                        game_state["pairs_found"] += 1
                    else:
                        logging.error(f"Failed to physically remove matched pair {card1_id} & {card2_id} from temp locations. State not updated.")
                        if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "error", "payload": f"Failed arm movement for matched pair {card1_id}/{card2_id}."})
                        # Attempt recovery? Return cards to temp?
                        # For now, just log error. State is inconsistent with physical world.
                else: # No match
                    logging.info(f"No match between {card1_id} ({obj1}) and {card2_id} ({obj2}). Returning cards.")
                    if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "message", "payload": f"No match. Returning cards {card1_id} & {card2_id}."})
                    success1 = await from_to(websocket, "temp1", "card", card1_id)
                    success2 = await from_to(websocket, "temp2", "card", card2_id)
                    if not (success1 and success2):
                         logging.error(f"Failed to return non-matching cards {card1_id} or {card2_id} to board.")
                         if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "error", "payload": f"Failed arm movement returning cards {card1_id}/{card2_id}."})
                    # No state change needed for isMatched or isFlippedBefore

                current_flipped.clear()
                await update_frontend_state()
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_json({"type": "cards_hidden", "payload": [card1_id, card2_id]})

            if game_state["pairs_found"] >= CARD_COUNT // 2:
                 logging.info("YOLO Game Finished!")
                 if websocket.client_state == WebSocketState.CONNECTED:
                      await websocket.send_json({"type": "game_over", "payload": "Congratulations! All pairs found."})
                 game_state["running"] = False
                 break

    except WebSocketDisconnect:
        logging.info("WebSocket disconnected during YOLO game.")
    except Exception as e:
        logging.error(f"Error during YOLO game loop: {e}", exc_info=True)
        if websocket.client_state == WebSocketState.CONNECTED:
           try: await websocket.send_json({"type": "error", "payload": f"Critical Game Error: {e}"})
           except Exception: pass
    finally:
        logging.info("Cleaning up YOLO game runner.")
        game_state["running"] = False # Ensure running flag is false
        # Cancel camera task politely
        if camera_task and not camera_task.done():
            logging.info("Attempting to cancel camera task for YOLO game.")
            camera_task.cancel()
            try:
                await camera_task
            except asyncio.CancelledError:
                logging.info("Camera task successfully cancelled.")
            except Exception as task_e:
                 logging.error(f"Error while awaiting cancelled camera task: {task_e}")

        # Final check to ensure arm is home
        logging.info("Ensuring arm is home after YOLO game.")
        try:
             await send_arm_command(arm_home[0], arm_home[1], arm_home[2], 0, 1)
        except Exception as final_home_e:
             logging.error(f"Failed to send final home command: {final_home_e}")
        logging.info("YOLO Game Runner finished cleanup.")


async def run_color_game(websocket: WebSocket):
    """Runs the Color Detection version of the Memory Game."""
    logging.info("Starting Color Game Logic...")
    game_state = active_games["color"]
    game_state["card_states"] = {i: {"isFlippedBefore": False, "color": None, "isMatched": False} for i in range(CARD_COUNT)}
    game_state["colors_found"] = {color: [] for color in COLOR_DEFINITIONS.keys()}
    game_state["pairs_found"] = 0
    game_state["current_flipped_cards"] = []
    game_state["running"] = True

    if websocket.client_state == WebSocketState.CONNECTED:
        await websocket.send_json({"type": "game_state", "payload": game_state["card_states"]})
        await websocket.send_json({"type": "message", "payload": "Color Game Started. Initializing arm..."})

    cam_url = CAMERA_URL
    camera_task = asyncio.create_task(capture_frames(websocket, cam_url, "color"))

    try:
        await send_arm_command(arm_home[0], arm_home[1], arm_home[2], 0, 1)

        while game_state["pairs_found"] < (CARD_COUNT // 2) and game_state["running"]:
            await asyncio.sleep(FLIP_DELAY_SECONDS)

            if websocket.client_state != WebSocketState.CONNECTED:
                 logging.warning("WebSocket disconnected during game loop.")
                 game_state["running"] = False
                 break

            current_flipped = game_state["current_flipped_cards"]
            card_states = game_state["card_states"]
            colors_found = game_state["colors_found"]

            logging.info(f"Color Loop: Flipped={current_flipped}, Pairs={game_state['pairs_found']}")

            async def update_frontend_state():
                payload = { "card_states": card_states, "pairs_found": game_state["pairs_found"], "current_flipped": current_flipped }
                if websocket.client_state == WebSocketState.CONNECTED:
                    try: await websocket.send_json({"type": "game_state", "payload": payload})
                    except Exception as send_e: logging.error(f"Failed to send game state update: {send_e}")

            def choose_random_card():
                 never_flipped = [i for i, s in card_states.items() if not s.get("isFlippedBefore") and not s.get("isMatched")]
                 if never_flipped: return random.choice(never_flipped)
                 unmatched = [i for i, s in card_states.items() if s.get("isFlippedBefore") and not s.get("isMatched") and i not in current_flipped]
                 if unmatched: return random.choice(unmatched)
                 return None

            def find_pair():
                 for color, ids in colors_found.items():
                     valid_ids = [i for i in ids if card_states.get(i,{}).get("isFlippedBefore") and not card_states.get(i,{}).get("isMatched")]
                     if len(valid_ids) >= 2:
                         return valid_ids[0], valid_ids[1]
                 return None

            def find_match(card_id):
                color = card_states.get(card_id,{}).get("color")
                if not color: return None
                return next((other_id for other_id in colors_found.get(color, [])
                             if other_id != card_id and card_states.get(other_id,{}).get("isFlippedBefore") and not card_states.get(other_id,{}).get("isMatched")),
                            None)

            # --- State Machine Logic (Color Version) ---
            if len(current_flipped) == 0:
                pair = find_pair()
                if pair:
                    card1_id, card2_id = pair
                    color = card_states[card1_id]['color']
                    logging.info(f"Strategy: Found known matching pair {card1_id} & {card2_id} for color '{color}'. Removing.")
                    if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "message", "payload": f"Found pair: {color}. Removing cards {card1_id} & {card2_id}."})
                    success1 = await from_to(websocket, "card", "trash", card1_id)
                    success2 = await from_to(websocket, "card", "trash", card2_id)
                    if success1 and success2:
                         card_states[card1_id]["isMatched"] = True
                         card_states[card2_id]["isMatched"] = True
                         game_state["pairs_found"] += 1
                    else:
                         logging.error(f"Failed to physically remove pair {card1_id} & {card2_id}. State not updated.")
                         if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "error", "payload": f"Failed arm movement for pair {card1_id}/{card2_id}."})
                    await update_frontend_state()
                    continue
                else:
                    card_to_flip = choose_random_card()
                    if card_to_flip is not None:
                        logging.info(f"Strategy: No known pair. Flipping random card {card_to_flip}.")
                        if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "message", "payload": f"Choosing card {card_to_flip}. Detecting..."})
                        detected_color = await detect_color_at_card(card_to_flip)
                        if detected_color is not None:
                            if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "message", "payload": f"Detected '{detected_color}' on card {card_to_flip}. Moving to Temp1."})
                            success = await from_to(websocket, "card", "temp1", card_to_flip)
                            if success:
                                card_states[card_to_flip]["color"] = detected_color
                                card_states[card_to_flip]["isFlippedBefore"] = True
                                if detected_color in colors_found:
                                    if card_to_flip not in colors_found[detected_color]:
                                        colors_found[detected_color].append(card_to_flip)
                                else: logging.warning(f"Detected color '{detected_color}' not in target list?")
                                current_flipped.append(card_to_flip)
                            else:
                                logging.error(f"Failed to move card {card_to_flip} to Temp1. State not updated.")
                                if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "error", "payload": f"Failed arm movement for card {card_to_flip}."})
                        else:
                             logging.warning(f"Failed to detect color on card {card_to_flip}. Cannot proceed.")
                             if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "message", "payload": f"Could not detect color on card {card_to_flip}. Choosing another."})
                             # card_states[card_to_flip]["isFlippedBefore"] = True
                             # card_states[card_to_flip]["color"] = "DETECT_FAIL"
                        await update_frontend_state()
                    else:
                        logging.info("No more cards to choose, but game not over? Checking state.")
                        await asyncio.sleep(2)

            elif len(current_flipped) == 1:
                 first_card_id = current_flipped[0]
                 match_id = find_match(first_card_id)
                 if match_id is not None:
                     color = card_states[first_card_id]['color']
                     logging.info(f"Strategy: Card {first_card_id} is flipped. Found known match: {match_id}. Removing pair.")
                     if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "message", "payload": f"Found match for '{color}'. Removing cards {first_card_id} & {match_id}."})
                     success1 = await from_to(websocket, "temp1", "trash", first_card_id)
                     success2 = await from_to(websocket, "card", "trash", match_id)
                     if success1 and success2:
                          card_states[first_card_id]["isMatched"] = True
                          card_states[match_id]["isMatched"] = True
                          card_states[match_id]["isFlippedBefore"] = True
                          game_state["pairs_found"] += 1
                          current_flipped.clear()
                     else:
                          logging.error(f"Failed to physically remove matched pair {first_card_id} & {match_id}. State not updated.")
                          if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "error", "payload": f"Failed arm movement for matched pair {first_card_id}/{match_id}."})
                          await from_to(websocket, "temp1", "card", first_card_id) # Try return
                          current_flipped.clear()
                     await update_frontend_state()
                     continue
                 else:
                    card_to_flip = choose_random_card()
                    if card_to_flip is not None:
                         logging.info(f"Strategy: No known match for {first_card_id}. Flipping random card {card_to_flip}.")
                         if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "message", "payload": f"No known match. Choosing card {card_to_flip}. Detecting..."})
                         detected_color = await detect_color_at_card(card_to_flip)
                         if detected_color is not None:
                             if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "message", "payload": f"Detected '{detected_color}' on card {card_to_flip}. Moving to Temp2."})
                             success = await from_to(websocket, "card", "temp2", card_to_flip)
                             if success:
                                 card_states[card_to_flip]["color"] = detected_color
                                 card_states[card_to_flip]["isFlippedBefore"] = True
                                 if detected_color in colors_found:
                                      if card_to_flip not in colors_found[detected_color]:
                                          colors_found[detected_color].append(card_to_flip)
                                 current_flipped.append(card_to_flip)
                             else:
                                 logging.error(f"Failed to move card {card_to_flip} to Temp2. State not updated.")
                                 if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "error", "payload": f"Failed arm movement for card {card_to_flip}."})
                         else:
                             logging.warning(f"Failed to detect color on card {card_to_flip}. Cannot flip second card.")
                             if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "message", "payload": f"Could not detect color on card {card_to_flip}. Will try again later."})
                             # card_states[card_to_flip]["isFlippedBefore"] = True
                             # card_states[card_to_flip]["color"] = "DETECT_FAIL"
                         await update_frontend_state()
                    else:
                        logging.info(f"One card flipped ({first_card_id}), but no other cards left? Returning first card.")
                        await from_to(websocket, "temp1", "card", first_card_id)
                        current_flipped.clear()
                        await update_frontend_state()

            elif len(current_flipped) == 2:
                card1_id, card2_id = current_flipped[0], current_flipped[1]
                color1 = card_states.get(card1_id,{}).get("color")
                color2 = card_states.get(card2_id,{}).get("color")
                logging.info(f"Strategy: Two cards flipped: {card1_id} ({color1}) and {card2_id} ({color2}). Checking match.")

                if color1 is not None and color1 == color2:
                    logging.info(f"Match confirmed between {card1_id} and {card2_id} ({color1}). Removing pair.")
                    if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "message", "payload": f"Match found: {color1}! Removing cards {card1_id} & {card2_id}."})
                    success1 = await from_to(websocket, "temp1", "trash", card1_id)
                    success2 = await from_to(websocket, "temp2", "trash", card2_id)
                    if success1 and success2:
                        card_states[card1_id]["isMatched"] = True
                        card_states[card2_id]["isMatched"] = True
                        game_state["pairs_found"] += 1
                    else:
                        logging.error(f"Failed to physically remove matched pair {card1_id} & {card2_id} from temp locations. State not updated.")
                        if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "error", "payload": f"Failed arm movement for matched pair {card1_id}/{card2_id}."})
                else:
                    logging.info(f"No match between {card1_id} ({color1}) and {card2_id} ({color2}). Returning cards.")
                    if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "message", "payload": f"No match. Returning cards {card1_id} & {card2_id}."})
                    success1 = await from_to(websocket, "temp1", "card", card1_id)
                    success2 = await from_to(websocket, "temp2", "card", card2_id)
                    if not (success1 and success2):
                         logging.error(f"Failed to return non-matching cards {card1_id} or {card2_id} to board.")
                         if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_json({"type": "error", "payload": f"Failed arm movement returning cards {card1_id}/{card2_id}."})

                current_flipped.clear()
                await update_frontend_state()
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_json({"type": "cards_hidden", "payload": [card1_id, card2_id]})

            if game_state["pairs_found"] >= CARD_COUNT // 2:
                 logging.info("Color Game Finished!")
                 if websocket.client_state == WebSocketState.CONNECTED:
                      await websocket.send_json({"type": "game_over", "payload": "Congratulations! All pairs found."})
                 game_state["running"] = False
                 break

    except WebSocketDisconnect:
        logging.info("WebSocket disconnected during Color game.")
    except Exception as e:
        logging.error(f"Error during Color game loop: {e}", exc_info=True)
        if websocket.client_state == WebSocketState.CONNECTED:
            try: await websocket.send_json({"type": "error", "payload": f"Critical Game Error: {e}"})
            except Exception: pass
    finally:
        logging.info("Cleaning up Color game runner.")
        game_state["running"] = False
        if camera_task and not camera_task.done():
            logging.info("Attempting to cancel camera task for Color game.")
            camera_task.cancel()
            try: await camera_task
            except asyncio.CancelledError: logging.info("Camera task successfully cancelled.")
            except Exception as task_e: logging.error(f"Error awaiting cancelled camera task: {task_e}")

        logging.info("Ensuring arm is home after Color game.")
        try:
             await send_arm_command(arm_home[0], arm_home[1], arm_home[2], 0, 1)
        except Exception as final_home_e:
             logging.error(f"Failed to send final home command: {final_home_e}")
        logging.info("Color Game Runner finished cleanup.")


# --- WebSocket Endpoint ---
@app.websocket("/ws/{game_version}")
async def websocket_endpoint(websocket: WebSocket, game_version: str):
    try:
        await websocket.accept()
    except Exception as accept_e:
         logging.error(f"WebSocket accept failed: {accept_e}")
         return # Cannot proceed if accept fails

    logging.info(f"WebSocket connection accepted for game version: {game_version}")

    if game_version not in ["color", "yolo"]:
        logging.error(f"Invalid game version requested: {game_version}")
        await websocket.send_json({"type": "error", "payload": "Invalid game version"})
        await websocket.close(code=1008)
        return

    lock = game_locks[game_version]
    if lock.locked():
        logging.warning(f"{game_version.capitalize()} game already in progress.")
        await websocket.send_json({"type": "error", "payload": f"{game_version.capitalize()} game already running."})
        await websocket.close(code=1008)
        return

    runner_task = None
    async with lock:
        if not setup_serial():
             logging.error("Failed to initialize serial port. Closing connection.")
             await websocket.send_json({"type": "error", "payload": "Failed to initialize serial port."})
             await websocket.close(code=1011)
             return

        active_games[game_version] = {"running": True}

        try:
            if game_version == "yolo":
                runner_task = asyncio.create_task(run_yolo_game(websocket))
            elif game_version == "color":
                runner_task = asyncio.create_task(run_color_game(websocket))
            else: # Should not happen due to check above, but safety first
                 raise ValueError("Invalid game version reached runner start.")

            # Keep connection alive while runner is active
            while active_games.get(game_version, {}).get("running") and runner_task and not runner_task.done():
                 try:
                      # Periodically check state or wait for a potential client message
                      # await websocket.receive_text() # Uncomment to wait for client messages
                      await asyncio.sleep(1)
                 except WebSocketDisconnect:
                      logging.info(f"WebSocket disconnected by client during {game_version} game.")
                      active_games[game_version]["running"] = False # Signal runner to stop
                      break

            if runner_task: # Wait for task completion if it hasn't finished/failed
                 await runner_task

        except WebSocketDisconnect:
            logging.info(f"WebSocket disconnect handled for {game_version} (outer loop).")
            active_games[game_version]["running"] = False # Ensure flag is set
        except Exception as e:
            logging.error(f"Unexpected error in WebSocket handler for {game_version}: {e}", exc_info=True)
            active_games[game_version]["running"] = False # Signal stop on error
            if websocket.client_state == WebSocketState.CONNECTED:
                try: await websocket.send_json({"type": "error", "payload": f"Server error: {e}"})
                except Exception: pass
        finally:
            logging.info(f"Cleaning up {game_version} game resources...")
            # Ensure runner task is handled
            if runner_task and not runner_task.done():
                 logging.info(f"Cancelling runner task for {game_version}.")
                 runner_task.cancel()
                 try: await runner_task
                 except asyncio.CancelledError: pass
                 except Exception as task_e: logging.error(f"Error during task cleanup: {task_e}")

            active_games[game_version] = {} # Clear game state
            global ser
            if ser and ser.is_open:
                logging.info("Closing serial port.")
                try: ser.close()
                except Exception as close_e: logging.error(f"Error closing serial port: {close_e}")
                ser = None

            if websocket.client_state != WebSocketState.DISCONNECTED:
                 try: await websocket.close()
                 except Exception: pass
            logging.info(f"WebSocket connection processing finished for {game_version}.")


# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server...")
    print("Ensure your React app is built and in the 'build' directory relative to this script.")
    print(f"Access the game at http://<your-ip>:{8000}") # Port hardcoded for now
    uvicorn.run("memory_matching_backend:app", host="0.0.0.0", port=8000, reload=False) # Changed app name