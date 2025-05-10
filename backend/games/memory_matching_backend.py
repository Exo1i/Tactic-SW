import asyncio
import base64
import json
import logging
import random
import threading
import time
from typing import Dict, Any, Optional, List, Tuple
import os

import cv2
import numpy as np
import serial # For type hinting; actual instance comes from main.py
from fastapi import WebSocket
from starlette.websockets import WebSocketState, WebSocketDisconnect # For WS state checking

# --- Configuration Constants (Defaults, can be overridden by config from main.py) ---
CAMERA_URL_DEFAULT = 'http://192.168.2.19:4747/video'
# This YOLO_MODEL_PATH is relative to the 'games' directory.
# main.py will construct the absolute path before passing it to GameSession.
YOLO_MODEL_PATH = "yolov5s.pt"

# --- Game Constants ---
import serial
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from starlette.websockets import WebSocketState
from starlette.exceptions import HTTPException # Added for catch_all


# --- Configuration ---
SERIAL_PORT = '/dev/ttyUSB0'  # Adjust as needed (e.g., 'COM3' on Windows)
BAUD_RATE = 9600
CAMERA_URL = 'http://192.168.49.1:4747/video' # Primary Camera URL

YOLO_MODEL_PATH = "./yolov5s.pt" # Or your specific model path

# --- Constants ---
GRID_ROWS = 2
GRID_COLS = 4
CARD_COUNT = GRID_ROWS * GRID_COLS
FLIP_DELAY_SECONDS = 0.5

# --- Arm Control Values (remain module-level constants) ---
arm_values = [[110, 40, 125], [87, 65, 120], [87, 110, 120], [110, 140, 125],
              [150, 55, 155], [130, 80, 140], [130, 105, 140], [150, 125, 155]]
arm_home = [180, 90, 0]
arm_temp1 = [90, 10, 120]
arm_temp2 = [90, 170, 120]
arm_trash = [140, 0, 140]

ARM_SYNC_STEP_DELAY = 0.3
ARM_MAX_RETRIES = 10
ARM_RETRY_DELAY_SECONDS = 1.0
DETECTION_MAX_RETRIES = 1000
DETECTION_RETRY_DELAY_SECONDS = 0.75
DETECTION_PERMANENT_FAIL_STATE = "PERMA_FAIL"

YOLO_TARGET_LABELS = ['orange', 'apple', 'cat', 'car', 'umbrella', 'banana', 'fire hydrant', 'person']
YOLO_FRAME_WIDTH = 640
YOLO_FRAME_HEIGHT = 480

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

# --- Logging Setup (module-level) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Board Detection / Transformation Helper Functions (module-level is fine) ---
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
            corners_np = np.array([p[0] for p in approx], dtype=np.float32) # Renamed to avoid conflict
            return sort_corners(corners_np)
        return None
    except Exception as e: logging.error(f"Error finding board corners: {e}"); return None

def sort_corners(corners_np: np.ndarray) -> np.ndarray: # Renamed parameter
    rect = np.zeros((4, 2), dtype="float32")
    s = corners_np.sum(axis=1); rect[0] = corners_np[np.argmin(s)]; rect[2] = corners_np[np.argmax(s)]
    diff = np.diff(corners_np, axis=1); rect[1] = corners_np[np.argmin(diff)]; rect[3] = corners_np[np.argmax(diff)]
    return rect

def transform_board(frame: np.ndarray, corners_np: np.ndarray) -> Optional[np.ndarray]: # Renamed parameter
    if frame is None or corners_np is None: return None
    try:
        dst_points = np.array([
            [0, 0], [COLOR_BOARD_DETECT_WIDTH - 1, 0],
            [COLOR_BOARD_DETECT_WIDTH - 1, COLOR_BOARD_DETECT_HEIGHT - 1],
            [0, COLOR_BOARD_DETECT_HEIGHT - 1]
        ], dtype=np.float32)
        M = cv2.getPerspectiveTransform(corners_np, dst_points)
        warped = cv2.warpPerspective(frame, M, (COLOR_BOARD_DETECT_WIDTH, COLOR_BOARD_DETECT_HEIGHT))
        return warped
    except Exception as e: logging.error(f"Error during perspective transform: {e}"); return None


class MemoryMatching:
    def __init__(self, config: Dict[str, Any]):
        # Use a simple string representation for serial_instance in logs if it's complex
        log_config = {k: (str(v) if isinstance(v, serial.Serial) else v) for k, v in config.items()}
        logging.info(f"MemoryMatching GameSession initializing with config: {json.dumps(log_config, default=lambda o: '<unserializable>')}")

        self.game_mode = config.get("mode", "color")
        self.serial_instance: Optional[serial.Serial] = config.get("serial_instance")
        self.webcam_ip = config.get("webcam_ip", CAMERA_URL_DEFAULT)
        self.yolo_model_path_abs = YOLO_MODEL_PATH

        self.yolo_model: Optional[Any] = None
        self.ultralytics_yolo_class = None # To store the YOLO class from ultralytics

        # Store frames in a dictionary for clarity
        self._latest_frame_data: Dict[str, Optional[np.ndarray]] = {
            "raw": None,
            "transformed": None
        }
        self._frame_lock = asyncio.Lock() # For async access to frames
        self._serial_lock = threading.Lock() # For thread-safe serial commands

        self.game_state_internal: Dict[str, Any] = {}
        self._websocket: Optional[WebSocket] = None # FastAPI WebSocket type
        self._async_loop: Optional[asyncio.AbstractEventLoop] = None
        self._capture_task: Optional[asyncio.Task] = None
        self._game_runner_task: Optional[asyncio.Task] = None

        self._stop_event = asyncio.Event() # For signaling async tasks to stop

        self.serial_port_config_from_main = config.get("serial_port_config", {})

        if not self.serial_instance or not self.serial_instance.is_open:
            logging.warning("MemoryMatching: Serial instance not provided or not open. Arm control may fail.")
        elif self.serial_instance and self.serial_instance.is_open and self.serial_port_config_from_main.get("type") == "usb":
             logging.info("MemoryMatching: USB serial detected, ensuring Arduino reset delay.")
             time.sleep(2) # Wait for Arduino to reset if it's a direct USB to an Arduino
             try:
                self.serial_instance.reset_input_buffer()
                self.serial_instance.reset_output_buffer()
                if self.serial_instance.in_waiting > 0:
                    initial_data = self.serial_instance.read(self.serial_instance.in_waiting).decode(errors='replace')
                    logging.info(f"MemoryMatching: Cleared initial serial data from provided instance: {initial_data.strip()}")
             except Exception as e:
                 logging.error(f"MemoryMatching: Error resetting buffers on provided serial instance: {e}")

    async def _load_yolo_model_if_needed(self):
        if self.game_mode == "yolo" and self.yolo_model is None:
            if not self.yolo_model_path_abs:
                logging.error("YOLO mode selected but no model path provided.")
                return False
            try:
                import importlib
                try: importlib.import_module('ultralytics')
                except ImportError:
                    logging.error("Ultralytics library not found. YOLO mode unavailable.")
                    return False

                from ultralytics import YOLO as UltralyticsYOLO
                self.ultralytics_yolo_class = UltralyticsYOLO

                if not os.path.exists(self.yolo_model_path_abs):
                    logging.error(f"YOLO model file not found: {self.yolo_model_path_abs}. YOLO mode unavailable.")
                    return False

                logging.info(f"Loading YOLO model for GameSession from: {self.yolo_model_path_abs}")
                self.yolo_model = self.ultralytics_yolo_class(self.yolo_model_path_abs)

                logging.info("Warming up YOLO model for GameSession...")
                dummy_img = np.zeros((64, 64, 3), dtype=np.uint8) # Small dummy image
                # Run predict in a thread to avoid blocking asyncio loop
                await asyncio.to_thread(self.yolo_model.predict, dummy_img, verbose=False, device='cpu') # Added device='cpu' for broader compatibility
                logging.info("GameSession YOLO model loaded and warmed up successfully.")
                return True
            except Exception as e:
                logging.error(f"Error loading GameSession YOLO model: {e}", exc_info=True)
                self.yolo_model = None
                return False
        return True # Return True if not yolo_mode or model already loaded

    async def manage_game_loop(self, websocket: WebSocket): # WebSocket type from FastAPI
        self._websocket = websocket
        self._async_loop = asyncio.get_running_loop()
        self._stop_event.clear()

        logging.info(f"MemoryMatching ({self.game_mode}): manage_game_loop started.")

        if not await self._load_yolo_model_if_needed(): # Handles yolo_mode check internally
            if self.game_mode == "yolo": # Only error out if it was supposed to load for yolo
                await self._send_error_to_client("YOLO model failed to load.")
                await self.cleanup() # Ensure cleanup if we exit early
                return

        self.game_state_internal = {"running": True}

        try:
            if self.game_mode == "yolo":
                self._game_runner_task = asyncio.create_task(self._run_yolo_game_logic())
            elif self.game_mode == "color":
                self._game_runner_task = asyncio.create_task(self._run_color_game_logic())
            else:
                await self._send_error_to_client(f"Invalid game mode: {self.game_mode}")
                await self.cleanup()
                return

            logging.info(f"MemoryMatching ({self.game_mode}): Game runner task created.")

            # Monitor loop: Keep connection open while runner is active
            while not self._stop_event.is_set():
                if self._game_runner_task.done():
                    logging.info(f"MemoryMatching ({self.game_mode}): Runner task finished.")
                    try: self._game_runner_task.result() # Raise exceptions from the task if any occurred
                    except asyncio.CancelledError:
                        logging.info(f"MemoryMatching ({self.game_mode}): Runner task was cancelled.")
                    except Exception as runner_exception:
                        logging.error(f"MemoryMatching ({self.game_mode}): Runner task failed with exception: {runner_exception}", exc_info=True)
                        await self._send_error_to_client(f"Game ended due to server error: {runner_exception}")
                    break # Exit monitor loop

                if self._websocket.client_state != WebSocketState.CONNECTED:
                    logging.info(f"MemoryMatching ({self.game_mode}): WebSocket disconnected by client.")
                    self._stop_event.set() # Signal runner to stop
                    break # Exit monitor loop

                await asyncio.sleep(0.5) # Keepalive check

        except WebSocketDisconnect: # This can be raised by await websocket.receive_* if used
            logging.info(f"MemoryMatching ({self.game_mode}): WebSocket disconnected (WebSocketDisconnect).")
            self._stop_event.set()
        except Exception as e:
            logging.error(f"MemoryMatching ({self.game_mode}): Error in manage_game_loop: {e}", exc_info=True)
            self._stop_event.set()
            await self._send_error_to_client(f"Server connection error: {e}")
        finally:
            logging.info(f"MemoryMatching ({self.game_mode}): manage_game_loop ending. Initiating cleanup.")
            await self.cleanup()

    async def _send_json_to_client(self, data: Dict):
        if self._websocket and self._websocket.client_state == WebSocketState.CONNECTED:
            try:
                await self._websocket.send_json(data)
            except WebSocketDisconnect:
                logging.warning(f"MemoryMatching ({self.game_mode}): WS disconnected while trying to send JSON.")
            except Exception as e:
                logging.error(f"MemoryMatching ({self.game_mode}): Error sending JSON to client: {e}")

    async def _send_error_to_client(self, message: str):
        await self._send_json_to_client({"type": "error", "payload": message})

    def _send_arm_command_sync(self, degree1: int, degree2: int, degree3: int, magnet: int, movement: int) -> Optional[str]:
        if not self.serial_instance or not self.serial_instance.is_open:
            logging.error("Serial port not available or not open for sending command.")
            return None
        if not (0 <= degree1 <= 180 and 0 <= degree2 <= 180 and 0 <= degree3 <= 180):
            logging.error(f"Invalid servo degrees: ({degree1}, {degree2}, {degree3}). Must be 0-180.")
            return None
        if magnet not in [0, 1]:
            logging.error(f"Invalid magnet value: {magnet}. Must be 0 or 1.")
            return None

        command = f"{degree1},{degree2},{degree3},{magnet},{movement}\n"
        command_bytes = command.encode('utf-8')
        command_strip = command.strip()

        attempt = 0
        while attempt < ARM_MAX_RETRIES:
            attempt += 1
            logging.info(f"MM Arm Sending (Attempt {attempt}/{ARM_MAX_RETRIES}): {command_strip}")
            with self._serial_lock: # Use instance serial_lock
                try:
                    if not self.serial_instance or not self.serial_instance.is_open: # Re-check inside lock
                        logging.error(f"MM Arm: Serial port became unavailable before attempt {attempt}.")
                        attempt = ARM_MAX_RETRIES; continue

                    self.serial_instance.reset_input_buffer() # Clear buffer before reading response
                    self.serial_instance.write(command_bytes)
                    self.serial_instance.flush() # Ensure data is sent

                    response = b''
                    start_time = time.time()
                    timeout_response = 12.0 # Timeout FOR THIS ATTEMPT

                    while time.time() - start_time < timeout_response:
                        if self.serial_instance.in_waiting > 0:
                            chunk = self.serial_instance.read(self.serial_instance.in_waiting)
                            response += chunk
                            if b"done" in response.lower():
                                break
                        time.sleep(0.02) # Short sleep to avoid busy-waiting

                    decoded_response = response.decode('utf-8', errors='replace').strip()
                    logging.debug(f"MM Arm Attempt {attempt} raw response: {response}")
                    logging.info(f"MM Arm Attempt {attempt} decoded response: '{decoded_response}'")

                    if "done" in decoded_response.lower():
                        logging.info(f"MM Arm Command '{command_strip}' successful on attempt {attempt}.")
                        return decoded_response
                    else:
                        logging.warning(f"MM Arm Cmd '{command_strip}' attempt {attempt} failed: 'done' not received. Resp: '{decoded_response}'")

                except serial.SerialException as e:
                    logging.error(f"MM Arm Serial comm error during attempt {attempt}: {e}")
                    # Let retries happen. If it's truly dead, main.py might re-init or game fails.
                    attempt = ARM_MAX_RETRIES # Or make it fatal for this game session
                except Exception as e:
                    logging.error(f"MM Arm Unexpected error serial cmd attempt {attempt}: {e}", exc_info=True)

            if attempt < ARM_MAX_RETRIES:
                logging.info(f"MM Arm: Waiting {ARM_RETRY_DELAY_SECONDS}s before retry...")
                time.sleep(ARM_RETRY_DELAY_SECONDS)

        logging.error(f"MM Arm Command '{command_strip}' FAILED after {ARM_MAX_RETRIES} attempts.")
        return None

    def _from_to_sync(self, src: str, dest: str, card_id: int) -> bool:
        logging.info(f"MM SYNC movement sequence: card {card_id} from {src} to {dest}")
        success = True
        if src not in ["card", "temp1", "temp2", "home"] or dest not in ["card", "temp1", "temp2", "trash", "home"]:
            logging.error(f"Invalid src ('{src}') or dest ('{dest}') location.")
            return False
        if src == "card" or dest == "card":
            if not (0 <= card_id < len(arm_values)):
                logging.error(f"Invalid card_id {card_id} for arm_values length {len(arm_values)}")
                return False
        try:
            if src == "card" and dest == "temp1":
                if self._send_arm_command_sync(arm_values[card_id][0], arm_values[card_id][1], arm_values[card_id][2], 1, 0) is None: success = False
                if success: time.sleep(ARM_SYNC_STEP_DELAY)
                if success and self._send_arm_command_sync(arm_home[0], arm_home[1], arm_home[2], 1, 1) is None: success = False
                if success: time.sleep(ARM_SYNC_STEP_DELAY)
                if success and self._send_arm_command_sync(arm_temp1[0], arm_temp1[1], arm_temp1[2], 0, 0) is None: success = False
                if success: time.sleep(ARM_SYNC_STEP_DELAY)
                if success and self._send_arm_command_sync(arm_home[0], arm_home[1], arm_home[2], 0, 1) is None: success = False
            elif src == "card" and dest == "temp2":
                if self._send_arm_command_sync(arm_values[card_id][0], arm_values[card_id][1], arm_values[card_id][2], 1, 0) is None: success = False
                if success: time.sleep(ARM_SYNC_STEP_DELAY)
                if success and self._send_arm_command_sync(arm_home[0],arm_home[1], arm_home[2], 1, 1) is None: success = False
                if success: time.sleep(ARM_SYNC_STEP_DELAY)
                if success and self._send_arm_command_sync(arm_temp2[0], arm_temp2[1], arm_temp2[2], 0, 0) is None: success = False
                if success: time.sleep(ARM_SYNC_STEP_DELAY)
                if success and self._send_arm_command_sync(arm_home[0], arm_home[1], arm_home[2], 0, 1) is None: success = False
            elif src == "card" and dest == "trash":
                if self._send_arm_command_sync(arm_values[card_id][0], arm_values[card_id][1], arm_values[card_id][2], 1, 0) is None: success = False
                if success: time.sleep(ARM_SYNC_STEP_DELAY)
                if success and self._send_arm_command_sync(arm_home[0], arm_home[1], arm_home[2], 1, 1) is None: success = False
                if success: time.sleep(ARM_SYNC_STEP_DELAY)
                if success and self._send_arm_command_sync(arm_trash[0], arm_trash[1], arm_trash[2], 0, 0) is None: success = False
                if success: time.sleep(ARM_SYNC_STEP_DELAY)
                if success and self._send_arm_command_sync(arm_home[0], arm_home[1], arm_home[2], 0, 1) is None: success = False
            elif src == "temp1" and dest == "trash":
                if self._send_arm_command_sync(arm_temp1[0], arm_temp1[1], arm_temp1[2], 1, 0) is None: success = False
                if success: time.sleep(ARM_SYNC_STEP_DELAY)
                if success and self._send_arm_command_sync(arm_home[0], arm_home[1], arm_home[2], 1, 1) is None: success = False
                if success: time.sleep(ARM_SYNC_STEP_DELAY)
                if success and self._send_arm_command_sync(arm_trash[0], arm_trash[1], arm_trash[2], 0, 0) is None: success = False
                if success: time.sleep(ARM_SYNC_STEP_DELAY)
                if success and self._send_arm_command_sync(arm_home[0], arm_home[1], arm_home[2], 0, 1) is None: success = False
            elif src == "temp2" and dest == "trash":
                if self._send_arm_command_sync(arm_temp2[0], arm_temp2[1], arm_temp2[2], 1, 0) is None: success = False
                if success: time.sleep(ARM_SYNC_STEP_DELAY)
                if success and self._send_arm_command_sync(arm_home[0], arm_home[1], arm_home[2], 1, 1) is None: success = False
                if success: time.sleep(ARM_SYNC_STEP_DELAY)
                if success and self._send_arm_command_sync(arm_trash[0], arm_trash[1], arm_trash[2], 0, 0) is None: success = False
                if success: time.sleep(ARM_SYNC_STEP_DELAY)
                if success and self._send_arm_command_sync(arm_home[0], arm_home[1], arm_home[2], 0, 1) is None: success = False
            elif src == "temp1" and dest == "card":
                if self._send_arm_command_sync(arm_temp1[0], arm_temp1[1], arm_temp1[2], 1, 0) is None: success = False
                if success: time.sleep(ARM_SYNC_STEP_DELAY)
                if success and self._send_arm_command_sync(arm_home[0], arm_home[1], arm_home[2], 1, 1) is None: success = False
                if success: time.sleep(ARM_SYNC_STEP_DELAY)
                if success and self._send_arm_command_sync(arm_values[card_id][0], arm_values[card_id][1], arm_values[card_id][2], 0, 0) is None: success = False
                if success: time.sleep(ARM_SYNC_STEP_DELAY)
                if success and self._send_arm_command_sync(arm_home[0], arm_home[1], arm_home[2], 0, 1) is None: success = False
            elif src == "temp2" and dest == "card":
                if self._send_arm_command_sync(arm_temp2[0], arm_temp2[1], arm_temp2[2], 1, 0) is None: success = False
                if success: time.sleep(ARM_SYNC_STEP_DELAY)
                if success and self._send_arm_command_sync(arm_home[0], arm_home[1], arm_home[2], 1, 1) is None: success = False
                if success: time.sleep(ARM_SYNC_STEP_DELAY)
                if success and self._send_arm_command_sync(arm_values[card_id][0], arm_values[card_id][1], arm_values[card_id][2], 0, 0) is None: success = False
                if success: time.sleep(ARM_SYNC_STEP_DELAY)
                if success and self._send_arm_command_sync(arm_home[0], arm_home[1], arm_home[2], 0, 1) is None: success = False
            elif src == "home" and dest == "home":
                if self._send_arm_command_sync(arm_home[0], arm_home[1], arm_home[2], 0, 1) is None: success = False
            else:
                logging.error(f"Invalid/unhandled src/dest combination: {src} -> {dest}"); success = False

            if not success:
                logging.error(f"MM SYNC movement sequence FAILED: A command failed persistently for card {card_id} ({src} -> {dest})")
                if not (src == "home" and dest == "home"):
                    logging.warning("Attempting to return arm home after sequence failure.")
                    self._send_arm_command_sync(arm_home[0], arm_home[1], arm_home[2], 0, 1) # Try recovery
                return False
            logging.info(f"MM SYNC movement sequence COMPLETED successfully: card {card_id} ({src} -> {dest})")
            return True
        except Exception as e:
            logging.error(f"Unexpected error during MM SYNC sequence ({src} -> {dest}, card {card_id}): {e}", exc_info=True)
            logging.warning("Attempting to return arm home after unexpected sequence error.")
            self._send_arm_command_sync(arm_home[0], arm_home[1], arm_home[2], 0, 1) # Try recovery
            return False

    async def _from_to(self, src: str, dest: str, card_id: int) -> bool:
        action_name = f"{src}_to_{dest}"
        logging.info(f"MM ASYNC wrapper for arm movement: card {card_id} [{action_name}]")
        move_successful = False
        try:
            move_successful = await asyncio.to_thread(self._from_to_sync, src, dest, card_id)
        except Exception as e:
            logging.error(f"Error calling/executing _from_to_sync via asyncio.to_thread for {action_name}: {e}", exc_info=True)
            move_successful = False
            logging.warning("MM: Attempting safe return home after thread execution error.")
            try: await asyncio.to_thread(self._send_arm_command_sync, arm_home[0], arm_home[1], arm_home[2], 0, 1)
            except Exception as home_e: logging.error(f"MM: Failed to return arm home after thread error: {home_e}")

        await self._send_json_to_client({
            "type": "arm_status",
            "payload": {"status": "finished", "success": move_successful, "action": action_name, "card_id": card_id}
        })
        logging.info(f"MM ASYNC wrapper finished for {action_name} (card {card_id}). Success: {move_successful}")
        return move_successful

    async def _capture_frames_task(self):
        cap = None
        logging.info(f"MM Starting frame capture for {self.game_mode} from {self.webcam_ip}")
        frame_count = 0; last_log_time = time.time()

        while not self._stop_event.is_set() and self.game_state_internal.get("running", False):
            warped_board_for_send = None # For sending to client
            try:
                if cap is None or not cap.isOpened():
                    logging.info(f"MM Attempting to open camera: {self.webcam_ip}")
                    cap = await asyncio.to_thread(cv2.VideoCapture, self.webcam_ip)
                    await asyncio.sleep(1.5) # Give camera time to initialize
                    if not cap.isOpened():
                        logging.error(f"MM Cannot open camera: {self.webcam_ip}. Retrying in 5s.")
                        cap = None
                        await self._send_error_to_client(f"Cannot open camera: {self.webcam_ip}")
                        await asyncio.sleep(5); continue
                    else:
                        logging.info(f"MM Camera {self.webcam_ip} opened successfully.")
                        await asyncio.to_thread(cap.set, cv2.CAP_PROP_FRAME_WIDTH, YOLO_FRAME_WIDTH)
                        await asyncio.to_thread(cap.set, cv2.CAP_PROP_FRAME_HEIGHT, YOLO_FRAME_HEIGHT)

                ret, frame = await asyncio.to_thread(cap.read) # cap.read() is blocking
                if not ret or frame is None:
                    logging.warning("MM Failed to grab frame. Releasing and retrying...")
                    if cap: await asyncio.to_thread(cap.release); cap = None
                    async with self._frame_lock: self._latest_frame_data["raw"], self._latest_frame_data["transformed"] = None, None
                    await asyncio.sleep(1); continue

                frame_count += 1
                current_frame_copy = frame.copy()
                local_latest_transformed = None

                # Board detection and transformation are CPU-bound, could be threaded if they become a bottleneck
                corners_found = find_board_corners(current_frame_copy) # This is a module-level function
                if corners_found is not None:
                    warped = transform_board(current_frame_copy, corners_found) # Module-level
                    if warped is not None:
                        local_latest_transformed = warped
                        warped_board_for_send = warped.copy() # Keep a copy for sending

                async with self._frame_lock:
                    self._latest_frame_data["raw"] = current_frame_copy
                    self._latest_frame_data["transformed"] = local_latest_transformed

                processed_frame_for_send = current_frame_copy # Use the raw frame for main view
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(processed_frame_for_send, timestamp, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),1,cv2.LINE_AA)
                cv2.putText(processed_frame_for_send, f"Mode: {self.game_mode.upper()}", (processed_frame_for_send.shape[1]-150,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),1,cv2.LINE_AA)
                if corners_found is not None:
                    cv2.polylines(processed_frame_for_send, [np.int32(corners_found)], True, (0,255,255),2)

                encode_param = [cv2.IMWRITE_JPEG_QUALITY, 75]
                # Encoding can be blocking, run in thread
                ret_main, buffer_main = await asyncio.to_thread(cv2.imencode, '.jpg', processed_frame_for_send, encode_param)
                jpg_main_as_text = base64.b64encode(buffer_main).decode('utf-8') if ret_main else None

                jpg_transformed_as_text = None
                if warped_board_for_send is not None: # Use the copy for sending
                    ret_trans, buffer_trans = await asyncio.to_thread(cv2.imencode, '.jpg', warped_board_for_send, encode_param)
                    if ret_trans: jpg_transformed_as_text = base64.b64encode(buffer_trans).decode('utf-8')

                if jpg_main_as_text:
                    payload_to_send = {"frame": jpg_main_as_text}
                    if jpg_transformed_as_text: payload_to_send["transformed_frame"] = jpg_transformed_as_text
                    await self._send_json_to_client({"type": "frame_update", "payload": payload_to_send})

                current_time_fps = time.time() # Renamed to avoid conflict with timestamp string
                if current_time_fps - last_log_time >= 5.0:
                    fps = frame_count / (current_time_fps - last_log_time)
                    logging.info(f"MM Camera FPS: {fps:.2f}"); frame_count=0; last_log_time=current_time_fps
                await asyncio.sleep(0.035) # Adjust sleep based on desired capture rate

            except WebSocketDisconnect: logging.info("MM WS disconnected in capture loop."); self._stop_event.set(); break
            except asyncio.CancelledError: logging.info("MM Capture task cancelled."); break
            except Exception as e:
                logging.error(f"Error in MM frame capture loop for {self.game_mode}: {e}", exc_info=True)
                if cap: await asyncio.to_thread(cap.release); cap = None # Ensure release on error
                await asyncio.sleep(1)

        if cap: await asyncio.to_thread(cap.release)
        async with self._frame_lock: self._latest_frame_data["raw"], self._latest_frame_data["transformed"] = None, None
        logging.info(f"MM Frame capture task stopped for {self.game_mode}.")

    async def _detect_object_at_card(self, card_id: int) -> Optional[str]:
        if self.yolo_model is None:
            logging.error("YOLO model not loaded for GameSession. Cannot perform detection.")
            return None # Indicate model is missing

        attempt = 0
        while attempt < DETECTION_MAX_RETRIES and not self._stop_event.is_set():
            attempt += 1; warped_board = None
            async with self._frame_lock: # Get latest WARPED frame
                if self._latest_frame_data["transformed"] is not None:
                    warped_board = self._latest_frame_data["transformed"].copy()

            if warped_board is None:
                logging.warning(f"MM YOLO Attempt {attempt}: No transformed board available for card {card_id}.")
                if attempt < DETECTION_MAX_RETRIES: await asyncio.sleep(DETECTION_RETRY_DELAY_SECONDS); continue
                else: break # Max retries for getting board

            try:
                row, col = card_id // GRID_COLS, card_id % GRID_COLS
                cell_w = COLOR_BOARD_DETECT_WIDTH // GRID_COLS; cell_h = COLOR_BOARD_DETECT_HEIGHT // GRID_ROWS
                x1,y1 = col*cell_w, row*cell_h; x2,y2 = x1+cell_w, y1+cell_h
                pad=5; roi_x1,roi_y1 = max(0,x1+pad), max(0,y1+pad)
                roi_x2,roi_y2 = min(COLOR_BOARD_DETECT_WIDTH-1, x2-pad), min(COLOR_BOARD_DETECT_HEIGHT-1, y2-pad)
                if roi_x1 >= roi_x2 or roi_y1 >= roi_y2: continue
                card_roi = warped_board[roi_y1:roi_y2, roi_x1:roi_x2]
                if card_roi.size == 0: continue

                # YOLO predict is blocking, run in thread
                def sync_predict_yolo(model, roi_img): # Renamed roi to roi_img
                    target_indices = [i for i, lbl in enumerate(model.names.values()) if lbl.lower() in YOLO_TARGET_LABELS]
                    return model.predict(roi_img, conf=0.45, verbose=False, device='cpu', classes=target_indices if target_indices else None)

                yolo_results = await asyncio.to_thread(sync_predict_yolo, self.yolo_model, card_roi)

                detected_obj_label = None; highest_conf = 0.0
                if yolo_results:
                    for res_item in yolo_results: # Iterate through prediction results
                        boxes = getattr(res_item, 'boxes', None); names_dict = getattr(res_item, 'names', {})  # Class names dictionary
                        if boxes: # Check if boxes were found
                            for box_item in boxes: # Iterate through detected boxes
                                cls_tensor = getattr(box_item, 'cls', None); conf_tensor = getattr(box_item, 'conf', None)
                                if cls_tensor is not None and conf_tensor is not None and cls_tensor.numel() > 0 and conf_tensor.numel() > 0:
                                    try:
                                        label_idx = int(cls_tensor[0].item()); score = conf_tensor[0].item()
                                        label_name = names_dict.get(label_idx, f"unknown_idx_{label_idx}").lower() # Get label name
                                        if label_name in YOLO_TARGET_LABELS and score > highest_conf:
                                            highest_conf = score; detected_obj_label = label_name
                                    except Exception as proc_err:
                                        logging.error(f"MM YOLO Attempt {attempt}: Error processing YOLO box data for card {card_id}: {proc_err}")

                if detected_obj_label:
                    logging.info(f"MM YOLO Successful Detection on attempt {attempt}: '{detected_obj_label}' (conf: {highest_conf:.2f}) for card {card_id}")
                    return detected_obj_label
                else: # No target object detected meeting criteria
                    logging.warning(f"MM YOLO Attempt {attempt}: No target object detected on card {card_id}.")
                    if attempt < DETECTION_MAX_RETRIES: await asyncio.sleep(DETECTION_RETRY_DELAY_SECONDS)
            except cv2.error as cv_err:
                logging.error(f"MM YOLO Attempt {attempt}: OpenCV error during YOLO detection for card {card_id}: {cv_err}", exc_info=True)
            except Exception as e: # Catch errors in ROI calc, etc.
                logging.error(f"MM YOLO Attempt {attempt}: Unexpected error during YOLO detection for card {card_id}: {e}", exc_info=True)
            # If detection failed on this attempt and more retries allowed, wait (done if no detected_obj_label)

        logging.error(f"MM YOLO Detection FAILED permanently for card {card_id} after {DETECTION_MAX_RETRIES} attempts.")
        return DETECTION_PERMANENT_FAIL_STATE

    async def _detect_color_at_card(self, card_id: int) -> Optional[str]:
        attempt = 0
        while attempt < DETECTION_MAX_RETRIES and not self._stop_event.is_set():
            attempt += 1; warped_board = None
            async with self._frame_lock:
                if self._latest_frame_data["transformed"] is not None:
                    warped_board = self._latest_frame_data["transformed"].copy()

            if warped_board is None:
                logging.warning(f"MM Color Attempt {attempt}: No transformed board available for card {card_id}.")
                if attempt < DETECTION_MAX_RETRIES: await asyncio.sleep(DETECTION_RETRY_DELAY_SECONDS); continue
                else: break

            try:
                row, col = card_id // GRID_COLS, card_id % GRID_COLS
                cell_w = COLOR_BOARD_DETECT_WIDTH // GRID_COLS; cell_h = COLOR_BOARD_DETECT_HEIGHT // GRID_ROWS
                x1,y1 = col*cell_w, row*cell_h; x2,y2 = x1+cell_w, y1+cell_h
                pad=5; roi_x1,roi_y1 = max(0,x1+pad), max(0,y1+pad)
                roi_x2,roi_y2 = min(COLOR_BOARD_DETECT_WIDTH-1, x2-pad), min(COLOR_BOARD_DETECT_HEIGHT-1, y2-pad)
                if roi_x1 >= roi_x2 or roi_y1 >= roi_y2: continue
                cell_roi = warped_board[roi_y1:roi_y2, roi_x1:roi_x2]
                if cell_roi.size == 0: continue

                hsv_roi = cv2.cvtColor(cell_roi, cv2.COLOR_BGR2HSV) # CPU bound, could thread if slow
                detected_colors_count: Dict[str, int] = {}
                for color_def_item in COLOR_RANGES: # Renamed to avoid conflict
                    color_name = color_def_item['name']
                    total_mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
                    for l_b, u_b in zip(color_def_item['lower'], color_def_item['upper']):
                        mask_part = cv2.inRange(hsv_roi, np.array(l_b), np.array(u_b))
                        total_mask = cv2.bitwise_or(total_mask, mask_part)
                    px_count = cv2.countNonZero(total_mask)
                    if px_count > 0: detected_colors_count[color_name] = px_count

                black_thresh = COLOR_CELL_THRESHOLD * 0.7
                if "black" in detected_colors_count and detected_colors_count["black"] > black_thresh:
                    logging.info(f"MM Color Attempt {attempt}: Card {card_id} detected as 'black'.")
                    return "black" # Return black immediately

                dominant_color_val = None; max_pixels = 0 # Renamed to avoid conflict
                for color_name_iter, px_count_iter in detected_colors_count.items(): # Renamed to avoid conflict
                    if color_name_iter != "black" and px_count_iter >= COLOR_CELL_THRESHOLD:
                        if px_count_iter > max_pixels: max_pixels = px_count_iter; dominant_color_val = color_name_iter

                if dominant_color_val:
                    logging.info(f"MM Color Successful Detection on attempt {attempt}: '{dominant_color_val}' (pixels: {max_pixels}) for card {card_id}")
                    return dominant_color_val
                else: # No dominant face color found meeting threshold
                    relevant_counts = {k:v for k,v in detected_colors_count.items() if k!='black' and v > 10}
                    logging.warning(f"MM Color Attempt {attempt}: No dominant face color found on card {card_id}. Counts(<Thr): {relevant_counts}")
                    if attempt < DETECTION_MAX_RETRIES: await asyncio.sleep(DETECTION_RETRY_DELAY_SECONDS)
            except cv2.error as cv_err:
                logging.error(f"MM Color Attempt {attempt}: OpenCV error during color detection: {cv_err}", exc_info=True)
            except Exception as e:
                logging.error(f"MM Color Attempt {attempt}: Unexpected error during color detection: {e}", exc_info=True)
            # If detection failed (no board, no color, error) and more retries allowed, wait (done if no dominant_color_val)

        logging.error(f"MM Color Detection FAILED permanently for card {card_id} after {DETECTION_MAX_RETRIES} attempts.")
        return DETECTION_PERMANENT_FAIL_STATE

    async def _run_yolo_game_logic(self):
        gs_key = self.game_mode.upper() # For logging prefix
        logging.info(f"[{gs_key}] Starting YOLO Game Logic...")
        gs = self.game_state_internal # gs is a shortcut to self.game_state_internal

        gs.update({
            "card_states": {i: {"isFlippedBefore": False, "object": None, "isMatched": False} for i in range(CARD_COUNT)},
            "objects_found": {obj: [] for obj in YOLO_TARGET_LABELS}, "pairs_found": 0, "current_flipped_cards": [],
            "running": True, "last_detect_fail_id": None,
        })
        logging.info(f"[{gs_key}] Initialized game state.")

        await self._send_json_to_client({"type": "game_state", "payload": {
            "card_states": gs["card_states"], "pairs_found": gs["pairs_found"], "current_flipped_cards": gs["current_flipped_cards"]
        }})
        await self._send_json_to_client({"type": "message", "payload": "YOLO Game Started. Initializing arm..."})

        self._capture_task = asyncio.create_task(self._capture_frames_task())
        init_home_success = await self._from_to("home", "home", -1) # Use self method
        if not init_home_success:
            gs["running"] = False; 
            # await self._send_error_to_client("Arm init failed.")
            # return # Stop game if arm fails
        await self._send_json_to_client({"type": "message", "payload": "Arm ready. Starting game."})

        try:
            while gs.get("pairs_found",0) < (CARD_COUNT//2) and gs.get("running",False) and not self._stop_event.is_set():
                await asyncio.sleep(FLIP_DELAY_SECONDS)
                if self._websocket.client_state != WebSocketState.CONNECTED: gs["running"]=False; break

                current_flipped = gs.get("current_flipped_cards", [])
                card_states = gs.get("card_states", {})
                objects_found = gs.get("objects_found", {})
                pairs_found = gs.get("pairs_found", 0)
                logging.info(f"[{gs_key}] Loop Start: Flipped={current_flipped}, Pairs={pairs_found}/{CARD_COUNT // 2}")

                async def update_frontend_state(extra_message: Optional[str] = None):
                    payload = {"card_states": card_states, "pairs_found": gs.get("pairs_found",0), "current_flipped": current_flipped} # Use gs for pairs_found
                    await self._send_json_to_client({"type": "game_state", "payload": payload})
                    if extra_message: await self._send_json_to_client({"type": "message", "payload": extra_message})

                def choose_random_card() -> Optional[int]:
                    available = [i for i,s in card_states.items() if not s.get("isMatched") and i not in current_flipped and s.get("object") != DETECTION_PERMANENT_FAIL_STATE]
                    if not available: return None
                    never_flipped = [i for i in available if not card_states[i].get("isFlippedBefore")]
                    if never_flipped:
                        chosen = random.choice(never_flipped)
                        if chosen == gs.get("last_detect_fail_id") and len(never_flipped) > 1:
                            chosen = random.choice([c for c in never_flipped if c != chosen])
                        gs["last_detect_fail_id"] = None; return chosen
                    previously_flipped = available
                    if previously_flipped:
                        chosen = random.choice(previously_flipped)
                        if chosen == gs.get("last_detect_fail_id") and len(previously_flipped) > 1:
                            chosen = random.choice([c for c in previously_flipped if c != chosen])
                        gs["last_detect_fail_id"] = None; return chosen
                    return None

                def find_pair() -> Optional[Tuple[int, int]]:
                    for obj, ids in objects_found.items():
                        if obj in [None, "DETECT_FAIL", DETECTION_PERMANENT_FAIL_STATE]: continue
                        valid = [i for i in ids if card_states.get(i,{}).get("isFlippedBefore") and \
                                 not card_states.get(i,{}).get("isMatched") and i not in current_flipped]
                        if len(valid) >= 2:
                            logging.info(f"[{gs_key}] Found known pair for '{obj}': {valid[0]},{valid[1]}")
                            return valid[0], valid[1]
                    return None

                def find_match(card_id_to_match: int) -> Optional[int]:
                    obj = card_states.get(card_id_to_match,{}).get("object")
                    if obj in [None, "DETECT_FAIL", DETECTION_PERMANENT_FAIL_STATE]: return None
                    for other_id in objects_found.get(obj, []):
                        if other_id != card_id_to_match and \
                                card_states.get(other_id,{}).get("isFlippedBefore") and \
                                not card_states.get(other_id,{}).get("isMatched") and \
                                other_id not in current_flipped:
                            logging.info(f"[{gs_key}] Found match for {card_id_to_match} ('{obj}'): {other_id}")
                            return other_id
                    return None

                # === STATE 0: No cards flipped ===
                if len(current_flipped) == 0:
                    known_pair = find_pair()
                    if known_pair:
                        card1_id, card2_id = known_pair; obj = card_states[card1_id]['object']
                        logging.info(f"[{gs_key}] State 0 -> Strategy: Remove known pair {card1_id}&{card2_id} ('{obj}')")
                        await update_frontend_state(f"Found pair: {obj}. Removing {card1_id}&{card2_id}.")
                        success1 = await self._from_to("card", "trash", card1_id)
                        if not success1: logging.error(f"[{gs_key}] Move fail {card1_id}"); await update_frontend_state(f"Arm fail {card1_id}."); continue
                        success2 = await self._from_to("card", "trash", card2_id)
                        if not success2:
                            logging.error(f"[{gs_key}] Move fail {card2_id}. {card1_id} gone!"); card_states[card1_id]["isMatched"] = True
                            await update_frontend_state(f"Arm fail {card2_id}."); continue
                        card_states[card1_id]["isMatched"]=True; card_states[card2_id]["isMatched"]=True
                        gs["pairs_found"] = pairs_found + 1; logging.info(f"Pairs found: {gs['pairs_found']}")
                        await update_frontend_state(); continue
                    else:
                        card_to_flip = choose_random_card()
                        if card_to_flip is not None:
                            logging.info(f"[{gs_key}] State 0 -> Strategy: Flip card {card_to_flip}.")
                            await update_frontend_state(f"Choosing card {card_to_flip}. Detecting...")
                            detected_obj = await self._detect_object_at_card(card_to_flip) # Use self method
                            if detected_obj == DETECTION_PERMANENT_FAIL_STATE:
                                logging.error(f"[{gs_key}] PERMANENT detection failure for card {card_to_flip}.")
                                await update_frontend_state(f"Critical detection fail on {card_to_flip}. Skipping card.")
                                card_states[card_to_flip]["isFlippedBefore"]=True; card_states[card_to_flip]["object"]=DETECTION_PERMANENT_FAIL_STATE; card_states[card_to_flip]["isMatched"]=True
                                gs["last_detect_fail_id"] = card_to_flip; await update_frontend_state()
                            elif detected_obj is not None:
                                logging.info(f"Detected '{detected_obj}'. Moving card {card_to_flip} to Temp1.")
                                await update_frontend_state(f"Detected '{detected_obj}'. Moving card {card_to_flip} to Temp1.")
                                success_move = await self._from_to("card", "temp1", card_to_flip) # Use self method
                                if success_move:
                                    card_states[card_to_flip]["object"]=detected_obj; card_states[card_to_flip]["isFlippedBefore"]=True
                                    if detected_obj not in objects_found: objects_found[detected_obj] = []
                                    if card_to_flip not in objects_found[detected_obj]: objects_found[detected_obj].append(card_to_flip)
                                    current_flipped.append(card_to_flip); logging.info(f"Card {card_to_flip} ('{detected_obj}') in Temp1. Flipped: {current_flipped}")
                                else: logging.error(f"[{gs_key}] Arm fail: {card_to_flip} to Temp1."); await update_frontend_state(f"Arm fail move {card_to_flip}.")
                                await update_frontend_state()
                            else: # Should not happen if model loaded, detect returns string or PERMA_FAIL
                                logging.error(f"[{gs_key}] detect_object_at_card returned unexpected None for card {card_to_flip}.")
                                await update_frontend_state(f"Internal error during detection for {card_to_flip}."); gs["last_detect_fail_id"] = card_to_flip
                        else: logging.info(f"[{gs_key}] State 0: No known pair & no available card to flip. Waiting."); await asyncio.sleep(1)

                # === STATE 1: One card flipped ===
                elif len(current_flipped) == 1:
                    first_card_id = current_flipped[0]; first_object = card_states.get(first_card_id,{}).get("object", "UNKNOWN")
                    logging.info(f"[{gs_key}] State 1: Card {first_card_id} ('{first_object}') in Temp1.")
                    if first_object in ["UNKNOWN", DETECTION_PERMANENT_FAIL_STATE]:
                        logging.warning(f"First card {first_card_id} has invalid state '{first_object}'. Returning."); await update_frontend_state(f"Problem with card {first_card_id}. Returning.")
                        await self._from_to("temp1", "card", first_card_id); current_flipped.clear(); await update_frontend_state(); continue
                    match_id = find_match(first_card_id)
                    if match_id is not None:
                        logging.info(f"[{gs_key}] State 1 -> Strategy: Found known match {match_id}. Removing pair.")
                        await update_frontend_state(f"Found match for '{first_object}': {match_id}. Removing.")
                        success1 = await self._from_to("temp1", "trash", first_card_id)
                        if not success1: 
                            logging.error(f"Arm fail temp1->trash {first_card_id}. Return."); await update_frontend_state(f"Arm fail {first_card_id}. Return.")
                            await self._from_to("temp1", "card", first_card_id); current_flipped.clear(); await update_frontend_state(); continue
                        success2 = await self._from_to("card", "trash", match_id)
                        if not success2:
                            logging.error(f"Arm fail card->trash {match_id}. {first_card_id} gone!"); card_states[first_card_id]["isMatched"]=True
                            await update_frontend_state(f"Arm fail {match_id}."); current_flipped.clear(); await update_frontend_state(); continue
                        card_states[first_card_id]["isMatched"]=True; card_states[match_id]["isMatched"]=True
                        card_states[match_id]["isFlippedBefore"]=True; card_states[match_id]["object"]=first_object
                        gs["pairs_found"] = pairs_found + 1; current_flipped.clear(); logging.info(f"Pairs found: {gs['pairs_found']}")
                        await update_frontend_state(); continue
                    else:
                        card_to_flip = choose_random_card()
                        if card_to_flip is not None:
                            logging.info(f"[{gs_key}] State 1 -> Strategy: No known match. Flipping {card_to_flip}.")
                            await update_frontend_state(f"No match for '{first_object}'. Choosing {card_to_flip}...")
                            detected_obj = await self._detect_object_at_card(card_to_flip)
                            if detected_obj == DETECTION_PERMANENT_FAIL_STATE:
                                logging.error(f"[{gs_key}] PERMANENT detection fail on second card {card_to_flip}."); await update_frontend_state(f"Critical detect fail {card_to_flip}. Returning first card.")
                                card_states[card_to_flip]["isFlippedBefore"]=True; card_states[card_to_flip]["object"]=DETECTION_PERMANENT_FAIL_STATE; card_states[card_to_flip]["isMatched"]=True
                                gs["last_detect_fail_id"] = card_to_flip; await self._from_to("temp1", "card", first_card_id); current_flipped.clear()
                            elif detected_obj is not None:
                                logging.info(f"Detected '{detected_obj}'. Moving {card_to_flip} to Temp2.")
                                await update_frontend_state(f"Detected '{detected_obj}'. Moving {card_to_flip} to Temp2.")
                                success_move = await self._from_to("card", "temp2", card_to_flip)
                                if success_move:
                                    card_states[card_to_flip]["object"]=detected_obj; card_states[card_to_flip]["isFlippedBefore"]=True
                                    if detected_obj not in objects_found: objects_found[detected_obj] = []
                                    if card_to_flip not in objects_found[detected_obj]: objects_found[detected_obj].append(card_to_flip)
                                    current_flipped.append(card_to_flip); logging.info(f"Card {card_to_flip} ('{detected_obj}') in Temp2. Flipped: {current_flipped}")
                                else: 
                                    logging.error(f"Arm fail card->temp2 {card_to_flip}. Return first."); await update_frontend_state(f"Arm fail {card_to_flip}. Return first.")
                                    await self._from_to("temp1", "card", first_card_id); current_flipped.clear()
                            else: 
                                logging.error(f"Detect returned None for {card_to_flip}. Return first."); await update_frontend_state(f"Internal error detect {card_to_flip}. Return first.")
                                await self._from_to("temp1", "card", first_card_id); current_flipped.clear(); gs["last_detect_fail_id"] = card_to_flip
                            await update_frontend_state()
                        else: 
                            logging.warning(f"State 1: No second card? Return {first_card_id}."); await update_frontend_state(f"No second card. Return {first_card_id}.")
                            await self._from_to("temp1", "card", first_card_id); current_flipped.clear(); await update_frontend_state()

                # === STATE 2: Two cards flipped ===
                elif len(current_flipped) == 2:
                    card1_id, card2_id = current_flipped[0], current_flipped[1]
                    obj1 = card_states.get(card1_id,{}).get("object"); obj2 = card_states.get(card2_id,{}).get("object")
                    logging.info(f"[{gs_key}] State 2: Cards {card1_id} ('{obj1}') & {card2_id} ('{obj2}') in Temp1/2. Checking for match.")
                    is_match = (obj1 is not None and obj1 != DETECTION_PERMANENT_FAIL_STATE and \
                                obj2 is not None and obj2 != DETECTION_PERMANENT_FAIL_STATE and obj1 == obj2)
                    if is_match:
                        logging.info(f"MATCH FOUND: {card1_id}&{card2_id} ('{obj1}'). Removing from Temp1/2."); await update_frontend_state(f"Match: {obj1}! Removing {card1_id}&{card2_id}.")
                        success1 = await self._from_to("temp1", "trash", card1_id)
                        if not success1:
                            logging.error(f"Arm fail temp1->trash {card1_id}. Return both."); await update_frontend_state(f"Arm fail {card1_id}. Return both.")
                            await self._from_to("temp1", "card", card1_id); await self._from_to("temp2", "card", card2_id); current_flipped.clear(); await update_frontend_state(); continue
                        success2 = await self._from_to("temp2", "trash", card2_id)
                        if not success2:
                            logging.error(f"Arm fail temp2->trash {card2_id}. {card1_id} gone!"); card_states[card1_id]["isMatched"]=True
                            await update_frontend_state(f"Arm fail {card2_id}. Return it."); await self._from_to("temp2", "card", card2_id); current_flipped.clear(); await update_frontend_state(); continue
                        card_states[card1_id]["isMatched"]=True; card_states[card2_id]["isMatched"]=True
                        gs["pairs_found"] = pairs_found + 1; current_flipped.clear(); logging.info(f"Pairs found: {gs['pairs_found']}")
                        await update_frontend_state()
                        if self._websocket.client_state == WebSocketState.CONNECTED: await self._send_json_to_client({"type": "cards_hidden", "payload": [card1_id, card2_id]})
                    else:
                        logging.info(f"NO MATCH ('{obj1}' vs '{obj2}'). Returning {card1_id} from Temp1 & {card2_id} from Temp2."); await update_frontend_state(f"No match. Returning {card1_id}&{card2_id}.")
                        success_ret1 = await self._from_to("temp1", "card", card1_id)
                        success_ret2 = await self._from_to("temp2", "card", card2_id)
                        if not (success_ret1 and success_ret2): logging.error(f"Failed return {card1_id} or {card2_id}."); await update_frontend_state(f"Warn: Arm fail return {card1_id}/{card2_id}.")
                        current_flipped.clear(); await update_frontend_state()
                        if self._websocket.client_state == WebSocketState.CONNECTED: await self._send_json_to_client({"type": "cards_hidden", "payload": [card1_id, card2_id]})

                # === Invalid State ===
                elif len(current_flipped) > 2:
                    logging.error(f"Invalid State: {len(current_flipped)} cards flipped: {current_flipped}. Attempting recovery."); await update_frontend_state("Error: Invalid state detected. Returning cards.")
                    if len(current_flipped)>0 and card_states.get(current_flipped[0], {}).get("object"): await self._from_to("temp1", "card", current_flipped[0])
                    if len(current_flipped)>1 and card_states.get(current_flipped[1], {}).get("object"): await self._from_to("temp2", "card", current_flipped[1])
                    current_flipped.clear(); await update_frontend_state()

                if gs.get("pairs_found", 0) >= CARD_COUNT // 2:
                    logging.info(f"[{self.game_mode.upper()}] Game Finished! All pairs found.")
                    await update_frontend_state() # Send final state
                    await self._send_json_to_client({"type": "game_over", "payload": "Congratulations! All pairs found."})
                    gs["running"] = False; await asyncio.sleep(1.0); break # Signal loop to stop

            if gs.get("pairs_found", 0) >= CARD_COUNT // 2:
                 logging.info(f"[{self.game_mode.upper()}] Game Finished successfully path.")
                 # Final messages already sent if logic is complete.

        except asyncio.CancelledError: logging.info(f"[{self.game_mode.upper()}] YOLO game logic task cancelled.")
        except Exception as e:
            logging.error(f"[{self.game_mode.upper()}] CRITICAL YOLO Loop Error: {e}", exc_info=True)
            await self._send_error_to_client(f"Game Error: {e}")
        finally:
            logging.info(f"[{self.game_mode.upper()}] YOLO game logic runner cleanup...")
            gs["running"] = False # Ensure it's set
            self._stop_event.set() # Signal all related tasks to stop as well

    async def _run_color_game_logic(self):
        gs_key = self.game_mode.upper() # For logging prefix
        logging.info(f"[{gs_key}] Starting Color Game Logic...")
        gs = self.game_state_internal

        gs.update({
            "card_states": {i: {"isFlippedBefore": False, "color": None, "isMatched": False} for i in range(CARD_COUNT)},
            "colors_found": {color_name: [] for color_name in COLOR_DEFINITIONS.keys()}, # Use color_name from keys
            "pairs_found": 0, "current_flipped_cards": [],
            "running": True, "last_detect_fail_id": None,
        })
        logging.info(f"[{gs_key}] Initialized game state.")

        await self._send_json_to_client({"type": "game_state", "payload": {
             "card_states": gs["card_states"], "pairs_found": gs["pairs_found"], "current_flipped_cards": gs["current_flipped_cards"]
        }})
        await self._send_json_to_client({"type": "message", "payload": "Color Game Started. Initializing arm..."})

        self._capture_task = asyncio.create_task(self._capture_frames_task())
        init_home_success = await self._from_to("home", "home", -1)
        if not init_home_success:
            gs["running"] = False; 
            await self._send_error_to_client("Arm init failed.")
            return
        await self._send_json_to_client({"type": "message", "payload": "Arm ready. Starting game."})

        try:
            while gs.get("pairs_found",0) < (CARD_COUNT//2) and gs.get("running",False) and not self._stop_event.is_set():
                await asyncio.sleep(FLIP_DELAY_SECONDS)
                if self._websocket.client_state != WebSocketState.CONNECTED: gs["running"]=False; break

                current_flipped = gs.get("current_flipped_cards", [])
                card_states = gs.get("card_states", {})
                colors_found = gs.get("colors_found", {}) # Ensure this is correctly populated
                pairs_found = gs.get("pairs_found", 0)
                logging.info(f"[{gs_key}] Loop Start: Flipped={current_flipped}, Pairs={pairs_found}/{CARD_COUNT // 2}")

                async def update_frontend_state(extra_message: Optional[str] = None):
                    payload = {"card_states": card_states, "pairs_found": gs.get("pairs_found",0), "current_flipped": current_flipped}
                    await self._send_json_to_client({"type": "game_state", "payload": payload})
                    if extra_message: await self._send_json_to_client({"type": "message", "payload": extra_message})

                def choose_random_card() -> Optional[int]: # Same logic as YOLO's, but checks 'color' field for PERMA_FAIL
                    available = [i for i, s in card_states.items() if not s.get("isMatched") and i not in current_flipped and s.get("color") != DETECTION_PERMANENT_FAIL_STATE]
                    if not available: return None
                    never_flipped = [i for i in available if not card_states[i].get("isFlippedBefore")]
                    if never_flipped:
                        chosen = random.choice(never_flipped)
                        if chosen == gs.get("last_detect_fail_id") and len(never_flipped) > 1: chosen = random.choice([c for c in never_flipped if c != chosen])
                        gs["last_detect_fail_id"] = None; return chosen
                    previously_flipped = available
                    if previously_flipped:
                        chosen = random.choice(previously_flipped)
                        if chosen == gs.get("last_detect_fail_id") and len(previously_flipped) > 1: chosen = random.choice([c for c in previously_flipped if c != chosen])
                        gs["last_detect_fail_id"] = None; return chosen
                    return None

                def find_pair() -> Optional[Tuple[int, int]]:
                    for color_val, ids in colors_found.items(): # Use color_val to avoid conflict
                        if color_val in [None, "DETECT_FAIL", DETECTION_PERMANENT_FAIL_STATE, "black"]: continue
                        valid = [i for i in ids if card_states.get(i,{}).get("isFlippedBefore") and \
                                 not card_states.get(i,{}).get("isMatched") and i not in current_flipped]
                        if len(valid) >= 2:
                            logging.info(f"[{gs_key}] Found known pair for '{color_val}': {valid[0]},{valid[1]}")
                            return valid[0], valid[1]
                    return None

                def find_match(card_id_to_match: int) -> Optional[int]:
                    color_val = card_states.get(card_id_to_match,{}).get("color") # Use color_val
                    if color_val in [None, "DETECT_FAIL", DETECTION_PERMANENT_FAIL_STATE, "black"]: return None
                    for other_id in colors_found.get(color_val, []):
                        if other_id != card_id_to_match and \
                                card_states.get(other_id,{}).get("isFlippedBefore") and \
                                not card_states.get(other_id,{}).get("isMatched") and \
                                other_id not in current_flipped:
                            logging.info(f"[{gs_key}] Found match for {card_id_to_match} ('{color_val}'): {other_id}")
                            return other_id
                    return None

                # === STATE 0: No cards flipped ===
                if len(current_flipped) == 0:
                    known_pair = find_pair()
                    if known_pair:
                        card1_id, card2_id = known_pair; color_val = card_states[card1_id]['color']
                        logging.info(f"[{gs_key}] State 0 -> Strategy: Remove known pair {card1_id}&{card2_id} ('{color_val}')")
                        await update_frontend_state(f"Found pair: {color_val}. Removing {card1_id}&{card2_id}.")
                        success1 = await self._from_to("card", "trash", card1_id)
                        if not success1: logging.error(f"Move fail {card1_id}"); await update_frontend_state(f"Arm fail {card1_id}."); continue
                        success2 = await self._from_to("card", "trash", card2_id)
                        if not success2:
                            logging.error(f"Move fail {card2_id}. {card1_id} gone!"); card_states[card1_id]["isMatched"] = True
                            await update_frontend_state(f"Arm fail {card2_id}."); continue
                        card_states[card1_id]["isMatched"]=True; card_states[card2_id]["isMatched"]=True
                        gs["pairs_found"] = pairs_found + 1; logging.info(f"Pairs found: {gs['pairs_found']}")
                        await update_frontend_state(); continue
                    else:
                        card_to_flip = choose_random_card()
                        if card_to_flip is not None:
                            logging.info(f"[{gs_key}] State 0 -> Strategy: Flip card {card_to_flip}.")
                            await update_frontend_state(f"Choosing card {card_to_flip}. Detecting color...")
                            detected_color_val = await self._detect_color_at_card(card_to_flip) # Use new name
                            if detected_color_val == DETECTION_PERMANENT_FAIL_STATE:
                                logging.error(f"[{gs_key}] PERMANENT color detect fail {card_to_flip}.")
                                await update_frontend_state(f"Critical color detect fail {card_to_flip}. Skipping.")
                                card_states[card_to_flip]["isFlippedBefore"]=True; card_states[card_to_flip]["color"]=DETECTION_PERMANENT_FAIL_STATE; card_states[card_to_flip]["isMatched"]=True
                                gs["last_detect_fail_id"] = card_to_flip; await update_frontend_state()
                            elif detected_color_val == "black":
                                logging.warning(f"Detected black back on {card_to_flip}. Marking, not moving.")
                                await update_frontend_state(f"Detected back of card {card_to_flip}.")
                                card_states[card_to_flip]["isFlippedBefore"] = True; card_states[card_to_flip]["color"] = "black"
                                gs["last_detect_fail_id"] = card_to_flip; await update_frontend_state()
                            elif detected_color_val is not None: # Success (found a face color)
                                logging.info(f"Detected '{detected_color_val}'. Moving {card_to_flip} to Temp1.")
                                await update_frontend_state(f"Detected '{detected_color_val}'. Moving {card_to_flip} to Temp1.")
                                success_move = await self._from_to("card", "temp1", card_to_flip)
                                if success_move:
                                    card_states[card_to_flip]["color"]=detected_color_val; card_states[card_to_flip]["isFlippedBefore"]=True
                                    if detected_color_val not in colors_found: colors_found[detected_color_val] = [] # Use new name
                                    if card_to_flip not in colors_found[detected_color_val]: colors_found[detected_color_val].append(card_to_flip)
                                    current_flipped.append(card_to_flip); logging.info(f"Card {card_to_flip} ('{detected_color_val}') in Temp1. Flipped: {current_flipped}")
                                else: logging.error(f"Arm fail card->temp1 {card_to_flip}."); await update_frontend_state(f"Arm fail {card_to_flip}.")
                                await update_frontend_state()
                            else: # Should not happen
                                logging.error(f"Detect_color returned None for {card_to_flip}."); await update_frontend_state(f"Internal error detect {card_to_flip}.")
                                gs["last_detect_fail_id"] = card_to_flip
                        else: logging.info(f"State 0: No pair & no available card."); await asyncio.sleep(1)

                # === STATE 1: One card flipped ===
                elif len(current_flipped) == 1:
                    first_card_id = current_flipped[0]; first_color = card_states.get(first_card_id,{}).get("color", "UNKNOWN")
                    logging.info(f"[{gs_key}] State 1: Card {first_card_id} ('{first_color}') in Temp1.")
                    if first_color in ["UNKNOWN",DETECTION_PERMANENT_FAIL_STATE,"black"]:
                        logging.warning(f"First card {first_card_id} has invalid state '{first_color}'. Returning."); await update_frontend_state(f"Problem with {first_card_id}. Returning.")
                        await self._from_to("temp1","card",first_card_id);current_flipped.clear();await update_frontend_state();continue
                    match_id = find_match(first_card_id)
                    if match_id is not None:
                        logging.info(f"State 1 -> Strategy: Found known match {match_id}. Removing pair."); await update_frontend_state(f"Found match for '{first_color}': {match_id}. Removing.")
                        success1 = await self._from_to("temp1","trash",first_card_id)
                        if not success1: logging.error(f"Arm fail temp1->trash {first_card_id}."); await update_frontend_state(f"Arm fail {first_card_id}. Return.");await self._from_to("temp1","card",first_card_id);current_flipped.clear();await update_frontend_state();continue
                        success2 = await self._from_to("card","trash",match_id)
                        if not success2: logging.error(f"Arm fail card->trash {match_id}. {first_card_id} gone!"); card_states[first_card_id]["isMatched"] = True;await update_frontend_state(f"Arm fail {match_id}.");current_flipped.clear();await update_frontend_state();continue
                        card_states[first_card_id]["isMatched"]=True;card_states[match_id]["isMatched"]=True;card_states[match_id]["isFlippedBefore"]=True;card_states[match_id]["color"]=first_color
                        gs["pairs_found"]=pairs_found+1;current_flipped.clear();logging.info(f"Pairs found: {gs['pairs_found']}")
                        await update_frontend_state();continue
                    else:
                        card_to_flip = choose_random_card()
                        if card_to_flip is not None:
                            logging.info(f"State 1 -> Strategy: No known match. Flipping {card_to_flip}."); await update_frontend_state(f"No match for '{first_color}'. Choosing {card_to_flip}...")
                            detected_color_val = await self._detect_color_at_card(card_to_flip)
                            if detected_color_val == DETECTION_PERMANENT_FAIL_STATE:
                                logging.error(f"PERMANENT color detect fail {card_to_flip}. Return first."); await update_frontend_state(f"Critical detect fail {card_to_flip}. Return first.")
                                card_states[card_to_flip]["isFlippedBefore"]=True;card_states[card_to_flip]["color"]=DETECTION_PERMANENT_FAIL_STATE;card_states[card_to_flip]["isMatched"]=True
                                gs["last_detect_fail_id"] = card_to_flip; await self._from_to("temp1","card",first_card_id);current_flipped.clear()
                            elif detected_color_val == "black":
                                logging.warning(f"Detected black back on {card_to_flip}. Return first."); await update_frontend_state(f"Detected back of {card_to_flip}. Return first.")
                                card_states[card_to_flip]["isFlippedBefore"] = True; card_states[card_to_flip]["color"] = "black"
                                gs["last_detect_fail_id"] = card_to_flip; await self._from_to("temp1","card",first_card_id);current_flipped.clear()
                            elif detected_color_val is not None: # Success (found face color)
                                logging.info(f"Detected '{detected_color_val}'. Moving {card_to_flip} to Temp2."); await update_frontend_state(f"Detected '{detected_color_val}'. Moving {card_to_flip} to Temp2.")
                                success_move = await self._from_to("card","temp2",card_to_flip)
                                if success_move:
                                    card_states[card_to_flip]["color"]=detected_color_val;card_states[card_to_flip]["isFlippedBefore"]=True
                                    if detected_color_val not in colors_found: colors_found[detected_color_val]=[]
                                    if card_to_flip not in colors_found[detected_color_val]: colors_found[detected_color_val].append(card_to_flip)
                                    current_flipped.append(card_to_flip);logging.info(f"Card {card_to_flip} ('{detected_color_val}') in Temp2. Flipped: {current_flipped}")
                                else: 
                                    logging.error(f"Arm fail card->temp2 {card_to_flip}. Return first."); await update_frontend_state(f"Arm fail {card_to_flip}. Return first.")
                                    await self._from_to("temp1","card",first_card_id);current_flipped.clear()
                            else: # Should not happen
                                logging.error(f"Detect_color returned None for {card_to_flip}. Return first."); await update_frontend_state(f"Internal error detect {card_to_flip}. Return first.")
                                await self._from_to("temp1","card",first_card_id);current_flipped.clear(); gs["last_detect_fail_id"] = card_to_flip
                            await update_frontend_state()
                        else: 
                            logging.warning(f"State 1: No second card available? Return {first_card_id}."); await update_frontend_state(f"No second card. Return {first_card_id}."); await self._from_to("temp1","card",first_card_id);current_flipped.clear();await update_frontend_state()

                # === STATE 2: Two cards flipped ===
                elif len(current_flipped) == 2:
                    card1_id, card2_id = current_flipped[0], current_flipped[1]
                    color1 = card_states.get(card1_id,{}).get("color"); color2 = card_states.get(card2_id,{}).get("color")
                    logging.info(f"[{gs_key}] State 2: Cards {card1_id} ('{color1}') & {card2_id} ('{color2}') in Temp1/2. Checking.")
                    is_match = (color1 is not None and color1 not in ["UNKNOWN", DETECTION_PERMANENT_FAIL_STATE, "black"] and \
                                color2 is not None and color2 not in ["UNKNOWN", DETECTION_PERMANENT_FAIL_STATE, "black"] and color1 == color2)
                    if is_match:
                        logging.info(f"MATCH FOUND (Color): {card1_id}&{card2_id} ('{color1}'). Removing from Temp."); await update_frontend_state(f"Match: {color1}! Removing {card1_id}&{card2_id}.")
                        success1 = await self._from_to("temp1", "trash", card1_id)
                        if not success1: logging.error(f"Arm fail temp1->trash {card1_id}. Return both."); await update_frontend_state(f"Arm fail {card1_id}. Return both."); await self._from_to("temp1", "card", card1_id); await self._from_to("temp2", "card", card2_id); current_flipped.clear(); await update_frontend_state(); continue
                        success2 = await self._from_to("temp2", "trash", card2_id)
                        if not success2: logging.error(f"Arm fail temp2->trash {card2_id}. {card1_id} gone!"); card_states[card1_id]["isMatched"]=True; await update_frontend_state(f"Arm fail {card2_id}. Return it."); await self._from_to("temp2", "card", card2_id); current_flipped.clear(); await update_frontend_state(); continue
                        card_states[card1_id]["isMatched"]=True; card_states[card2_id]["isMatched"]=True
                        gs["pairs_found"] = pairs_found + 1; current_flipped.clear(); logging.info(f"Pairs found: {gs['pairs_found']}")
                        await update_frontend_state()
                        if self._websocket.client_state == WebSocketState.CONNECTED: await self._send_json_to_client({"type": "cards_hidden", "payload": [card1_id, card2_id]})
                    else:
                        logging.info(f"NO MATCH (Color) ('{color1}' vs '{color2}'). Returning {card1_id}&{card2_id}."); await update_frontend_state(f"No match. Returning {card1_id}&{card2_id}.")
                        success_ret1 = await self._from_to("temp1", "card", card1_id)
                        success_ret2 = await self._from_to("temp2", "card", card2_id)
                        if not (success_ret1 and success_ret2): logging.error(f"Failed return {card1_id} or {card2_id}."); await update_frontend_state(f"Warn: Arm fail return {card1_id}/{card2_id}.")
                        current_flipped.clear(); await update_frontend_state()
                        if self._websocket.client_state == WebSocketState.CONNECTED: await self._send_json_to_client({"type": "cards_hidden", "payload": [card1_id, card2_id]})

                # === Invalid State ===
                elif len(current_flipped) > 2:
                    logging.error(f"Invalid State: {len(current_flipped)} flipped: {current_flipped}. Recovering."); await update_frontend_state("Error: Invalid state. Returning cards.")
                    if len(current_flipped)>0 and card_states.get(current_flipped[0],{}).get("color"): await self._from_to("temp1","card",current_flipped[0])
                    if len(current_flipped)>1 and card_states.get(current_flipped[1],{}).get("color"): await self._from_to("temp2","card",current_flipped[1])
                    current_flipped.clear();await update_frontend_state()

                if gs.get("pairs_found",0) >= CARD_COUNT//2:
                    logging.info(f"[{self.game_mode.upper()}] Game Finished! All pairs found.")
                    await update_frontend_state()
                    await self._send_json_to_client({"type": "game_over", "payload": "Congratulations! All pairs found."})
                    gs["running"]=False;await asyncio.sleep(1.0);break

            if gs.get("pairs_found", 0) >= CARD_COUNT // 2:
                 logging.info(f"[{self.game_mode.upper()}] Game Finished successfully path.")
                 # Final messages already sent.

        except asyncio.CancelledError: logging.info(f"[{self.game_mode.upper()}] Color game logic task cancelled.")
        except Exception as e:
            logging.error(f"[{self.game_mode.upper()}] CRITICAL Color Loop Error: {e}", exc_info=True)
            await self._send_error_to_client(f"Game Error: {e}")
        finally:
            logging.info(f"[{self.game_mode.upper()}] Color game logic runner cleanup...")
            gs["running"] = False
            self._stop_event.set()

    async def cleanup(self):
        logging.info(f"MemoryMatching ({self.game_mode}): Initiating cleanup...")
        self._stop_event.set() # Signal all tasks to stop

        if self.game_state_internal: # Mark game as not running
            self.game_state_internal["running"] = False

        tasks_to_await = []
        if self._capture_task and not self._capture_task.done():
            logging.info(f"MemoryMatching ({self.game_mode}): Cancelling capture task...")
            self._capture_task.cancel()
            tasks_to_await.append(self._capture_task)

        if self._game_runner_task and not self._game_runner_task.done():
            logging.info(f"MemoryMatching ({self.game_mode}): Cancelling game runner task...")
            self._game_runner_task.cancel()
            tasks_to_await.append(self._game_runner_task)

        if tasks_to_await:
            try:
                # Wait for tasks to complete cancellation/cleanup
                await asyncio.gather(*tasks_to_await, return_exceptions=True)
                logging.info(f"MemoryMatching ({self.game_mode}): All tasks awaited in cleanup.")
            except asyncio.CancelledError:
                 logging.info(f"MemoryMatching ({self.game_mode}): Tasks confirmed cancelled during gather.")
            except Exception as e:
                 logging.error(f"MemoryMatching ({self.game_mode}): Error awaiting tasks in cleanup: {e}")

        # Ensure arm is sent home if it was used and serial is available
        if self.serial_instance and self.serial_instance.is_open:
            logging.info(f"MemoryMatching ({self.game_mode}): Ensuring arm home post-game.")
            try:
                # Run sync command in thread to avoid blocking cleanup if it's truly synchronous
                await asyncio.to_thread(self._from_to_sync, "home", "home", -1)
            except Exception as e:
                logging.error(f"MemoryMatching ({self.game_mode}): Error sending arm home during cleanup: {e}")

        # Do not close self.serial_instance, it's managed by main.py
        # Do not close self._websocket, it's managed by main.py's endpoint wrapper

        logging.info(f"MemoryMatching ({self.game_mode}): Cleanup complete.")

# Removed: All FastAPI application code, old WebSocket endpoint, old GameSession adapter,
# and global state variables like ser, yolo_model_global, active_games, frame_lock (now instance var), etc.
