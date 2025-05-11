import cv2
import numpy as np
from ultralytics import YOLO
import time
import math
import base64
import os
import threading
import asyncio
import json
from starlette.websockets import WebSocketState, WebSocketDisconnect
import queue

# --- Default Constants ---
CONF_THRESHOLD_DEFAULT = 0.7
FOCAL_LENGTH_DEFAULT = 580
BALLOON_WIDTH_DEFAULT = 0.18
TARGET_COLOR_DEFAULT = "red"
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
X_CAMERA_FOV = 86
Y_CAMERA_FOV = 53
LASER_OFFSET_CM_X_DEFAULT = 4
LASER_OFFSET_CM_Y_DEFAULT = 18
KP_X_DEFAULT = 0.05
KP_Y_DEFAULT = 0.05
CENTER_TOLERANCE_DEFAULT = 10
MAX_ANGLE_CHANGE_DEFAULT = 5
INIT_PAN = 90
INIT_TILT = 90
MOVEMENT_COOLDOWN = 0.2
NO_BALLOON_TIMEOUT = 10
RETURN_TO_CENTER_DELAY = 1.0
CAPTURE_FPS_TARGET = 30

# Global model path
YOLO_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "games/TargetDetection/runs/detect/train/weights/best.pt"
)

class VideoStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else (False, None)

    def release(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()

class GameSession:
    def __init__(self, config, esp32_client=None):
        self.esp32_client = esp32_client
        self.webcam_ip = "http://192.168.86.206:4747/video"

        # Use global YOLO_MODEL_PATH, ignore config for model path
        model_path_abs = YOLO_MODEL_PATH
        if not os.path.exists(model_path_abs):
            print(f"Warning: Model not found at {model_path_abs}. Using fallback relative path.")
            model_path_abs = "games/TargetDetection/runs/detect/train/weights/best.pt"
        try:
            self.model = YOLO(model_path_abs)
            print(f"Loaded YOLO model from: {model_path_abs}")
        except Exception as e:
            print(f"Error loading YOLO model from {model_path_abs}: {e}")
            self.model = None

        self.focal_length = float(config.get("focal_length", FOCAL_LENGTH_DEFAULT))
        self.laser_offset_cm_x = float(config.get("laser_offset_cm_x", LASER_OFFSET_CM_X_DEFAULT))
        self.laser_offset_cm_y = float(config.get("laser_offset_cm_y", LASER_OFFSET_CM_Y_DEFAULT))
        self.target_color = config.get("target_color", TARGET_COLOR_DEFAULT)
        self.kp_x = float(config.get("kp_x", KP_X_DEFAULT))
        self.kp_y = float(config.get("kp_y", KP_Y_DEFAULT))
        self.conf_threshold = float(config.get("conf_threshold", CONF_THRESHOLD_DEFAULT))
        self.center_tolerance = float(config.get("center_tolerance", CENTER_TOLERANCE_DEFAULT))
        self.max_angle_change = float(config.get("max_angle_change", MAX_ANGLE_CHANGE_DEFAULT))
        self.balloon_width = float(config.get("balloon_width", BALLOON_WIDTH_DEFAULT))
        self.no_balloon_timeout_setting = float(config.get("no_balloon_timeout", NO_BALLOON_TIMEOUT))

        self.current_pan = INIT_PAN
        self.current_tilt = INIT_TILT
        self.depth = 150.0
        self.shot_angles = []
        self.last_movement_time = 0
        self.last_balloon_detected_time = time.time()
        self.game_over_due_to_timeout = False
        self.game_requested_stop = False

        self.shot_balloons = []
        self.current_target_initial_coords = None
        self.is_at_initial_position = True

        self.websocket = None
        self.capture_thread = None
        self.stop_event = threading.Event()
        self._async_loop = None
        self.frame_queue = queue.Queue(maxsize=100)
        self.video_stream = None

    async def _send_switch_command(self):
        if self.esp32_client is None:
            print("[ESP32] No ESP32 client available, skipping switch command")
            return False
        try:
            print("[ESP32] Sending switch command to activate SHOOTER mode")
            await self.esp32_client.send_json({
                "action": "switch",
                "game": "SHOOTER"
            })
            return True
        except Exception as e:
            print(f"[ESP32] Error sending switch command: {e}")
            return False

    async def manage_game_loop(self, websocket):
        self.websocket = websocket
        # Store the event loop from the main thread so we can use it from background threads
        self._async_loop = asyncio.get_running_loop()
        print("TargetShooter: Stored main event loop for cross-thread communication")

        print("TargetShooter: Starting game loop.")
        if self.esp32_client is not None:
            await self._send_switch_command()

        self.stop_event.clear()
        self.game_requested_stop = False
        self.game_over_due_to_timeout = False
        self.last_balloon_detected_time = time.time()
        self.shot_angles = []

        self.capture_thread = threading.Thread(target=self._capture_and_process_task, daemon=True)
        self.capture_thread.start()
        print("TargetShooter: Capture and processing thread started.")

        try:
            while not self.stop_event.is_set():
                if self.websocket.client_state != WebSocketState.CONNECTED:
                    print("TargetShooter: WebSocket disconnected by client.")
                    break
                
                message_text = await self.websocket.receive_text()
                try:
                    command_data = json.loads(message_text)
                    response = self.process_command(command_data)
                    if self.websocket and self.websocket.client_state == WebSocketState.CONNECTED:
                        await self.websocket.send_json(response)
                    
                    if self.game_requested_stop:
                        print("TargetShooter: Game stop requested by command.")
                        break 
                except json.JSONDecodeError:
                    print(f"TargetShooter: Received non-JSON command: {message_text}")
                    if self.websocket and self.websocket.client_state == WebSocketState.CONNECTED:
                        await self.websocket.send_json({"status": "error", "message": "Invalid command format, expected JSON."})
                except Exception as cmd_e:
                    print(f"TargetShooter: Error processing command: {cmd_e}")
                    if self.websocket and self.websocket.client_state == WebSocketState.CONNECTED:
                        await self.websocket.send_json({"status": "error", "message": f"Error processing command: {str(cmd_e)}"})

        except WebSocketDisconnect:
            print("TargetShooter: WebSocket disconnected in manage_game_loop.")
        except Exception as e:
            print(f"TargetShooter: Error in manage_game_loop: {e}")
        finally:
            print("TargetShooter: manage_game_loop ending. Stopping resources.")
            self.stop()

    def _capture_and_process_task(self):
        print(f"TargetShooter: Attempting to open camera: {self.webcam_ip}")
        self.video_stream = VideoStream(self.webcam_ip)
        time.sleep(1)

        ret, frame = self.video_stream.read()
        if not ret or frame is None:
            print(f"TargetShooter: CRITICAL - Failed to open camera stream at {self.webcam_ip}")
            error_msg = {"status": "error", "message": f"Failed to open camera: {self.webcam_ip}"}
            if self.websocket and self._async_loop:
                asyncio.run_coroutine_threadsafe(self.websocket.send_json(error_msg), self._async_loop)
            self.stop_event.set()
            return

        print(f"TargetShooter: Camera {self.webcam_ip} opened successfully.")
        frame_delay = 1.0 / 30

        while not self.stop_event.is_set():
            loop_start_time = time.time()
            try:
                ret, frame = self.video_stream.read()
                if not ret or frame is None:
                    print("TargetShooter: Failed to grab frame, retrying...")
                    time.sleep(1)
                    continue

                response_data = self._process_single_frame_logic(frame)

                if response_data.get("processed_frame"):
                    try:
                        if self.frame_queue.full():
                            try:
                                self.frame_queue.get_nowait()
                            except queue.Empty:
                                pass
                        self.frame_queue.put_nowait(response_data["processed_frame"])
                    except Exception as qerr:
                        print(f"Frame queue error: {qerr}")

                if self.websocket and self.websocket.client_state == WebSocketState.CONNECTED and self._async_loop:
                    asyncio.run_coroutine_threadsafe(self.websocket.send_json(response_data), self._async_loop)
                
                if response_data.get("status") == "ended" or self.game_over_due_to_timeout or self.game_requested_stop:
                    print("TargetShooter: Game ended based on frame processing. Stopping capture task.")
                    self.stop_event.set()
                    break

                elapsed_time = time.time() - loop_start_time

            except Exception as e:
                print(f"TargetShooter: Error in capture/process task: {e}")
                time.sleep(1)

        self.video_stream.release()
        print("TargetShooter: Capture and processing thread finished.")
        self.stop_event.set()

    def get_stream_generator(self):
        """
        Yields multipart JPEG frames for HTTP streaming with corrected formatting
        using pre-defined byte strings for robustness.
        Ensures cv2.imencode results are converted to bytes.
        """
        boundary_str = "frame"
        boundary = b"--" + boundary_str.encode()
        boundary_end = b"--" + boundary_str.encode() + b"--\r\n"
        
        content_type_header = b"Content-Type: image/jpeg\r\n"
        crlf = b"\r\n"

        print("Stream generator started")
        frame_count = 0
        
        # Initial greeting frame
        img = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        cv2.putText(img, "Target Shooter Stream Starting...", (50, IMAGE_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        ret_val, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        
        jpg_bytes_initial = b''
        if ret_val:
            jpg_bytes_initial = buffer.tobytes()
        else:
            print("Stream generator: Failed to encode initial frame.")
        
        # Send initial frame
        yield boundary + crlf
        yield content_type_header
        yield f"Content-Length: {len(jpg_bytes_initial)}\r\n".encode() 
        yield crlf
        yield jpg_bytes_initial
        yield crlf
        
        while not self.stop_event.is_set():
            try:
                b64_frame = self.frame_queue.get(timeout=0.5) 
                
                if not b64_frame:
                    print("Stream generator: got None from queue, skipping.")
                    continue
                    
                jpg_bytes = base64.b64decode(b64_frame)
                
                yield boundary + crlf
                yield content_type_header
                yield f"Content-Length: {len(jpg_bytes)}\r\n".encode()
                yield crlf
                yield jpg_bytes
                yield crlf
                
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"Streamed {frame_count} frames")
                    
            except queue.Empty:
                img_placeholder = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
                cv2.putText(img_placeholder, "Waiting for frames...", (50, IMAGE_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
                ret_val_ph, buffer_ph = cv2.imencode('.jpg', img_placeholder, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                
                jpg_bytes_placeholder = b''
                if ret_val_ph:
                    jpg_bytes_placeholder = buffer_ph.tobytes()
                else:
                    print("Stream generator: Failed to encode placeholder frame.")

                yield boundary + crlf
                yield content_type_header
                yield f"Content-Length: {len(jpg_bytes_placeholder)}\r\n".encode()
                yield crlf
                yield jpg_bytes_placeholder
                yield crlf
            except base64.binascii.Error as b64_error:
                print(f"Stream generator: Base64 decode error: {b64_error}. Skipping frame.")
            except Exception as e:
                print(f"Stream generator error: {e}")
                if self.stop_event.is_set():
                    break
                
        yield boundary_end
        print("Stream generator exiting.")

    def send_to_arduino(self, pan_angle, tilt_angle, shoot_command=False):
        if self.esp32_client is not None:
            try:
                if shoot_command:
                    command = "SHOOT"
                else:
                    command = f"{int(round(pan_angle))},{int(round(tilt_angle))}"
                print(f"[ESP32] Sending command: {command}")
                
                if self._async_loop:
                    asyncio.run_coroutine_threadsafe(
                        self.esp32_client.send_json({
                            "action": "command",
                            "command": command
                        }),
                        self._async_loop
                    )
                    time.sleep(0.05)
                    return True
                else:
                    print("[ESP32] No event loop available, cannot send command")
                    return False
            except Exception as e:
                print(f"[ESP32] Error sending command: {e}")
                return False
        print("[ESP32] No ESP32 client available, skipping command")
        return False

    def wait_for_ack(self, timeout=40):
        if self.esp32_client is not None:
            print("[ESP32] Skipping wait_for_ack (not implemented for ESP32)")
            return True
        return False

    def _calculate_error(self, target_x, target_y):
        center_x = IMAGE_WIDTH / 2
        center_y = IMAGE_HEIGHT / 2
        error_x = target_x - center_x
        error_y = target_y - center_y
        return error_x, error_y

    def _calculate_new_angles(self, error_x, error_y):
        pan_adjustment = -self.kp_x * error_x
        tilt_adjustment = self.kp_y * error_y
        
        pan_adjustment = max(-self.max_angle_change, min(self.max_angle_change, pan_adjustment))
        tilt_adjustment = max(-self.max_angle_change, min(self.max_angle_change, tilt_adjustment))
        
        new_pan = self.current_pan + pan_adjustment
        new_tilt = self.current_tilt + tilt_adjustment
        
        new_pan = max(0, min(180, new_pan))
        new_tilt = max(0, min(180, new_tilt))
        return new_pan, new_tilt

    def _get_color_name(self, bgr):
        r, g, b = bgr[2], bgr[1], bgr[0]
        if r > 150 and g > 70 and g<150 and b > 70 and b<150:
         return "red"
        elif r < 140 and g > 150 and b < 170:
            return "green"
        elif r < 175 and g < 175 and b > 150:
            return "blue"
        elif r > 160 and g > 140 and  b < 100:
            return "yellow"
        elif r < 50 and g < 50 and b < 50:
            return "black"
        elif r > 200 and g > 200 and b > 200:
            return "white"
        else:
            return f"rgb({int(r)}, {int(g)}, {int(b)})"

    def _is_target_centered(self, error_x, error_y):
        return abs(error_x) < self.center_tolerance and abs(error_y) < self.center_tolerance

    def _is_angle_already_shot(self, pan, tilt, threshold=5):
        for shot_pan, shot_tilt in self.shot_angles:
            if abs(pan - shot_pan) < threshold and abs(tilt - shot_tilt) < threshold:
                return True
        return False
    
    def _is_balloon_already_shot(self, center_x, center_y, threshold=50):
        for shot_x, shot_y in self.shot_balloons:
            distance = math.sqrt((center_x - shot_x) ** 2 + (center_y - shot_y) ** 2)
            if distance < threshold:
                return True
        return False

    def _draw_crosshair(self, frame):
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        size = 20; thickness = 2; color=(0,0,255)
        cv2.line(frame, (center_x - size, center_y), (center_x + size, center_y), color, thickness)
        cv2.line(frame, (center_x, center_y - size), (center_x, center_y + size), color, thickness)

        fov_rad_x = math.radians(X_CAMERA_FOV)
        fov_rad_y = math.radians(Y_CAMERA_FOV)
        width_at_target = 2 * self.depth * math.tan(fov_rad_x / 2) if self.depth > 0 else 1
        height_at_target = 2 * self.depth * math.tan(fov_rad_y / 2) if self.depth > 0 else 1
        
        pixels_per_cm_x = IMAGE_WIDTH / width_at_target if width_at_target > 0 else 1
        pixels_per_cm_y = IMAGE_HEIGHT / height_at_target if height_at_target > 0 else 1

        shooter_x_offset = int(self.laser_offset_cm_x * pixels_per_cm_x)
        shooter_y_offset = int(self.laser_offset_cm_y * pixels_per_cm_y)
        shooter_x = center_x + shooter_x_offset
        shooter_y = center_y + shooter_y_offset

        cv2.circle(frame, (shooter_x, shooter_y), 5, (0, 255, 255), -1)
        cv2.line(frame, (shooter_x - 10, shooter_y), (shooter_x + 10, shooter_y), (0, 255, 255), 2)
        cv2.line(frame, (shooter_x, shooter_y - 10), (shooter_x, shooter_y + 10), (0, 255, 255), 2)

    def process_command(self, command_data):
        action = command_data.get("action")
        response_message = "Command received"
        
        if action == "initial_config" or action == "update_params":
            self.focal_length = float(command_data.get("focal_length", self.focal_length))
            self.laser_offset_cm_x = float(command_data.get("laser_offset_cm_x", self.laser_offset_cm_x))
            self.laser_offset_cm_y = float(command_data.get("laser_offset_cm_y", self.laser_offset_cm_y))
            self.target_color = command_data.get("target_color", self.target_color)
            self.kp_x = float(command_data.get("kp_x", self.kp_x))
            self.kp_y = float(command_data.get("kp_y", self.kp_y))
            self.no_balloon_timeout_setting = float(command_data.get("no_balloon_timeout", self.no_balloon_timeout_setting))
            response_message = f"Parameters updated. FL:{self.focal_length}, OffsetX:{self.laser_offset_cm_x}, OffsetY:{self.laser_offset_cm_y}, Color:{self.target_color}, Timeout: {self.no_balloon_timeout_setting}"
            
            if action == "initial_config":
                print("TargetShooter: Received initial_config, resetting game state for new session.")
                self.shot_angles = []
                self.game_over_due_to_timeout = False
                self.game_requested_stop = False
                self.last_balloon_detected_time = time.time()
        elif action == "reset_shot_angles":
            self.shot_angles = []
            response_message = "Shot angles reset."
        elif action == "end_game":
            self.game_requested_stop = True
            self.stop_event.set()
            response_message = "Game end requested. Stopping..."
        elif action == "emergency_stop":
            self.game_requested_stop = True
            self.stop_event.set()
            response_message = "Emergency stop: Game stopping."
        else:
            response_message = f"Unknown command: {action}"
        
        return {"status": "command_processed", "message": response_message, "game_state": self.get_current_game_state_dict()}

    def get_current_game_state_dict(self):
        return {
            "pan": self.current_pan,
            "tilt": self.current_tilt,
            "depth_cm": self.depth,
            "target_color": self.target_color,
            "shot_angles_count": len(self.shot_angles),
            "game_over_timeout": self.game_over_due_to_timeout,
            "game_requested_stop": self.game_requested_stop,
            "focal_length": self.focal_length,
            "laser_offset_cm_x": self.laser_offset_cm_x,
            "laser_offset_cm_y": self.laser_offset_cm_y,
            "kp_x": self.kp_x,
            "kp_y": self.kp_y,
            "no_balloon_timeout_setting": self.no_balloon_timeout_setting,
            "conf_threshold": self.conf_threshold,
            "center_tolerance": self.center_tolerance,
        }

    def _process_single_frame_logic(self, frame_mat):
        if self.model is None:
            return {"status": "error", "message": "YOLO Model not loaded.", "processed_frame": None, "game_state": self.get_current_game_state_dict()}
        
        if self.game_over_due_to_timeout or self.game_requested_stop or self.stop_event.is_set():
            msg = "Game ended: Timeout." if self.game_over_due_to_timeout else "Game ended: User request."
            if self.stop_event.is_set() and not (self.game_over_due_to_timeout or self.game_requested_stop):
                 msg = "Game ended: System stop."
            return {"status": "ended", "message": msg, "processed_frame": None, "game_state": self.get_current_game_state_dict()}

        frame = frame_mat
        
        if frame.shape[1] != IMAGE_WIDTH or frame.shape[0] != IMAGE_HEIGHT:
            frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))

        current_time = time.time()
        balloon_detected_this_frame = False
        status_message = "Processing..."

        results = self.model.predict(source=frame, show=False, verbose=False, conf=self.conf_threshold)
        self._draw_crosshair(frame)

        best_target = None
        target_found = False

        best_confidence = 0
        if abs(self.current_pan - INIT_PAN) < 1 and abs(self.current_tilt - INIT_TILT) < 1:
          is_at_initial_position = True
        else:
          is_at_initial_position = False

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()

            for i, box in enumerate(boxes):
                label = self.model.names[int(cls_ids[i])]
                if label.lower() != 'balloon':
                    continue

                x1, y1, x2, y2 = box.astype(int)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # if self.is_at_initial_position and self._is_balloon_already_shot(center_x, center_y):
                #     print(f"Skipping balloon at ({center_x}, {center_y}) - already shot")
                #     continue

                error_x_calc, error_y_calc = self._calculate_error(center_x, center_y)
                target_pan_for_check, target_tilt_for_check = self._calculate_new_angles(error_x_calc, error_y_calc)
                
                balloon_detected_this_frame = True

                pixel_width = x2 - x1
                estimated_depth = (self.focal_length * self.balloon_width) / pixel_width if pixel_width > 0 else 0.0
                estimated_depth *= 100
                if 30 < estimated_depth < 1000: self.depth = estimated_depth

                roi = frame[y1:y2, x1:x2]
                if roi.size == 0: continue
                mean_color_bgr = [int(round(x)) for x in cv2.mean(roi)[:3]]
                color_name = self._get_color_name(mean_color_bgr)

                if color_name != self.target_color:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
                    cv2.putText(frame, f"{label} ({color_name})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128,128,128),1)
                    continue

                info_text = f"{label} ({color_name}) {confs[i]:.2f} D:{self.depth:.1f}cm"
                
                if confs[i] > best_confidence:
                    best_confidence = confs[i]
                    best_target = {
                    'pos': (center_x, center_y),
                    'box': (x1, y1, x2, y2),
                    'info': info_text,
                    'initial_coords': (center_x, center_y) if self.is_at_initial_position else self.current_target_initial_coords
                }
                target_found = True
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, info_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        
        if balloon_detected_this_frame:
            self.last_balloon_detected_time = current_time
        elif current_time - self.last_balloon_detected_time > self.no_balloon_timeout_setting:
            status_message = f"No unshot {self.target_color} balloons for {self.no_balloon_timeout_setting}s. Game Over."
            print(status_message)
            self.game_over_due_to_timeout = True
            self.send_to_arduino(INIT_PAN, INIT_TILT)
        
        current_time = time.time()
        if target_found and (current_time - self.last_movement_time) > MOVEMENT_COOLDOWN:
            self.last_movement_time = current_time
            center_x, center_y = best_target['pos']
            if self.is_at_initial_position:
                 self.current_target_initial_coords = best_target['initial_coords']
        
            error_x, error_y = self._calculate_error(center_x, center_y)
            
            frame_center_x, frame_center_y = IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2
            cv2.line(frame, (frame_center_x, frame_center_y), (center_x, center_y), (255, 0, 0), 2)

            if self._is_target_centered(error_x, error_y):
                status_message = "TARGET LOCKED!"
                cv2.putText(frame, status_message, (frame_center_x - 80, frame_center_y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                if not self._is_angle_already_shot(self.current_pan, self.current_tilt):
                    print(f"FIRE! at angles: Pan={self.current_pan:.1f}, Tilt={self.current_tilt:.1f}, Depth: {self.depth:.1f}cm")
                    if self.send_to_arduino(self.current_pan, self.current_tilt, shoot_command=True):
                        status_message = "SHOOT command sent."
                        if self.wait_for_ack():
                            status_message = "Balloon shot! ACK received."
                            if self.current_target_initial_coords is not None:
                                self.shot_balloons.append(self.current_target_initial_coords)
                                print(f"Stored initial coords: {self.current_target_initial_coords}")
                    
                            self.current_target_initial_coords = None
                                                            
                            print("Returning to initial position...")
                            self.current_pan, self.current_tilt = INIT_PAN, INIT_TILT
                            self.send_to_arduino(self.current_pan, self.current_tilt)
                            time.sleep(RETURN_TO_CENTER_DELAY)
                            self.last_balloon_detected_time = time.time()
                        else:
                            status_message = "Shoot sent, but NO ACK. Retrying aim or hold."
                    else:
                        status_message = "Failed to send SHOOT command."
            else:
                new_pan, new_tilt = self._calculate_new_angles(error_x, error_y)
                if abs(new_pan - self.current_pan) > 0.5 or abs(new_tilt - self.current_tilt) > 0.5:
                    self.current_pan, self.current_tilt = new_pan, new_tilt
                    self.send_to_arduino(self.current_pan, self.current_tilt)
                    status_message = f"Moving: Pan={int(self.current_pan)} Tilt={int(self.current_tilt)}"
                else:
                    status_message = "Target found, holding position."
        elif best_target:
            status_message = "Target found, waiting for movement cooldown."
        elif not balloon_detected_this_frame and not self.game_over_due_to_timeout and not self.game_requested_stop:
             status_message = f"Scanning for {self.target_color} balloons..."

        cv2.putText(frame, f"Pan: {int(self.current_pan)} Tilt: {int(self.current_tilt)}", (10, IMAGE_HEIGHT - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Status: {status_message}", (10, IMAGE_HEIGHT - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Timeout in: {max(0, self.no_balloon_timeout_setting - (current_time - self.last_balloon_detected_time)):.1f}s", (10, IMAGE_HEIGHT - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,100,100),1)

        _, jpg_frame = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        b64_frame = base64.b64encode(jpg_frame).decode('utf-8')

        current_status_is_ended = self.game_over_due_to_timeout or self.game_requested_stop or self.stop_event.is_set()
        return {
            "status": "ended" if current_status_is_ended else "ok",
            "message": status_message,
            "processed_frame": b64_frame,
            "game_state": self.get_current_game_state_dict()
        }

    def stop(self):
        if not self.stop_event.is_set():
            print("TargetShooter: Initiating stop sequence...")
            self.stop_event.set()

            if self.capture_thread and self.capture_thread.is_alive():
                print("TargetShooter: Waiting for capture thread to join...")
                self.capture_thread.join(timeout=3.0)
                if self.capture_thread.is_alive():
                    print("TargetShooter: Capture thread did not join in time.")

            self.game_requested_stop = True

            self.send_to_arduino(INIT_PAN, INIT_TILT)
            print("TargetShooter: Sent home command to ESP32 on stop.")



        else:            print("TargetShooter: Stop sequence complete.")

