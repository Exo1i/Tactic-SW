# games/rubiks_cube_game.py

from typing import Optional, Dict, Any, List, Tuple # Added List, Tuple
import serial # type: ignore
import time
import cv2
import numpy as np
import kociemba # type: ignore
import json
import base64
import os
import random
from collections import Counter

class RubiksCubeGame:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Standard operational variables
        self.mode = "idle"
        self.status_message = "Ready"
        self.error_message = None
        self.stop_requested: bool = False
        self.serial_connection: Optional[serial.Serial] = None
        self.websocket: Optional[Any] = None
        self.last_motor_move_time: float = 0.0
        
        # Frame processing & detection parameters
        self.WINDOW_SIZE: tuple = self.config.get('window_size', (320, 240))
        self.current_frame_for_detection: Optional[np.ndarray] = None
        self.prev_contour_scan: Optional[np.ndarray] = None # For stability check in scanning

        # New: Zoom and Distance control parameters
        self.zoom_crop_factor: float = float(self.config.get('zoom_crop_factor', 1.0))
        self.relative_detection_distance: float = float(self.config.get('relative_detection_distance', 1.0))
        
        # Base contour area thresholds (for nominal distance)
        self.BASE_MIN_CONTOUR_AREA: int = self.config.get('base_min_contour_area', 2000) 
        self.BASE_MAX_CONTOUR_AREA: int = self.config.get('base_max_contour_area', 80000)
        
        # Effective contour area thresholds, calculated based on relative_detection_distance
        self.MIN_CONTOUR_AREA: int = 0 
        self.MAX_CONTOUR_AREA: int = 0 
        self._update_effective_contour_areas() 

        # Calibration variables
        self.calibration_step = 0
        self.last_valid_grid_info_for_calibration: Optional[tuple] = None
        self.last_processed_frame_for_calibration: Optional[np.ndarray] = None
        self.calibration_roi_scale: float = float(self.config.get('calibration_roi_scale', 0.12))
        self.COLOR_NAMES_CALIBRATION: list = ["W", "R", "G", "Y", "O", "B"] 

        # Scanning variables
        self.current_scan_idx = 0
        self.SCAN_COOLDOWN: float = float(self.config.get('scan_cooldown', 0.5))
        self.MOTOR_STABILIZATION_TIME: float = float(self.config.get('motor_stabilization_time', 1.5))
        self.STABILITY_THRESHOLD: int = self.config.get('stability_threshold', 2)
        self.stability_counter: int = 0
        self.last_scan_time: float = time.time()
        self.prev_face_colors_scan: Optional[List[str]] = None # Type hint
        self.u_scans: List[List[str]] = [[] for _ in range(12)] # Type hint
        self.rotation_sequence: list = self.config.get('rotation_sequence', [
            "", "R L'", "B F'", "R L'", "B F'", "R L'", "B F'", "R L'", "B F'", "R L'", "B F'", "R L'"
        ])

        # Solving variables
        self.solution: Optional[str] = None
        self.current_solve_move_index = 0
        self.total_solve_moves = 0
        
        # Serial communication
        self.serial_port: str = self.config.get('serial_port', 'COM9')
        self.serial_baudrate: int = self.config.get('serial_baudrate', 9600)
        self.init_serial()

        # Color detection
        self.color_ranges: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = self._load_color_ranges_from_file() # type: ignore
        if not self.color_ranges:
            print("Using default color ranges for STICKERLESS CUBE.")
            self.color_ranges = {
                "W": [(np.array([0, 0, 150]), np.array([180, 70, 255]))],
                "R": [(np.array([0, 80, 70]), np.array([10, 255, 255])),
                      (np.array([160, 80, 70]), np.array([179, 255, 255]))],
                "G": [(np.array([35, 60, 60]), np.array([85, 255, 255]))],
                "Y": [(np.array([18, 80, 100]), np.array([35, 255, 255]))],
                "O": [(np.array([3, 80, 70]), np.array([20, 255, 255]))],
                "B": [(np.array([80, 60, 60]), np.array([135, 255, 255]))]
            }
        else:
            print("Color ranges loaded from file.")
        
        self.set_zoom_crop_factor(self.zoom_crop_factor) 
        self.set_relative_detection_distance(self.relative_detection_distance)

    # --- Mode Control Methods ---
    def start_scanning_mode(self) -> bool:
        if self.mode not in ["idle", "error", "calibrating"]:
            self.error_message = f"Cannot start scanning, current mode: {self.mode}"
            self.status_message = self.error_message
            print(self.error_message)
            return False
        
        self._reset_scan_state()
        self._reset_solve_state() # A new scan invalidates any old solution
        self.mode = "scanning"
        self.status_message = "Scanning initiated. Present cube to camera."
        self.error_message = None
        print("Scanning mode started.")
        return True

    def start_calibration_mode(self) -> bool:
        if self.mode not in ["idle", "error"]:
             self.error_message = f"Cannot start calibration, current mode: {self.mode}"
             self.status_message = self.error_message
             print(self.error_message)
             return False
        
        self.mode = "calibrating"
        self.calibration_step = 0
        self.color_ranges = {} # Clear old ranges before starting new calibration
        self.status_message = f"Calibration: Aim {self.COLOR_NAMES_CALIBRATION[0]} at box."
        self.error_message = None
        self.last_valid_grid_info_for_calibration = None 
        print("Calibration mode started.")
        return True

    def _update_effective_contour_areas(self):
        if self.relative_detection_distance <= 0: 
            squared_factor = 0.01 
        else:
            squared_factor = self.relative_detection_distance ** 2
        
        self.MIN_CONTOUR_AREA = int(self.BASE_MIN_CONTOUR_AREA / squared_factor)
        self.MAX_CONTOUR_AREA = int(self.BASE_MAX_CONTOUR_AREA / squared_factor)
        
        self.MIN_CONTOUR_AREA = max(100, self.MIN_CONTOUR_AREA) 
        self.MAX_CONTOUR_AREA = max(self.MIN_CONTOUR_AREA + 100, self.MAX_CONTOUR_AREA) 
        
        total_pixels = self.WINDOW_SIZE[0] * self.WINDOW_SIZE[1]
        self.MAX_CONTOUR_AREA = min(self.MAX_CONTOUR_AREA, int(total_pixels * 0.95))

    def set_zoom_crop_factor(self, factor: float):
        original_factor = factor
        if factor < 1.0: factor = 1.0
        elif factor > 10.0: factor = 10.0 # Corrected from 1.0, means it's clamped between 1 and 10
        if original_factor != factor:
            print(f"Info: Zoom crop factor ({original_factor}) corrected to {factor}.")
            
        self.zoom_crop_factor = factor
        self.status_message = f"Zoom crop factor set to {self.zoom_crop_factor:.2f}"

    def set_relative_detection_distance(self, factor: float):
        original_factor = factor
        if factor <= 0.1: factor = 0.1
        elif factor > 5.0: factor = 5.0
        if original_factor != factor:
            print(f"Info: Relative detection distance ({original_factor}) corrected to {factor}.")

        self.relative_detection_distance = factor
        self._update_effective_contour_areas() 
        self.status_message = (f"Detection distance factor: {self.relative_detection_distance:.2f}. "
                               f"Areas: {self.MIN_CONTOUR_AREA}-{self.MAX_CONTOUR_AREA}")

    def init_serial(self) -> bool:
        try:
            if self.serial_connection and self.serial_connection.is_open:
                self.serial_connection.close()
            self.serial_connection = serial.Serial(self.serial_port, self.serial_baudrate, timeout=1)
            time.sleep(2) 
            self.status_message = f"Serial connected: {self.serial_port}"
            print(self.status_message)
            self.error_message = None 
            return True
        except serial.SerialException as e: 
            self.error_message = f"Serial Error: {str(e)}"
            self.status_message = f"Failed to connect: {self.serial_port}"
            self.serial_connection = None
            print(self.error_message)
            return False
        
    def set_websocket(self, websocket: Any): 
        self.websocket = websocket
        print("WebSocket instance set for RubiksCubeGame.")
        
    def send_arduino_command(self, cmd: str, wait_for_ack: bool = True, is_solution: bool = False) -> bool:
        if not self.serial_connection or not self.serial_connection.is_open:
            print(f"Serial not connected. Command '{cmd}' not sent (SIMULATING).")
            simulated_delay = 0
            if is_solution: simulated_delay = 3 + cmd.count(' ') * 0.1 
            elif " " in cmd or len(cmd) > 2 : simulated_delay = 0.5 + cmd.count(' ') * 0.2 
            else: simulated_delay = self.MOTOR_STABILIZATION_TIME 
            time.sleep(simulated_delay)
            self.last_motor_move_time = time.time()
            if is_solution: 
                self.mode = "idle"
                self.status_message = "Solution SIMULATED as completed."
                self.solution = None 
                self.current_solve_move_index = self.total_solve_moves if self.total_solve_moves > 0 else 0
            return True

        try:
            self.serial_connection.write(f"{cmd}\n".encode('utf-8'))
            self.serial_connection.flush() 
            self.last_motor_move_time = time.time() 

            if not wait_for_ack:
                return True

            timeout_duration = 90 if is_solution else 15 + cmd.count(' ') * 2 
            start_wait_time = time.time()
            ack_buffer = ""

            while time.time() - start_wait_time < timeout_duration:
                if self.stop_requested and is_solution: 
                    print(f"Stop requested during Arduino command: {cmd}")
                    self.stop_requested = False 
                    return False 

                if self.serial_connection.in_waiting > 0:
                    try:
                        byte_chunk = self.serial_connection.read(self.serial_connection.in_waiting)
                        ack_buffer += byte_chunk.decode("latin-1", errors='replace')
                        
                        while '\n' in ack_buffer:
                            line, ack_buffer = ack_buffer.split('\n', 1)
                            line = line.strip()
                            if not line: continue

                            if "completed" in line.lower() or "executed" in line.lower():
                                print(f"Command '{cmd[:30]}...' ACKNOWLEDGED as completed by Arduino.")
                                if is_solution:
                                    print("SOLUTION execution reported complete by Arduino.")
                                    self.mode = "idle"
                                    self.status_message = "Solution completed successfully."
                                    self.solution = None 
                                    self.current_solve_move_index = self.total_solve_moves if self.total_solve_moves > 0 else 0
                                return True
                            elif "error" in line.lower():
                                self.error_message = f"Arduino error for '{cmd[:30]}...': {line}"
                                print(self.error_message)
                                return False
                    except Exception as e: 
                        print(f"Error reading/decoding from Arduino for '{cmd[:30]}...': {e}. Buffer: {ack_buffer}")
                        ack_buffer = "" 
                time.sleep(0.02) 

            self.error_message = f"TIMEOUT waiting for Arduino ack for '{cmd[:30]}...' (waited {timeout_duration}s)."
            print(self.error_message)
            return False
        except serial.SerialTimeoutException: 
            self.error_message = f"Serial write timeout for command '{cmd[:30]}...'. Check connection."
            print(self.error_message)
            self.init_serial() 
            return False
        except Exception as e:
            self.error_message = f"EXCEPTION sending Arduino command '{cmd[:30]}...': {type(e).__name__} - {e}"
            print(self.error_message)
            self.init_serial() 
            return False

    def _send_compound_move(self, move: str):
        if move:
            self.send_arduino_command(move, wait_for_ack=True)


    def process_frame(self, frame_data: bytes) -> Dict[str, Any]:
        try:
            nparr = np.frombuffer(frame_data, np.uint8)
            raw_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if raw_frame is None: raise ValueError("Failed to decode frame data from client.")

            self.last_processed_frame_for_calibration = raw_frame.copy() 
            
            input_frame_for_resize = raw_frame
            if self.zoom_crop_factor > 1.0 and self.zoom_crop_factor is not None: 
                h_orig, w_orig = raw_frame.shape[:2]
                crop_w = int(w_orig / self.zoom_crop_factor)
                crop_h = int(h_orig / self.zoom_crop_factor)
                crop_w = max(1, crop_w)
                crop_h = max(1, crop_h)
                start_x = (w_orig - crop_w) // 2
                start_y = (h_orig - crop_h) // 2
                end_x = start_x + crop_w
                end_y = start_y + crop_h
                start_x = np.clip(start_x, 0, w_orig -1)
                start_y = np.clip(start_y, 0, h_orig -1)
                end_x = np.clip(end_x, start_x + 1, w_orig) 
                end_y = np.clip(end_y, start_y + 1, h_orig) 
                
                if end_y > start_y and end_x > start_x: 
                    input_frame_for_resize = raw_frame[start_y:end_y, start_x:end_x]

            frame_for_processing = cv2.resize(input_frame_for_resize, self.WINDOW_SIZE, interpolation=cv2.INTER_AREA)
            
            self.current_frame_for_detection = frame_for_processing.copy() 
            display_overlay_frame = frame_for_processing.copy() 

            if self.mode == "calibrating": display_overlay_frame = self._draw_calibration_overlay(display_overlay_frame)
            elif self.mode == "scanning": display_overlay_frame = self._process_scanning_step(display_overlay_frame) 
            elif self.mode in ["solving", "scrambling"]: display_overlay_frame = self._draw_solving_overlay(display_overlay_frame)
            else: 
                status_to_show = f"Mode: {self.mode.capitalize()}"
                if self.status_message and self.status_message != status_to_show : status_to_show += f" - {self.status_message}"
                cv2.putText(display_overlay_frame, status_to_show, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,80), 1)

            _, buffer = cv2.imencode('.jpg', display_overlay_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            current_state_payload = self.get_state() 
            current_state_payload["processed_frame"] = frame_b64 
            return current_state_payload
        except Exception as e:
            self.error_message = f"Frame proc error: {type(e).__name__} - {e}"
            print(f"! CRITICAL ERROR in process_frame: {self.error_message}")
            # import traceback # Optional: for more detailed server-side error logging
            # traceback.print_exc()
            error_display_frame = np.zeros((self.WINDOW_SIZE[1], self.WINDOW_SIZE[0], 3), dtype=np.uint8)
            cv2.putText(error_display_frame, "ERROR", (self.WINDOW_SIZE[0]//2 - 50, self.WINDOW_SIZE[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            _, buffer = cv2.imencode('.jpg', error_display_frame)
            error_frame_b64 = base64.b64encode(buffer).decode('utf-8')
            error_state_payload = self.get_state() 
            error_state_payload["mode"] = "error" 
            error_state_payload["error_message"] = self.error_message 
            error_state_payload["status_message"] = "Frame processing failed."
            error_state_payload["processed_frame"] = error_frame_b64
            return error_state_payload

    def get_state(self) -> Dict[str, Any]:
        error_to_send = self.error_message
        if self.error_message: self.error_message = None 

        return {
            "mode": self.mode, "status_message": self.status_message, "error_message": error_to_send,
            "calibration_step": self.calibration_step,
            "current_color_calibrating": self.COLOR_NAMES_CALIBRATION[self.calibration_step] if self.mode == "calibrating" and self.calibration_step < len(self.COLOR_NAMES_CALIBRATION) else None,
            "scan_index": self.current_scan_idx,
            "solve_move_index": self.current_solve_move_index,
            "total_solve_moves": self.total_solve_moves,
            "solution_preview": self.solution[:30] + "..." if self.solution and len(self.solution) > 30 else self.solution,
            "serial_connected": self.serial_connection is not None and self.serial_connection.is_open,
            "zoom_crop_factor": self.zoom_crop_factor,
            "relative_detection_distance": self.relative_detection_distance,
            "effective_min_contour_area": self.MIN_CONTOUR_AREA,
            "effective_max_contour_area": self.MAX_CONTOUR_AREA,
        }

    def _find_best_cube_contour(self, frame_to_process: np.ndarray) -> Optional[np.ndarray]:
        processed_bgr = cv2.bilateralFilter(frame_to_process, d=7, sigmaColor=35, sigmaSpace=35)
        hsv = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2HSV)
        combined_mask = np.zeros((frame_to_process.shape[0], frame_to_process.shape[1]), dtype=np.uint8)
        
        for _, ranges_list in self.color_ranges.items(): 
            for lower, upper in ranges_list: 
                mask_part = cv2.inRange(hsv, lower, upper)
                combined_mask |= mask_part

        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_contour_approx = None
        max_area_found = 0 

        for contour in contours:
            area = cv2.contourArea(contour)
            if self.MIN_CONTOUR_AREA < area < self.MAX_CONTOUR_AREA: 
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.03 * peri, True) # Simpler approximation for candidate selection
                
                if len(approx) >= 4 and len(approx) <= 12: # Allow more vertices for initial complex shapes
                    x, y, w, h = cv2.boundingRect(approx)
                    if w == 0 or h == 0: continue
                    aspect_ratio = float(w) / h
                    
                    hull = cv2.convexHull(contour) 
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area) / hull_area if hull_area > 0 else 0.0

                    # Looser criteria here, stricter checks with auto-squaring later
                    if 0.60 < aspect_ratio < 1.40 and solidity > 0.65: 
                        if area > max_area_found: 
                            max_area_found = area
                            # Store the *original contour* or its hull if `approx` is too simple for `_detect_corners`
                            # Using `approx` is fine as `_detect_corners` will use its hull.
                            best_contour_approx = approx
        return best_contour_approx

    def _detect_corners(self, contour_approx: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if contour_approx is None: return None
        
        # Use convex hull for more robust corner detection from potentially complex contour_approx
        hull = cv2.convexHull(contour_approx)
        
        # Attempt to simplify the hull to 4 points
        epsilon_hull = 0.05 * cv2.arcLength(hull, True) 
        approx_hull = cv2.approxPolyDP(hull, epsilon_hull, True)
        
        corners = None
        if len(approx_hull) == 4:
            corners = approx_hull.reshape(4,2).astype(np.float32)
        else:
            # Fallback to minAreaRect of the hull if approxPolyDP doesn't yield 4 points
            rect = cv2.minAreaRect(hull) 
            corners = cv2.boxPoints(rect).astype(np.float32)
        
        if corners is None or corners.shape[0] != 4 : return None # Should ideally not happen if minAreaRect is used
        if self.current_frame_for_detection is None: return None # Should not happen if called from process_frame
        
        gray = cv2.cvtColor(self.current_frame_for_detection, cv2.COLOR_BGR2GRAY)
        h_img, w_img = gray.shape[:2]

        # Clip corners to image boundaries before refinement
        for i in range(corners.shape[0]): 
            corners[i, 0] = np.clip(corners[i, 0], 0, w_img - 1)
            corners[i, 1] = np.clip(corners[i, 1], 0, h_img - 1)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        try:
            # Ensure corners are float32 and contiguous for cornerSubPix
            corners_float32 = np.ascontiguousarray(corners, dtype=np.float32)
            refined_corners = cv2.cornerSubPix(gray, corners_float32, (5, 5), (-1,-1), criteria) 
        except cv2.error: 
            # print("Warning: cv2.error in cornerSubPix. Using unrefined corners.")
            refined_corners = corners # Use unrefined if subpixel fails

        # Clip refined corners again
        for i in range(refined_corners.shape[0]): 
            refined_corners[i, 0] = np.clip(refined_corners[i, 0], 0, w_img - 1)
            refined_corners[i, 1] = np.clip(refined_corners[i, 1], 0, h_img - 1)
        
        # Order corners: tl, tr, br, bl
        pts_sorted_y = refined_corners[np.argsort(refined_corners[:, 1])] 
        top_pts = pts_sorted_y[:2][np.argsort(pts_sorted_y[:2, 0])] 
        bottom_pts = pts_sorted_y[2:][np.argsort(pts_sorted_y[2:, 0])] 
        ordered_corners = np.array([top_pts[0], top_pts[1], bottom_pts[1], bottom_pts[0]], dtype=np.float32)
        return ordered_corners

    def _force_ideal_square_from_detected_corners(self, corners: np.ndarray) -> Optional[np.ndarray]:
        if corners is None or not isinstance(corners, np.ndarray) or corners.shape != (4, 2):
            return None

        rect = cv2.minAreaRect(corners) 
        box_center = np.array(rect[0], dtype=np.float32)
        rect_width, rect_height = rect[1]
        angle_deg = rect[2]

        ideal_side = (rect_width + rect_height) / 2.0

        min_sensible_side = max(30.0, np.sqrt(self.MIN_CONTOUR_AREA) * 0.5) 
        if ideal_side < min_sensible_side :
            # print(f"Debug: Ideal side {ideal_side:.1f} (from {rect_width:.1f}x{rect_height:.1f}) < min {min_sensible_side:.1f}. AutoSq Reject.")
            return None
        
        max_sensible_side = np.sqrt(self.MAX_CONTOUR_AREA) * 1.2 
        if ideal_side > max_sensible_side:
            # print(f"Debug: Ideal side {ideal_side:.1f} > max {max_sensible_side:.1f}. AutoSq Clamp.")
            ideal_side = max_sensible_side

        angle_rad = np.deg2rad(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        half_side = ideal_side / 2.0
        
        canonical_pts_at_origin = np.array([
            [-half_side, -half_side], [half_side, -half_side],
            [half_side, half_side], [-half_side, half_side]
        ], dtype=np.float32)
        
        new_corners = np.zeros((4, 2), dtype=np.float32)
        for i, pt_orig in enumerate(canonical_pts_at_origin):
            x_rot = pt_orig[0] * cos_a - pt_orig[1] * sin_a
            y_rot = pt_orig[0] * sin_a + pt_orig[1] * cos_a
            new_corners[i, 0] = x_rot + box_center[0]
            new_corners[i, 1] = y_rot + box_center[1]
            
        img_h, img_w = self.WINDOW_SIZE[1], self.WINDOW_SIZE[0]
        new_corners[:, 0] = np.clip(new_corners[:, 0], 0, img_w - 1)
        new_corners[:, 1] = np.clip(new_corners[:, 1], 0, img_h - 1)
        
        return new_corners

    def _predict_square_length(self, corners: Optional[np.ndarray]) -> int:
        if corners is None or len(corners) != 4: return 50 
        dists = [np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]
        avg_len = int(np.mean(dists))
        return max(30, avg_len) # Ensure a minimum size for the warped image

    def _perspective_transform(self, frame_to_transform: np.ndarray, detected_corners: np.ndarray) -> Optional[np.ndarray]:
        # Parameter `detected_corners` is now the primary input for corners.
        if detected_corners is None or detected_corners.shape != (4,2) : return None
        
        # predict_square_length calculates the side length of the target square based on the input corners' side lengths.
        # If detected_corners are already an ideal square, this will correctly use that side length.
        side_length = self._predict_square_length(detected_corners)
        if side_length < 20: # If predicted side length is too small, warp might be meaningless
            # print(f"Warning: Perspective transform predicted side length {side_length} too small.")
            return None 
            
        dst_points = np.array([[0,0], [side_length-1,0], [side_length-1,side_length-1], [0,side_length-1]], dtype=np.float32)
        try:
            transform_matrix = cv2.getPerspectiveTransform(detected_corners, dst_points)
            warped_image = cv2.warpPerspective(frame_to_transform, transform_matrix, (side_length, side_length), flags=cv2.INTER_LANCZOS4)
            return warped_image
        except cv2.error as e:
            print(f"! OpenCV Error in perspective transform: {e}")
            return None

    def _load_color_ranges_from_file(self, filename: str = "color_ranges.json") -> Optional[Dict[str, list]]:
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    loaded_data = json.load(f)
                parsed_ranges = {}
                for color, ranges_data_list in loaded_data.items(): 
                    if isinstance(ranges_data_list, list):
                        parsed_ranges[color] = []
                        for r_data_pair in ranges_data_list:
                            if isinstance(r_data_pair, list) and len(r_data_pair) == 2:
                                lower = np.array(r_data_pair[0], dtype=np.uint8)
                                upper = np.array(r_data_pair[1], dtype=np.uint8)
                                if lower.shape == (3,) and upper.shape == (3,):
                                    parsed_ranges[color].append((lower, upper))
                                else: print(f"Warning: Invalid shape for range in color '{color}' in {filename}.")
                            else: print(f"Warning: Invalid range pair format for color '{color}' in {filename}.")
                    else: print(f"Warning: Invalid data structure for color '{color}' in {filename}.")
                return parsed_ranges if parsed_ranges else None
            except json.JSONDecodeError as e: print(f"Error decoding JSON from {filename}: {e}")
            except Exception as e: print(f"Error loading/parsing color ranges from {filename}: {type(e).__name__} - {e}")
        return None

    def _save_color_ranges_to_file(self, filename: str = "color_ranges.json"):
        serializable_ranges = {}
        for color, ranges_list_tuples in self.color_ranges.items():
            serializable_ranges[color] = [] 
            for lower_np, upper_np in ranges_list_tuples:
                 serializable_ranges[color].append((lower_np.tolist(), upper_np.tolist()))
        try:
            with open(filename, 'w') as f:
                json.dump(serializable_ranges, f, indent=4)
            print(f"Color ranges saved to {filename}")
        except Exception as e: print(f"Error saving color ranges to {filename}: {e}")

    def _draw_calibration_overlay(self, display_frame: np.ndarray) -> np.ndarray:
        height, width = display_frame.shape[:2]
        roi_dimension = int(min(width, height) * self.calibration_roi_scale)
        roi_dimension = max(20, roi_dimension) 
        roi_x_start = (width - roi_dimension) // 2
        roi_y_start = (height - roi_dimension) // 2
        roi_x_end = roi_x_start + roi_dimension
        roi_y_end = roi_y_start + roi_dimension
        self.last_valid_grid_info_for_calibration = (roi_x_start, roi_y_start, roi_x_end, roi_y_end)
        cv2.rectangle(display_frame, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (0, 255, 0), 2)
        center_x, center_y = roi_x_start + roi_dimension // 2, roi_y_start + roi_dimension // 2
        cv2.line(display_frame, (center_x - 5, center_y), (center_x + 5, center_y), (0, 255, 0), 1)
        cv2.line(display_frame, (center_x, center_y - 5), (center_x, center_y + 5), (0, 255, 0), 1)
        
        if self.calibration_step < len(self.COLOR_NAMES_CALIBRATION):
            current_color = self.COLOR_NAMES_CALIBRATION[self.calibration_step]
            instruction = f"Aim {current_color} at green box. Press Capture."
            cv2.putText(display_frame, instruction, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        else: 
            cv2.putText(display_frame, "Calibration Complete! Saved.", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return display_frame

    def _process_scanning_step(self, display_frame_overlay: np.ndarray) -> np.ndarray:
        frame_to_detect_on = self.current_frame_for_detection 
        if frame_to_detect_on is None:
             cv2.putText(display_frame_overlay, "Err: No detection frame", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
             return display_frame_overlay

        best_contour_approx = self._find_best_cube_contour(frame_to_detect_on)
        cube_detected_initial = best_contour_approx is not None
        
        is_shape_acceptable_for_scan = False 
        scan_debug_info = "" 
        corners_for_transform = None

        if cube_detected_initial:
            cv2.drawContours(display_frame_overlay, [best_contour_approx], -1, (0, 255, 0), 1) # Green for original contour
            x_c, y_c, w_c, h_c = cv2.boundingRect(best_contour_approx)
            if w_c > 0 and h_c > 0: 
                cell_w, cell_h = w_c // 3, h_c // 3
                for i in range(1, 3):
                    cv2.line(display_frame_overlay, (x_c, y_c + i * cell_h), (x_c + w_c, y_c + i * cell_h), (0,80,0),1)
                    cv2.line(display_frame_overlay, (x_c + i * cell_w, y_c), (x_c + i * cell_w, y_c + h_c), (0,80,0),1)

            num_vertices = len(best_contour_approx)
            aspect_ratio_br = float(w_c) / h_c if h_c > 0 else 0.0
            
            detected_quad_corners = self._detect_corners(best_contour_approx)
            
            scan_debug_info = f"Orig(V:{num_vertices}, ARbr:{aspect_ratio_br:.2f}, QuadCrns:{'4' if detected_quad_corners is not None and len(detected_quad_corners)==4 else 'No'})"

            if detected_quad_corners is not None and len(detected_quad_corners) == 4:
                q_x, q_y, q_w, q_h = cv2.boundingRect(detected_quad_corners.astype(np.int32))
                q_ar = float(q_w)/q_h if q_h > 0 else 0.0
                q_area = cv2.contourArea(detected_quad_corners.astype(np.int32))

                is_quad_good_enough = (0.85 < q_ar < 1.15 and 
                                       self.MIN_CONTOUR_AREA * 0.7 < q_area < self.MAX_CONTOUR_AREA * 1.3)

                if is_quad_good_enough:
                    is_shape_acceptable_for_scan = True
                    corners_for_transform = detected_quad_corners
                    scan_debug_info += " (Quad OK)"
                else:
                    scan_debug_info += f" (Quad not ideal AR:{q_ar:.2f} Area:{q_area}, try ForceSq)"
                    forced_corners = self._force_ideal_square_from_detected_corners(detected_quad_corners)
                    if forced_corners is not None:
                        fc_x, fc_y, fc_w, fc_h = cv2.boundingRect(forced_corners.astype(np.int32))
                        fc_ar = float(fc_w)/fc_h if fc_h > 0 else 0.0
                        fc_area = cv2.contourArea(forced_corners.astype(np.int32))

                        if (0.80 < fc_ar < 1.20 and 
                            self.MIN_CONTOUR_AREA * 0.5 < fc_area < self.MAX_CONTOUR_AREA * 1.5):
                            is_shape_acceptable_for_scan = True
                            corners_for_transform = forced_corners
                            cv2.drawContours(display_frame_overlay, [forced_corners.astype(np.int32)], -1, (255, 0, 255), 1) # Magenta for forced
                            scan_debug_info += " (ForceSq OK)"
                        else:
                            is_shape_acceptable_for_scan = False
                            scan_debug_info += f" (ForceSq REJ AR:{fc_ar:.2f} Area:{fc_area})"
                    else:
                        is_shape_acceptable_for_scan = False
                        scan_debug_info += " (ForceSq FAIL)"
            else: # detected_quad_corners was None or not 4 points
                is_shape_acceptable_for_scan = False
        
        # Stability check based on the original `best_contour_approx`
        if cube_detected_initial and is_shape_acceptable_for_scan: # Use is_shape_acceptable_for_scan to ensure we have a good candidate FOR STABILITY
            if self.prev_contour_scan is not None:
                try: 
                    M_curr = cv2.moments(best_contour_approx); M_prev = cv2.moments(self.prev_contour_scan)
                    if M_curr["m00"] > 0 and M_prev["m00"] > 0:
                        c_curr = (int(M_curr["m10"]/M_curr["m00"]), int(M_curr["m01"]/M_curr["m00"]))
                        c_prev = (int(M_prev["m10"]/M_prev["m00"]), int(M_prev["m01"]/M_prev["m00"]))
                        pos_diff = np.sqrt((c_curr[0]-c_prev[0])**2 + (c_curr[1]-c_prev[1])**2)
                        area_curr = cv2.contourArea(best_contour_approx)
                        area_prev = cv2.contourArea(self.prev_contour_scan)
                        area_diff_ratio = abs(area_curr - area_prev) / max(1.0, (area_curr + area_prev) / 2.0)

                        if pos_diff < 10 and area_diff_ratio < 0.20 : # Added area stability
                             self.stability_counter += 1
                        else: self.stability_counter = 0
                    else: self.stability_counter = 0 
                except: self.stability_counter = 0 
            else:
                self.stability_counter = 1 
            self.prev_contour_scan = best_contour_approx.copy() # Store the raw contour for next frame's stability
        else: # No acceptable contour or shape rejected
            self.stability_counter = 0
            self.prev_contour_scan = None


        current_time = time.time()
        time_since_last_scan_attempt = current_time - self.last_scan_time
        time_since_motor = current_time - self.last_motor_move_time
        
        scan_status_text = f"Scan {self.current_scan_idx + 1}/12: "
        ready_to_capture_this_frame = False
        status_color = (220, 220, 220) 

        if not cube_detected_initial:
            scan_status_text += "Detecting..."
        elif not is_shape_acceptable_for_scan: # This now considers if forced square was accepted
            scan_status_text += f"Shape rej. {scan_debug_info}"
            status_color = (0, 165, 255) 
        elif time_since_motor < self.MOTOR_STABILIZATION_TIME:
            scan_status_text += f"Motor ({self.MOTOR_STABILIZATION_TIME - time_since_motor:.1f}s)"
        elif time_since_last_scan_attempt < self.SCAN_COOLDOWN:
            scan_status_text += f"Cooldown ({self.SCAN_COOLDOWN - time_since_last_scan_attempt:.1f}s)"
        elif self.stability_counter < self.STABILITY_THRESHOLD:
            scan_status_text += f"Stabilizing ({self.stability_counter}/{self.STABILITY_THRESHOLD}) {scan_debug_info}"
        else: 
            scan_status_text += f"CAPTURING... {scan_debug_info}"
            ready_to_capture_this_frame = True
            status_color = (0,255,0) 
        
        cv2.putText(display_frame_overlay, scan_status_text, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        self.status_message = scan_status_text 

        if ready_to_capture_this_frame and self.current_scan_idx < 12:
            if corners_for_transform is None:
                self.status_message = scan_status_text.replace("CAPTURING...", "No valid corners for warp")
                # print("Error: corners_for_transform is None during capture logic.") # Debug
                return display_frame_overlay

            warped_face = self._perspective_transform(frame_to_detect_on, corners_for_transform)
            if warped_face is not None and warped_face.size > 0:
                grid_s = min(warped_face.shape[0], warped_face.shape[1])
                cell_s = max(1, grid_s // 3)
                if grid_s < 21 or cell_s < 7: 
                    self.status_message = scan_status_text.replace("CAPTURING...", "Warped too small")
                    return display_frame_overlay 
                
                current_face_sticker_colors: List[str] = [] # Type hint
                valid_rois = True
                for r_idx in range(3): 
                    for c_idx in range(3): 
                        y_s, y_e = r_idx * cell_s, (r_idx + 1) * cell_s
                        x_s, x_e = c_idx * cell_s, (c_idx + 1) * cell_s
                        pad = max(1, cell_s // 6) 
                        
                        roi_y_s = min(y_s + pad, warped_face.shape[0] - 1)
                        roi_y_e = max(roi_y_s + 1, min(y_e - pad, warped_face.shape[0])) 
                        roi_x_s = min(x_s + pad, warped_face.shape[1] - 1)
                        roi_x_e = max(roi_x_s + 1, min(x_e - pad, warped_face.shape[1])) 

                        if roi_y_e <= roi_y_s or roi_x_e <= roi_x_s: current_face_sticker_colors.append('X'); valid_rois=False; break
                        sticker_roi = warped_face[roi_y_s:roi_y_e, roi_x_s:roi_x_e]
                        if sticker_roi.size==0: current_face_sticker_colors.append('X'); valid_rois=False; break
                        
                        detected_color = self._detect_color(sticker_roi)
                        current_face_sticker_colors.append(detected_color)
                    if not valid_rois: break
                
                non_center_unknown = False
                if len(current_face_sticker_colors) == 9:
                    for i, color_char in enumerate(current_face_sticker_colors):
                        if i == 4: continue # Skip center sticker for this particular check
                        if color_char == 'X':
                            non_center_unknown = True
                            break
                
                if valid_rois and len(current_face_sticker_colors) == 9:
                    if non_center_unknown: 
                        self.status_message = scan_status_text.replace("CAPTURING...", f"Unk.clr(edge/corn): {''.join(current_face_sticker_colors)}")
                    elif self.prev_face_colors_scan is None or tuple(current_face_sticker_colors) != tuple(self.prev_face_colors_scan):
                        self.u_scans[self.current_scan_idx] = list(current_face_sticker_colors)
                        self.prev_face_colors_scan = list(current_face_sticker_colors)
                        print(f"Scan {self.current_scan_idx + 1} OK: {''.join(current_face_sticker_colors)}")
                        
                        self.current_scan_idx += 1
                        self.stability_counter = 0 
                        self.last_scan_time = time.time() 

                        if self.current_scan_idx < 12:
                            next_rotation_cmd = self.rotation_sequence[self.current_scan_idx] 
                            if next_rotation_cmd:
                                self.status_message = f"Scan {self.current_scan_idx} done. Rotating..." # current_scan_idx already incremented
                                self._send_compound_move(next_rotation_cmd) 
                        else: 
                            self._send_compound_move("F' B")
                            self.status_message = "All scans done. Finalizing..."
                            constructed_state = self._construct_cube_state_from_scans()
                            if constructed_state:
                                self._print_cube_state_visual(constructed_state, "Constructed (FRBLUD) for Solving")
                                solution_str = self._solve_cube_with_kociemba(constructed_state)
                                if solution_str is not None: 
                                    self.solution = solution_str 
                                    self.mode = "solving"
                                    self.total_solve_moves = len(self.solution.split()) if self.solution else 0
                                    self.current_solve_move_index = 0 
                                    self.status_message = f"Solution ({self.total_solve_moves}m). Sending..."
                                    print(f"Attempting to send solution to Arduino: {self.solution}")
                                    self.send_arduino_command(f"SOLUTION:{self.solution}", is_solution=True)
                                else: 
                                    self.mode = "idle" 
                                    self.status_message = self.error_message or "Could not solve. Check scans/colors or cube state."
                                    self._reset_scan_state() 
                            else: 
                                self.mode = "idle" 
                                self.error_message = self.error_message or "Cube construction failed. Scan may be invalid."
                                self.status_message = self.error_message
                                self._reset_scan_state()
                    else: self.status_message = scan_status_text.replace("CAPTURING...", "Duplicate")
                elif not valid_rois: self.status_message = scan_status_text.replace("CAPTURING...","Invalid ROIs")
                else: self.status_message = scan_status_text.replace("CAPTURING...","Sticker count err")
            else: self.status_message = scan_status_text.replace("CAPTURING...","Warp failed")
        return display_frame_overlay

    def _draw_solving_overlay(self, display_frame: np.ndarray) -> np.ndarray:
        mode_text = self.mode.capitalize()
        progress_text = f"{mode_text}: "
        if self.total_solve_moves > 0 :
            progress_text += f"{self.current_solve_move_index}/{self.total_solve_moves}"
        else: progress_text += "Starting..."
        
        if self.mode == "solving" and self.solution:
            preview = self.solution[:25] + ('...' if len(self.solution)>25 else '')
            cv2.putText(display_frame, f"Sol: {preview}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,0), 1)

        cv2.putText(display_frame, progress_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,50), 1)
        return display_frame

    def _detect_color(self, roi_img: np.ndarray) -> str:
        if roi_img is None or roi_img.size == 0 or roi_img.shape[0] < 5 or roi_img.shape[1] < 5: 
            return 'X' 

        median_ksize = 3 
        if min(roi_img.shape[:2]) <= median_ksize: 
            processed_roi = roi_img
        else:
            processed_roi = cv2.medianBlur(roi_img, median_ksize)

        hsv_roi = cv2.cvtColor(processed_roi, cv2.COLOR_BGR2HSV)

        h, w = hsv_roi.shape[:2]
        ch_s, ch_e = max(0, int(h*0.25)), min(h, int(h*0.75)) if h > 4 else h 
        cw_s, cw_e = max(0, int(w*0.25)), min(w, int(w*0.75)) if w > 4 else w
        
        if ch_e <= ch_s or cw_e <= cw_s : 
            hsv_sample_area = hsv_roi 
            if hsv_sample_area.size == 0: return 'X'
        else: 
            hsv_sample_area = hsv_roi[ch_s:ch_e, cw_s:cw_e]
        
        if hsv_sample_area.size == 0: return 'X'

        best_match_color = 'X'; highest_match_strength = -1.0

        for color_name, hsv_ranges_list in self.color_ranges.items():
            current_color_pixel_count = 0
            for lower_hsv, upper_hsv in hsv_ranges_list:
                if not (isinstance(lower_hsv, np.ndarray) and isinstance(upper_hsv, np.ndarray) and
                        lower_hsv.shape == (3,) and upper_hsv.shape == (3,)):
                    continue # Should not happen if loaded correctly

                mask = cv2.inRange(hsv_sample_area, lower_hsv, upper_hsv)
                current_color_pixel_count += cv2.countNonZero(mask)
            
            total_pixels_in_sample = hsv_sample_area.shape[0] * hsv_sample_area.shape[1]
            if total_pixels_in_sample == 0: continue 
            match_percentage = current_color_pixel_count / total_pixels_in_sample

            if match_percentage > highest_match_strength:
                highest_match_strength = match_percentage
                best_match_color = color_name
        
        # Lowered threshold for more lenient matching if needed, or adjust specific color ranges.
        # Original was 0.35. If colors are too similar or lighting varies, this might be an issue.
        if highest_match_strength < 0.30: # Slightly more permissive
            return 'X'
            
        return best_match_color

    def capture_calibration_color(self) -> bool:
        if self.mode != "calibrating" or self.calibration_step >= len(self.COLOR_NAMES_CALIBRATION):
            self.error_message = "Not in calibration mode or calibration already complete."
            self.status_message = self.error_message; return False
        if self.last_processed_frame_for_calibration is None: 
            self.error_message = "No raw frame available for calibration capture."
            self.status_message = self.error_message; return False
        if self.last_valid_grid_info_for_calibration is None:
            self.error_message = "Calibration ROI not set (internal error)."
            self.status_message = self.error_message; return False

        try:
            resized_frame_for_cal_roi = cv2.resize(self.last_processed_frame_for_calibration, 
                                                   self.WINDOW_SIZE, interpolation=cv2.INTER_AREA)
            
            x_s, y_s, x_e, y_e = self.last_valid_grid_info_for_calibration 
            h_cal, w_cal = resized_frame_for_cal_roi.shape[:2]
            x_s = np.clip(x_s, 0, w_cal -1)
            y_s = np.clip(y_s, 0, h_cal -1)
            x_e = np.clip(x_e, x_s + 1, w_cal)
            y_e = np.clip(y_e, y_s + 1, h_cal)

            if y_e <= y_s or x_e <= x_s:
                 self.error_message = "Calibration ROI became invalid after clipping."
                 self.status_message = self.error_message; return False

            roi_to_calibrate = resized_frame_for_cal_roi[y_s:y_e, x_s:x_e]

            if roi_to_calibrate.size == 0:
                self.error_message = "Calibration ROI is empty after extraction."
                self.status_message = self.error_message; return False
            
            roi_filtered = cv2.bilateralFilter(roi_to_calibrate, d=5, sigmaColor=30, sigmaSpace=30)
            hsv_roi = cv2.cvtColor(roi_filtered, cv2.COLOR_BGR2HSV)

            h_roi, w_roi = hsv_roi.shape[:2]
            center_crop_h_start, center_crop_h_end = max(0, int(h_roi*0.25)), min(h_roi, int(h_roi*0.75))
            center_crop_w_start, center_crop_w_end = max(0, int(w_roi*0.25)), min(w_roi, int(w_roi*0.75))
            
            if center_crop_h_end <= center_crop_h_start or center_crop_w_end <= center_crop_w_start:
                hsv_sample_for_median = hsv_roi
            else:
                hsv_sample_for_median = hsv_roi[center_crop_h_start:center_crop_h_end, center_crop_w_start:center_crop_w_end]

            if hsv_sample_for_median.size == 0:
                self.error_message = "Calibration sample area is empty."
                self.status_message = self.error_message; return False

            avg_h = int(np.median(hsv_sample_for_median[:,:,0])) 
            avg_s = int(np.median(hsv_sample_for_median[:,:,1]))
            avg_v = int(np.median(hsv_sample_for_median[:,:,2]))
            print(f"Median HSV for calib ROI ({self.COLOR_NAMES_CALIBRATION[self.calibration_step]}): H={avg_h}, S={avg_s}, V={avg_v}")

            current_color_to_cal = self.COLOR_NAMES_CALIBRATION[self.calibration_step]
            
            h_delta = 10; s_abs_delta = 55; v_abs_delta = 65 
            s_min_default = 50; v_min_default = 50

            if current_color_to_cal == "W": 
                h_delta = 30; s_max_cap = 80; v_min_cap = 140 
                lower_s = max(0, avg_s - s_abs_delta); upper_s = min(s_max_cap, avg_s + s_abs_delta)
                lower_v = max(v_min_cap, avg_v - v_abs_delta); upper_v = min(255, avg_v + v_abs_delta)
            elif current_color_to_cal in ["R", "O"]: # Red and Orange might need tighter Hue for Red, careful with wrap-around
                 h_delta = 7 # Smaller hue delta for R, O
                 lower_s = max(s_min_default + 20, avg_s - s_abs_delta); upper_s = min(255, avg_s + s_abs_delta)
                 lower_v = max(v_min_default + 10, avg_v - v_abs_delta); upper_v = min(255, avg_v + v_abs_delta)
            else: # G, Y, B
                 lower_s = max(s_min_default, avg_s - s_abs_delta); upper_s = min(255, avg_s + s_abs_delta)
                 lower_v = max(v_min_default, avg_v - v_abs_delta); upper_v = min(255, avg_v + v_abs_delta)

            lower_s = np.clip(lower_s, 0, 255); upper_s = np.clip(upper_s, 0, 255)
            lower_v = np.clip(lower_v, 0, 255); upper_v = np.clip(upper_v, 0, 255)
            if lower_s > upper_s: lower_s = max(0, upper_s - 10) 
            if lower_v > upper_v: lower_v = max(0, upper_v - 10)

            final_ranges = []
            if current_color_to_cal == "R": 
                hue_low_threshold = h_delta + 5 
                hue_high_threshold = 179 - (h_delta + 5) 
                
                if avg_h < hue_low_threshold: 
                     final_ranges.append((np.array([0, lower_s, lower_v]), 
                                          np.array([min(179, avg_h + h_delta), upper_s, upper_v])))
                     final_ranges.append((np.array([max(0, 179 - (h_delta - (avg_h % 180) ) ), lower_s, lower_v]), 
                                          np.array([179, upper_s, upper_v])))
                elif avg_h > hue_high_threshold: 
                     final_ranges.append((np.array([max(0, avg_h - h_delta), lower_s, lower_v]), 
                                          np.array([179, upper_s, upper_v])))
                     final_ranges.append((np.array([0, lower_s, lower_v]), 
                                          np.array([min(179, (avg_h + h_delta) % 180), upper_s, upper_v])))
                else: 
                     final_ranges.append((np.array([max(0,avg_h-h_delta),lower_s,lower_v]), 
                                          np.array([min(179,avg_h+h_delta),upper_s,upper_v])))
                final_ranges = [(l,u) for l,u in final_ranges if np.all(l<=u) and l[0]<=u[0] and l[0] < 180 and u[0] < 180 and l[0] != u[0]] # Ensure range is valid
            else: 
                l_h = max(0, avg_h - h_delta); u_h = min(179, avg_h + h_delta)
                final_ranges.append((np.array([l_h, lower_s, lower_v]), np.array([u_h, upper_s, upper_v])))

            if not final_ranges: 
                print(f"! Warning: No valid ranges generated for {current_color_to_cal}. Using wide fallback based on median HSV.")
                final_ranges.append((np.array([max(0,avg_h-15),max(0,avg_s-70),max(0,avg_v-80)]), 
                                     np.array([min(179,avg_h+15),min(255,avg_s+70),min(255,avg_v+80)])))

            self.color_ranges[current_color_to_cal] = final_ranges
            print(f"Calibrated {current_color_to_cal}: {self.color_ranges[current_color_to_cal]}")

            self.calibration_step += 1
            if self.calibration_step >= len(self.COLOR_NAMES_CALIBRATION):
                self._save_color_ranges_to_file()
                self.mode = "idle"
                self.status_message = "Calibration complete & saved."
            else:
                next_color = self.COLOR_NAMES_CALIBRATION[self.calibration_step]
                self.status_message = f"Calibrated {current_color_to_cal}. Next: {next_color}"
            self.error_message = None 
            return True

        except Exception as e:
            self.error_message = f"Calib. capture error: {type(e).__name__} - {e}"
            self.status_message = "Calibration capture failed."
            print(f"Error in capture_calibration_color: {self.error_message}")
            # import traceback # For detailed debugging
            # traceback.print_exc()
            return False

    def _reset_scan_state(self):
        self.current_scan_idx = 0
        self.u_scans = [[] for _ in range(12)]
        self.prev_face_colors_scan = None
        self.prev_contour_scan = None
        self.stability_counter = 0
        self.last_scan_time = time.time()
        self.last_motor_move_time = time.time() 
        print("Scan state has been reset.")

    def _reset_solve_state(self):
        self.solution = None
        self.current_solve_move_index = 0
        self.total_solve_moves = 0
        print("Solve state has been reset.")

    def _validate_kociemba_string(self, s: str, name: str = "Cube String"):
        if not isinstance(s, str): raise ValueError(f"{name} must be a string.")
        if len(s) != 54: raise ValueError(f"{name} must be 54 chars long, got {len(s)}: '{s[:10]}...'")
        allowed_chars = {'U', 'R', 'F', 'D', 'L', 'B'}
        if not all(c in allowed_chars for c in s):
            invalid_chars = set(c for c in s if c not in allowed_chars)
            raise ValueError(f"{name} contains invalid chars: {invalid_chars}. String: '{s[:10]}...'")
        counts = Counter(s)
        if not all(count == 9 for count in counts.values()):
            raise ValueError(f"{name} must have 9 of each URFDLB char. Counts: {counts}")
        
        # Kociemba string is URFDLB. Centers are at U(4), R(13), F(22), D(31), L(40), B(49)
        # in this URFDLB ordered string.
        centers_kociemba_ordered = [s[4], s[13], s[22], s[31], s[40], s[49]] 
        if len(set(centers_kociemba_ordered)) != 6:
            raise ValueError(f"{name} center pieces are not unique: {centers_kociemba_ordered}")
        
        # Check if U-face center is 'U', R-face center is 'R', etc.
        if (centers_kociemba_ordered[0]!='U' or centers_kociemba_ordered[1]!='R' or \
            centers_kociemba_ordered[2]!='F' or centers_kociemba_ordered[3]!='D' or \
            centers_kociemba_ordered[4]!='L' or centers_kociemba_ordered[5]!='B'):
            raise ValueError(f"{name} center pieces are not U,R,F,D,L,B in Kociemba URFDLB face order. Got: {centers_kociemba_ordered}")
    
    def _validate_cube(self, cube_str: str, name: str = "Physical Cube String"):
        if len(cube_str) != 54:
            raise ValueError(f"{name} must be 54 characters, got {len(cube_str)}")
        counts = Counter(cube_str)
        if 'X' in counts: 
            raise ValueError(f"{name} contains 'X' (unknown) stickers. Counts: {counts}")
        if len(counts) != 6:
             raise ValueError(f"{name} must have exactly 6 unique colors, found {len(counts)}. Counts: {counts}")
        if not all(count == 9 for count in counts.values()):
            raise ValueError(f"{name} must have 9 of each color. Counts: {counts}")


    def _remap_colors_to_kociemba(self, cube_frblud_physical_colors: str):
        self._validate_cube(cube_frblud_physical_colors, "FRBLUD")
        centers = [cube_frblud_physical_colors[i] for i in [4, 13, 22, 31, 40, 49]]
        color_map = {
            centers[4]: 'U', centers[1]: 'R', centers[0]: 'F',
            centers[5]: 'D', centers[3]: 'L', centers[2]: 'B'
        }
        return color_map, ''.join(color_map[c] for c in cube_frblud_physical_colors)


    def _remap_cube_to_kociemba_face_order(self, cube_frblud_with_kociemba_chars: str):
        front, right, back, left, up, down = [cube_frblud_with_kociemba_chars[i:i + 9] for i in range(0, 54, 9)]
        return up + right + front + down + left + back

    def _get_solved_kociemba_string(self):
        return 'U'*9 + 'R'*9 + 'F'*9 + 'D'*9 + 'L'*9 + 'B'*9

    def _is_cube_solved_by_kociemba_string(self, kociemba_str: str) -> bool:
        if len(kociemba_str) != 54: return False
        # Kociemba string is URFDLB. Centers are at U(4), R(13), F(22), D(31), L(40), B(49).
        expected_centers = ['U', 'R', 'F', 'D', 'L', 'B']
        actual_centers = [kociemba_str[4], kociemba_str[13], kociemba_str[22], 
                          kociemba_str[31], kociemba_str[40], kociemba_str[49]]
        if actual_centers != expected_centers: return False 
        return kociemba_str == self._get_solved_kociemba_string()
        
    def _solve_cube_with_kociemba(self, cube_frblud_physical_colors: str):
        try:
            self._validate_cube(cube_frblud_physical_colors, "Input FRBLUD Physical Colors")
            
            _, frblud_string_with_kociemba_chars = self._remap_colors_to_kociemba(cube_frblud_physical_colors)
            
            temp_counts = Counter(frblud_string_with_kociemba_chars)
            if not all(count == 9 for count in temp_counts.values()) or len(temp_counts) != 6:
                 raise ValueError(f"FRBLUD string with Kociemba Chars is invalid (after phys->kociemba char map). Counts: {temp_counts}")

            scrambled_kociemba_format_str = self._remap_cube_to_kociemba_face_order(frblud_string_with_kociemba_chars)
            self._validate_kociemba_string(scrambled_kociemba_format_str, "Scrambled Kociemba Format String")

            if self._is_cube_solved_by_kociemba_string(scrambled_kociemba_format_str):
                print("\nCube is already solved! No moves needed.")
                return "" 

            solution = kociemba.solve(scrambled_kociemba_format_str) # Uses default solved state URFDLB

            print(f"Kociemba raw solution: {solution}")

            u_replacement = "R L F2 B2 R' L' D R L F2 B2 R' L'"       
            u_prime_replacement = "R L F2 B2 R' L' D' R L F2 B2 R' L'" 
            u2_replacement = "R L F2 B2 R' L' D2 R L F2 B2 R' L'"     

            moves = solution.split()
            expanded_solution_moves = []
            for move in moves:
                if move == "U": expanded_solution_moves.append(u_replacement)
                elif move == "U'": expanded_solution_moves.append(u_prime_replacement)
                elif move == "U2": expanded_solution_moves.append(u2_replacement)
                else: expanded_solution_moves.append(move)
            
            expanded_solution_str = " ".join(m for m in expanded_solution_moves if m)
            final_simplified_solution = self._simplify_cube_moves_basic(expanded_solution_str)

            print(f"Expanded solution length: {len(expanded_solution_str.split())}")
            print(f"Final simplified solution: {final_simplified_solution}")
            print(f"Final simplified solution length: {len(final_simplified_solution.split())}")

            return final_simplified_solution

        except ValueError as ve: 
            self.error_message = f"Kociemba validation/logic error: {str(ve)}"
            print(f"! Error solving cube: {self.error_message}")
            # import traceback; traceback.print_exc() # For detailed debugging
            return None
        except Exception as e: 
            self.error_message = f"Kociemba general error: {type(e).__name__} - {str(e)}"
            print(f"! Error solving cube: {self.error_message}")
            # import traceback; traceback.print_exc() # For detailed debugging
            return None
        
    def _is_cube_solved_by_face_colors(self, cube_state_str: str) -> bool: 
        if len(cube_state_str) != 54: return False
        for i in range(0, 54, 9): 
            face = cube_state_str[i : i+9]
            if not all(s == face[4] for s in face): 
                return False
        return True

    def _simplify_cube_moves_basic(self, moves_str: str) -> str:
        moves = [m for m in moves_str.strip().split() if m] 
        if not moves: return ""
        
        simplified_pass1 = [] 
        i = 0
        while i < len(moves):
            current_move_full = moves[i]
            face = current_move_full[0]
            
            if face not in ['F', 'B', 'R', 'L', 'D', 'U']: 
                simplified_pass1.append(current_move_full)
                i += 1
                continue

            net_rot = 0
            j = i
            while j < len(moves):
                m_full = moves[j]
                if not m_full or m_full[0] != face: 
                    break
                
                val = 1
                if len(m_full) > 1:
                    if m_full[1] == '2': val = 2
                    elif m_full[1] == "'": val = 3 
                net_rot = (net_rot + val) % 4
                j += 1
            
            if net_rot == 1: simplified_pass1.append(face)
            elif net_rot == 2: simplified_pass1.append(face + "2")
            elif net_rot == 3: simplified_pass1.append(face + "'")
            i = j
        
        return " ".join(simplified_pass1)


    def scramble_cube(self) -> bool:
        if self.mode not in ["idle", "error"]:
            self.error_message = "Cannot scramble, busy."; self.status_message = self.error_message; return False
        try:
            possible_faces = ['F', 'B', 'R', 'L', 'D']; modifiers = ['', "'", '2'] 
            scramble_moves_list = []
            
            last_face_scrambled = None
            for _ in range(random.randint(18, 22)): 
                chosen_face = random.choice(possible_faces)
                # Ensure not to pick the same face consecutively (simple prevention)
                if scramble_moves_list and chosen_face == last_face_scrambled:
                    available_faces = [f for f in possible_faces if f != chosen_face]
                    if not available_faces: continue # Should not happen with 5 faces
                    chosen_face = random.choice(available_faces)
                
                chosen_modifier = random.choice(modifiers)
                scramble_moves_list.append(chosen_face + chosen_modifier)
                last_face_scrambled = chosen_face 
            
            scramble_sequence = " ".join(scramble_moves_list)
            # Further simplify if moves like R R R appear by chance (unlikely but good practice)
            scramble_sequence = self._simplify_cube_moves_basic(scramble_sequence)

            self.mode = "scrambling"; self.status_message = f"Scrambling: {scramble_sequence[:30]}..."
            self._reset_solve_state() # Reset any previous solution
            self.total_solve_moves = len(scramble_sequence.split()) # Display scramble length
            print(f"Generated scramble: {scramble_sequence}")
            
            # Send scramble to Arduino; treat it like a solution for ACK purposes
            if self.send_arduino_command(scramble_sequence, is_solution=True): 
                # If send_arduino_command sets mode to idle on success for solution, this is fine.
                if self.mode != "idle": # Check if it was already set to idle by send_arduino_command
                    self.mode = "idle"
                    self.status_message = "Scramble completed."
                self._reset_solve_state() # Clear scramble progress display
                return True
            else:
                self.mode = "idle" # Or "error" if send_arduino_command set an error_message
                self.error_message = self.error_message or "Failed to execute scramble on Arduino."
                self.status_message = self.error_message
                self._reset_solve_state()
                return False
        except Exception as e:
            self.error_message = f"Scramble error: {type(e).__name__} - {e}"; self.status_message = "Scramble gen failed."
            self.mode = "error"; return False

    def stop_current_operation(self):
        print(f"Stop requested. Current mode: {self.mode}")
        self.stop_requested = True
        if self.serial_connection and self.serial_connection.is_open:
            try: 
                self.serial_connection.write("STOP\n".encode('utf-8')) 
                self.serial_connection.flush(); print("Sent STOP command to Arduino.")
            except Exception as e: print(f"Error sending STOP command: {e}")

        self.mode = "idle"; self.status_message = "Operation stopped by user."
        self.error_message = None
        self._reset_solve_state()
        self._reset_scan_state()
        self.stop_requested = False 
        print(self.status_message)
    
    def _construct_cube_state_from_scans(self) -> Optional[str]:
        cube_state = [''] * 54
        cube_state[4] = 'B'
        cube_state[13] = 'O'
        cube_state[22] = 'G'
        cube_state[31] = 'R'
        cube_state[40] = 'W'
        cube_state[49] = 'Y'
        cube_state[36:45] = self.u_scans[0]
        for i in range(54):
            if not cube_state[i]:
                cube_state[i] = '-'
        cube_state[0] = self.u_scans[1][0]
        cube_state[2] = self.u_scans[1][2]
        cube_state[3] = self.u_scans[1][3]
        cube_state[5] = self.u_scans[1][5]
        cube_state[6] = self.u_scans[1][6]
        cube_state[8] = self.u_scans[1][8]
        cube_state[9] = self.u_scans[2][0]
        cube_state[10] = self.u_scans[2][1]
        cube_state[11] = self.u_scans[2][2]
        cube_state[15] = self.u_scans[2][6]
        cube_state[16] = self.u_scans[2][7]
        cube_state[17] = self.u_scans[2][8]
        cube_state[47] = self.u_scans[3][0]
        cube_state[53] = self.u_scans[3][2]
        cube_state[1] = self.u_scans[3][3]
        cube_state[7] = self.u_scans[3][5]
        cube_state[45] = self.u_scans[3][6]
        cube_state[51] = self.u_scans[3][8]
        cube_state[24] = self.u_scans[4][0]
        cube_state[12] = self.u_scans[4][1]
        cube_state[18] = self.u_scans[4][2]
        cube_state[26] = self.u_scans[4][6]
        cube_state[14] = self.u_scans[4][7]
        cube_state[20] = self.u_scans[4][8]
        cube_state[33] = self.u_scans[5][0]
        cube_state[27] = self.u_scans[5][2]
        cube_state[50] = self.u_scans[5][3]
        cube_state[48] = self.u_scans[5][5]
        cube_state[35] = self.u_scans[5][6]
        cube_state[29] = self.u_scans[5][8]
        cube_state[36] = self.u_scans[6][0]
        cube_state[46] = self.u_scans[6][1]
        cube_state[38] = self.u_scans[6][2]
        cube_state[42] = self.u_scans[6][6]
        cube_state[52] = self.u_scans[6][7]
        cube_state[44] = self.u_scans[6][8]
        cube_state[21] = self.u_scans[7][3]
        cube_state[23] = self.u_scans[7][5]
        cube_state[34] = self.u_scans[8][1]
        cube_state[28] = self.u_scans[8][7]
        cube_state[25] = self.u_scans[9][3]
        cube_state[19] = self.u_scans[9][5]
        cube_state[30] = self.u_scans[10][1]
        cube_state[32] = self.u_scans[10][7]
        cube_state[39] = self.u_scans[11][3]
        cube_state[41] = self.u_scans[11][5]
        return ''.join(cube_state)
            
    def _print_cube_state_visual(self, state_str_frblud: str, title: str = "Cube State (FRBLUD)"):
        if len(state_str_frblud) != 54:
            print(f"Error printing cube state: Expected 54 chars, got {len(state_str_frblud)}")
            return

        F = state_str_frblud[0:9]; R = state_str_frblud[9:18]; B = state_str_frblud[18:27]
        L = state_str_frblud[27:36]; U = state_str_frblud[36:45]; D = state_str_frblud[45:54]

        print(f"\n--- {title} ---")
        def print_face_row(face_str, row_idx, prefix="   "): print(f"{prefix}{face_str[row_idx*3]} {face_str[row_idx*3+1]} {face_str[row_idx*3+2]}")
        # Print U face
        for r_idx in range(3): print_face_row(U, r_idx, "      ") 
        print("      ---------")
        # Print L, F, R, B faces in a row
        for r_idx in range(3): 
            l_s = f"{L[r_idx*3]} {L[r_idx*3+1]} {L[r_idx*3+2]}"
            f_s = f"{F[r_idx*3]} {F[r_idx*3+1]} {F[r_idx*3+2]}"
            r_s = f"{R[r_idx*3]} {R[r_idx*3+1]} {R[r_idx*3+2]}"
            b_s = f"{B[r_idx*3]} {B[r_idx*3+1]} {B[r_idx*3+2]}"
            print(f"{l_s:<5} | {f_s:<5} | {r_s:<5} | {b_s:<5}")
        print("      ---------")
        # Print D face
        for r_idx in range(3): print_face_row(D, r_idx, "      ") 
        print("--- End of Cube State ---\n")

# Example usage (retains original example structure)
if __name__ == '__main__':
    game = RubiksCubeGame()
    print("RubiksCubeGame instance created.")
    
    game.start_calibration_mode()
    print(f"Mode after starting calibration: {game.mode}, Step: {game.calibration_step}")
    # Assuming calibration mode would be exited or completed before scanning in a real scenario.
    # For this test, directly start scanning to see its reset behavior.
    game.mode = "idle" # Simulate exiting calibration
    game.start_scanning_mode() 
    print(f"Mode after starting scan: {game.mode}, Scan Index: {game.current_scan_idx}")

    # Test Kociemba logic with a physically solved cube string (using assumed standard center colors)
    # F=Green, R=Red, B=Blue, L=Orange, U=White, D=Yellow
    solved_frblud_physical = ("G"*9) + ("R"*9) + ("B"*9) + ("O"*9) + ("W"*9) + ("Y"*9)
    print(f"\nTesting solve with physically solved FRBLUD: {solved_frblud_physical[:10]}...")
    solution = game._solve_cube_with_kociemba(solved_frblud_physical)
    print(f"Solution for solved cube: '{solution}' (Expected empty string or similar indication)")
    if game.error_message: print(f"Error during test solve: {game.error_message}")
    game.error_message = None # Reset error for next test

    print("\n--- Simplify Moves Tests ---")
    test_moves_simple = "F B' R R R L2 L2 D' D D D F2" 
    simplified = game._simplify_cube_moves_basic(test_moves_simple)
    print(f"Original: '{test_moves_simple}' -> Simplified: '{simplified}' (Expected: F B' R' D2 F2)")

    test_moves_cancel = "F F' R R2 B B B" 
    simplified_cancel = game._simplify_cube_moves_basic(test_moves_cancel)
    print(f"Original: '{test_moves_cancel}' -> Simplified: '{simplified_cancel}' (Expected: R' B')")

    print("\n--- Auto-Squaring Test ---")
    mock_corners_slightly_off = np.array([
        [50, 50], [155, 55], [150, 160], [45, 155] 
    ], dtype=np.float32)
    
    forced_sq = game._force_ideal_square_from_detected_corners(mock_corners_slightly_off)
    if forced_sq is not None:
        print(f"Mock corners (slightly off): \n{mock_corners_slightly_off}")
        print(f"Forced ideal square: \n{forced_sq}")
        fx, fy, fw, fh = cv2.boundingRect(forced_sq.astype(np.int32))
        print(f"Forced square BRect: x={fx}, y={fy}, w={fw}, h={fh}, AR={fw/fh if fh > 0 else 0:.2f}")
    else:
        print("Forcing square failed for mock_corners_slightly_off (or deemed too small/large).")

    mock_corners_too_small = np.array([
        [100,100], [110,100], [110,110], [100,110]
    ], dtype=np.float32)
    # Ensure MIN_CONTOUR_AREA would make this "too small" for testing.
    # game.MIN_CONTOUR_AREA = 20*20 # Example: an area of 400. 10x10 square has area 100.
    # The heuristic in _force_ideal_square_from_detected_corners also uses sqrt(MIN_CONTOUR_AREA)*0.5
    # so if MIN_CONTOUR_AREA is 2000 (default base), sqrt is ~44.7, *0.5 is ~22.3.
    # Our 10x10 mock_corners_too_small (ideal_side=10) should be rejected.
    forced_sq_small = game._force_ideal_square_from_detected_corners(mock_corners_too_small)
    if forced_sq_small is None:
        print("Forcing square correctly rejected for too_small corners.")
    else:
        print(f"Forcing square INCORRECTLY PRODUCED for too_small corners: \n{forced_sq_small}")