# games/rubiks_cube_game.py

from typing import Optional, Dict, Any
import serial
import time
import cv2
import numpy as np
import kociemba
import json
import base64
import os
import random
from collections import Counter
import math # For sqrt in Delta E

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
        self.prev_contour_scan: Optional[np.ndarray] = None 

        # Zoom and Distance control parameters
        self.zoom_crop_factor: float = float(self.config.get('zoom_crop_factor', 1.0))
        self.relative_detection_distance: float = float(self.config.get('relative_detection_distance', 1.0))
        
        # Base contour area thresholds
        self.BASE_MIN_CONTOUR_AREA: int = self.config.get('base_min_contour_area', 2000)
        self.BASE_MAX_CONTOUR_AREA: int = self.config.get('base_max_contour_area', 80000)
        
        # Effective contour area thresholds
        self.MIN_CONTOUR_AREA: int = 0 
        self.MAX_CONTOUR_AREA: int = 0 
        self._update_effective_contour_areas() 

        # NEW: Contour detection cropping parameters
        self.contour_crop_enabled: bool = self.config.get('contour_crop_enabled', True)
        self.contour_crop_center_x_factor: float = float(self.config.get('contour_crop_center_x_factor', 0.5))
        self.contour_crop_center_y_factor: float = float(self.config.get('contour_crop_center_y_factor', 0.5))
        self.contour_crop_relative_width: float = float(self.config.get('contour_crop_relative_width', 0.5))
        self.contour_crop_relative_height: float = float(self.config.get('contour_crop_relative_height', 0.5))
        self._validate_contour_crop_params()

        # Calibration variables
        self.calibration_step = 0
        self.last_valid_grid_info_for_calibration: Optional[tuple] = None
        self.last_processed_frame_for_calibration: Optional[np.ndarray] = None
        self.calibration_roi_scale: float = float(self.config.get('calibration_roi_scale', 0.12))
        self.COLOR_NAMES_CALIBRATION: list = ["W", "R", "G", "Y", "O", "B"] # Standard order for UI calibration

        # Scanning variables
        self.current_scan_idx = 0
        self.SCAN_COOLDOWN: float = float(self.config.get('scan_cooldown', 0.5))
        self.MOTOR_STABILIZATION_TIME: float = float(self.config.get('motor_stabilization_time', 0.5))
        self.STABILITY_THRESHOLD: int = self.config.get('stability_threshold', 2)
        self.stability_counter: int = 0
        self.last_scan_time: float = time.time()
        self.prev_face_colors_scan: Optional[list] = None
        self.u_scans: list = [[] for _ in range(12)]
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

        # --- Color Detection Systems ---
        # 1. HSV ranges for broad contour finding (loaded from color_ranges_hsv_contour.json or defaults)
        # This was previously self.color_ranges in your original code
        self.hsv_color_ranges_for_contour: Dict[str, list] = self._load_hsv_color_ranges_for_contour()
        if not self.hsv_color_ranges_for_contour: # Should ideally always have defaults
            print("CRITICAL WARNING: No HSV ranges for contour detection. Using minimal defaults.")
            self.hsv_color_ranges_for_contour = { "W": [(np.array([0,0,100]),np.array([180,50,255]))] } # Minimal default
        
        # 2. Calibrated Lab values for precise color identification (loaded from calibrated_lab_colors.json)
        self.calibrated_lab_values: Dict[str, list] = self._load_calibrated_lab_values()
        self.DELTA_E_THRESHOLD: float = float(self.config.get('delta_e_threshold', 45.0)) # Perceptual color difference threshold

        if not self.calibrated_lab_values:
            print("INFO: Lab color values not loaded or empty. Calibration is highly recommended for accurate color detection.")
        else:
            print(f"Lab color values loaded for: {list(self.calibrated_lab_values.keys())}")
        
        # Initial validation of factors
        self.set_zoom_crop_factor(self.zoom_crop_factor)
        self.set_relative_detection_distance(self.relative_detection_distance)

    # --- Public Mode Control Methods ---
    def start_idle_mode(self):
        if self.mode in ["solving", "scrambling", "scanning"]:
            print(f"Request to switch to IDLE from {self.mode}. Stopping current operation.")
            self.stop_current_operation() # This will set mode to idle
        else:
            self.mode = "idle"
        self.status_message = "Switched to Idle mode."
        self.error_message = None
        print(self.status_message)

    def start_scanning_mode(self):
        if self.mode in ["solving", "scrambling"]:
            self.error_message = f"Cannot start scanning, robot is busy ({self.mode})."
            self.status_message = self.error_message
            print(self.error_message)
            return
        if not self.calibrated_lab_values or len(self.calibrated_lab_values) < 6: # Ensure all 6 colors are calibrated
            self.error_message = "Cannot start scanning: Lab colors are not fully calibrated (need all 6)."
            self.status_message = self.error_message
            self.mode = "idle" 
            print(self.error_message)
            return

        self.mode = "scanning"
        self._reset_scan_state()
        self.status_message = "Scanning mode started. Waiting for cube..."
        self.error_message = None
        print(self.status_message)

    def start_calibration_mode(self):
        if self.mode in ["solving", "scrambling", "scanning"]:
            self.error_message = f"Cannot start calibration, robot is busy ({self.mode}). Stop current operation first."
            self.status_message = self.error_message
            print(self.error_message)
            return

        self.mode = "calibrating"
        self.calibration_step = 0 
        self.calibrated_lab_values = {} # Clear old lab values for a fresh calibration
        self.error_message = None
        print("Calibration mode started. Calibrate White (W) first.")
        if self.calibration_step < len(self.COLOR_NAMES_CALIBRATION):
            self.status_message = f"Aim {self.COLOR_NAMES_CALIBRATION[self.calibration_step]} at green box. Press Capture."
        else: # Should not happen with step 0
            self.status_message = "Ready to calibrate."


    def _validate_contour_crop_params(self):
        self.contour_crop_center_x_factor = np.clip(self.contour_crop_center_x_factor, 0.0, 1.0)
        self.contour_crop_center_y_factor = np.clip(self.contour_crop_center_y_factor, 0.0, 1.0)
        self.contour_crop_relative_width = np.clip(self.contour_crop_relative_width, 0.1, 1.0)
        self.contour_crop_relative_height = np.clip(self.contour_crop_relative_height, 0.1, 1.0)

    def set_contour_crop_params(self, enabled: bool, cx_factor: float, cy_factor: float, rel_width: float, rel_height: float):
        self.contour_crop_enabled = enabled
        self.contour_crop_center_x_factor = float(cx_factor)
        self.contour_crop_center_y_factor = float(cy_factor)
        self.contour_crop_relative_width = float(rel_width)
        self.contour_crop_relative_height = float(rel_height)
        self._validate_contour_crop_params()
        self.status_message = (f"Contour crop: {'On' if self.contour_crop_enabled else 'Off'}, "
                               f"C=({self.contour_crop_center_x_factor:.2f},{self.contour_crop_center_y_factor:.2f}), "
                               f"Size=({self.contour_crop_relative_width:.2f}x{self.contour_crop_relative_height:.2f})")
        print(self.status_message)

    def _update_effective_contour_areas(self):
        if self.relative_detection_distance <= 0: squared_factor = 0.01 
        else: squared_factor = self.relative_detection_distance ** 2
        self.MIN_CONTOUR_AREA = int(self.BASE_MIN_CONTOUR_AREA / squared_factor)
        self.MAX_CONTOUR_AREA = int(self.BASE_MAX_CONTOUR_AREA / squared_factor)
        self.MIN_CONTOUR_AREA = max(100, self.MIN_CONTOUR_AREA) 
        self.MAX_CONTOUR_AREA = max(self.MIN_CONTOUR_AREA + 100, self.MAX_CONTOUR_AREA) 
        total_pixels = self.WINDOW_SIZE[0] * self.WINDOW_SIZE[1]
        self.MAX_CONTOUR_AREA = min(self.MAX_CONTOUR_AREA, int(total_pixels * 0.95)) 

    def set_zoom_crop_factor(self, factor: float):
        original_factor = factor
        if factor < 1.0: factor = 1.0
        elif factor > 10.0: factor = 10.0
        if original_factor != factor: print(f"Info: Zoom crop factor ({original_factor}) corrected to {factor}.")
        self.zoom_crop_factor = factor
        self.status_message = f"Zoom crop factor set to {self.zoom_crop_factor:.2f}"

    def set_relative_detection_distance(self, factor: float):
        original_factor = factor
        if factor <= 0.1: factor = 0.1
        elif factor > 5.0: factor = 5.0
        if original_factor != factor: print(f"Info: Relative detection distance ({original_factor}) corrected to {factor}.")
        self.relative_detection_distance = factor
        self._update_effective_contour_areas()
        self.status_message = (f"Detection distance factor: {self.relative_detection_distance:.2f}. "
                               f"Areas: {self.MIN_CONTOUR_AREA}-{self.MAX_CONTOUR_AREA}")

    def set_delta_e_threshold(self, threshold: float):
        original_threshold = threshold
        if threshold < 5.0: threshold = 5.0
        elif threshold > 100.0: threshold = 100.0
        if original_threshold != threshold: print(f"Info: Delta E threshold ({original_threshold}) corrected to {threshold}.")
        self.DELTA_E_THRESHOLD = threshold
        self.status_message = f"Delta E threshold set to {self.DELTA_E_THRESHOLD:.1f}"

    def init_serial(self) -> bool:
        try:
            if self.serial_connection and self.serial_connection.is_open: self.serial_connection.close()
            self.serial_connection = serial.Serial(self.serial_port, self.serial_baudrate, timeout=1)
            time.sleep(2); self.status_message = f"Serial connected: {self.serial_port}"; print(self.status_message)
            self.error_message = None; return True
        except serial.SerialException as e: 
            self.error_message = f"Serial Error: {str(e)}"; self.status_message = f"Failed to connect: {self.serial_port}"
            self.serial_connection = None; print(self.error_message); return False
        
    def set_websocket(self, websocket: Any): self.websocket = websocket; print("WebSocket instance set.")
        
    def send_arduino_command(self, cmd: str, wait_for_ack: bool = True, is_solution: bool = False) -> bool:
        # This method is from your original code, unchanged.
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
            self.serial_connection.write(f"{cmd}\n".encode('utf-8')); self.serial_connection.flush(); self.last_motor_move_time = time.time() 
            if not wait_for_ack: return True
            timeout_duration = 90 if is_solution else 15 + cmd.count(' ') * 2; start_wait_time = time.time(); ack_buffer = ""
            while time.time() - start_wait_time < timeout_duration:
                if self.stop_requested and is_solution: print(f"Stop requested during Arduino command: {cmd}"); self.stop_requested=False; return False 
                if self.serial_connection.in_waiting > 0:
                    try:
                        ack_buffer += self.serial_connection.read(self.serial_connection.in_waiting).decode("latin-1", errors='replace')
                        while '\n' in ack_buffer:
                            line, ack_buffer = ack_buffer.split('\n', 1); line = line.strip()
                            if not line: continue
                            if "completed" in line.lower() or "executed" in line.lower():
                                print(f"Command '{cmd[:30]}...' ACKNOWLEDGED as completed by Arduino.");
                                if is_solution:
                                    print("SOLUTION execution reported complete by Arduino.")
                                    self.mode="idle"; self.status_message="Solution completed successfully."; self.solution=None; self.current_solve_move_index = self.total_solve_moves if self.total_solve_moves > 0 else 0
                                return True
                            elif "error" in line.lower(): self.error_message=f"Arduino error for '{cmd[:30]}...': {line}"; print(self.error_message); return False
                    except Exception as e: print(f"Error reading/decoding from Arduino for '{cmd[:30]}...': {e}. Buffer: {ack_buffer}"); ack_buffer = "" 
                time.sleep(0.02) 
            self.error_message = f"TIMEOUT waiting for Arduino ack for '{cmd[:30]}...' (waited {timeout_duration}s)."; print(self.error_message); return False
        except serial.SerialTimeoutException: self.error_message=f"Serial write timeout for command '{cmd[:30]}...'. Check connection."; print(self.error_message); self.init_serial(); return False
        except Exception as e: self.error_message=f"EXCEPTION sending Arduino command '{cmd[:30]}...': {type(e).__name__} - {e}"; print(self.error_message); self.init_serial(); return False

    def _send_compound_move(self, move: str): # From original
        if move: self.send_arduino_command(move, wait_for_ack=True)

    def process_frame(self, frame_data: bytes) -> Dict[str, Any]:
        try:
            nparr = np.frombuffer(frame_data, np.uint8)
            raw_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if raw_frame is None: raise ValueError("Failed to decode frame data from client.")
            
            self.last_processed_frame_for_calibration = raw_frame.copy() 
            
            input_frame_for_resize = raw_frame
            if self.zoom_crop_factor > 1.0: # zoom_crop_factor is already checked for None in __init__ or setter
                h_orig, w_orig = raw_frame.shape[:2]
                crop_w = int(w_orig / self.zoom_crop_factor); crop_h = int(h_orig / self.zoom_crop_factor)
                crop_w = max(1, crop_w); crop_h = max(1, crop_h)
                start_x = (w_orig-crop_w)//2; start_y = (h_orig-crop_h)//2
                end_x = start_x+crop_w; end_y = start_y+crop_h
                start_x=np.clip(start_x,0,w_orig-1); start_y=np.clip(start_y,0,h_orig-1)
                end_x=np.clip(end_x,start_x+1,w_orig); end_y=np.clip(end_y,start_y+1,h_orig)
                if end_y > start_y and end_x > start_x: input_frame_for_resize = raw_frame[start_y:end_y, start_x:end_x]

            frame_for_processing = cv2.resize(input_frame_for_resize, self.WINDOW_SIZE, interpolation=cv2.INTER_AREA)
            self.current_frame_for_detection = frame_for_processing.copy() 
            display_overlay_frame = frame_for_processing.copy() 

            # Draw contour crop ROI if enabled
            if self.contour_crop_enabled and self.mode != "calibrating":
                frame_h, frame_w = display_overlay_frame.shape[:2]
                crop_rect_w = int(frame_w * self.contour_crop_relative_width)
                crop_rect_h = int(frame_h * self.contour_crop_relative_height)
                crop_rect_x = int(frame_w * self.contour_crop_center_x_factor - crop_rect_w / 2)
                crop_rect_y = int(frame_h * self.contour_crop_center_y_factor - crop_rect_h / 2)
                crop_rect_x1=np.clip(crop_rect_x,0,frame_w-1); crop_rect_y1=np.clip(crop_rect_y,0,frame_h-1)
                crop_rect_x2=np.clip(crop_rect_x+crop_rect_w,crop_rect_x1+1,frame_w); crop_rect_y2=np.clip(crop_rect_y+crop_rect_h,crop_rect_y1+1,frame_h)
                if crop_rect_x2 > crop_rect_x1 and crop_rect_y2 > crop_rect_y1:
                     cv2.rectangle(display_overlay_frame, (crop_rect_x1, crop_rect_y1), (crop_rect_x2, crop_rect_y2), (255, 0, 255), 1)

            if self.mode == "calibrating": display_overlay_frame = self._draw_calibration_overlay(display_overlay_frame)
            elif self.mode == "scanning": display_overlay_frame = self._process_scanning_step(display_overlay_frame) 
            elif self.mode in ["solving", "scrambling"]: display_overlay_frame = self._draw_solving_overlay(display_overlay_frame)
            else: 
                status_to_show = f"Mode: {self.mode.capitalize()}"
                if self.status_message and self.status_message != status_to_show : status_to_show += f" - {self.status_message}"
                cv2.putText(display_overlay_frame, status_to_show, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,80), 1)

            _, buffer = cv2.imencode('.jpg', display_overlay_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            current_state_payload = self.get_state(); current_state_payload["processed_frame"] = frame_b64 
            return current_state_payload
        except Exception as e:
            self.error_message = f"Frame proc error: {type(e).__name__} - {e}"; print(f"! CRITICAL ERROR in process_frame: {self.error_message}")
            error_display_frame = np.zeros((self.WINDOW_SIZE[1], self.WINDOW_SIZE[0], 3), dtype=np.uint8)
            cv2.putText(error_display_frame, "ERROR", (self.WINDOW_SIZE[0]//2-50, self.WINDOW_SIZE[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            _, buffer = cv2.imencode('.jpg', error_display_frame); error_frame_b64 = base64.b64encode(buffer).decode('utf-8')
            error_state_payload = self.get_state(); error_state_payload["mode"]="error"; error_state_payload["error_message"]=self.error_message 
            error_state_payload["status_message"]="Frame processing failed."; error_state_payload["processed_frame"]=error_frame_b64
            return error_state_payload

    def get_state(self) -> Dict[str, Any]:
        error_to_send = self.error_message
        if self.error_message: self.error_message = None 
        return {
            "mode": self.mode, "status_message": self.status_message, "error_message": error_to_send,
            "calibration_step": self.calibration_step,
            "current_color_calibrating": self.COLOR_NAMES_CALIBRATION[self.calibration_step] if self.mode == "calibrating" and self.calibration_step < len(self.COLOR_NAMES_CALIBRATION) else None,
            "scan_index": self.current_scan_idx,
            "solve_move_index": self.current_solve_move_index, "total_solve_moves": self.total_solve_moves,
            "solution_preview": self.solution[:30]+"..." if self.solution and len(self.solution)>30 else self.solution,
            "serial_connected": self.serial_connection is not None and self.serial_connection.is_open,
            "zoom_crop_factor": self.zoom_crop_factor, "relative_detection_distance": self.relative_detection_distance,
            "effective_min_contour_area": self.MIN_CONTOUR_AREA, "effective_max_contour_area": self.MAX_CONTOUR_AREA,
            "delta_e_threshold": self.DELTA_E_THRESHOLD, "lab_colors_calibrated_count": len(self.calibrated_lab_values), # Send count
            "contour_crop_enabled": self.contour_crop_enabled,
            "contour_crop_center_x_factor": self.contour_crop_center_x_factor,
            "contour_crop_center_y_factor": self.contour_crop_center_y_factor,
            "contour_crop_relative_width": self.contour_crop_relative_width,
            "contour_crop_relative_height": self.contour_crop_relative_height,
        }

    def _find_best_cube_contour(self, frame_to_process: np.ndarray) -> Optional[np.ndarray]:
        orig_h, orig_w = frame_to_process.shape[:2]
        search_frame = frame_to_process; offset_x, offset_y = 0, 0

        if self.contour_crop_enabled:
            crop_w = int(orig_w * self.contour_crop_relative_width); crop_h = int(orig_h * self.contour_crop_relative_height)
            crop_x = int(orig_w * self.contour_crop_center_x_factor - crop_w/2); crop_y = int(orig_h * self.contour_crop_center_y_factor - crop_h/2)
            crop_x=np.clip(crop_x,0,orig_w-1); crop_y=np.clip(crop_y,0,orig_h-1)
            crop_w=np.clip(crop_w,1,orig_w-crop_x); crop_h=np.clip(crop_h,1,orig_h-crop_y)
            if crop_w>0 and crop_h>0: search_frame=frame_to_process[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]; offset_x,offset_y=crop_x,crop_y
            
        processed_bgr = cv2.bilateralFilter(search_frame, d=7, sigmaColor=35, sigmaSpace=35)
        hsv = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2HSV)
        combined_mask = np.zeros((search_frame.shape[0], search_frame.shape[1]), dtype=np.uint8)
        
        # Use self.hsv_color_ranges_for_contour (formerly self.color_ranges)
        for _, ranges_list in self.hsv_color_ranges_for_contour.items(): 
            for lower, upper in ranges_list: 
                mask_part = cv2.inRange(hsv, lower, upper); combined_mask |= mask_part

        kernel = np.ones((3,3), np.uint8) # Original kernel size was 3
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_contour_approx = None; max_area_found = 0 

        for contour in contours:
            area = cv2.contourArea(contour)
            if self.MIN_CONTOUR_AREA < area < self.MAX_CONTOUR_AREA:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.03 * peri, True) 
                if len(approx) >= 4 and len(approx) <= 8: 
                    x,y,w,h = cv2.boundingRect(approx)
                    if w==0 or h==0: continue
                    aspect_ratio = float(w)/h; hull=cv2.convexHull(contour); hull_area=cv2.contourArea(hull)
                    solidity = float(area)/hull_area if hull_area > 0 else 0.0
                    if 0.70 < aspect_ratio < 1.30 and solidity > 0.70:
                        if area > max_area_found: max_area_found=area; best_contour_approx=approx 
        
        if best_contour_approx is not None and self.contour_crop_enabled and (offset_x > 0 or offset_y > 0):
            best_contour_approx[:, 0, 0] += offset_x
            best_contour_approx[:, 0, 1] += offset_y
        return best_contour_approx

    def _detect_corners(self, contour_approx: Optional[np.ndarray]) -> Optional[np.ndarray]: # From original
        if contour_approx is None: return None
        if len(contour_approx) == 4: corners = contour_approx.reshape(4,2).astype(np.float32)
        else:
            hull = cv2.convexHull(contour_approx); epsilon_hull = 0.05*cv2.arcLength(hull,True); approx_hull=cv2.approxPolyDP(hull,epsilon_hull,True)
            if len(approx_hull)==4: corners=approx_hull.reshape(4,2).astype(np.float32)
            else: rect=cv2.minAreaRect(hull); corners=cv2.boxPoints(rect).astype(np.float32)
        if self.current_frame_for_detection is None: return None
        gray=cv2.cvtColor(self.current_frame_for_detection,cv2.COLOR_BGR2GRAY);h_img,w_img=gray.shape[:2]
        for i in range(corners.shape[0]): corners[i,0]=np.clip(corners[i,0],0,w_img-1); corners[i,1]=np.clip(corners[i,1],0,h_img-1)
        criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.01)
        try: refined_corners=cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria) 
        except cv2.error: refined_corners=corners 
        for i in range(refined_corners.shape[0]): refined_corners[i,0]=np.clip(refined_corners[i,0],0,w_img-1); refined_corners[i,1]=np.clip(refined_corners[i,1],0,h_img-1)
        pts_sorted_y=refined_corners[np.argsort(refined_corners[:,1])]; top_pts=pts_sorted_y[:2][np.argsort(pts_sorted_y[:2,0])] 
        bottom_pts=pts_sorted_y[2:][np.argsort(pts_sorted_y[2:,0])]; return np.array([top_pts[0],top_pts[1],bottom_pts[1],bottom_pts[0]],dtype=np.float32)

    def _predict_square_length(self, corners: Optional[np.ndarray]) -> int: # From original
        if corners is None or len(corners)!=4: return 50 
        dists=[np.linalg.norm(corners[i]-corners[(i+1)%4]) for i in range(4)]; return max(30,int(np.mean(dists)))

    def _perspective_transform(self, frame_to_transform: np.ndarray, contour_approx: np.ndarray) -> Optional[np.ndarray]: # From original
        corners=self._detect_corners(contour_approx);
        if corners is None: return None
        side_length=self._predict_square_length(corners)
        if side_length<20: return None
        dst_points=np.array([[0,0],[side_length-1,0],[side_length-1,side_length-1],[0,side_length-1]],dtype=np.float32)
        try: M=cv2.getPerspectiveTransform(corners,dst_points); return cv2.warpPerspective(frame_to_transform,M,(side_length,side_length),flags=cv2.INTER_LANCZOS4)
        except cv2.error as e: print(f"! OpenCV Error in perspective transform: {e}"); return None

    # Renamed from _load_color_ranges_from_file
    def _load_hsv_color_ranges_for_contour(self, filename: str = "color_ranges_hsv_contour.json") -> Dict[str, list]:
        # This is the original _load_color_ranges_from_file logic, for HSV contour ranges.
        default_hsv_ranges = {
            "W": [(np.array([0,0,150]),np.array([180,70,255]))], "R": [(np.array([0,80,70]),np.array([10,255,255])),(np.array([160,80,70]),np.array([179,255,255]))],
            "G": [(np.array([35,60,60]),np.array([85,255,255]))], "Y": [(np.array([18,80,100]),np.array([35,255,255]))],
            "O": [(np.array([3,80,70]),np.array([20,255,255]))], "B": [(np.array([80,60,60]),np.array([135,255,255]))]
        }
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f: loaded_data = json.load(f)
                parsed_ranges = {}
                for color, ranges_data_list in loaded_data.items(): 
                    if isinstance(ranges_data_list, list):
                        parsed_ranges[color] = []
                        for r_data_pair in ranges_data_list:
                            if isinstance(r_data_pair, list) and len(r_data_pair) == 2:
                                lower = np.array(r_data_pair[0], dtype=np.uint8); upper = np.array(r_data_pair[1], dtype=np.uint8)
                                if lower.shape==(3,) and upper.shape==(3,): parsed_ranges[color].append((lower, upper))
                                else: print(f"Warn: Invalid shape for HSV range (color '{color}') in {filename}.")
                            else: print(f"Warn: Invalid HSV range pair format (color '{color}') in {filename}.")
                    else: print(f"Warn: Invalid data structure for HSV color '{color}' in {filename}.")
                if parsed_ranges: print(f"HSV ranges for contour loaded from {filename}."); return parsed_ranges
                else: print(f"Warn: {filename} empty/invalid. Using default HSV ranges for contour."); return default_hsv_ranges
            except Exception as e: print(f"Err loading HSV ranges from {filename}: {e}. Using defaults."); return default_hsv_ranges
        else: print(f"{filename} not found. Using default HSV ranges for contour detection."); return default_hsv_ranges
        
    # Renamed from _save_color_ranges_to_file
    def _save_hsv_color_ranges_for_contour(self, filename: str = "color_ranges_hsv_contour.json"):
        # This is the original _save_color_ranges_to_file logic, for HSV contour ranges.
        serializable_ranges = {}
        for color, ranges_list_tuples in self.hsv_color_ranges_for_contour.items(): # Use the renamed attribute
            serializable_ranges[color] = [(lower_np.tolist(), upper_np.tolist()) for lower_np, upper_np in ranges_list_tuples]
        try:
            with open(filename, 'w') as f: json.dump(serializable_ranges, f, indent=4)
            print(f"HSV contour color ranges saved to {filename}")
        except Exception as e: print(f"Error saving HSV contour color ranges to {filename}: {e}")

    def _load_calibrated_lab_values(self, filename: str = "calibrated_lab_colors.json") -> Dict[str, list]:
        if os.path.exists(filename):
            try:
                with open(filename,'r') as f: loaded_data=json.load(f)
                valid_data = {k.upper():[float(v) for v in val_list] for k,val_list in loaded_data.items() if isinstance(val_list,list) and len(val_list)==3 and all(isinstance(v,(int,float)) for v in val_list)}
                return valid_data
            except Exception as e: print(f"Err loading Lab values from {filename}: {e}")
        return {}

    def _save_calibrated_lab_values(self, filename: str = "calibrated_lab_colors.json"):
        try:
            with open(filename,'w') as f: json.dump(self.calibrated_lab_values, f, indent=4)
            print(f"Calibrated Lab values saved to {filename}")
        except Exception as e: print(f"Err saving Lab values to {filename}: {e}")

    def _draw_calibration_overlay(self, display_frame: np.ndarray) -> np.ndarray: # Updated for Lab
        h,w=display_frame.shape[:2]; roi_dim=max(20,int(min(w,h)*self.calibration_roi_scale)) 
        roi_xs=(w-roi_dim)//2; roi_ys=(h-roi_dim)//2; roi_xe=roi_xs+roi_dim; roi_ye=roi_ys+roi_dim
        self.last_valid_grid_info_for_calibration=(roi_xs,roi_ys,roi_xe,roi_ye)
        cv2.rectangle(display_frame,(roi_xs,roi_ys),(roi_xe,roi_ye),(0,255,0),2)
        cx,cy=roi_xs+roi_dim//2,roi_ys+roi_dim//2
        cv2.line(display_frame,(cx-5,cy),(cx+5,cy),(0,255,0),1); cv2.line(display_frame,(cx,cy-5),(cx,cy+5),(0,255,0),1)
        if self.calibration_step < len(self.COLOR_NAMES_CALIBRATION):
            instr = f"Aim {self.COLOR_NAMES_CALIBRATION[self.calibration_step]} for Lab calib. Press Capture."
            cv2.putText(display_frame, instr, (10,25), cv2.FONT_HERSHEY_SIMPLEX,0.45,(220,220,220),1) # Slightly smaller font
        else: cv2.putText(display_frame,"Lab Calibration Complete!",(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        return display_frame

    def _process_scanning_step(self, display_frame_overlay: np.ndarray) -> np.ndarray:
        frame_to_detect_on = self.current_frame_for_detection 
        if frame_to_detect_on is None: cv2.putText(display_frame_overlay, "Err: No detection frame", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1); return display_frame_overlay

        best_contour_approx = self._find_best_cube_contour(frame_to_detect_on)
        cube_detected = best_contour_approx is not None; status_color = (220,220,220) 

        if cube_detected: # Logic from original
            cv2.drawContours(display_frame_overlay, [best_contour_approx], -1, (0,255,0),1)
            xc,yc,wc,hc = cv2.boundingRect(best_contour_approx)
            if wc>0 and hc>0:
                cellw,cellh = wc//3, hc//3
                for i in range(1,3): cv2.line(display_frame_overlay,(xc,yc+i*cellh),(xc+wc,yc+i*cellh),(0,80,0),1); cv2.line(display_frame_overlay,(xc+i*cellw,yc),(xc+i*cellw,yc+hc),(0,80,0),1)
            if self.prev_contour_scan is not None:
                try: 
                    Mc=cv2.moments(best_contour_approx); Mp=cv2.moments(self.prev_contour_scan)
                    if Mc["m00"]>0 and Mp["m00"]>0:
                        cc=(int(Mc["m10"]/Mc["m00"]),int(Mc["m01"]/Mc["m00"])); cp=(int(Mp["m10"]/Mp["m00"]),int(Mp["m01"]/Mp["m00"]))
                        if np.sqrt((cc[0]-cp[0])**2 + (cc[1]-cp[1])**2) < 10: self.stability_counter+=1
                        else: self.stability_counter=0
                    else: self.stability_counter=0
                except: self.stability_counter=0 
            else: self.stability_counter=1 
            self.prev_contour_scan = best_contour_approx.copy() if best_contour_approx is not None else None
        else: self.stability_counter=0; self.prev_contour_scan=None

        curr_t=time.time(); t_last_scan=curr_t-self.last_scan_time; t_motor=curr_t-self.last_motor_move_time
        scan_status_txt = f"Scan {self.current_scan_idx+1}/12: "; ready_to_cap = False

        # Check for Lab calibration before allowing capture
        if not self.calibrated_lab_values or len(self.calibrated_lab_values) < 6: scan_status_txt += "NEEDS FULL LAB CALIBRATION! "
        elif not cube_detected: scan_status_txt += "Detecting..."
        elif t_motor < self.MOTOR_STABILIZATION_TIME: scan_status_txt += f"Motor ({self.MOTOR_STABILIZATION_TIME-t_motor:.1f}s)"
        elif t_last_scan < self.SCAN_COOLDOWN: scan_status_txt += f"Cooldown ({self.SCAN_COOLDOWN-t_last_scan:.1f}s)"
        elif self.stability_counter < self.STABILITY_THRESHOLD: scan_status_txt += f"Stabilizing ({self.stability_counter}/{self.STABILITY_THRESHOLD})"
        else: scan_status_txt += "CAPTURING..."; ready_to_cap=True; status_color=(0,255,0) 
        
        cv2.putText(display_frame_overlay, scan_status_txt, (10,25), cv2.FONT_HERSHEY_SIMPLEX,0.5,status_color,1); self.status_message=scan_status_txt 

        if ready_to_cap and self.current_scan_idx < 12 and self.calibrated_lab_values and len(self.calibrated_lab_values) == 6:
            # Rest of scanning logic from original, using the new _detect_color (Lab-based)
            if best_contour_approx is None: self.status_message=scan_status_txt.replace("CAPTURING...","Contour Lost"); return display_frame_overlay
            warped_face = self._perspective_transform(frame_to_detect_on, best_contour_approx)
            if warped_face is not None and warped_face.size > 0:
                grid_s = min(warped_face.shape[0],warped_face.shape[1]); cell_s = max(1,grid_s//3)
                if grid_s < 21 or cell_s < 7: self.status_message=scan_status_txt.replace("CAPTURING...","Warped small"); return display_frame_overlay 
                face_colors=[]; valid_rois=True
                for r_idx in range(3): 
                    for c_idx in range(3): 
                        ys,ye=r_idx*cell_s,(r_idx+1)*cell_s; xs,xe=c_idx*cell_s,(c_idx+1)*cell_s; pad=max(1,cell_s//6) 
                        roi_ys=min(ys+pad,warped_face.shape[0]-1); roi_ye=max(roi_ys+1,min(ye-pad,warped_face.shape[0])) 
                        roi_xs=min(xs+pad,warped_face.shape[1]-1); roi_xe=max(roi_xs+1,min(xe-pad,warped_face.shape[1])) 
                        if roi_ye<=roi_ys or roi_xe<=roi_xs: face_colors.append('X'); valid_rois=False; break
                        sticker_roi=warped_face[roi_ys:roi_ye,roi_xs:roi_xe]
                        if sticker_roi.size==0: face_colors.append('X'); valid_rois=False; break
                        face_colors.append(self._detect_color(sticker_roi)) # Uses new Lab-based _detect_color
                    if not valid_rois: break
                
                non_center_unk = any(c=='X' for i,c in enumerate(face_colors) if i!=4 and len(face_colors)==9)
                
                if valid_rois and len(face_colors)==9:
                    if non_center_unk: self.status_message=scan_status_txt.replace("CAPTURING...",f"Unk.clr(edge/corn): {''.join(face_colors)}")
                    elif self.prev_face_colors_scan is None or tuple(face_colors) != tuple(self.prev_face_colors_scan):
                        self.u_scans[self.current_scan_idx]=list(face_colors); self.prev_face_colors_scan=list(face_colors)
                        print(f"Scan {self.current_scan_idx+1} OK: {''.join(face_colors)}")
                        self.current_scan_idx+=1; self.stability_counter=0; self.last_scan_time=time.time() 
                        if self.current_scan_idx < 12:
                            next_rot = self.rotation_sequence[self.current_scan_idx] 
                            if next_rot: self.status_message=f"Scan {self.current_scan_idx} done. Rotating..."; self._send_compound_move(next_rot) 
                        else: 
                            self.status_message="All scans done. Finalizing..."; self._send_compound_move("B F'") 
                            cube_state = self._construct_cube_state_from_scans() # Original logic
                            if cube_state:
                                sol_str = self._solve_cube_with_kociemba(cube_state) # Original logic
                                if sol_str is not None: 
                                    self.solution=sol_str; self.mode="solving"; self.total_solve_moves=len(sol_str.split()) if sol_str else 0; self.current_solve_move_index=0 
                                    self.status_message=f"Solution ({self.total_solve_moves}m). Sending..."; self.send_arduino_command(f"SOLUTION:{self.solution}",is_solution=True)
                                else: self.mode="idle"; self.error_message="Could not solve."; self.status_message=self.error_message; self._reset_scan_state() 
                            else: self.mode="idle"; self.error_message="Cube construction failed."; self.status_message=self.error_message; self._reset_scan_state()
                    else: self.status_message=scan_status_txt.replace("CAPTURING...","Duplicate")
                elif not valid_rois: self.status_message=scan_status_txt.replace("CAPTURING...","Invalid ROIs")
                else: self.status_message=scan_status_txt.replace("CAPTURING...","Sticker count err")
            else: self.status_message=scan_status_txt.replace("CAPTURING...","Warp failed")
        return display_frame_overlay

    def _draw_solving_overlay(self, display_frame: np.ndarray) -> np.ndarray: # From original
        mode_txt = self.mode.capitalize(); prog_txt = f"{mode_txt}: "
        if self.total_solve_moves > 0: prog_txt += f"{self.current_solve_move_index}/{self.total_solve_moves}"
        else: prog_txt += "Starting..."
        if self.mode == "solving" and self.solution: cv2.putText(display_frame, f"Sol: {self.solution[:25]+('...'if len(self.solution)>25 else '')}", (10,50),cv2.FONT_HERSHEY_SIMPLEX,0.4,(200,200,0),1)
        cv2.putText(display_frame, prog_txt, (10,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(220,220,50),1); return display_frame

    def _detect_color(self, roi_img: np.ndarray) -> str: # New Lab-based detection
        if roi_img is None or roi_img.size==0 or roi_img.shape[0]<3 or roi_img.shape[1]<3: return 'X' 
        if not self.calibrated_lab_values or len(self.calibrated_lab_values) < 6 : return 'X' # Need full calibration

        processed_roi = cv2.medianBlur(roi_img,3) if min(roi_img.shape[:2])>3 else roi_img # Keep blur minimal
        lab_roi = cv2.cvtColor(processed_roi,cv2.COLOR_BGR2LAB)
        
        h,w=lab_roi.shape[:2]; chs,che=max(0,int(h*0.3)),min(h,int(h*0.7)) if h>4 else h # Central 40% (0.3 to 0.7)
        cws,cwe=max(0,int(w*0.3)),min(w,int(w*0.7)) if w>4 else w
        lab_sample = lab_roi[chs:che,cws:cwe] if chs<che and cws<cwe else lab_roi
        if lab_sample.size==0: return 'X'
        
        # OpenCV LAB values are L:[0,255], a:[0,255], b:[0,255]
        roi_lab_L = np.median(lab_sample[:,:,0])
        roi_lab_a = np.median(lab_sample[:,:,1])
        roi_lab_b = np.median(lab_sample[:,:,2])
        roi_lab_color = np.array([roi_lab_L, roi_lab_a, roi_lab_b])
        
        best_match_color='X'; min_delta_e=float('inf')

        for color_name, calibrated_lab_list in self.calibrated_lab_values.items():
            calibrated_lab_val = np.array(calibrated_lab_list)
            delta_e = math.sqrt(np.sum((roi_lab_color - calibrated_lab_val)**2))
            if delta_e < min_delta_e:
                min_delta_e = delta_e
                best_match_color = color_name
        
        if min_delta_e <= self.DELTA_E_THRESHOLD:
            return best_match_color
        else:
            # For debugging: print(f"Unknown color: ROI Lab({roi_lab_color}), Best match {best_match_color}, DeltaE {min_delta_e:.1f}")
            return 'X'

    def capture_calibration_color(self) -> bool: # Updated for Lab calibration
        if self.mode!="calibrating" or self.calibration_step>=len(self.COLOR_NAMES_CALIBRATION): self.error_message="Not in calib. mode or complete."; self.status_message=self.error_message; return False
        if self.last_processed_frame_for_calibration is None: self.error_message="No raw frame for cal."; self.status_message=self.error_message; return False
        if self.last_valid_grid_info_for_calibration is None: self.error_message="Cal. ROI not set."; self.status_message=self.error_message; return False
        try:
            resized_frame = cv2.resize(self.last_processed_frame_for_calibration, self.WINDOW_SIZE, interpolation=cv2.INTER_AREA)
            xs,ys,xe,ye = self.last_valid_grid_info_for_calibration; hcal,wcal=resized_frame.shape[:2]
            xs=np.clip(xs,0,wcal-1);ys=np.clip(ys,0,hcal-1);xe=np.clip(xe,xs+1,wcal);ye=np.clip(ye,ys+1,hcal)
            if ye<=ys or xe<=xs: self.error_message="Cal. ROI invalid clip."; self.status_message=self.error_message; return False
            roi_to_calibrate = resized_frame[ys:ye,xs:xe]
            if roi_to_calibrate.size==0: self.error_message="Cal. ROI empty."; self.status_message=self.error_message; return False
            
            roi_filt = cv2.bilateralFilter(roi_to_calibrate, d=5, sigmaColor=30, sigmaSpace=30)
            lab_roi = cv2.cvtColor(roi_filt, cv2.COLOR_BGR2LAB) # Convert to LAB
            hr,wr=lab_roi.shape[:2]; chs,che=max(0,int(hr*0.25)),min(hr,int(hr*0.75)); cws,cwe=max(0,int(wr*0.25)),min(wr,int(wr*0.75))
            lab_sample_area = lab_roi[chs:che,cws:cwe] if chs<che and cws<cwe else lab_roi
            if lab_sample_area.size==0: self.error_message="Cal. sample empty."; self.status_message=self.error_message; return False

            median_l = float(np.median(lab_sample_area[:,:,0]))
            median_a = float(np.median(lab_sample_area[:,:,1]))
            median_b = float(np.median(lab_sample_area[:,:,2]))
            
            current_color_to_cal = self.COLOR_NAMES_CALIBRATION[self.calibration_step]
            self.calibrated_lab_values[current_color_to_cal] = [median_l, median_a, median_b]
            print(f"Calibrated {current_color_to_cal} (Lab): L={median_l:.1f}, a={median_a:.1f}, b={median_b:.1f}")

            self.calibration_step+=1
            if self.calibration_step >= len(self.COLOR_NAMES_CALIBRATION):
                self._save_calibrated_lab_values() # Save Lab values
                self.mode="idle"
                self.status_message="Lab calibration complete & saved."
                # Optionally, save default HSV ranges if they were modified or if file doesn't exist
                # self._save_hsv_color_ranges_for_contour() 
            else:
                next_color = self.COLOR_NAMES_CALIBRATION[self.calibration_step]
                self.status_message=f"Calibrated {current_color_to_cal}. Next: {next_color} for Lab."
            self.error_message=None; return True
        except Exception as e:
            self.error_message=f"Cal. capture err: {type(e).__name__} - {e}"; self.status_message="Cal. capture failed."; print(f"Err in capture_cal_color: {self.error_message}"); return False

    def _reset_scan_state(self): # From original
        self.current_scan_idx=0; self.u_scans=[[] for _ in range(12)]; self.prev_face_colors_scan=None
        self.prev_contour_scan=None; self.stability_counter=0; self.last_scan_time=time.time()
        self.last_motor_move_time=time.time(); print("Scan state has been reset.")

    def _reset_solve_state(self): # From original
        self.solution=None; self.current_solve_move_index=0; self.total_solve_moves=0; print("Solve state has been reset.")

    def _validate_kociemba_string(self, s: str, name: str = "Cube String"): # From original
        if not isinstance(s, str): raise ValueError(f"{name} must be a string.")
        if len(s) != 54: raise ValueError(f"{name} must be 54 chars long, got {len(s)}: '{s[:10]}...'")
        allowed_chars = {'U','R','F','D','L','B'}; counts = Counter(s)
        if not all(c in allowed_chars for c in s): raise ValueError(f"{name} contains invalid chars: {set(c for c in s if c not in allowed_chars)}. String: '{s[:10]}...'")
        if not all(count == 9 for count in counts.values()): raise ValueError(f"{name} must have 9 of each URFDLB char. Counts: {counts}")
        centers = [s[4],s[13],s[22],s[31],s[40],s[49]]
        if len(set(centers))!=6: raise ValueError(f"{name} center pieces not unique: {centers}")
        if centers[0]!='U' or centers[1]!='R' or centers[2]!='F' or centers[3]!='D' or centers[4]!='L' or centers[5]!='B': raise ValueError(f"{name} center pieces not URFDLB order. Got: {centers}")
    
    def _validate_cube(self, cube, order_name): # From original
        if len(cube)!=54: raise ValueError(f"{order_name} must be 54 characters")
        counts=Counter(cube)
        if len(counts)!=6 or any(count!=9 for count in counts.values()): raise ValueError(f"{order_name} invalid: {counts} (need 9 of each of 6 colors)")

    def _remap_colors_to_kociemba(self, cube_frblud): # From original
        self._validate_cube(cube_frblud, "FRBLUD")
        centers=[cube_frblud[i] for i in [4,13,22,31,40,49]] # F,R,B,L,U,D centers in FRBLUD string order
        if len(set(centers))!=6: raise ValueError(f"Cannot remap: FRBLUD center colors are not unique: {centers}.")
        # Maps actual color on a Kociemba-face-position to Kociemba's URFDLB characters
        color_map={
            centers[4]: 'U', # Actual color of U-center (cube_frblud[40]) maps to 'U'
            centers[1]: 'R', # Actual color of R-center (cube_frblud[13]) maps to 'R'
            centers[0]: 'F', # Actual color of F-center (cube_frblud[4])  maps to 'F'
            centers[5]: 'D', # Actual color of D-center (cube_frblud[49]) maps to 'D'
            centers[3]: 'L', # Actual color of L-center (cube_frblud[31]) maps to 'L'
            centers[2]: 'B'  # Actual color of B-center (cube_frblud[22]) maps to 'B'
        }
        return color_map, ''.join(color_map[c] for c in cube_frblud)

    def _remap_cube_to_kociemba(self, cube_frblud_remapped): # From original
        # Takes FRBLUD string (where colors are now URFDLB chars) and reorders to URFDLB string
        front,right,back,left,up,down = [cube_frblud_remapped[i:i+9] for i in range(0,54,9)]
        return up+right+front+down+left+back

    def _get_solved_state(self, cube_frblud, color_map): # From original
        # Generates a solved state string in FRBLUD order, using actual color characters.
        # color_map here is {actual_U_color: 'U', actual_R_color: 'R', ...}
        # We need the inverse: {'U': actual_U_color, ...}
        inv_color_map = {v: k for k, v in color_map.items()}
        return (inv_color_map['F']*9 + inv_color_map['R']*9 + inv_color_map['B']*9 +
                inv_color_map['L']*9 + inv_color_map['U']*9 + inv_color_map['D']*9)

    def _is_cube_solved(self, cube_state_frblud): # From original (was _is_cube_solved_by_face_colors)
        if len(cube_state_frblud)!=54: return False
        for i in range(0,54,9): # Iterate through faces F,R,B,L,U,D in FRBLUD string
            face=cube_state_frblud[i:i+9]; center_color=face[4]
            if not all(sticker==center_color for sticker in face): return False
        return True

    def _simplify_cube_moves(self, moves_str): # From original (the more complex one)
        moves = moves_str.strip().split()
        def move_value(move): return 2 if move.endswith("2") else (-1 if move.endswith("'") else 1)
        def value_to_move(face, value):
            value %= 4; return None if value==0 else (face if value==1 else (face+"2" if value==2 else face+"'"))
        face_groups=[['L','R'],['F','B'],['U','D']]; simplified=[]; i=0
        while i < len(moves): # Pass 1: Combine same face
            current_face=moves[i][0]; current_value=move_value(moves[i]); j=i+1
            while j < len(moves) and moves[j][0]==current_face: current_value+=move_value(moves[j]); j+=1
            move=value_to_move(current_face,current_value);
            if move: simplified.append(move)
            i=j
        final_simplified=[]; i=0
        while i < len(simplified): # Pass 2: Combine face groups (not strictly necessary for Kociemba output but part of original)
            current_face=simplified[i][0]; face_group=next((g for g in face_groups if current_face in g),None)
            if face_group:
                counts={face:0 for face in face_group}; j=i
                while j<len(simplified) and simplified[j][0] in face_group: counts[simplified[j][0]]+=move_value(simplified[j]); j+=1
                for face_in_group in face_group: # Maintain order like L then R for this pass
                    move=value_to_move(face_in_group,counts[face_in_group]);
                    if move: final_simplified.append(move)
                i=j
            else: final_simplified.append(simplified[i]); i+=1
        return " ".join(final_simplified) if final_simplified else "No moves"   

    def _solve_cube_with_kociemba(self, cube_frblud_actual_colors): # From original
        try:
            if self._is_cube_solved(cube_frblud_actual_colors): print("\nCube is already solved!"); return ""
            # color_map maps actual colors to URFDLB, cube_frblud_kociemba_chars is FRBLUD string with URFDLB chars
            color_map, cube_frblud_kociemba_chars = self._remap_colors_to_kociemba(cube_frblud_actual_colors)
            # scrambled_kociemba_input is URFDLB string with URFDLB chars
            scrambled_kociemba_input = self._remap_cube_to_kociemba(cube_frblud_kociemba_chars)
            
            # Target for Kociemba is always the standard UUU...RRR... string
            # The original code derived this via _get_solved_state and remapping.
            # This is equivalent to:
            solved_kociemba_target = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"

            self._validate_kociemba_string(scrambled_kociemba_input, "Scrambled Kociemba") # Validates URFDLB string
            # self._validate_kociemba_string(solved_kociemba_target, "Solved Kociemba Target") # This is always valid

            solution = kociemba.solve(scrambled_kociemba_input, solved_kociemba_target)

            u_replacement = "R L F2 B2 R' L' D R L F2 B2 R' L'"
            u_prime_replacement = "R L F2 B2 R' L' D' R L F2 B2 R' L'"
            u2_replacement = "R L F2 B2 R' L' D2 R L F2 B2 R' L'"
            moves = solution.split(); modified_solution=[]
            for move in moves:
                if move=="U": modified_solution.append(u_replacement)
                elif move=="U'": modified_solution.append(u_prime_replacement)
                elif move=="U2": modified_solution.append(u2_replacement)
                else: modified_solution.append(move)
            final_solution=" ".join(modified_solution)
            optimized_solution=self._simplify_cube_moves(final_solution) # Use the more complex simplify
            print(f"\nKociemba raw sol: {solution} ({len(moves)}m)")
            # print(f"Expanded sol: {final_solution} ({len(final_solution.split())}m)")
            print(f"Optimized robot sol: {optimized_solution} ({len(optimized_solution.split())}m)")
            return optimized_solution
        except ValueError as ve: self.error_message=f"Kociemba validation err: {ve}"; print(f"Error: {self.error_message}"); self._print_cube_state_visual(cube_frblud_actual_colors, "Problematic Cube (FRBLUD Actual Colors)"); return None
        except Exception as e: self.error_message=f"Err solving with Kociemba: {type(e).__name__} - {e}"; print(f"Error: {self.error_message}"); self._print_cube_state_visual(cube_frblud_actual_colors, "Problematic Cube (FRBLUD Actual Colors)"); return None
        
    def _simplify_cube_moves_basic(self, moves_str: str) -> str: # From original
        moves = [m for m in moves_str.strip().split() if m]; simplified = []
        i=0
        while i < len(moves):
            current_move = moves[i]; face = current_move[0]
            if face not in ['F','B','R','L','D','U']: simplified.append(current_move); i+=1; continue
            net_rot = 0; j = i
            while j < len(moves):
                m = moves[j]
                if not m or m[0] != face: break
                val = 1
                if len(m) > 1:
                    if m[1] == '2': val = 2
                    elif m[1] == "'": val = 3 
                net_rot = (net_rot + val) % 4; j += 1
            if net_rot == 1: simplified.append(face)
            elif net_rot == 2: simplified.append(face + "2")
            elif net_rot == 3: simplified.append(face + "'")
            i = j
        return " ".join(simplified)

    def scramble_cube(self) -> bool: # From original
        if self.mode not in ["idle","error"]: self.error_message="Cannot scramble, busy."; self.status_message=self.error_message; return False
        try:
            possible_faces=['F','B','R','L','D']; modifiers=['',"'",'2']; scramble_moves_list=[]
            for _ in range(random.randint(18,22)): 
                chosen_face=random.choice(possible_faces)
                if scramble_moves_list and chosen_face==scramble_moves_list[-1][0]:
                    available_faces=[f for f in possible_faces if f!=chosen_face]; chosen_face=random.choice(available_faces if available_faces else possible_faces)
                scramble_moves_list.append(chosen_face+random.choice(modifiers))
            scramble_sequence=" ".join(scramble_moves_list); self.mode="scrambling"; self.status_message=f"Scrambling: {scramble_sequence[:30]}..."
            self._reset_solve_state(); self.total_solve_moves=len(scramble_moves_list); print(f"Generated scramble: {scramble_sequence}")
            if self.send_arduino_command(scramble_sequence,is_solution=True): 
                if self.mode!="idle": self.mode="idle"; self.status_message="Scramble completed."
                self._reset_solve_state(); return True
            else: self.mode="idle"; self.error_message=self.error_message or "Failed to execute scramble."; self.status_message=self.error_message; self._reset_solve_state(); return False
        except Exception as e: self.error_message=f"Scramble err: {type(e).__name__}-{e}"; self.status_message="Scramble gen failed."; self.mode="error"; return False

    def stop_current_operation(self): # From original
        print(f"Stop requested. Current mode: {self.mode}"); self.stop_requested=True
        if self.serial_connection and self.serial_connection.is_open:
            try: self.serial_connection.write("STOP\n".encode('utf-8')); self.serial_connection.flush(); print("Sent STOP command to Arduino.")
            except Exception as e: print(f"Error sending STOP command: {e}")
        self.mode="idle"; self.status_message="Operation stopped by user."; self.error_message=None
        self._reset_solve_state(); self._reset_scan_state(); self.stop_requested=False; print(self.status_message)
    
    def _construct_cube_state_from_scans(self) -> Optional[str]: # From original (CRITICAL: Uses original predefined_centers)
        if len(self.u_scans)!=12 or not all(self.u_scans[i] and len(self.u_scans[i])==9 for i in range(12)):
            self.error_message="Invalid or incomplete scan data for cube construction."
            print(f"! CONSTRUCT ERROR: {self.error_message}"); return None
        
        temp_cube_state=['X']*54 
        # THIS IS THE CRUCIAL PART FROM YOUR ORIGINAL CODE for center color definition
        predefined_centers = {'F':'B', 'R':'O', 'B':'G', 'L':'R', 'U':'W', 'D':'Y'}
        temp_cube_state[4] =predefined_centers['F']; temp_cube_state[13]=predefined_centers['R'] 
        temp_cube_state[22]=predefined_centers['B']; temp_cube_state[31]=predefined_centers['L'] 
        temp_cube_state[40]=predefined_centers['U']; temp_cube_state[49]=predefined_centers['D'] 

        # Fill U face initially from u_scans[0] then apply specific overrides. From original.
        for i in range(9):
            if i==4: continue 
            temp_cube_state[36+i]=self.u_scans[0][i]
        
        try: # Piece assignments copied exactly from your original code
            temp_cube_state[0]=self.u_scans[1][0];temp_cube_state[1]=self.u_scans[3][3];temp_cube_state[2]=self.u_scans[1][2];temp_cube_state[3]=self.u_scans[1][3];temp_cube_state[5]=self.u_scans[1][5];temp_cube_state[6]=self.u_scans[1][6];temp_cube_state[7]=self.u_scans[3][5];temp_cube_state[8]=self.u_scans[1][8]
            temp_cube_state[9]=self.u_scans[2][0];temp_cube_state[10]=self.u_scans[2][1];temp_cube_state[11]=self.u_scans[2][2];temp_cube_state[12]=self.u_scans[4][1];temp_cube_state[14]=self.u_scans[4][7];temp_cube_state[15]=self.u_scans[2][6];temp_cube_state[16]=self.u_scans[2][7];temp_cube_state[17]=self.u_scans[2][8]
            temp_cube_state[18]=self.u_scans[4][2];temp_cube_state[19]=self.u_scans[9][5];temp_cube_state[20]=self.u_scans[4][8];temp_cube_state[21]=self.u_scans[7][3];temp_cube_state[23]=self.u_scans[7][5];temp_cube_state[24]=self.u_scans[4][0];temp_cube_state[25]=self.u_scans[9][3];temp_cube_state[26]=self.u_scans[4][6]
            temp_cube_state[27]=self.u_scans[5][2];temp_cube_state[28]=self.u_scans[8][7];temp_cube_state[29]=self.u_scans[5][8];temp_cube_state[30]=self.u_scans[10][1];temp_cube_state[32]=self.u_scans[10][7];temp_cube_state[33]=self.u_scans[5][0];temp_cube_state[34]=self.u_scans[8][1];temp_cube_state[35]=self.u_scans[5][6]
            temp_cube_state[36]=self.u_scans[6][0];temp_cube_state[37]=self.u_scans[0][1];temp_cube_state[38]=self.u_scans[6][2];temp_cube_state[39]=self.u_scans[11][3];temp_cube_state[41]=self.u_scans[11][5];temp_cube_state[42]=self.u_scans[6][6];temp_cube_state[43]=self.u_scans[0][7];temp_cube_state[44]=self.u_scans[6][8]
            temp_cube_state[45]=self.u_scans[3][6];temp_cube_state[46]=self.u_scans[6][1];temp_cube_state[47]=self.u_scans[3][0];temp_cube_state[48]=self.u_scans[5][5];temp_cube_state[50]=self.u_scans[5][3];temp_cube_state[51]=self.u_scans[3][8];temp_cube_state[52]=self.u_scans[6][7];temp_cube_state[53]=self.u_scans[3][2]
        except IndexError as e:
            self.error_message=f"IndexError in cube construct: {e}. A scan might be missing/malformed."
            print(f"! CONSTRUCT ERR: {self.error_message}"); return None
        
        final_cube_state_str="".join(temp_cube_state); counts=Counter(final_cube_state_str)
        expected_actual_colors_from_centers=set(predefined_centers.values()); valid_colors_ok=True
        for color_char in expected_actual_colors_from_centers:
            if counts.get(color_char,0)!=9: valid_colors_ok=False; break
        if not valid_colors_ok or 'X' in counts: 
            self.error_message=f"Constructed state invalid. Counts:{counts}. Final:{final_cube_state_str}"
            print(f"! CONSTRUCT ERR: {self.error_message}"); return None
        print("Cube state constructed (FRBLUD):", final_cube_state_str) # Added print for success
        return final_cube_state_str

    def _print_cube_state_visual(self, state_str_frblud: str, title: str = "Cube State (FRBLUD)"): # From original
        if len(state_str_frblud)!=54: print(f"Err print state: Exp 54, got {len(state_str_frblud)}"); return
        F,R,B,L,U,D = [state_str_frblud[i:i+9] for i in range(0,54,9)]
        print(f"\n--- {title} ---")
        def pf(fc,r,p="   "): print(f"{p}{fc[r*3]} {fc[r*3+1]} {fc[r*3+2]}")
        for ri in range(3): pf(U,ri,"      ")
        print("      ---------")
        for ri in range(3): print(f"{L[ri*3]} {L[ri*3+1]} {L[ri*3+2]} | {F[ri*3]} {F[ri*3+1]} {F[ri*3+2]} | {R[ri*3]} {R[ri*3+1]} {R[ri*3+2]} | {B[ri*3]} {B[ri*3+1]} {B[ri*3+2]}")
        print("      ---------")
        for ri in range(3): pf(D,ri,"      ")
        print("--- End of Cube State ---\n")

if __name__ == '__main__':
    game = RubiksCubeGame()
    print("RubiksCubeGame instance created.")
    
    # Test mode changes
    game.start_calibration_mode()
    print(f"Mode: {game.mode}, Status: {game.status_message}")
    game.start_scanning_mode() # Should fail if not calibrated
    print(f"Mode: {game.mode}, Status: {game.status_message}, Error: {game.error_message}")
    
    # Simulate calibration for testing scanning mode
    if game.mode == "calibrating": # If previous start_scanning_mode failed and reverted to calib or stayed idle and we force calib
        print("Simulating Lab calibration...")
        game.calibrated_lab_values = { "W": [240,128,128],"R": [120,190,160],"G": [120,80,160],"Y": [220,120,190],"O": [150,170,190],"B": [80,140,80] }
        game.calibration_step = 6 # Mark as complete
        game.start_idle_mode() # Go to idle after fake calibration
        print("Simulated Lab calibration complete.")

    game.start_scanning_mode()
    print(f"Mode: {game.mode}, Status: {game.status_message}")
    
    game.set_contour_crop_params(enabled=False, cx_factor=0.4, cy_factor=0.6, rel_width=0.5, rel_height=0.5)
    
    # Example of using get_state to see current parameters
    # current_game_state = game.get_state()
    # print("\nCurrent game state via get_state():")
    # for key, value in current_game_state.items():
    #     if key != "processed_frame": # Don't print the long frame string
    #         print(f"  {key}: {value}")

    # Test original simplify basic
    # test_moves_simple = "F B' R R R L2 L2 D' D D D F2"
    # simplified = game._simplify_cube_moves_basic(test_moves_simple)
    # print(f"\nOriginal simple: '{test_moves_simple}' -> Simplified basic: '{simplified}'")

    # Test original simplify (complex)
    # test_moves_complex = "F B' R R R L2 L2 U D' D D D U' U' U' F2" # U moves are handled by kociemba expansion now
    # simplified_complex = game._simplify_cube_moves(test_moves_complex)
    # print(f"Original complex: '{test_moves_complex}' -> Simplified complex: '{simplified_complex}'")