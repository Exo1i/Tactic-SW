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

class RubiksCubeGame:
    def __init__(self, config: Dict[str, Any] = None):
        # Initialize state variables
        self.mode = "idle"
        self.status_message = "Ready"
        self.error_message = None
        self.calibration_step = 0
        self.current_scan_idx = 0
        self.current_solve_move_index = 0
        self.total_solve_moves = 0
        self.solution = None
        self.stop_requested = False
        self.serial_connection = None
        self.last_valid_grid = None
        self.last_processed_frame = None
        
        # Constants
        self.WINDOW_SIZE = (640, 480)
        self.SCAN_COOLDOWN = 0.5
        self.MOTOR_STABILIZATION_TIME = 0.5
        self.STABILITY_THRESHOLD = 1
        self.MIN_CONTOUR_AREA = 4000
        self.MAX_CONTOUR_AREA = 60000
        self.COLOR_NAMES = ["W", "R", "G", "Y", "O", "B"]
        
        # Rotation sequence for scanning
        self.rotation_sequence = [
            "",        # Scan 1: Initial U face
            "R L'",    # Scan 2
            "B F'",    # Scan 3
            "R L'",    # Scan 4
            "B F'",    # Scan 5
            "R L'",    # Scan 6
            "B F'",    # Scan 7
            "R L'",    # Scan 8
            "B F'",    # Scan 9
            "R L'",    # Scan 10
            "B F'",    # Scan 11
            "R L'"     # Scan 12
        ]
        
        # Load configuration
        self.config = config or {}
        self.serial_port = self.config.get('serial_port', 'COM7')
        self.serial_baudrate = self.config.get('serial_baudrate', 9600)
        
        # Initialize Arduino connection
        self.init_serial()
        
        # Load or set default color ranges
        self.color_ranges = self.load_color_ranges()
        if not self.color_ranges:
            self.color_ranges = {
                "W": (np.array([0, 0, 200]), np.array([180, 30, 255])),      # White
                "R": (np.array([170, 120, 70]), np.array([180, 255, 255])),  # Red
                "G": (np.array([35, 50, 50]), np.array([85, 255, 255])),     # Green
                "Y": (np.array([20, 100, 100]), np.array([30, 255, 255])),   # Yellow
                "O": (np.array([5, 150, 150]), np.array([15, 255, 255])),    # Orange
                "B": (np.array([100, 100, 50]), np.array([130, 255, 255]))   # Blue
            }
        
        # Initialize scanning variables
        self.last_scan_time = time.time()
        self.stability_counter = 0
        self.prev_face_colors = None
        self.u_scans = [[] for _ in range(12)]
    
    def init_serial(self) -> bool:
        """Initialize serial connection to Arduino."""
        try:
            if self.serial_connection and self.serial_connection.is_open:
                self.serial_connection.close()
            
            self.serial_connection = serial.Serial(self.serial_port, self.serial_baudrate, timeout=1)
            time.sleep(2)  # Wait for Arduino to reset
            return True
        except Exception as e:
            self.error_message = f"Serial Error: {str(e)}"
            self.status_message = f"Error connecting to Arduino on {self.serial_port}"
            self.serial_connection = None
            return False
    
    def send_arduino_command(self, cmd: str) -> bool:
        """Send command to Arduino and wait for acknowledgment."""
        try:
            if self.serial_connection and self.serial_connection.is_open:
                self.serial_connection.write(f"{cmd}\n".encode())
                self.serial_connection.flush()
                # Wait for acknowledgment
                while True:
                    if self.serial_connection.in_waiting:
                        response = self.serial_connection.readline().decode().strip()
                        if "completed" in response.lower() or "executed" in response.lower():
                            return True
                    time.sleep(0.1)
            return False
        except Exception as e:
            self.error_message = f"Arduino command error: {str(e)}"
            return False
    
    def load_color_ranges(self, filename="color_ranges.json"):
        """Load color ranges from a JSON file."""
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    serializable_ranges = json.load(f)
                return {
                    color: (np.array(lower), np.array(upper)) 
                    for color, (lower, upper) in serializable_ranges.items()
                }
            except Exception:
                return None
        return None
    
    def save_color_ranges(self, filename="color_ranges.json"):
        """Save color ranges to a JSON file."""
        serializable_ranges = {
            color: (lower.tolist(), upper.tolist()) 
            for color, (lower, upper) in self.color_ranges.items()
        }
        with open(filename, 'w') as f:
            json.dump(serializable_ranges, f, indent=4)
    
    def process_frame(self, frame_data: bytes) -> Dict[str, Any]:
        """Process incoming frame data and return game state."""
        try:
            # Decode frame
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Failed to decode frame")
            
            # Resize frame
            frame = cv2.resize(frame, self.WINDOW_SIZE, interpolation=cv2.INTER_AREA)
            processed_frame = frame.copy()
            
            # Process frame based on current mode
            if self.mode == "calibrating":
                processed_frame = self.process_calibration_frame(frame, processed_frame)
            elif self.mode == "scanning":
                processed_frame = self.process_scanning_frame(frame, processed_frame)
            elif self.mode in ["solving", "scrambling"]:
                processed_frame = self.process_solving_frame(frame, processed_frame)
            else:  # idle or error mode
                processed_frame = self.process_idle_frame(frame, processed_frame)
            
            # Store last processed frame
            self.last_processed_frame = processed_frame.copy()
            
            # Encode processed frame
            _, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # Return current state
            return {
                "mode": self.mode,
                "status_message": self.status_message,
                "calibration_step": self.calibration_step if self.mode == "calibrating" else None,
                "current_color": self.COLOR_NAMES[self.calibration_step] if self.mode == "calibrating" and self.calibration_step < len(self.COLOR_NAMES) else None,
                "scan_index": self.current_scan_idx if self.mode == "scanning" else None,
                "solve_move_index": self.current_solve_move_index if self.mode in ["solving", "scrambling"] else 0,
                "total_solve_moves": self.total_solve_moves if self.mode in ["solving", "scrambling"] else 0,
                "solution": self.solution if self.solution else None,
                "error_message": self.error_message if self.mode == "error" else None,
                "serial_connected": self.serial_connection is not None and self.serial_connection.is_open,
                "processed_frame": frame_b64
            }
            
        except Exception as e:
            self.error_message = f"Frame processing error: {str(e)}"
            return {
                "mode": "error",
                "error_message": self.error_message,
                "status_message": "Error processing frame"
            }
    
    def process_calibration_frame(self, frame, display_frame):
        """Process frame during calibration mode."""
        # Draw calibration grid
        grid_size = int(min(self.WINDOW_SIZE) * 0.4)
        grid_cell_size = grid_size // 3
        pad_x, pad_y = 20, 50
        
        # Draw a small box around the center cell (1,1)
        center_y_start = pad_y + grid_cell_size
        center_y_end = pad_y + 2 * grid_cell_size
        center_x_start = pad_x + grid_cell_size
        center_x_end = pad_x + 2 * grid_cell_size
        
        # Draw the center cell box in green
        cv2.rectangle(display_frame, 
                     (center_x_start, center_y_start), 
                     (center_x_end, center_y_end), 
                     (0, 255, 0), 2)
        
        # Add instruction text
        current_color = self.COLOR_NAMES[self.calibration_step]
        instruction = f"Show {current_color} center, then click Capture"
        cv2.putText(display_frame, instruction, 
                   (pad_x, pad_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Store grid position for calibration
        self.last_valid_grid = (pad_x, pad_y, grid_size)
        
        return display_frame
    
    def process_scanning_frame(self, frame, display_frame):
        """Process frame during scanning mode."""
        # Create a clean copy for display
        display_frame = frame.copy()
        
        # Smooth the frame to reduce noise
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # Color detection and cube processing code
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        combined_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        
        # Create mask for all colors more efficiently
        for color, (lower, upper) in self.color_ranges.items():
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Clean up mask more aggressively
        kernel = np.ones((5,5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours with less detail for better stability
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cube_detected = False
        best_contour = None
        best_score = float('inf')
        current_grid = self.last_valid_grid  # Keep previous grid if no new detection
        
        # Process only the largest contours
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
            area = cv2.contourArea(contour)
            if self.MIN_CONTOUR_AREA < area < self.MAX_CONTOUR_AREA:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                if len(approx) >= 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = float(w) / h
                    squareness_score = abs(1 - aspect_ratio)
                    
                    if 0.7 < aspect_ratio < 1.3 and squareness_score < best_score:
                        best_score = squareness_score
                        best_contour = contour
                        cube_detected = True
                        
                        # Smooth the grid position using previous position if available
                        grid_size = min(w, h)
                        center_x = x + w // 2
                        center_y = y + h // 2
                        pad_x = center_x - grid_size // 2
                        pad_y = center_y - grid_size // 2
                        
                        if self.last_valid_grid:
                            prev_x, prev_y, prev_size = self.last_valid_grid
                            # Smooth the transition
                            pad_x = int(0.7 * prev_x + 0.3 * pad_x)
                            pad_y = int(0.7 * prev_y + 0.3 * pad_y)
                            grid_size = int(0.7 * prev_size + 0.3 * grid_size)
                        
                        current_grid = (pad_x, pad_y, grid_size)
                        break
        
        if cube_detected and best_contour is not None:
            self.last_valid_grid = current_grid
            pad_x, pad_y, grid_size = current_grid
            
            # Draw contour and grid with anti-aliasing for smoother appearance
            cv2.drawContours(display_frame, [best_contour], -1, (0, 255, 0), 2, cv2.LINE_AA)
            
            grid_cell_size = grid_size // 3
            
            # Draw grid with anti-aliasing
            for i in range(1, 3):
                cv2.line(display_frame, 
                        (pad_x + i * grid_cell_size, pad_y), 
                        (pad_x + i * grid_cell_size, pad_y + grid_size), 
                        (0, 255, 0), 2, cv2.LINE_AA)
                cv2.line(display_frame, 
                        (pad_x, pad_y + i * grid_cell_size), 
                        (pad_x + grid_size, pad_y + i * grid_cell_size), 
                        (0, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(display_frame, 
                        (pad_x, pad_y), 
                        (pad_x + grid_size, pad_y + grid_size), 
                        (0, 255, 0), 2, cv2.LINE_AA)
            
            # Check stability based on contour shape and position
            if hasattr(self, 'prev_contour') and self.prev_contour is not None:
                shape_diff = cv2.matchShapes(best_contour, self.prev_contour, 1, 0.0)
                position_diff = abs(pad_x - self.prev_x) + abs(pad_y - self.prev_y)
                
                if shape_diff < 0.3 and position_diff < 20:
                    # Only increment if below threshold
                    if self.stability_counter < self.STABILITY_THRESHOLD:
                        self.stability_counter += 1
                else:
                    self.stability_counter = max(0, self.stability_counter - 1)  # Gradual decrease
            
            self.prev_contour = best_contour
            self.prev_x, self.prev_y = pad_x, pad_y
            
            # Get face colors and check for auto-scanning
            current_time = time.time()
            time_since_last_scan = current_time - self.last_scan_time
            
            if (self.stability_counter >= self.STABILITY_THRESHOLD and 
                time_since_last_scan >= self.SCAN_COOLDOWN):
                
                face_colors = []
                for i in range(3):
                    for j in range(3):
                        y_start = pad_y + i * grid_cell_size
                        y_end = pad_y + (i + 1) * grid_cell_size
                        x_start = pad_x + j * grid_cell_size
                        x_end = pad_x + (j + 1) * grid_cell_size
                        
                        padding = grid_cell_size // 8
                        y_start += padding
                        y_end -= padding
                        x_start += padding
                        x_end -= padding
                        
                        roi = frame[y_start:y_end, x_start:x_end]
                        color = self.detect_color(roi)
                        if color:
                            face_colors.append(color)
                            # Draw detected color with anti-aliasing
                            cv2.putText(display_frame, 
                                      color, 
                                      (x_start + grid_cell_size//4, y_start + grid_cell_size//2),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                
                if len(face_colors) == 9:
                    if not hasattr(self, 'prev_face_colors'):
                        self.prev_face_colors = face_colors
                        self.stability_counter = 1
                    elif face_colors == self.prev_face_colors:
                        self.u_scans[self.current_scan_idx] = face_colors
                        self.current_scan_idx += 1
                        self.last_scan_time = current_time
                        self.stability_counter = 0
                        self.prev_face_colors = None
                        
                        # Send rotation command if not last scan
                        if self.current_scan_idx < len(self.rotation_sequence):
                            move = self.rotation_sequence[self.current_scan_idx]
                            if move:
                                self.send_arduino_command(move)
                        
                        # Check if all scans complete
                        if self.current_scan_idx >= 12:
                            # Process solution
                            cube_state = self.construct_cube_state()
                            if cube_state:
                                solution = self.solve_cube(cube_state)
                                if solution:
                                    self.solution = solution
                                    self.mode = "solving"
                                    self.current_solve_move_index = 0
                                    self.total_solve_moves = len(solution.split())
                                    self.status_message = "Solution found, executing moves"
                                    self.send_arduino_command(solution)
                                else:
                                    self.mode = "error"
                                    self.error_message = "Could not find solution"
                            else:
                                self.mode = "error"
                                self.error_message = "Invalid cube state"
                    else:
                        self.prev_face_colors = face_colors
                        self.stability_counter = 1
        else:
            self.stability_counter = max(0, self.stability_counter - 1)  # Gradual decrease
            self.prev_contour = None
        
        # Add scan progress and stability text with anti-aliasing
        if self.current_scan_idx < 12:
            if cube_detected:
                status = f"Scan {self.current_scan_idx + 1}/12 - Stability: {min(self.stability_counter, self.STABILITY_THRESHOLD)}/{self.STABILITY_THRESHOLD}"
            else:
                status = f"Position cube for scan {self.current_scan_idx + 1}/12"
            cv2.putText(display_frame, status, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        return display_frame
    
    def process_solving_frame(self, frame, display_frame):
        """Process frame during solving/scrambling mode."""
        # Draw progress text
        mode_text = "Solving" if self.mode == "solving" else "Scrambling"
        progress_text = f"{mode_text} ({self.current_solve_move_index}/{self.total_solve_moves})"
        cv2.putText(display_frame, progress_text, 
                   (11, self.WINDOW_SIZE[1] - 19), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(display_frame, progress_text, 
                   (10, self.WINDOW_SIZE[1] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        return display_frame
    
    def process_idle_frame(self, frame, display_frame):
        """Process frame during idle mode."""
        # Draw cube detection overlay
        if self.last_valid_grid:
            pad_x, pad_y, grid_size = self.last_valid_grid
            cv2.rectangle(display_frame, 
                         (pad_x, pad_y), 
                         (pad_x + grid_size, pad_y + grid_size), 
                         (0, 255, 0), 2)
            
            grid_cell_size = grid_size // 3
            for i in range(1, 3):
                cv2.line(display_frame, 
                        (pad_x + i * grid_cell_size, pad_y), 
                        (pad_x + i * grid_cell_size, pad_y + grid_size), 
                        (0, 255, 0), 1)
                cv2.line(display_frame, 
                        (pad_x, pad_y + i * grid_cell_size), 
                        (pad_x + grid_size, pad_y + i * grid_cell_size), 
                        (0, 255, 0), 1)
        
        return display_frame
    
    def detect_cube(self, frame):
        """Detect the Rubik's cube in the frame and update last_valid_grid."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        combined_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        
        # Create mask for all colors
        for color, (lower, upper) in self.color_ranges.items():
            combined_mask |= cv2.inRange(hsv, lower, upper)
        
        # Clean up mask
        kernel = np.ones((5,5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_contour = None
        best_score = float('inf')
        cube_detected = False
        best_pos = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.MIN_CONTOUR_AREA < area < self.MAX_CONTOUR_AREA:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                if len(approx) >= 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = float(w) / h
                    squareness_score = abs(1 - aspect_ratio)
                    
                    if 0.7 < aspect_ratio < 1.3 and squareness_score < best_score:
                        best_score = squareness_score
                        best_contour = contour
                        best_pos = (x, y, w, h)
                        cube_detected = True
        
        if cube_detected and best_pos:
            x, y, w, h = best_pos
            grid_size = min(w, h)
            self.last_valid_grid = (x, y, grid_size)
            return True
        
        return False
    
    def detect_color(self, roi):
        """Detect the dominant color in a region of interest (ROI)."""
        if len(roi.shape) == 3 and roi.shape[2] == 3:
            if roi.dtype != np.uint8:
                roi = np.uint8(roi)
            
            h, w = roi.shape[:2]
            center_roi = roi[h//4:3*h//4, w//4:3*w//4]
            hsv_roi = cv2.cvtColor(center_roi, cv2.COLOR_BGR2HSV)
            
            color_matches = {}
            for color, (lower, upper) in self.color_ranges.items():
                mask = cv2.inRange(hsv_roi, lower, upper)
                match_percentage = cv2.countNonZero(mask) / (center_roi.shape[0] * center_roi.shape[1])
                color_matches[color] = match_percentage * 100
            
            best_color = max(color_matches, key=color_matches.get)
            best_match = color_matches[best_color]
            
            if best_match > 10:
                return best_color
        
        return None
    
    def cleanup(self):
        """Clean up resources."""
        if self.serial_connection and self.serial_connection.is_open:
            try:
                self.serial_connection.write("STOP\n".encode('ascii'))
                self.serial_connection.flush()
                time.sleep(0.1)
                self.serial_connection.close()
            except:
                pass 
    
    def calibrate_color(self) -> bool:
        """Capture and calibrate the current color."""
        if self.mode != "calibrating" or self.calibration_step >= len(self.COLOR_NAMES):
            return False
        
        if not self.last_valid_grid or not self.last_processed_frame is not None:
            return False
        
        try:
            pad_x, pad_y, grid_size = self.last_valid_grid
            grid_cell_size = grid_size // 3
            
            # Get ROI from center cell (1,1)
            center_y_start = pad_y + grid_cell_size
            center_y_end = pad_y + 2 * grid_cell_size
            center_x_start = pad_x + grid_cell_size
            center_x_end = pad_x + 2 * grid_cell_size
            
            roi = self.last_processed_frame[center_y_start:center_y_end, 
                                         center_x_start:center_x_end]
            
            if roi.size == 0:
                return False
            
            # Calculate HSV ranges
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            avg_hsv = np.mean(hsv_roi, axis=(0, 1))
            
            current_color = self.COLOR_NAMES[self.calibration_step]
            h_range = 10 if current_color != "W" else 90  # Wider hue range for white
            s_range = 50
            v_range = 50
            
            # Calculate lower and upper bounds
            lower = np.array([
                max(0, avg_hsv[0] - h_range),
                max(0, avg_hsv[1] - s_range),
                max(0, avg_hsv[2] - v_range)
            ])
            upper = np.array([
                min(180, avg_hsv[0] + h_range),
                min(255, avg_hsv[1] + s_range),
                min(255, avg_hsv[2] + v_range)
            ])
            
            # Special handling for white
            if current_color == "W":
                lower[1] = 0  # Allow low saturation for white
                lower[2] = max(150, avg_hsv[2] - v_range)  # Must be bright
            
            self.color_ranges[current_color] = (lower, upper)
            self.calibration_step += 1
            
            if self.calibration_step >= len(self.COLOR_NAMES):
                self.save_color_ranges()
                self.mode = "idle"
                self.status_message = "Calibration completed"
            else:
                self.status_message = f"Calibrated {current_color}. Now show {self.COLOR_NAMES[self.calibration_step]}"
            
            return True
            
        except Exception as e:
            self.error_message = f"Calibration error: {str(e)}"
            return False
    
    def start_scanning(self) -> bool:
        """Start the scanning process."""
        if self.mode not in ["idle", "error"]:
            return False
        
        self.mode = "scanning"
        self.current_scan_idx = 0
        self.u_scans = [[] for _ in range(12)]
        self.status_message = "Position cube for first scan"
        return True
    
    def capture_scan(self) -> bool:
        """Capture the current face scan."""
        if self.mode != "scanning" or self.current_scan_idx >= 12:
            return False
        
        if not self.last_valid_grid or not self.last_processed_frame is not None:
            return False
        
        try:
            pad_x, pad_y, grid_size = self.last_valid_grid
            grid_cell_size = grid_size // 3
            face_colors = []
            
            for i in range(3):
                for j in range(3):
                    y_start = pad_y + i * grid_cell_size
                    y_end = pad_y + (i + 1) * grid_cell_size
                    x_start = pad_x + j * grid_cell_size
                    x_end = pad_x + (j + 1) * grid_cell_size
                    
                    padding = grid_cell_size // 8
                    y_start += padding
                    y_end -= padding
                    x_start += padding
                    x_end -= padding
                    
                    roi = self.last_processed_frame[y_start:y_end, x_start:x_end]
                    color = self.detect_color(roi)
                    if not color:
                        return False
                    face_colors.append(color)
            
            self.u_scans[self.current_scan_idx] = face_colors
            self.current_scan_idx += 1
            
            if self.current_scan_idx < 12:
                # Send rotation command to Arduino
                rotation_sequence = [
                    "",        # Scan 1: Initial U face
                    "R L'",    # Scan 2
                    "B F'",    # Scan 3
                    "R L'",    # Scan 4
                    "B F'",    # Scan 5
                    "R L'",    # Scan 6
                    "B F'",    # Scan 7
                    "R L'",    # Scan 8
                    "B F'",    # Scan 9
                    "R L'",    # Scan 10
                    "B F'",    # Scan 11
                    "R L'"     # Scan 12
                ]
                if self.current_scan_idx < len(rotation_sequence):
                    move = rotation_sequence[self.current_scan_idx]
                    if move:
                        self.send_arduino_command(move)
                
                self.status_message = f"Scan {self.current_scan_idx + 1}/12 captured"
            else:
                # All scans complete, construct cube state and solve
                cube_state = self.construct_cube_state()
                if cube_state:
                    solution = self.solve_cube(cube_state)
                    if solution:
                        self.solution = solution
                        self.mode = "solving"
                        self.current_solve_move_index = 0
                        self.total_solve_moves = len(solution.split())
                        self.status_message = "Solution found, executing moves"
                        self.send_arduino_command(f"SOLUTION:{solution}")
                    else:
                        self.mode = "error"
                        self.error_message = "Could not find solution"
                else:
                    self.mode = "error"
                    self.error_message = "Invalid cube state"
            
            return True
            
        except Exception as e:
            self.error_message = f"Scan error: {str(e)}"
            return False
    
    def construct_cube_state(self) -> Optional[str]:
        """Construct the full cube state from scanned faces."""
        if len(self.u_scans) != 12 or not all(len(scan) == 9 for scan in self.u_scans):
            return None
        
        cube_state = [''] * 54
        # Set centers (fixed)
        cube_state[4] = 'B'   # F center
        cube_state[13] = 'O'  # R center
        cube_state[22] = 'G'  # B center
        cube_state[31] = 'R'  # L center
        cube_state[40] = 'W'  # U center
        cube_state[49] = 'Y'  # D center
        
        # Map scanned faces to cube state
        cube_state[36:45] = self.u_scans[0]  # First U face scan
        
        # Map remaining scans
        scan_mappings = [
            (1, [0,2,3,5,6,8]),  # Scan 2
            (2, [9,10,11,15,16,17]),  # Scan 3
            (3, [47,53,1,7,45,51]),  # Scan 4
            (4, [24,12,18,26,14,20]),  # Scan 5
            (5, [33,27,50,48,35,29]),  # Scan 6
            (6, [36,46,38,42,52,44]),  # Scan 7
            (7, [21,23]),  # Scan 8
            (8, [34,28]),  # Scan 9
            (9, [25,19]),  # Scan 10
            (10, [30,32]),  # Scan 11
            (11, [39,41])  # Scan 12
        ]
        
        for scan_idx, positions in scan_mappings:
            scan = self.u_scans[scan_idx]
            for i, pos in enumerate(positions):
                cube_state[pos] = scan[i]
        
        return ''.join(cube_state)
    
    def solve_cube(self, cube_state: str) -> Optional[str]:
        """Solve the cube using kociemba algorithm."""
        try:
            # Validate cube state
            if len(cube_state) != 54:
                raise ValueError(f"Invalid cube state length: {len(cube_state)}")
            if not all(c in self.COLOR_NAMES for c in cube_state):
                raise ValueError(f"Invalid colors in cube state")
            
            # Convert color notation to kociemba notation
            color_to_face = {
                'W': 'U',  # White is Up
                'R': 'R',  # Red is Right
                'G': 'F',  # Green is Front
                'Y': 'D',  # Yellow is Down
                'O': 'L',  # Orange is Left
                'B': 'B'   # Blue is Back
            }
            kociemba_state = ''.join(color_to_face[c] for c in cube_state)
            
            # Get solution
            solution = kociemba.solve(kociemba_state)
            if not solution:
                return None
            
            # Replace U moves with alternative sequences
            moves = solution.split()
            modified_solution = []
            for move in moves:
                if move == "U":
                    modified_solution.append("R L F2 B2 R' L' D R L F2 B2 R' L'")
                elif move == "U'":
                    modified_solution.append("R L F2 B2 R' L' D' R L F2 B2 R' L'")
                elif move == "U2":
                    modified_solution.append("R L F2 B2 R' L' D2 R L F2 B2 R' L'")
                else:
                    modified_solution.append(move)
            
            return " ".join(modified_solution)
            
        except Exception as e:
            self.error_message = f"Solve error: {str(e)}"
            return None
    
    def scramble_cube(self) -> bool:
        """Generate and execute a random scramble sequence."""
        if self.mode not in ["idle", "error"]:
            return False
        
        try:
            # Generate random scramble (no U moves)
            basic_moves = ['F', 'B', 'R', 'L', 'D']  # No U moves as we can't execute them
            modifiers = ['', '\'', '2']
            scramble = []
            last_face = None
            
            for _ in range(20):  # 20 moves scramble
                # Don't allow same face moves consecutively
                available_moves = [move for move in basic_moves if move != last_face]
                face = random.choice(available_moves)
                modifier = random.choice(modifiers)
                scramble.append(face + modifier)
                last_face = face
            
            scramble_sequence = " ".join(scramble)
            self.status_message = "Executing scramble sequence"
            
            # Send scramble to Arduino and wait for completion
            self.send_arduino_command(scramble_sequence)
            
            # Return to idle mode
            self.mode = "idle"
            self.status_message = "Scramble completed"
            return True
            
        except Exception as e:
            self.error_message = f"Scramble error: {str(e)}"
            self.mode = "error"
            return False
    
    def stop(self):
        """Stop any ongoing operation."""
        self.stop_requested = True
        if self.serial_connection and self.serial_connection.is_open:
            try:
                self.serial_connection.write("STOP\n".encode('ascii'))
                self.serial_connection.flush()
            except:
                pass
        
        self.mode = "idle"
        self.status_message = "Operation stopped"
        self.error_message = None
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current game state."""
        return {
            "mode": self.mode,
            "status_message": self.status_message,
            "calibration_step": self.calibration_step if self.mode == "calibrating" else None,
            "current_color": self.COLOR_NAMES[self.calibration_step] if self.mode == "calibrating" and self.calibration_step < len(self.COLOR_NAMES) else None,
            "scan_index": self.current_scan_idx if self.mode == "scanning" else None,
            "solve_move_index": self.current_solve_move_index if self.mode in ["solving", "scrambling"] else 0,
            "total_solve_moves": self.total_solve_moves if self.mode in ["solving", "scrambling"] else 0,
            "solution": self.solution if self.solution else None,
            "error_message": self.error_message if self.mode == "error" else None,
            "serial_connected": self.serial_connection is not None and self.serial_connection.is_open
        } 