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
        
        # Constants (updated from RubiksCubeSolver)
        self.WINDOW_SIZE = (640, 480)
        self.SCAN_COOLDOWN = 0.5
        self.MOTOR_STABILIZATION_TIME = 0.5
        self.STABILITY_THRESHOLD = 3
        self.MIN_CONTOUR_AREA = 5000
        self.MAX_CONTOUR_AREA = 50000
        self.COLOR_NAMES = ["W", "R", "G", "Y", "O", "B"]
        
        # Rotation sequence for scanning
        self.rotation_sequence = [
            "", "R L'", "B F'", "R L'", "B F'", "R L'", "B F'", "R L'", "B F'", "R L'", "B F'", "R L'"
        ]
        
        # Load configuration
        self.config = config or {}
        self.serial_port = self.config.get('serial_port', 'COM7')
        self.serial_baudrate = self.config.get('serial_baudrate', 9600)
        
        # Initialize Arduino connection
        self.init_serial()
        
        # Tighter HSV color ranges from RubiksCubeSolver
        self.color_ranges = {
            "W": (np.array([0, 0, 180]), np.array([180, 50, 255])),      # White: High value, low saturation
            "R": (np.array([0, 120, 100]), np.array([10, 255, 255])),    # Red: Narrow hue range
            "G": (np.array([40, 100, 100]), np.array([80, 255, 255])),   # Green: Adjusted hue
            "Y": (np.array([25, 120, 120]), np.array([35, 255, 255])),   # Yellow: Higher saturation
            "O": (np.array([10, 120, 120]), np.array([20, 255, 255])),   # Orange: Narrower hue range
            "B": (np.array([90, 100, 100]), np.array([120, 255, 255]))   # Blue: Adjusted hue
        }
        
        # Initialize scanning variables
        self.last_scan_time = time.time()
        self.last_motor_move_time = 0
        self.stability_counter = 0
        self.prev_face_colors = None
        self.prev_contour = None
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
                print(f"\nSending to Arduino: {cmd}")
                self.serial_connection.write(f"{cmd}\n".encode())
                self.serial_connection.flush()
                
                # Wait for acknowledgment with a longer timeout for solutions
                start_time = time.time()
                timeout = 30 if len(cmd.split()) > 10 else 10  # Longer timeout for solutions
                
                while True:
                    if time.time() - start_time > timeout:
                        print(f"Timeout waiting for Arduino response after {timeout} seconds")
                        return False
                    
                    if self.serial_connection.in_waiting:
                        response = self.serial_connection.readline().decode().strip()
                        print(f"Arduino response: {response}")
                        
                        # For long solutions, keep reading until we get completion
                        if "completed" in response.lower() or "executed" in response.lower():
                            print("Command execution completed")
                            # Reset to idle mode after solution is complete
                            if len(cmd.split()) > 10:  # If this was a solution
                                print("Solution completed, resetting to idle mode")
                                self.mode = "idle"
                                self.status_message = "Solution completed - Ready for next command"
                                self.current_solve_move_index = 0
                                self.total_solve_moves = 0
                                self.solution = None
                            return True
                        elif "error" in response.lower():
                            print(f"Arduino reported error: {response}")
                            return False
                    
                    # Small delay to prevent busy waiting
                    time.sleep(0.1)
            else:
                print("Serial connection not available")
                return False
                
        except Exception as e:
            self.error_message = f"Arduino command error: {str(e)}"
            print(f"Error sending command: {e}")
            return False
    
    def send_compound_move(self, move: str):
        """Send a compound move command to Arduino."""
        if move:
            self.send_arduino_command(move)
    
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
        """Process frame during scanning mode (updated from RubiksCubeSolver.solve)."""
        # Create a clean copy for display
        display_frame = frame.copy()
        
        # Convert to HSV and create combined mask
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        combined_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        
        for color, (lower, upper) in self.color_ranges.items():
            combined_mask |= cv2.inRange(hsv, lower, upper)
        
        # Clean up mask
        kernel = np.ones((5,5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cube_detected = False
        best_contour = None
        best_score = float('inf')
        current_grid = self.last_valid_grid
        
        # Process contours
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
                        cube_detected = True
                        break
        
        if cube_detected and best_contour is not None:
            x, y, w, h = cv2.boundingRect(best_contour)
            grid_size = min(w, h)
            
            center_x = x + w // 2
            center_y = y + h // 2
            pad_x = center_x - grid_size // 2
            pad_y = center_y - grid_size // 2
            
            self.last_valid_grid = (pad_x, pad_y, grid_size)
            current_grid = self.last_valid_grid
            
            # Draw contour and grid
            cv2.drawContours(display_frame, [best_contour], -1, (0, 255, 0), 2)
            
            grid_cell_size = grid_size // 3
            for i in range(1, 3):
                cv2.line(display_frame, 
                        (pad_x + i * grid_cell_size, pad_y), 
                        (pad_x + i * grid_cell_size, pad_y + grid_size), 
                        (0, 255, 0), 2)
                cv2.line(display_frame, 
                        (pad_x, pad_y + i * grid_cell_size), 
                        (pad_x + grid_size, pad_y + i * grid_cell_size), 
                        (0, 255, 0), 2)
            cv2.rectangle(display_frame, 
                         (pad_x, pad_y), 
                         (pad_x + grid_size, pad_y + grid_size), 
                         (0, 255, 0), 2)
            
            # Check stability
            if self.prev_DIMcontour is not None:
                shape_diff = cv2.matchShapes(best_contour, self.prev_contour, 1, 0.0)
                position_diff = abs(x - self.prev_x) + abs(y - self.prev_y)
                
                if shape_diff < 0.3 and position_diff < 20:
                    self.stability_counter += 1
                else:
                    self.stability_counter = 0
            
            self.prev_contour = best_contour
            self.prev_x, self.prev_y = x, y
        else:
            self.stability_counter = 0
            self.prev_contour = None
        
        # Check scanning conditions
        current_time = time.time()
        time_since_last_scan = current_time - self.last_scan_time
        time_since_last_move = current_time - self.last_motor_move_time
        
        if self.current_scan_idx < 12:
            if cube_detected:
                status = f"Scan {self.current_scan_idx + 1}/12 - "
                if time_since_last_move < self.MOTOR_STABILIZATION_TIME:
                    status += f"Waiting for cube to stabilize: {self.MOTOR_STABILIZATION_TIME - time_since_last_move:.1f}s"
                elif time_since_last_scan < self.SCAN_COOLDOWN:
                    status += f"Cooldown: {self.SCAN_COOLDOWN - time_since_last_scan:.1f}s"
                else:
                    status += f"Stability: {self.stability_counter}/{self.STABILITY_THRESHOLD}"
            else:
                status = f"Position cube for scan {self.current_scan_idx + 1}/12"
            
            cv2.putText(display_frame, status, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Perform scan if conditions are met
        if (cube_detected and 
            self.stability_counter >= self.STABILITY_THRESHOLD and 
            self.current_scan_idx < 12 and 
            time_since_last_scan >= self.SCAN_COOLDOWN and 
            time_since_last_move >= self.MOTOR_STABILIZATION_TIME):
            
            if self.last_valid_grid is not None:
                face_colors = []
                pad_x, pad_y, grid_size = self.last_valid_grid
                grid_cell_size = grid_size // 3
                
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
                        face_colors.append(color)
                        cv2.putText(display_frame, 
                                   color, 
                                   (x_start + grid_cell_size//4, y_start + grid_cell_size//2), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                
                if self.prev_face_colors is None or not all(a == b for a, b in zip(face_colors, self.prev_face_colors)):
                    self.u_scans[self.current_scan_idx] = face_colors
                    self.prev_face_colors = face_colors
                    
                    if self.current_scan_idx < 11:
                        self.send_compound_move(self.rotation_sequence[self.current_scan_idx + 1])
                        self.last_motor_move_time = time.time()
                    
                    self.current_scan_idx += 1
                    if self.current_scan_idx == 12:
                        print("\nAll scans completed! Processing solution...")
                        self.send_compound_move("B F'")
                        self.last_motor_move_time = time.time()
                        
                        cube_state = self.construct_cube_from_u_scans(self.u_scans)
                        try:
                            solution = self.solve_cube_frblud(cube_state)
                            if solution:
                                self.solution = solution
                                self.mode = "solving"
                                self.current_solve_move_index = 0
                                self.total_solve_moves = len(solution.split())
                                self.status_message = "Solution found, executing moves"
                                print(f"\nSending solution command: SOLUTION:{solution}")
                                self.send_arduino_command(f"SOLUTION:{solution}")
                            else:
                                self.mode = "error"
                                self.error_message = "Could not find solution"
                        except Exception as e:
                            self.mode = "error"
                            self.error_message = f"Failed to solve: {str(e)}"
                    
                    self.last_scan_time = current_time
                    self.stability_counter = 0
                else:
                    print("Duplicate face detected, skipping")
        
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
    
    def detect_color(self, roi):
        """Detect the dominant color in a region of interest (ROI) with stricter criteria (from RubiksCubeSolver._detect_color)."""
        if len(roi.shape) == 3 and roi.shape[2] == 3:
            if roi.dtype != np.uint8:
                roi = np.uint8(roi)
            
            # Preprocess ROI with Gaussian blur to reduce noise
            roi = cv2.GaussianBlur(roi, (5, 5), 0)
            
            h, w = roi.shape[:2]
            center_roi = roi[h//4:3*h//4, w//4:3*w//4]
            hsv_roi = cv2.cvtColor(center_roi, cv2.COLOR_BGR2HSV)
            
            # Range-based method with higher threshold
            color_matches = {}
            for color, (lower, upper) in self.color_ranges.items():
                mask = cv2.inRange(hsv_roi, lower, upper)
                match_percentage = cv2.countNonZero(mask) / (center_roi.shape[0] * center_roi.shape[1])
                color_matches[color] = match_percentage * 100
            
            range_best_color = max(color_matches, key=color_matches.get)
            range_best_match = color_matches[range_best_color]
            
            # Dominant HSV method with stricter distance threshold
            pixels = hsv_roi.reshape((-1, 3))
            pixel_list = [tuple(p) for p in pixels]
            most_common_hsv = Counter(pixel_list).most_common(1)[0][0]
            
            dominant_color = None
            min_distance = float('inf')
            
            for color, (lower, upper) in self.color_ranges.items():
                middle_hsv = (lower + upper) / 2
                h_dist = min(abs(most_common_hsv[0] - middle_hsv[0]), 
                            180 - abs(most_common_hsv[0] - middle_hsv[0]))
                s_dist = abs(most_common_hsv[1] - middle_hsv[1])
                v_dist = abs(most_common_hsv[2] - middle_hsv[2])
                
                # Stricter weights for non-white colors
                if color == "W":
                    distance = 0.1 * h_dist + 0.3 * s_dist + 0.6 * v_dist
                else:
                    distance = 0.7 * h_dist + 0.2 * s_dist + 0.1 * v_dist
                
                if distance < min_distance:
                    min_distance = distance
                    dominant_color = color
            
            # Stricter criteria: require higher range match or low distance
            if range_best_match > 20:
                return range_best_color
            elif dominant_color and min_distance < 30:
                return dominant_color
            
            # Fallback: Average HSV with strict criteria
            avg_hsv = np.mean(hsv_roi, axis=(0,1))
            closest_color = None
            min_distance = float('inf')
            
            for color, (lower, upper) in self.color_ranges.items():
                middle_hsv = (lower + upper) / 2
                h_dist = min(abs(avg_hsv[0] - middle_hsv[0]), 
                            180 - abs(avg_hsv[0] - middle_hsv[0]))
                s_dist = abs(avg_hsv[1] - middle_hsv[1])
                v_dist = abs(avg_hsv[2] - middle_hsv[2])
                
                if color == "W":
                    distance = 0.1 * h_dist + 0.3 * s_dist + 0.6 * v_dist
                else:
                    distance = 0.7 * h_dist + 0.2 * s_dist + 0.1 * v_dist
                
                if distance < min_distance:
                    min_distance = distance
                    closest_color = color
            
            if min_distance < 30:
                return closest_color
            return "W"
        
        return "W"
    
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
        """Capture and calibrate the current color (updated from RubiksCubeSolver._calibrate_colors)."""
        if self.mode != "calibrating" or self.calibration_step >= len(self.COLOR_NAMES):
            return False
        
        if not self.last_valid_grid or self.last_processed_frame is None:
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
            h_range = 5 if current_color != "W" else 90  # Tighter hue range for non-white
            s_range = 40
            v_range = 40
            
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
        self.prev_face_colors = None
        self.prev_contour = None
        self.stability_counter = 0
        self.last_scan_time = time.time()
        self.last_motor_move_time = 0
        return True
    
    def capture_scan(self) -> bool:
        """Capture the current face scan."""
        if self.mode != "scanning" or self.current_scan_idx >= 12:
            return False
        
        if not self.last_valid_grid or self.last_processed_frame is None:
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
            self.prev_face_colors = face_colors
            self.current_scan_idx += 1
            
            if self.current_scan_idx < 12:
                move = self.rotation_sequence[self.current_scan_idx]
                if move:
                    print(f"\nSending rotation command: {move}")
                    self.send_compound_move(move)
                    self.last_motor_move_time = time.time()
                
                self.status_message = f"Scan {self.current_scan_idx + 1}/12 captured"
            else:
                print("\nAll scans completed! Processing solution...")
                print("\nSending final B F' command...")
                if not self.send_compound_move("B F'"):
                    self.mode = "error"
                    self.error_message = "Failed to execute final B F' command"
                    return False
                
                print("Waiting for motors to stabilize...")
                time.sleep(self.MOTOR_STABILIZATION_TIME)
                
                cube_state = self.construct_cube_from_u_scans(self.u_scans)
                if cube_state:
                    solution = self.solve_cube_frblud(cube_state)
                    if solution:
                        self.solution = solution
                        self.mode = "solving"
                        self.current_solve_move_index = 0
                        self.total_solve_moves = len(solution.split())
                        self.status_message = "Solution found, executing moves"
                        print(f"\nSending solution command: SOLUTION:{solution}")
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
    
    def construct_cube_from_u_scans(self, u_scans) -> Optional[str]:
        """Construct the full cube state from 12 U face scans (from RubiksCubeSolver._construct_cube_from_u_scans)."""
        if len(u_scans) != 12 or not all(len(scan) == 9 for scan in u_scans):
            return None
        
        cube_state = [''] * 54
        cube_state[4] = 'B'   # F center
        cube_state[13] = 'O'  # R center
        cube_state[22] = 'G'  # B center
        cube_state[31] = 'R'  # L center
        cube_state[40] = 'W'  # U center
        cube_state[49] = 'Y'  # D center
        
        cube_state[36:45] = u_scans[0]
        for i in range(54):
            if not cube_state[i]:
                cube_state[i] = '-'
        
        cube_state[0] = u_scans[1][0]
        cube_state[2] = u_scans[1][2]
        cube_state[3] = u_scans[1][3]
        cube_state[5] = u_scans[1][5]
        cube_state[6] = u_scans[1][6]
        cube_state[8] = u_scans[1][8]
        
        cube_state[9] = u_scans[2][0]
        cube_state[10] = u_scans[2][1]
        cube_state[11] = u_scans[2][2]
        cube_state[15] = u_scans[2][6]
        cube_state[16] = u_scans[2][7]
        cube_state[17] = u_scans[2][8]
        
        cube_state[47] = u_scans[3][0]
        cube_state[53] = u_scans[3][2]
        cube_state[1] = u_scans[3][3]
        cube_state[7] = u_scans[3][5]
        cube_state[45] = u_scans[3][6]
        cube_state[51] = u_scans[3][8]
        
        cube_state[24] = u_scans[4][0]
        cube_state[12] = u_scans[4][1]
        cube_state[18] = u_scans[4][2]
        cube_state[26] = u_scans[4][6]
        cube_state[14] = u_scans[4][7]
        cube_state[20] = u_scans[4][8]
        
        cube_state[33] = u_scans[5][0]
        cube_state[27] = u_scans[5][2]
        cube_state[50] = u_scans[5][3]
        cube_state[48] = u_scans[5][5]
        cube_state[35] = u_scans[5][6]
        cube_state[29] = u_scans[5][8]
        
        cube_state[36] = u_scans[6][0]
        cube_state[46] = u_scans[6][1]
        cube_state[38] = u_scans[6][2]
        cube_state[42] = u_scans[6][6]
        cube_state[52] = u_scans[6][7]
        cube_state[44] = u_scans[6][8]
        
        cube_state[21] = u_scans[7][3]
        cube_state[23] = u_scans[7][5]
        
        cube_state[34] = u_scans[8][1]
        cube_state[28] = u_scans[8][7]
        
        cube_state[25] = u_scans[9][3]
        cube_state[19] = u_scans[9][5]
        
        cube_state[30] = u_scans[10][1]
        cube_state[32] = u_scans[10][7]
        
        cube_state[39] = u_scans[11][3]
        cube_state[41] = u_scans[11][5]
        
        # Print the cube state for debugging
        print("\nConstructed Cube State:")
        print("Color mapping: W=White, R=Red, G=Green, Y=Yellow, O=Orange, B=Blue")
        print("\nRaw state:", ''.join(cube_state))
        print("\nVisual representation:")
        for i in range(3):
            start = 36 + i*3
            print("        " + " ".join(cube_state[start:start+3]))
        for i in range(3):
            line = ""
            for face_start in [0, 9, 18, 27]:
                start = face_start + i*3
                line += " ".join(cube_state[start:start+3]) + " | "
            print(line[:-3])
        for i in range(3):
            start = 45 + i*3
            print("        " + " ".join(cube_state[start:start+3]))
        
        return ''.join(cube_state)
    
    def print_cube_state(self, cube_state: str):
        """Print the cube state in a visual format (from RubiksCubeSolver._print_full_cube_state)."""
        print("\nFull cube state (Front, Right, Back, Left, Up, Down):")
        print("".join(cube_state))
        print("\nVisual representation:")
        idx = [0, 9, 18, 27, 36, 45]
        for i in range(3):
            start = idx[4] + i*3
            print("        " + " ".join(cube_state[start:start+3]))
        for i in range(3):
            line = ""
            for face_start in idx[:4]:
                start = face_start + i*3
                line += " ".join(cube_state[start:start+3]) + " | "
            print(line[:-3])
        for i in range(3):
            start = idx[5] + i*3
            print("        " + " ".join(cube_state[start:start+3]))
    
    def validate_cube(self, cube, order_name):
        """Validate the cube state (from RubiksCubeSolver._validate_cube)."""
        if len(cube) != 54:
            raise ValueError(f"{order_name} must be 54 characters")
        counts = Counter(cube)
        if len(counts) != 6 or any(count != 9 for count in counts.values()):
            raise ValueError(f"{order_name} invalid: {counts} (need 9 of each of 6 colors)")
    
    def remap_colors_to_kociemba(self, cube_frblud):
        """Remap cube colors to Kociemba notation (from RubiksCubeSolver._remap_colors_to_kociemba)."""
        self.validate_cube(cube_frblud, "FRBLUD")
        centers = [cube_frblud[i] for i in [4, 13, 22, 31, 40, 49]]
        color_map = {
            centers[4]: 'U', centers[1]: 'R', centers[0]: 'F',
            centers[5]: 'D', centers[3]: 'L', centers[2]: 'B'
        }
        return color_map, ''.join(color_map[c] for c in cube_frblud)
    
    def remap_cube_to_kociemba(self, cube_frblud_remapped):
        """Remap cube faces to Kociemba order (from RubiksCubeSolver._remap_cube_to_kociemba)."""
        front, right, back, left, up, down = [cube_frblud_remapped[i:i+9] for i in range(0, 54, 9)]
        return up + right + front + down + left + back
    
    def get_solved_state(self, cube_frblud, color_map):
        """Generate the solved cube state (from RubiksCubeSolver._get_solved_state)."""
        centers = [cube_frblud[i] for i in [4, 13, 22, 31, 40, 49]]
        return ''.join(c * 9 for c in centers)
    
    def is_cube_solved(self, cube_state):
        """Check if the cube is already solved (from RubiksCubeSolver._is_cube_solved)."""
        for i in range(0, 54, 9):
            face = cube_state[i:i+9]
            if not all(sticker == face[4] for sticker in face):
                return False
        return True
    
    def simplify_cube_moves(self, moves_str):
        """Simplify a sequence of cube moves (from RubiksCubeSolver._simplify_cube_moves)."""
        moves = moves_str.strip().split()
        
        def move_value(move):
            if move.endswith("2"):
                return 2
            elif move.endswith("'"):
                return -1
            return 1
        
        def value_to_move(face, value):
            value = value % 4
            if value == 0:
                return None
            elif value == 1:
                return face
            elif value == 2:
                return face + "2"
            elif value == 3:
                return face + "'"
        
        face_groups = [['L', 'R'], ['F', 'B'], ['U', 'D']]
        
        # First pass: Combine consecutive moves of the same face
        i = 0
        simplified = []
        while i < len(moves):
            current_face = moves[i][0]
            current_value = move_value(moves[i])
            
            j = i + 1
            while j < len(moves) and moves[j][0] == current_face:
                current_value += move_value(moves[j])
                j += 1
                
            move = value_to_move(current_face, current_value)
            if move:
                simplified.append(move)
                
            i = j
        
        # Second pass: Combine moves by face groups
        final_simplified = []
        i = 0
        while i < len(simplified):
            current_face = simplified[i][0]
            
            face_group = None
            for group in face_groups:
                if current_face in group:
                    face_group = group
                    break
            
            if face_group:
                counts = {face: 0 for face in face_group}
                j = i
                
                while j < len(simplified) and simplified[j][0] in face_group:
                    face = simplified[j][0]
                    counts[face] += move_value(simplified[j])
                    j += 1
                    
                for face in face_group:
                    move = value_to_move(face, counts[face])
                    if move:
                        final_simplified.append(move)
                        
                i = j
            else:
                final_simplified.append(simplified[i])
                i += 1
        
        return " ".join(final_simplified) if final_simplified else "No moves"
    
    def solve_cube_frblud(self, cube_frblud):
        """Solve the cube given its FRBLUD state (from RubiksCubeSolver._solve_cube_frblud)."""
        try:
            if self.is_cube_solved(cube_frblud):
                print("\nCube is already solved! No moves needed.")
                return ""
            
            color_map, cube_frblud_remapped = self.remap_colors_to_kociemba(cube_frblud)
            scrambled_kociemba = self.remap_cube_to_kociemba(cube_frblud_remapped)
            solved_frblud = self.get_solved_state(cube_frblud, color_map)
            _, solved_frblud_remapped = self.remap_colors_to_kociemba(solved_frblud)
            solved_kociemba = self.remap_cube_to_kociemba(solved_frblud_remapped)
            self.validate_cube(scrambled_kociemba, "Scrambled Kociemba")
            self.validate_cube(solved_kociemba, "Solved Kociemba")
            
            print("\nScrambled Kociemba state:", scrambled_kociemba)
            print("Solved Kociemba state:", solved_kociemba)
            
            solution = kociemba.solve(scrambled_kociemba, solved_kociemba)
            print("Raw solution from kociemba:", solution)
            
            u_replacement = "R L F2 B2 R' L' D R L F2 B2 R' L'"
            u_prime_replacement = "R L F2 B2 R' L' D' R L F2 B2 R' L'"
            u2_replacement = "R L F2 B2 R' L' D2 R L F2 B2 R' L'"
            
            moves = solution.split()
            modified_solution = []
            for move in moves:
                if move == "U":
                    modified_solution.append(u_replacement)
                elif move == "U'":
                    modified_solution.append(u_prime_replacement)
                elif move == "U2":
                    modified_solution.append(u2_replacement)
                else:
                    modified_solution.append(move)
            
            final_solution = " ".join(modified_solution)
            optimized_solution = self.simplify_cube_moves(final_solution)
            print("\nOriginal solution length:", len(final_solution.split()))
            print("Optimized solution length:", len(optimized_solution.split()))
            
            return optimized_solution
        
        except Exception as e:
            print(f"Error solving cube: {str(e)}")
            self.error_message = f"Solve error: {str(e)}"
            return None
    
    def scramble_cube(self) -> bool:
        """Generate and execute a random scramble sequence (from RubiksCubeSolver.scramble)."""
        if self.mode not in ["idle", "error"]:
            return False
        
        try:
            print("\nGenerating scramble sequence...")
            basic_moves = ['F', 'B', 'R', 'L', 'D']
            modifiers = ['', '\'', '2']
            scramble = []
            last_face = None
            
            for _ in range(20):
                available_moves = [move for move in basic_moves if move != last_face]
                face = random.choice(available_moves)
                modifier = random.choice(modifiers)
                scramble.append(face + modifier)
                last_face = face
            
            scramble_sequence = " ".join(scramble)
            self.mode = "scrambling"
            self.status_message = "Executing scramble sequence"
            self.current_solve_move_index = 0
            self.total_solve_moves = len(scramble)
            
            print(f"Scramble sequence: {scramble_sequence}")
            if not self.send_arduino_command(scramble_sequence):
                self.mode = "error"
                self.error_message = "Failed to execute scramble sequence"
                return False
            
            self.mode = "idle"
            self.status_message = "Scramble completed - Ready for next command"
            self.current_solve_move_index = 0
            self.total_solve_moves = 0
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