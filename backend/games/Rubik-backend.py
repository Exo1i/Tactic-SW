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
        self.last_scan_time = time.time()
        self.stability_counter = 0
        self.prev_face_colors = None
        self.u_scans = [[] for _ in range(12)]
        self.frame = None
        self.prev_contour = None
        self.last_motor_move_time = 0
        self.prev_face_colors = None
        self.scan_windows = []
        
        # Constants
        self.WINDOW_SIZE = (640, 480)
        self.SCAN_COOLDOWN = 1
        self.MOTOR_STABILIZATION_TIME = 1
        self.STABILITY_THRESHOLD = 2
        self.MIN_CONTOUR_AREA = 8000
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
                "W": (np.array([0, 0, 180]), np.array([180, 50, 255])),      # White: High value, low saturation
                "R": (np.array([0, 120, 100]), np.array([10, 255, 255])),    # Red: Narrow hue range
                "G": (np.array([40, 100, 100]), np.array([80, 255, 255])),   # Green: Adjusted hue
                "Y": (np.array([25, 120, 120]), np.array([35, 255, 255])),   # Yellow: Higher saturation
                "O": (np.array([10, 120, 120]), np.array([20, 255, 255])),   # Orange: Narrower hue range
                "B": (np.array([90, 100, 100]), np.array([120, 255, 255]))   # Blue: Adjusted hue
            }

        # Initialize scanning variables


    
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
    def _send_compound_move(self, move):
        """Send a compound move command to Arduino."""
        if move:
            self.send_arduino_command(move)

    def _find_best_cube_contour(self, frame):
        """Find the best contour that represents the cube's top face with improved accuracy."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Step 1: Create a combined mask from all color ranges
        combined_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        for color, ranges in self.color_ranges.items():
            for lower, upper in ranges:
                mask = cv2.inRange(hsv, lower, upper)
                combined_mask |= mask

        # Step 2: Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        # Step 3: Find contours with hierarchical information
        contours, hierarchy = cv2.findContours(combined_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # Step 4: Find the best square-like contour
        best_contour = None
        best_score = float('inf')

        for contour in contours:
            area = cv2.contourArea(contour)
            if self.MIN_CONTOUR_AREA < area < self.MAX_CONTOUR_AREA:
                # Calculate perimeter and approximate the contour
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

                # Check if it's quadrilateral-like (has 4 corners or close to it)
                if 4 <= len(approx) <= 6:
                    # Calculate various metrics to find the best square-like shape
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = float(w) / h
                    squareness_score = abs(1 - aspect_ratio)

                    # Calculate convexity - perfect square is convex
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area) / hull_area if hull_area > 0 else 0

                    # Combine scores - lower is better
                    combined_score = squareness_score + (1 - solidity)

                    if 0.7 < aspect_ratio < 1.3 and combined_score < best_score:
                        best_score = combined_score
                        best_contour = contour

        return best_contour

    def _detect_corners(self, contour):
        """Detect and order the four corners of the contour with improved accuracy and boundary checking."""
        # Step 1: Start with approximation
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Step 2: Refine if we don't get exactly 4 points
        if len(approx) != 4:
            # Get minimum area rectangle as fallback
            rect = cv2.minAreaRect(contour)
            corners = cv2.boxPoints(rect)
        else:
            corners = approx.reshape(4, 2)

        # Step 3: Refine corner positions using corner sub-pixel accuracy
        corners = corners.astype(np.float32)

        # Create a grayscale copy of the current frame for corner refinement
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        # Get image dimensions
        h, w = gray.shape[:2]

        # Ensure corners are within image boundaries
        corners = np.array([[
            max(0, min(w - 1, corner[0])),
            max(0, min(h - 1, corner[1]))
        ] for corner in corners], dtype=np.float32)

        # Define the refinement criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        try:
            # Refine the corner locations with sub-pixel accuracy
            refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Ensure refined corners are within image boundaries
            refined_corners = np.array([[
                max(0, min(w - 1, corner[0])),
                max(0, min(h - 1, corner[1]))
            ] for corner in refined_corners], dtype=np.float32)

            # Step 4: Order corners: top-left, top-right, bottom-right, bottom-left
            corners = sorted(refined_corners, key=lambda p: p[1])
            top_corners = sorted(corners[:2], key=lambda p: p[0])
            bottom_corners = sorted(corners[2:], key=lambda p: p[0])

            return np.array([top_corners[0], top_corners[1], bottom_corners[1], bottom_corners[0]], dtype=np.float32)
        except:
            # Fallback if cornerSubPix fails
            corners = sorted(corners, key=lambda p: p[1])
            top_corners = sorted(corners[:2], key=lambda p: p[0])
            bottom_corners = sorted(corners[2:], key=lambda p: p[0])

            return np.array([top_corners[0], top_corners[1], bottom_corners[1], bottom_corners[0]], dtype=np.float32)
    def _predict_square_length(self, corners):
        """Predict the side length of the cube's top face based on the detected corners."""
        distances = [
            np.linalg.norm(corners[0] - corners[1]),  # Top edge
            np.linalg.norm(corners[1] - corners[2]),  # Right edge
            np.linalg.norm(corners[2] - corners[3]),  # Bottom edge
            np.linalg.norm(corners[3] - corners[0])  # Left edge
        ]
        # Use the average of the edges to predict the square length
        return int(np.mean(distances))

    def _perspective_transform(self, frame, contour):
        """Transform the detected cube to a square frontal view using predicted length."""
        corners = self._detect_corners(contour)
        if len(corners) != 4:
            raise ValueError("Could not detect exactly 4 corners")

        # Predict the square length based on the corners
        side_length = self._predict_square_length(corners)

        # Define destination points as a square with the predicted length
        dst_points = np.array([
            [0, 0],
            [side_length - 1, 0],
            [side_length - 1, side_length - 1],
            [0, side_length - 1]
        ], dtype=np.float32)

        # Compute and apply the perspective transform
        matrix = cv2.getPerspectiveTransform(corners, dst_points)
        warped = cv2.warpPerspective(frame, matrix, (side_length, side_length))

        return warped
        
    def load_color_ranges(self, filename="color_ranges.json"):
        """Load color ranges from a JSON file."""
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    serializable_ranges = json.load(f)
                result = {}
                for color, value in serializable_ranges.items():
                    if isinstance(value, list) and len(value) == 2:
                        lower = np.array(value[0])
                        upper = np.array(value[1])
                        result[color] = [(lower, upper)]  # Wrap in list
                return result
            except Exception as e:
                print("Error loading color ranges:", e)
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
        height, width = display_frame.shape[:2]

        # Determine square grid size based on smaller dimension
        grid_size = int(min(self.WINDOW_SIZE))
        grid_cell_size = grid_size // 4

        # Center the grid
        pad_x = (width - grid_size) // 2
        pad_y = (height - grid_size) // 2

        # Draw a box around the center cell (1,1)
        center_y_start = pad_y + grid_cell_size
        center_y_end = pad_y + 2 * grid_cell_size
        center_x_start = pad_x + grid_cell_size
        center_x_end = pad_x + 2 * grid_cell_size

        # Draw the center cell box in green
        cv2.rectangle(display_frame,
                    (center_x_start, center_y_start),
                    (center_x_end, center_y_end),
                    (0, 255, 0), 2)

        # Add instruction text above the grid
        current_color = self.COLOR_NAMES[self.calibration_step]
        instruction = f"Show {current_color} center, then click Capture"
        text_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = (width - text_size[0]) // 2
        text_y = pad_y - 10

        cv2.putText(display_frame, instruction,
                (text_x, max(30, text_y)),  # avoid putting text off screen
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Store grid position for calibration
        self.last_valid_grid = (pad_x, pad_y, grid_size)

        return display_frame

    
    def process_scanning_frame(self, frame, display_frame):

        temp_dir = "cube_scans"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        self.current_scan_idx = 0
        self.prev_face_colors = None
        self.scan_windows = []
        self.last_motor_move_time = 0
        self.stability_counter = 0
        self.prev_contour = None
        self.last_valid_grid = None

        current_time = time.time()
        time_since_last_scan = current_time - self.last_scan_time
        time_since_last_move = current_time - self.last_motor_move_time
        # Store the original frame for corner detection
        self.frame = frame.copy()

        display_frame = cv2.resize(display_frame, self.WINDOW_SIZE, interpolation=cv2.INTER_AREA)
        display = display_frame.copy()
        # Use the improved contour detection method
        best_contour = self._find_best_cube_contour(self.frame)
        cube_detected = best_contour is not None
        if cube_detected:
            # Draw the contour on display
            cv2.drawContours(display, [best_contour], -1, (0, 255, 0), 2)
            # Create a grid based on the detected cube
            try:
                # Get perspective transformed view of the cube
                warped_image = self._perspective_transform(frame, best_contour)
                if warped_image is not None:
                    grid_size = min(warped_image.shape[0], warped_image.shape[1])
                    grid_cell_size = max(1, grid_size // 3)  # Ensure cell size is at least 1
                    # Ensure warped image is large enough for a 3x3 grid
                    if grid_size >= 30 and grid_cell_size >= 10:
                        self.last_valid_grid = (warped_image, grid_size, grid_cell_size)
                # Draw grid lines on the display safely
                x, y, w, h = cv2.boundingRect(best_contour)
                if w > 0 and h > 0:  # Ensure valid dimensions
                    cell_w = max(1, w // 3)  # Ensure cell width is at least 1
                    cell_h = max(1, h // 3)  # Ensure cell height is at least 1
                    # Draw simplified grid on the contour
                    for i in range(1, 3):
                        # Draw horizontal lines
                        cv2.line(display, (x, y + i * cell_h), (x + w, y + i * cell_h), (0, 255, 0), 2)
                        # Draw vertical lines
                        cv2.line(display, (x + i * cell_w, y), (x + i * cell_w, y + h), (0, 255, 0), 2)
                if self.prev_contour is not None:
                    try:
                        shape_diff = cv2.matchShapes(best_contour, self.prev_contour, 1, 0.0)
                        # Calculate position difference (center point difference)
                        M1 = cv2.moments(best_contour)
                        M2 = cv2.moments(self.prev_contour)
                        if M1["m00"] != 0 and M2["m00"] != 0:
                            cx1 = int(M1["m10"] / M1["m00"])
                            cy1 = int(M1["m01"] / M1["m00"])
                            cx2 = int(M2["m10"] / M2["m00"])
                            cy2 = int(M2["m01"] / M2["m00"])
                            position_diff = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
                        else:
                            position_diff = 100  # Large value to indicate instability
                        if shape_diff < 0.2 and position_diff < 15:
                            self.stability_counter += 1
                        else:
                            self.stability_counter = 0
                    except:
                        self.stability_counter = 0
                else:
                    self.stability_counter = 1
                self.prev_contour = best_contour
            except Exception as e:
                print(f"Error processing cube: {str(e)}")
                self.stability_counter = 0
        else:
            self.stability_counter = 0
            self.prev_contour = None
        
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
            cv2.putText(display, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        

        if (cube_detected and
                self.current_scan_idx < 12 and
                time_since_last_scan >= self.SCAN_COOLDOWN and
                time_since_last_move >= self.MOTOR_STABILIZATION_TIME):
            if self.last_valid_grid is not None:
                warped_image, grid_size, grid_cell_size = self.last_valid_grid
                face_colors = []
                try:
                    for i in range(3):
                        for j in range(3):
                            # Calculate bounds with safety checks
                            y_start = min(i * grid_cell_size, warped_image.shape[0] - 1)
                            y_end = min((i + 1) * grid_cell_size, warped_image.shape[0])
                            x_start = min(j * grid_cell_size, warped_image.shape[1] - 1)
                            x_end = min((j + 1) * grid_cell_size, warped_image.shape[1])
                            # Ensure we have valid regions
                            if y_end <= y_start or x_end <= x_start:
                                # Default to white if region is invalid
                                face_colors.append("W")
                                continue
                            padding = min(grid_cell_size // 8, 5)  # Limit padding to avoid going out of bounds
                            y_start = min(y_start + padding, warped_image.shape[0] - 1)
                            y_end = min(y_end - padding, warped_image.shape[0])
                            x_start = min(x_start + padding, warped_image.shape[1] - 1)
                            x_end = min(x_end - padding, warped_image.shape[1])
                            # Final safety check
                            if y_end <= y_start or x_end <= x_start:
                                face_colors.append("W")
                                continue
                            roi = warped_image[y_start:y_end, x_start:x_end]
                            color = self.detect_color(roi)
                            face_colors.append(color)
                    if len(face_colors) == 9:  # Ensure we have all 9 colors
                        print(face_colors)
                        if self.prev_face_colors is None or not all(
                                a == b for a, b in zip(face_colors, self.prev_face_colors)):
                            self.u_scans[self.current_scan_idx] = face_colors
                            self.prev_face_colors = face_colors
                            if self.current_scan_idx < 11:
                                self._send_compound_move(self.rotation_sequence[self.current_scan_idx + 1])
                                self.last_motor_move_time = time.time()
                            window_name = f"U Face Scan #{self.current_scan_idx + 1}"
                            self.scan_windows.append(window_name)
                            # Create a safe copy of warped image for visualization
                            scan_image = warped_image.copy()
                            # Draw a clean 3x3 grid on the scan image
                            for i in range(1, 3):
                                cv2.line(scan_image, (i * grid_cell_size, 0),
                                         (i * grid_cell_size, grid_size), (0, 0, 0), 2)
                                cv2.line(scan_image, (0, i * grid_cell_size),
                                         (grid_size, i * grid_cell_size), (0, 0, 0), 2)
                            # Add detected colors to the grid
                            for i in range(3):
                                for j in range(3):
                                    # Safely calculate text position
                                    text_x = j * grid_cell_size + grid_cell_size // 4
                                    text_y = (i + 1) * grid_cell_size - grid_cell_size // 3
                                    # Make sure positions are within bounds
                                    text_x = max(10, min(scan_image.shape[1] - 10, text_x))
                                    text_y = max(10, min(scan_image.shape[0] - 10, text_y))
                                    color = face_colors[i * 3 + j]
                                    cv2.putText(scan_image, color, (text_x, text_y),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                            cv2.imwrite(os.path.join(temp_dir, f"u_scan_{self.current_scan_idx + 1}_processed.jpg"),
                                        scan_image)
                            self.current_scan_idx += 1
                            if self.current_scan_idx == 12:
                                print("\nAll scans completed! Processing solution...")
                                self._send_compound_move("B F'")
                                self.last_motor_move_time = time.time()
                                cube_state = self._construct_cube_from_u_scans(self.u_scans)
                                try:
                                    solution = self._solve_cube_frblud(cube_state)
                                    if solution:
                                        print(f"\nSolution: {solution}")
                                        self.send_arduino_command(f"SOLUTION:{solution}")
                                except Exception as e:
                                    print(f"Failed to solve: {e}")
                                print("\nPress 'q' to return to main menu.")
                            self.last_scan_time = current_time
                            self.stability_counter = 0
                        else:
                            print("Duplicate face detected, skipping")
                except Exception as e:
                    print(f"Error processing grid cells: {str(e)}")
        return display
            

    
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
        """Detect the Rubik's cube in the frame and update self.last_valid_grid."""
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

            roi = cv2.GaussianBlur(roi, (5, 5), 0)
            h, w = roi.shape[:2]
            center_roi = roi[h // 4:3 * h // 4, w // 4:3 * w // 4]
            hsv_roi = cv2.cvtColor(center_roi, cv2.COLOR_BGR2HSV)

            # Range-based method with multiple ranges
            color_matches = {}
            for color, ranges in self.color_ranges.items():
                mask = np.zeros_like(hsv_roi[:, :, 0], dtype=np.uint8)
                for lower, upper in ranges:
                    mask |= cv2.inRange(hsv_roi, lower, upper)
                match_percentage = cv2.countNonZero(mask) / (center_roi.shape[0] * center_roi.shape[1])
                color_matches[color] = match_percentage * 100

            range_best_color = max(color_matches, key=color_matches.get)
            range_best_match = color_matches[range_best_color]

            # Dominant HSV method
            pixels = hsv_roi.reshape((-1, 3))
            pixel_list = [tuple(p) for p in pixels]
            most_common_hsv = Counter(pixel_list).most_common(1)[0][0]

            dominant_color = None
            min_distance = float('inf')

            for color, ranges in self.color_ranges.items():
                for lower, upper in ranges:
                    middle_hsv = (lower + upper) / 2
                    h_dist = min(abs(most_common_hsv[0] - middle_hsv[0]),
                                 180 - abs(most_common_hsv[0] - middle_hsv[0]))
                    s_dist = abs(most_common_hsv[1] - middle_hsv[1])
                    v_dist = abs(most_common_hsv[2] - middle_hsv[2])

                    if color == "W":
                        distance = 0.1 * h_dist + 0.3 * s_dist + 0.6 * v_dist
                    else:
                        distance = 0.7 * h_dist + 0.2 * s_dist + 0.1 * v_dist

                    if distance < min_distance:
                        min_distance = distance
                        dominant_color = color

            avg_hsv = np.mean(hsv_roi, axis=(0, 1))
            return dominant_color if min_distance < 30 else range_best_color

        return 'W'  # Default to white if invalid ROI

   
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
            
            # Calculate HSV ranges (from _calibrate_colors)
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            avg_hsv = np.mean(hsv_roi, axis=(0, 1))
            
            current_color = self.COLOR_NAMES[self.calibration_step]
            if current_color == "W":
                h_range = 90
            elif current_color == "R" and (avg_hsv[0] < 10 or avg_hsv[0] > 170):
                h_range = 10
            else:
                h_range = 5
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
                        print(f"\nSending rotation command: {move}")
                        self.send_arduino_command(move)
                
                self.status_message = f"Scan {self.current_scan_idx + 1}/12 captured"
            else:
                # All scans complete, send final B F' and process solution
                print("\nAll scans completed! Processing solution...")
                print("\nSending final B F' command...")
                if not self.send_arduino_command("B F'"):
                    self.mode = "error"
                    self.error_message = "Failed to execute final B F' command"
                    return False
                
                print("Waiting for motors to stabilize...")
                time.sleep(self.MOTOR_STABILIZATION_TIME)
                
                # Construct cube state and solve
                cube_state = self.construct_cube_state()
                if cube_state:
                    solution = self.solve_cube(cube_state)
                    if solution:
                        self.solution = solution
                        self.mode = "solving"
                        self.current_solve_move_index = 0
                        self.total_solve_moves = len(solution.split())
                        self.status_message = "Solution found, executing moves"
                        print(f"\nSending solution command: SOLUTION:{solution}")
                        self.send_arduino_command(solution)
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
        cube_state[4] = 'B'   # F center
        cube_state[13] = 'O'  # R center
        cube_state[22] = 'G'  # B center
        cube_state[31] = 'R'  # L center
        cube_state[40] = 'W'  # U center
        cube_state[49] = 'Y'  # D center
        
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
        
        # Print the cube state for debugging
        print("\nConstructed Cube State:")
        print("Color mapping: W=White, R=Red, G=Green, Y=Yellow, O=Orange, B=Blue")
        print("\nRaw state:", ''.join(cube_state))
        print("\nVisual representation:")
        # Print Up face
        for i in range(3):
            start = 36 + i*3  # Up face starts at index 36
            print("        " + " ".join(cube_state[start:start+3]))
        # Print middle faces (Front, Right, Back, Left)
        for i in range(3):
            line = ""
            for face_start in [0, 9, 18, 27]:  # F, R, B, L
                start = face_start + i*3
                line += " ".join(cube_state[start:start+3]) + " | "
            print(line[:-3])
        # Print Down face
        for i in range(3):
            start = 45 + i*3  # Down face starts at index 45
            print("        " + " ".join(cube_state[start:start+3]))
        
        return ''.join(cube_state)
    
    def print_cube_state(self, cube_state: str):
        """Print the cube state in a visual format."""
        print("\nCube State:")
        print("Color mapping: W=White, R=Red, G=Green, Y=Yellow, O=Orange, B=Blue")
        print("\nVisual representation:")
        # Print Up face
        for i in range(3):
            start = 36 + i*3  # Up face starts at index 36
            print("        " + " ".join(cube_state[start:start+3]))
        # Print middle faces (Front, Right, Back, Left)
        for i in range(3):
            line = ""
            for face_start in [0, 9, 18, 27]:  # F, R, B, L
                start = face_start + i*3
                line += " ".join(cube_state[start:start+3]) + " | "
            print(line[:-3])
        # Print Down face
        for i in range(3):
            start = 45 + i*3  # Down face starts at index 45
            print("        " + " ".join(cube_state[start:start+3]))
    
    def validate_cube(self, cube, order_name):
        if len(cube) != 54:
            raise ValueError(f"{order_name} must be 54 characters")
        counts = Counter(cube)
        if len(counts) != 6 or any(count != 9 for count in counts.values()):
            raise ValueError(f"{order_name} invalid: {counts} (need 9 of each of 6 colors)")

    def remap_colors_to_kociemba(self, cube_frblud):
        self.validate_cube(cube_frblud, "FRBLUD")
        centers = [cube_frblud[i] for i in [4, 13, 22, 31, 40, 49]]  # F, R, B, L, U, D
        color_map = {
            centers[4]: 'U',  # Up
            centers[1]: 'R',  # Right
            centers[0]: 'F',  # Front
            centers[5]: 'D',  # Down
            centers[3]: 'L',  # Left
            centers[2]: 'B'   # Back
        }
        return color_map, ''.join(color_map[c] for c in cube_frblud)

    def remap_cube_to_kociemba(self, cube_frblud_remapped):
        front, right, back, left, up, down = [cube_frblud_remapped[i:i+9] for i in range(0, 54, 9)]
        return up + right + front + down + left + back

    def get_solved_state(self, cube_frblud, color_map):
        centers = [cube_frblud[i] for i in [4, 13, 22, 31, 40, 49]]  # F, R, B, L, U, D
        return ''.join(c * 9 for c in centers)

    def is_cube_solved(self, cube_state):
        """Check if the cube is already solved by verifying each face has the same color."""
        # Check each face (9 stickers per face)
        for i in range(0, 54, 9):
            face = cube_state[i:i+9]
            # If any sticker is different from the center of the face, cube is not solved
            if not all(sticker == face[4] for sticker in face):
                return False
        return True

    def simplify_cube_moves(self, moves_str):
        # Split the string into individual moves
        moves = moves_str.strip().split()
        
        # Function to get the net effect of a single move
        def move_value(move):
            if move.endswith("2"):
                return 2
            elif move.endswith("'"):
                return -1
            return 1
        
        # Function to convert net value back to move notation
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
        
        # Process moves for each face type
        face_groups = [['L', 'R'], ['F', 'B'], ['U', 'D']]
        
        # First pass: Combine consecutive moves of the same face
        i = 0
        simplified = []
        while i < len(moves):
            current_face = moves[i][0]
            current_value = move_value(moves[i])
            
            # Look ahead for same face moves
            j = i + 1
            while j < len(moves) and moves[j][0] == current_face:
                current_value += move_value(moves[j])
                j += 1
                
            # Add the simplified move if needed
            move = value_to_move(current_face, current_value)
            if move:
                simplified.append(move)
                
            i = j
        
        # Second pass: Combine moves by face groups
        final_simplified = []
        i = 0
        while i < len(simplified):
            current_face = simplified[i][0]
            
            # Find which group this face belongs to
            face_group = None
            for group in face_groups:
                if current_face in group:
                    face_group = group
                    break
            
            if face_group:
                # Count moves for each face in this group
                counts = {face: 0 for face in face_group}
                j = i
                
                # Collect consecutive moves in this group
                while j < len(simplified) and simplified[j][0] in face_group:
                    face = simplified[j][0]
                    counts[face] += move_value(simplified[j])
                    j += 1
                    
                # Add simplified moves for this group
                for face in face_group:
                    move = value_to_move(face, counts[face])
                    if move:
                        final_simplified.append(move)
                        
                i = j
            else:
                # For faces not in any group (which shouldn't happen with standard cube notation)
                final_simplified.append(simplified[i])
                i += 1
        
        return " ".join(final_simplified) if final_simplified else "No moves"

    def solve_cube(self, cube_frblud):
        """Solve the cube using kociemba algorithm."""
        try:
            # First check if cube is already solved
            if self.is_cube_solved(cube_frblud):
                print("\nCube is already solved! No moves needed.")
                return ""
                
            # If not solved, proceed with normal solving process
            color_map, cube_frblud_remapped = self.remap_colors_to_kociemba(cube_frblud)
            scrambled_kociemba = self.remap_cube_to_kociemba(cube_frblud_remapped)
            solved_frblud = self.get_solved_state(cube_frblud, color_map)
            _, solved_frblud_remapped = self.remap_colors_to_kociemba(solved_frblud)
            solved_kociemba = self.remap_cube_to_kociemba(solved_frblud_remapped)
            
            print("\nValidating cube states...")
            self.validate_cube(scrambled_kociemba, "Scrambled Kociemba")
            self.validate_cube(solved_kociemba, "Solved Kociemba")
            
            print("\nScrambled Kociemba state:", scrambled_kociemba)
            print("Solved Kociemba state:", solved_kociemba)
            
            print("\nAttempting to solve with kociemba...")
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
            
            # Optimize the solution using simplify_cube_moves
            optimized_solution = self.simplify_cube_moves(final_solution)
            print("\nOriginal solution length:", len(final_solution.split()))
            print("Optimized solution length:", len(optimized_solution.split()))
            
            return optimized_solution
        
        except Exception as e:
            print(f"\nError in solve_cube: {str(e)}")
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
            self.mode = "scrambling"
            self.status_message = "Executing scramble sequence"
            self.current_solve_move_index = 0
            self.total_solve_moves = len(scramble)
            
            # Send scramble to Arduino and wait for completion
            if not self.send_arduino_command(scramble_sequence):
                self.mode = "error"
                self.error_message = "Failed to execute scramble sequence"
                return False
            
            # Return to idle mode
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