import cv2
import numpy as np
import time
import asyncio
import base64
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GameSession:
    def __init__(self, config=None, esp32_client=None):
        """Initialize the sorting game with optional configuration."""
        self.esp32_client = esp32_client
        self.switch_command_sent = False
        self.running = True
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.fps = 0
        self.status_message = "Game initialized. Please position the board."
        
        # Game state variables
        self.board_detected = False
        self.warped_board = None
        self.grid_shapes = [["Unknown" for _ in range(4)] for _ in range(2)]  # Default is Unknown
        self.required_swaps = []
        self.current_move = None
        self.game_completed = False
        self.game_started = False
        
        # Detection enhancement variables
        self.detection_frames_count = 0
        self.detection_attempt_interval = 5  # Try to detect every 5 frames
        self.last_corners = None  # Store last successful detection for stability
        self.corner_stability_threshold = 0.85  # Controls corner position stability (0-1)
        
        # Constants
        self.BOARD_HEIGHT = 200
        self.BOARD_WIDTH = 400  # Width is double the height
        self.GRID_ROWS = 2
        self.GRID_COLS = 4
        
        # Arm control values - positions mapped to grid cells
        self.arm_positions = {
            (0, 0): [150, 150, 170], (0, 1): [140, 120, 165], 
            (0, 2): [125, 90, 155], (0, 3): [125, 60, 155],
            (1, 0): [150,55,155], (1, 1): [130,80,140], 
            (1, 2): [130,105,140], (1, 3): [150,125,155]
        }
        self.ARM_HOME = [180, 0, 0] # User's original value
        
        # Enable/pickup signals interpretation:
        # ENABLE_ACTIVE (1): Arm is in a travel phase.
        # ENABLE_INACTIVE (0): Arm is at a key point for action (pickup/drop) or rest.
        # PICKUP_TRUE (1): Magnet is ON.
        # PICKUP_FALSE (0): Magnet is OFF.
        self.ENABLE_ACTIVE = 1
        self.ENABLE_INACTIVE = 0
        self.PICKUP_TRUE = 1
        self.PICKUP_FALSE = 0
        
        # Process config if provided
        if config:
            if isinstance(config, dict):
                if "board_width" in config:
                    self.BOARD_WIDTH = config["board_width"]
                if "board_height" in config:
                    self.BOARD_HEIGHT = config["board_height"]
        
        logger.info("Sorting Game initialized")

    async def _send_switch_command(self):
        """Send a switch command to ESP32 to activate ARM mode"""
        if self.esp32_client is None:
            logger.error("No ESP32 client available, skipping switch command")
            return False
            
        try:
            logger.info("Sending switch command to activate ARM mode")
            await self.esp32_client.send_json({
                "action": "switch",
                "game": "ARM"
            })
            self.switch_command_sent = True
            return True
        except Exception as e:
            logger.error(f"Error sending switch command: {e}")
            return False
    
    async def send_command_to_esp32(self, servo1_angle, servo2_angle, servo3_angle, enable_signal, pickup_flag):
        """Send servo command to ESP32 via WebSocket"""
        if self.esp32_client is None:
            logger.error("No ESP32 client available, skipping command")
            return False
            
        try:
            s1 = int(servo1_angle)
            s2 = int(servo2_angle)
            s3 = int(servo3_angle)
            enable = int(enable_signal)
            pickup = int(pickup_flag)
            
            command = f"{s1},{s2},{s3},{enable},{pickup}"
            logger.info(f"Sending command: {command}") # Logs the "arm values" sent
            
            # This is where the command, including angles, is packaged into a JSON
            # and sent via the esp32_client instance.
            await self.esp32_client.send_json({
                "action": "command",
                "command": command
            })
            return True
        except Exception as e:
            logger.error(f"Error sending command: {e}")
            return False
    
    async def move_robot_arm(self, from_pos, to_pos):
        """Move robot arm to swap pieces between two positions using a temporary position.
        Sends commands one by one to the ESP32.
        """
        if self.esp32_client is None:
            logger.error("No ESP32 client available, skipping robot movement")
            return False
        
        logger.info(f"SWAP OPERATION: Moving piece from {from_pos} to {to_pos}")
        
        if from_pos not in self.arm_positions or to_pos not in self.arm_positions:
            logger.error(f"Invalid position: {from_pos} or {to_pos} not in arm_positions")
            return False
        
        from_angles = self.arm_positions[from_pos]
        to_angles = self.arm_positions[to_pos]
        home_angles = self.ARM_HOME
        temp_position = [90, 0, 120] # Example temporary position
        
        move_delay = 2.0  # Time to wait after each command for ESP32 to process and arm to move
        
        sequence_steps = self._generate_swap_sequence_commands(from_pos, to_pos)
        
        try:
            for step in sequence_steps:
                logger.info(f"[ARM EXEC] {step['desc']}: Angles {step['angles']}, Enable {step['enable']}, Pickup {step['pickup']}")
                success = await self.send_command_to_esp32(
                    step['angles'][0], step['angles'][1], step['angles'][2],
                    step['enable'], step['pickup']
                )
                if not success:
                    logger.error(f"Failed to send command for step: {step['desc']}. Aborting swap.")
                    # Attempt to return arm to home on failure
                    await self.send_command_to_esp32(*home_angles, self.ENABLE_ACTIVE, self.PICKUP_FALSE) # Travel home empty
                    await asyncio.sleep(move_delay)
                    await self.send_command_to_esp32(*home_angles, self.ENABLE_INACTIVE, self.PICKUP_FALSE) # Rest at home empty
                    return False
                await asyncio.sleep(move_delay)
            
            logger.info(f"[ARM DEBUG] Swap sequence completed successfully: {from_pos} ⟷ {to_pos}")
            return True
            
        except Exception as e:
            logger.error(f"Error during robot movement: {e}", exc_info=True)
            try:
                logger.info("[ARM DEBUG] Error occurred - attempting emergency return to home")
                await self.send_command_to_esp32(*home_angles, self.ENABLE_ACTIVE, self.PICKUP_FALSE)
                await asyncio.sleep(move_delay)
                await self.send_command_to_esp32(*home_angles, self.ENABLE_INACTIVE, self.PICKUP_FALSE)
            except Exception as home_error:
                logger.error(f"Failed to return to home after error: {home_error}")
            return False

    def _generate_swap_sequence_commands(self, from_pos_tuple, to_pos_tuple):
        """
        Generates the precise list of arm commands for a swap operation.
        Each command is a dictionary: {"desc": str, "angles": list, "enable": int, "pickup": int}
        """
        from_angles = self.arm_positions[from_pos_tuple]
        to_angles = self.arm_positions[to_pos_tuple]
        home_angles = self.ARM_HOME
        temp_angles = [90, 0, 120] # Consistent temporary position

        commands = []

        # Part 1: Move piece from `from_pos_tuple` to `temp_angles`
        commands.extend([
            {"desc": f"Move to {from_pos_tuple} to pick", "angles": from_angles, "enable": self.ENABLE_INACTIVE, "pickup": self.PICKUP_TRUE},
            {"desc": f"Travel to home from {from_pos_tuple} (with piece)", "angles": home_angles, "enable": self.ENABLE_ACTIVE, "pickup": self.PICKUP_TRUE},
            {"desc": f"Move to temp {temp_angles} to release", "angles": temp_angles, "enable": self.ENABLE_INACTIVE, "pickup": self.PICKUP_FALSE},
            {"desc": "Travel to home from temp (empty)", "angles": home_angles, "enable": self.ENABLE_ACTIVE, "pickup": self.PICKUP_FALSE},
        ])

        # Part 2: Move piece from `to_pos_tuple` to `from_pos_tuple`
        commands.extend([
            {"desc": f"Move to {to_pos_tuple} to pick", "angles": to_angles, "enable": self.ENABLE_INACTIVE, "pickup": self.PICKUP_TRUE},
            {"desc": f"Travel to home from {to_pos_tuple} (with piece)", "angles": home_angles, "enable": self.ENABLE_ACTIVE, "pickup": self.PICKUP_TRUE},
            {"desc": f"Move to {from_pos_tuple} to release", "angles": from_angles, "enable": self.ENABLE_INACTIVE, "pickup": self.PICKUP_FALSE},
            {"desc": f"Travel to home from {from_pos_tuple} (empty)", "angles": home_angles, "enable": self.ENABLE_ACTIVE, "pickup": self.PICKUP_FALSE},
        ])

        # Part 3: Move piece from `temp_angles` to `to_pos_tuple`
        commands.extend([
            {"desc": f"Move to temp {temp_angles} to pick", "angles": temp_angles, "enable": self.ENABLE_INACTIVE, "pickup": self.PICKUP_TRUE},
            {"desc": "Travel to home from temp (with piece)", "angles": home_angles, "enable": self.ENABLE_ACTIVE, "pickup": self.PICKUP_TRUE},
            {"desc": f"Move to {to_pos_tuple} to release", "angles": to_angles, "enable": self.ENABLE_INACTIVE, "pickup": self.PICKUP_FALSE},
            {"desc": f"Travel to home from {to_pos_tuple} (empty, final)", "angles": home_angles, "enable": self.ENABLE_ACTIVE, "pickup": self.PICKUP_FALSE},
            {"desc": "Rest at home (final)", "angles": home_angles, "enable": self.ENABLE_INACTIVE, "pickup": self.PICKUP_FALSE}, # Ensure arm is stationary and magnet off
        ])
        return commands

    def find_board_corners(self, frame):
        """Find the four corners of a white board on a black background."""
        if frame is None:
            return None

        # Convert to grayscale and HSV for robust detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Threshold for white in HSV (tuned for white board on black bg)
        lower_white_hsv = np.array([0, 0, 180])
        upper_white_hsv = np.array([180, 50, 255])
        hsv_mask = cv2.inRange(hsv, lower_white_hsv, upper_white_hsv)

        # Morphological operations to clean up mask
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self.last_corners

        # Only consider the largest contour (should be the board)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        frame_area = frame.shape[0] * frame.shape[1]
        if area < frame_area * 0.05 or area > frame_area * 0.95:
            return self.last_corners

        # Approximate to polygon
        peri = cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
        if len(approx) != 4:
            # Try convex hull if not 4 corners
            hull = cv2.convexHull(largest_contour)
            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
            if len(approx) != 4:
                return self.last_corners

        corners = np.array([point[0] for point in approx], dtype=np.float32)
        sorted_corners = self.sort_corners(corners)
        self.last_corners = sorted_corners
        return sorted_corners

    def sort_corners(self, corners):
        """Sort corners in order: top-left, top-right, bottom-right, bottom-left."""
        # Use sum and diff for robust sorting
        s = corners.sum(axis=1)
        rect = np.zeros((4, 2), dtype="float32")
        rect[0] = corners[np.argmin(s)]  # Top-left
        rect[2] = corners[np.argmax(s)]  # Bottom-right
        diff = np.diff(corners, axis=1)
        rect[1] = corners[np.argmin(diff)]  # Top-right
        rect[3] = corners[np.argmax(diff)]  # Bottom-left
        return rect

    def transform_board(self, frame, corners):
        """Apply perspective transform to get top-down view of board."""
        if frame is None or corners is None:
            return None
        dst_points = np.array([
            [0, 0],
            [self.BOARD_WIDTH - 1, 0],
            [self.BOARD_WIDTH - 1, self.BOARD_HEIGHT - 1],
            [0, self.BOARD_HEIGHT - 1]
        ], dtype=np.float32)
        try:
            M = cv2.getPerspectiveTransform(corners, dst_points)
            warped = cv2.warpPerspective(frame, M, (self.BOARD_WIDTH, self.BOARD_HEIGHT))
            return warped
        except Exception as e:
            logger.error(f"Error during perspective transform: {e}")
            return None

    def detect_shapes(self, warped_board):
        """Detect shapes in a 2x4 grid on the warped board (which is now tightly fit to the board)."""
        if warped_board is None:
            return [["Unknown" for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)], None

        board_height, board_width = warped_board.shape[:2]
        cell_width = board_width // self.GRID_COLS
        cell_height = board_height // self.GRID_ROWS
        shapes_grid = [["Unknown" for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]

        display_grid = warped_board.copy()

        for i in range(self.GRID_ROWS):
            for j in range(self.GRID_COLS):
                x1, y1 = j * cell_width, i * cell_height
                x2, y2 = (j + 1) * cell_width, (i + 1) * cell_height

                # Draw cell boundaries (on warped board only)
                cv2.rectangle(display_grid, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Add some padding to avoid edge effects
                padding = 5
                x1 += padding
                y1 += padding
                x2 -= padding
                y2 -= padding

                if x1 < x2 and y1 < y2:
                    cell = warped_board[y1:y2, x1:x2]

                    # Process the cell
                    if cell.size > 0:
                        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
                        blur = cv2.GaussianBlur(gray, (5, 5), 0)

                        # Use adaptive thresholding for better shape segmentation
                        thresh = cv2.adaptiveThreshold(
                            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
                        )

                        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        shape = "Unknown"  # Default to Unknown
                        max_area = 0
                        best_contour = None

                        # Select the largest contour
                        for cnt in contours:
                            area = cv2.contourArea(cnt)
                            if area > max_area and area > 50:
                                max_area = area
                                best_contour = cnt

                        if best_contour is not None:
                            peri = cv2.arcLength(best_contour, True)
                            epsilon = 0.04 * peri  # Adjusted for finer approximation
                            approx = cv2.approxPolyDP(best_contour, epsilon, True)
                            corners = len(approx)

                            if corners == 3:
                                shape = "Triangle"
                            elif corners == 4:
                                shape = "Square"
                            elif corners == 5:
                                shape = "Pentagon"
                            else:
                                # Circle detection requires more careful handling
                                if len(best_contour) > 5:
                                    circularity = 4 * np.pi * max_area / (peri * peri)
                                    if circularity > 0.7:
                                        shape = "Circle"
                                    else:
                                        shape = "Unknown"
                                else:
                                    shape = "Unknown"

                            # Draw shape label on display grid
                            label_x = x1 + (x2 - x1) // 3
                            label_y = y1 + (y2 - y1) // 2
                            cv2.putText(display_grid, shape, (label_x, label_y), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                        shapes_grid[i][j] = shape

        return shapes_grid, display_grid

    def determine_swaps(self, grid):
        """Determine necessary swaps to make bottom row match top row."""
        if grid is None or len(grid) != 2 or len(grid[0]) != self.GRID_COLS:
            return []
            
        target_row = grid[0]  # First row as reference
        current_row = grid[1]  # Second row to be modified
        
        # Store which shapes we've already matched
        matched_indices = []
        swaps_needed = []
        
        # First, check for shapes that are already matched
        for i in range(self.GRID_COLS):
            if current_row[i] == target_row[i] and current_row[i] != "Unknown":
                matched_indices.append(i)
        
        # For each unmatched position
        for i in range(self.GRID_COLS):
            if i in matched_indices:
                continue
                
            target_shape = target_row[i]
            if target_shape == "Unknown":
                continue
                
            # Find this shape in the current row
            found = False
            for j in range(self.GRID_COLS):
                if j in matched_indices:
                    continue
                if current_row[j] == target_shape:
                    # Found a match, record the swap
                    if i != j:
                        swaps_needed.append(((1, j), (1, i)))
                    matched_indices.append(j)
                    found = True
                    break
        
        return swaps_needed
    
    def check_game_completion(self):
        """Check if the game is completed (all shapes in bottom row match top row)."""
        if not self.grid_shapes or len(self.grid_shapes) < 2:
            return False
            
        target_row = self.grid_shapes[0]
        current_row = self.grid_shapes[1]
        
        for i in range(self.GRID_COLS):
            if target_row[i] != current_row[i] or target_row[i] == "Unknown" or current_row[i] == "Unknown":
                return False
                
        return True
    
    def stop(self):
        """Stop the game processing."""
        self.running = False
    
    def get_next_swap_arm_values(self):
        """Return the arm values (angles and signals) for the next swap, or None if no swap."""
        if not self.required_swaps:
            return None
        from_pos, to_pos = self.required_swaps[0]
        return self._generate_swap_sequence_commands(from_pos, to_pos)

    async def process_frame(self, frame_bytes):
        """Process a frame to detect the board, shapes, and determine necessary moves."""
        # Send the switch command on the first frame if not sent already
        if not self.switch_command_sent and self.esp32_client is not None:
            await self._send_switch_command()

        try:
            # Decode the frame
            np_arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                return {"status": "error", "message": "Invalid frame"}
            
            # Calculate FPS
            current_time = time.time()
            self.frame_count += 1
            self.detection_frames_count += 1
            
            if current_time - self.last_frame_time >= 1.0:
                self.fps = self.frame_count / (current_time - self.last_frame_time)
                self.last_frame_time = current_time
                self.frame_count = 0

            # Create a display frame for visualization
            display_frame = frame.copy()
            
            # Try to find board corners on a regular basis
            corners = None
            if self.detection_frames_count >= self.detection_attempt_interval:
                self.detection_frames_count = 0
                corners = self.find_board_corners(frame)
            elif self.last_corners is not None:
                # Use last known corners between detection attempts for stability
                corners = self.last_corners
            
            warped_display = None
            
            if corners is not None:
                # Draw detected corners on the display frame
                for corner in corners:
                    cv2.circle(display_frame, (int(corner[0]), int(corner[1])), 5, (0, 255, 0), -1)
                cv2.polylines(display_frame, [corners.astype(np.int32)], True, (0, 255, 0), 2)
                
                # Transform board to top-down view
                self.warped_board = self.transform_board(frame, corners)
                if self.warped_board is not None:
                    # Detect shapes in warped board
                    self.grid_shapes, warped_display = self.detect_shapes(self.warped_board)
                    self.board_detected = True

                    # Always update required_swaps after detection
                    self.required_swaps = self.determine_swaps(self.grid_shapes)

                    # If game is started and not completed, update status and check completion
                    if self.game_started and not self.game_completed:
                        if len(self.required_swaps) > 0:
                            self.status_message = f"Detected {len(self.required_swaps)} swaps needed"
                        else:
                            if self.check_game_completion():
                                self.game_completed = True
                                self.status_message = "Game completed! All shapes match."
                            else:
                                self.status_message = "No swaps detected, but shapes don't match yet."
            else:
                self.board_detected = False
                self.warped_board = None
                self.status_message = "Board not detected. Please position the board."

            # Add status message to display frame
            cv2.putText(display_frame, f"FPS: {self.fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, self.status_message, (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
            # If game started, show the current state
            if self.game_started:
                cv2.putText(display_frame, "Game Started", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if self.current_move:
                    from_pos, to_pos = self.current_move
                    cv2.putText(display_frame, f"Current Move: {from_pos} -> {to_pos}", (10, 120), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            # If game completed, show congratulations
            if self.game_completed:
                cv2.putText(display_frame, "GAME COMPLETED!", (display_frame.shape[1]//4, display_frame.shape[0]//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            # Encode the frames for sending via WebSocket
            _, main_buffer = cv2.imencode('.jpg', display_frame)
            main_b64 = base64.b64encode(main_buffer).decode('utf-8')
            
            warped_b64 = None
            if warped_display is not None:
                _, warped_buffer = cv2.imencode('.jpg', warped_display)
                warped_b64 = base64.b64encode(warped_buffer).decode('utf-8')
            
            # Prepare response with current game state
            response = {
                "status": "ok",
                "frame": main_b64,
                "warped_frame": warped_b64,
                "board_detected": self.board_detected,
                "game_started": self.game_started,
                "game_completed": self.game_completed,
                "status_message": self.status_message,
                "grid_shapes": self.grid_shapes if self.grid_shapes else [["Unknown"] * 4, ["Unknown"] * 4],
                "required_swaps": self.required_swaps,  # Keep for debugging
                "total_swaps": len(self.required_swaps),
                "next_swap": self.required_swaps[0] if self.required_swaps else None,
                "current_move": self.current_move,
                "next_swap_arm_values": self.get_next_swap_arm_values(), 
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}", exc_info=True)
            return {"status": "error", "message": f"Error processing frame: {str(e)}"}
    
    async def process_command(self, command):
        """Process a command received from the WebSocket."""
        try:
            if not isinstance(command, dict):
                return {"status": "error", "message": "Invalid command format"}
            
            action = command.get("action")
            
            if action == "start_game":
                if not self.board_detected:
                    return {"status": "error", "message": "Board not detected. Please position the board first."}
                
                self.game_started = True
                self.game_completed = False
                # Calculate swaps immediately on game start
                self.required_swaps = self.determine_swaps(self.grid_shapes)
                if self.check_game_completion():
                    self.game_completed = True
                    self.status_message = "Game started! Board already solved."
                elif len(self.required_swaps) > 0:
                    self.status_message = f"Game started! {len(self.required_swaps)} swaps needed."
                else:
                    self.status_message = "Game started! Shapes appear in order or no valid swaps found."
                return {
                    "status": "ok",
                    "message": "Game started",
                    "total_swaps": len(self.required_swaps),
                    "next_swap": self.required_swaps[0] if self.required_swaps else None,
                }
                
            elif action == "reset_game":
                self.game_started = False
                self.game_completed = False
                self.required_swaps = []
                self.current_move = None
                self.grid_shapes = [["Unknown" for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
                self.status_message = "Game reset. Position the board to start again."
                return {
                    "status": "ok",
                    "message": "Game reset",
                    "total_swaps": 0,
                    "next_swap": None,
                }
                
            elif action == "execute_swap":
                if not self.game_started:
                    return {"status": "error", "message": "Game not started yet"}
                
                if self.game_completed:
                    return {"status": "error", "message": "Game already completed"}
                    
                if not self.required_swaps:
                    return {"status": "error", "message": "No swaps required"}
                
                # Get the next swap
                from_pos, to_pos = self.required_swaps[0]
                self.current_move = (from_pos, to_pos)
                
                logger.info(f"[SWAP DEBUG] Executing swap between positions: {from_pos} ({self.grid_shapes[from_pos[0]][from_pos[1]]}) and {to_pos} ({self.grid_shapes[to_pos[0]][to_pos[1]]})")
                
                self.status_message = f"Executing swap: {from_pos} -> {to_pos}"
                
                # Execute the swap using robot arm
                success = await self.move_robot_arm(from_pos, to_pos)
                
                if success:
                    # Update the grid to reflect the swap
                    i1, j1 = from_pos
                    i2, j2 = to_pos
                    
                    logger.info(f"[SWAP DEBUG] Swapping grid positions in software:")
                    logger.info(f"[SWAP DEBUG] Before - Grid[{i1}][{j1}]: {self.grid_shapes[i1][j1]}, Grid[{i2}][{j2}]: {self.grid_shapes[i2][j2]}")
                    
                    self.grid_shapes[i1][j1], self.grid_shapes[i2][j2] = self.grid_shapes[i2][j2], self.grid_shapes[i1][j1]
                    
                    logger.info(f"[SWAP DEBUG] After - Grid[{i1}][{j1}]: {self.grid_shapes[i1][j1]}, Grid[{i2}][{j2}]: {self.grid_shapes[i2][j2]}")
                    
                    # Remove the completed swap from the list
                    self.required_swaps.pop(0)
                    self.current_move = None
                    
                    # Check if game is completed
                    if self.check_game_completion():
                        self.game_completed = True
                        self.status_message = "Game completed! All shapes match."
                    elif not self.required_swaps:
                        self.status_message = "All swaps completed! Checking results..."
                    else:
                        self.status_message = f"{len(self.required_swaps)} swaps remaining"
                    
                    return {
                        "status": "ok",
                        "message": "Swap executed successfully",
                        "total_swaps": len(self.required_swaps),
                        "next_swap": self.required_swaps[0] if self.required_swaps else None,
                    }
                else:
                    self.current_move = None
                    self.status_message = "Failed to execute swap"
                    return {
                        "status": "error",
                        "message": "Failed to execute swap",
                        "total_swaps": len(self.required_swaps),
                        "next_swap": self.required_swaps[0] if self.required_swaps else None,
                    }
                    
            elif action == "get_state":
                return {
                    "status": "ok",
                    "board_detected": self.board_detected,
                    "game_started": self.game_started,
                    "game_completed": self.game_completed,
                    "status_message": self.status_message,
                    "grid_shapes": self.grid_shapes if self.grid_shapes else [["Unknown"] * 4, ["Unknown"] * 4],
                    "required_swaps": self.required_swaps,
                    "total_swaps": len(self.required_swaps),
                    "next_swap": self.required_swaps[0] if self.required_swaps else None,
                    "current_move": self.current_move,
                    "next_swap_arm_values": self.get_next_swap_arm_values(),
                }
            
            else:
                return {"status": "error", "message": f"Unknown action: {action}"}
                
        except Exception as e:
            logger.error(f"Error processing command: {str(e)}", exc_info=True)
            return {"status": "error", "message": f"Error processing command: {str(e)}"}
