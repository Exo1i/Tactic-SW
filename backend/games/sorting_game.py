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
        self.grid_shapes = [[None for _ in range(4)] for _ in range(2)]
        self.required_swaps = []
        self.current_move = None
        self.game_completed = False
        self.game_started = False
        
        # Constants
        self.BOARD_HEIGHT = 200
        self.BOARD_WIDTH = 400  # Width is double the height
        self.GRID_ROWS = 2
        self.GRID_COLS = 4
        
        # Arm control values - positions mapped to grid cells
        self.arm_positions = {
            (0, 0): [150, 150, 170], (0, 1): [140, 120, 165], 
            (0, 2): [125, 90, 155], (0, 3): [125, 60, 155],
            (1, 0): [90, 150, 130], (1, 1): [90, 120, 130], 
            (1, 2): [80, 90, 125], (1, 3): [80, 60, 125]
        }
        self.ARM_HOME = [180, 90, 0]
        
        # Enable/pickup signals
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
            logger.info(f"Sending command: {command}")
            
            await self.esp32_client.send_json({
                "action": "command",
                "command": command
            })
            return True
        except Exception as e:
            logger.error(f"Error sending command: {e}")
            return False
    
    async def move_robot_arm(self, from_pos, to_pos):
        """Move robot arm to pick up a shape from one position and move it to another"""
        if self.esp32_client is None:
            logger.error("No ESP32 client available, skipping robot movement")
            return False
        
        logger.info(f"Moving piece from {from_pos} to {to_pos}")
        
        # Get positions
        if from_pos not in self.arm_positions or to_pos not in self.arm_positions:
            logger.error(f"Invalid position: {from_pos} or {to_pos} not in arm_positions")
            return False
        
        from_angles = self.arm_positions[from_pos]
        to_angles = self.arm_positions[to_pos]
        home_angles = self.ARM_HOME
        
        move_delay = 2.0  # Time to wait between arm movements
        
        try:
            # Step 1: Move to source position with magnet off
            logger.info(f"Moving to source position {from_pos}")
            await self.send_command_to_esp32(*from_angles, self.ENABLE_ACTIVE, self.PICKUP_FALSE)
            await asyncio.sleep(move_delay)
            
            # Step 2: Turn on magnet to pick up the piece
            logger.info("Activating magnet to pick up piece")
            await self.send_command_to_esp32(*from_angles, self.ENABLE_ACTIVE, self.PICKUP_TRUE)
            await asyncio.sleep(move_delay)
            
            # Step 3: Move to home position with piece
            logger.info("Moving to home position")
            await self.send_command_to_esp32(*home_angles, self.ENABLE_ACTIVE, self.PICKUP_TRUE)
            await asyncio.sleep(move_delay)
            
            # Step 4: Move to destination position with piece
            logger.info(f"Moving to destination position {to_pos}")
            await self.send_command_to_esp32(*to_angles, self.ENABLE_ACTIVE, self.PICKUP_TRUE)
            await asyncio.sleep(move_delay)
            
            # Step 5: Release piece
            logger.info("Releasing piece")
            await self.send_command_to_esp32(*to_angles, self.ENABLE_ACTIVE, self.PICKUP_FALSE)
            await asyncio.sleep(move_delay)
            
            # Step 6: Return to home position
            logger.info("Returning to home position")
            await self.send_command_to_esp32(*home_angles, self.ENABLE_INACTIVE, self.PICKUP_FALSE)
            await asyncio.sleep(move_delay)
            
            logger.info(f"Movement from {from_pos} to {to_pos} completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during robot movement: {e}")
            # Try to return to home position in case of error
            try:
                await self.send_command_to_esp32(*home_angles, self.ENABLE_INACTIVE, self.PICKUP_FALSE)
            except Exception as home_error:
                logger.error(f"Failed to return to home after error: {home_error}")
            return False
    
    def find_board_corners(self, frame):
        """Find the four corners of a white frame on a dark background."""
        if frame is None:
            return None
            
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 50, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter by area to avoid small artifacts
            if area < 10000 or area > (frame.shape[0] * frame.shape[1] * 0.8):
                continue
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) == 4:
                candidates.append(approx)

        if not candidates:
            return None

        largest_contour = max(candidates, key=cv2.contourArea)
        corners = np.array([point[0] for point in largest_contour], dtype=np.float32)
        return self.sort_corners(corners)

    def sort_corners(self, corners):
        """Sort corners in order: top-left, top-right, bottom-right, bottom-left."""
        # First sort by sum of coordinates (x+y)
        s = corners.sum(axis=1)
        rect = np.zeros((4, 2), dtype="float32")
        rect[0] = corners[np.argmin(s)]  # Top-left has smallest sum
        rect[2] = corners[np.argmax(s)]  # Bottom-right has largest sum
        
        # Then sort by difference of coordinates (x-y)
        diff = np.diff(corners, axis=1)
        rect[1] = corners[np.argmin(diff)]  # Top-right has smallest difference
        rect[3] = corners[np.argmax(diff)]  # Bottom-left has largest difference
        
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
        """Detect shapes in a 2x4 grid on the warped board."""
        if warped_board is None:
            return None
            
        board_height, board_width = warped_board.shape[:2]
        cell_width = board_width // self.GRID_COLS
        cell_height = board_height // self.GRID_ROWS
        shapes_grid = [[None for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        
        display_grid = warped_board.copy()
        
        for i in range(self.GRID_ROWS):
            for j in range(self.GRID_COLS):
                x1, y1 = j * cell_width, i * cell_height
                x2, y2 = (j + 1) * cell_width, (i + 1) * cell_height
                
                # Draw cell boundaries
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
                        shape = "Unknown"
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
                                    # Check circularity using area/perimeter ratio
                                    circularity = 4 * np.pi * area / (peri * peri)
                                    if circularity > 0.7:  # Higher value means more circular
                                        shape = "Circle"
                            
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
            if current_time - self.last_frame_time >= 1.0:
                self.fps = self.frame_count / (current_time - self.last_frame_time)
                self.last_frame_time = current_time
                self.frame_count = 0

            # Create a display frame for visualization
            display_frame = frame.copy()
            
            # Try to find board corners
            corners = self.find_board_corners(frame)
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
                    
                    # Determine the required swaps if game is started
                    if self.game_started and not self.game_completed:
                        new_swaps = self.determine_swaps(self.grid_shapes)
                        
                        # Update required swaps if they change
                        if new_swaps != self.required_swaps:
                            self.required_swaps = new_swaps
                            if len(self.required_swaps) > 0:
                                self.status_message = f"Detected {len(self.required_swaps)} swaps needed"
                            else:
                                # Check if we've actually completed the game
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
                "required_swaps": self.required_swaps,
                "current_move": self.current_move
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
                self.status_message = "Game started! Analyzing shapes..."
                return {"status": "ok", "message": "Game started"}
                
            elif action == "reset_game":
                self.game_started = False
                self.game_completed = False
                self.required_swaps = []
                self.current_move = None
                self.status_message = "Game reset. Position the board to start again."
                return {"status": "ok", "message": "Game reset"}
                
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
                self.status_message = f"Executing swap: {from_pos} -> {to_pos}"
                
                # Execute the swap using robot arm
                success = await self.move_robot_arm(from_pos, to_pos)
                
                if success:
                    # Update the grid to reflect the swap
                    i1, j1 = from_pos
                    i2, j2 = to_pos
                    self.grid_shapes[i1][j1], self.grid_shapes[i2][j2] = self.grid_shapes[i2][j2], self.grid_shapes[i1][j1]
                    
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
                    
                    return {"status": "ok", "message": "Swap executed successfully"}
                else:
                    self.current_move = None
                    self.status_message = "Failed to execute swap"
                    return {"status": "error", "message": "Failed to execute swap"}
                    
            elif action == "get_state":
                return {
                    "status": "ok",
                    "board_detected": self.board_detected,
                    "game_started": self.game_started,
                    "game_completed": self.game_completed,
                    "status_message": self.status_message,
                    "grid_shapes": self.grid_shapes if self.grid_shapes else [["Unknown"] * 4, ["Unknown"] * 4],
                    "required_swaps": self.required_swaps,
                    "current_move": self.current_move
                }
            
            else:
                return {"status": "error", "message": f"Unknown action: {action}"}
                
        except Exception as e:
            logger.error(f"Error processing command: {str(e)}", exc_info=True)
            return {"status": "error", "message": f"Error processing command: {str(e)}"}
