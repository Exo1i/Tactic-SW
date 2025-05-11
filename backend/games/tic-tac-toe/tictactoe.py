import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import cv2
import numpy as np
import base64
import time
import asyncio
from tensorflow.keras.models import load_model
from .utils import detections, imutils
from alphabeta import Tic, get_enemy, determine

class GameSession:
    def __init__(self, config, esp32_client=None):
        try:
            model_path = config.get("model", "data/model.h5")
            if not os.path.exists(model_path):
                print(f"Model not found at {model_path}, searching in backend directory...")
                backend_model = os.path.join("backend", model_path)
                if os.path.exists(backend_model):
                    model_path = backend_model
            self.model = load_model(model_path)
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

        # Game state
        self.zoom = float(config.get("zoom", 1.0))
        self.check_interval = float(config.get("check_interval", 3.0))
        self.board = Tic()
        self.history = {}
        self.previous_state = [None] * 9
        self.last_check_time = time.time()
        self.move_detected_in_last_cycle = False
        self.paper_detection_threshold = 170
        self.grid_detection_threshold = 170
        
        # ESP32 WebSocket client
        self.esp32_client = esp32_client
        self.switch_command_sent = False
        
        # Servo position mapping
        self.angle_map = {
            0: [150, 110, 170], 1: [150, 90, 170], 2: [150, 72, 180],
            3: [120, 120, 132], 4: [110, 90, 128], 5: [125, 65, 132],
            6: [75, 125, 120], 7: [60, 90, 120], 8: [90, 50, 120],
            'home': [180, 0, 0],
            'pickup3': [135, 57, 140], 'pickup2': [145, 140, 145],
            'pickup1': [120, 45, 130], 'pickup0': [120, 150, 130],
        }
        self.next_pickup_index = 0
        self.NUM_PICKUP_POSITIONS = 4
        self.ENABLE_ACTIVE = 1
        self.ENABLE_INACTIVE = 0
        self.PICKUP_TRUE = 0
        self.PICKUP_FALSE = 1

    async def _send_switch_command(self):
        """Send a switch command to ESP32 to activate ARM mode"""
        if self.esp32_client is None:
            print("[ESP32] No ESP32 client available, skipping switch command")
            return False
            
        try:
            print("[ESP32] Sending switch command to activate ARM mode")
            await self.esp32_client.send_json({
                "action": "switch",
                "game": "ARM"
            })
            self.switch_command_sent = True
            return True
        except Exception as e:
            print(f"[ESP32] Error sending switch command: {e}")
            return False

    def zoom_frame(self, frame, zoom=2.0):
        """Zoom into the center of the frame by the specified factor."""
        if frame is None or zoom <= 1.0:
            return frame

        height, width = frame.shape[:2]
        if height == 0 or width == 0:
            return frame

        center_x, center_y = width // 2, height // 2
        roi_width = int(width / zoom)
        roi_height = int(height / zoom)
        
        roi_width = max(1, roi_width)
        roi_height = max(1, roi_height)
        
        x1 = center_x - roi_width // 2
        y1 = center_y - roi_height // 2
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x1 + roi_width)
        y2 = min(height, y1 + roi_height)
        
        x1 = max(0, x2 - roi_width)
        y1 = max(0, y2 - roi_height)
        
        try:
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return frame
            return cv2.resize(roi, (width, height), interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            return frame

    def find_sheet_paper(self, frame, thresh, add_margin=True):
        """Detect the sheet of paper and transform to bird's eye view."""
        if frame is None or thresh is None:
            return None, None

        stats = detections.find_corners(thresh)
        if stats is None or len(stats) < 5:
            return None, None

        # First point is center of coordinate system
        corners = stats[1:, :2]
        corners = imutils.order_points(corners)
        if corners is None:
            return None, None

        # Get bird's eye view transformation
        try:
            paper = imutils.four_point_transform(frame, corners)
        except Exception as e:
            print(f"Error in four_point_transform: {e}")
            return None, None

        if paper is None or paper.size == 0:
            return None, None

        # Add margin if needed
        if add_margin:
            h, w = paper.shape[:2]
            margin = 10
            if h > 2 * margin and w > 2 * margin:
                paper = paper[margin:-margin, margin:-margin]

        if paper is None or paper.size == 0:
            return None, None

        return paper, corners

    def find_shape(self, cell):
        """Classify the shape in a cell (X, O, or None)."""
        if cell is None or cell.size == 0 or self.model is None:
            return None

        # Convert to grayscale if needed
        if len(cell.shape) == 3 and cell.shape[2] == 3:
            gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        elif len(cell.shape) == 2:
            gray_cell = cell
        else:
            return None

        # Check if cell has enough content
        if cv2.countNonZero(gray_cell) < (gray_cell.size * 0.05):
            return None

        mapper = {0: None, 1: 'X', 2: 'O'}
        try:
            processed_cell = detections.preprocess_input(gray_cell)  # returns (1, 32, 32, 1)
            prediction = self.model.predict(processed_cell, verbose=0)
            idx = np.argmax(prediction)
            confidence = prediction[0][idx]
            confidence_threshold = 0.80
            if confidence < confidence_threshold:
                return None
            return mapper[idx]
        except Exception as e:
            print(f"Error in shape detection: {e}")
            return None

    def get_board_template(self, thresh):
        """Detect the 3x3 grid structure."""
        if thresh is None or thresh.size == 0:
            return []
            
        middle_center = detections.contoured_bbox(thresh)
        if middle_center is None:
            return []

        center_x, center_y, width, height = middle_center
        if width <= 5 or height <= 5:
            return []

        gap = int(max(width, height) * 0.05)
        eff_w = width + gap
        eff_h = height + gap

        left = center_x - eff_w
        right = center_x + eff_w
        top = center_y - eff_h
        bottom = center_y + eff_h

        grid_coords = [
            (left, top, width, height), (center_x, top, width, height), (right, top, width, height),
            (left, center_y, width, height), (center_x, center_y, width, height), (right, center_y, width, height),
            (left, bottom, width, height), (center_x, bottom, width, height), (right, bottom, width, height)
        ]

        paper_h, paper_w = thresh.shape[:2]
        for i, (x, y, w, h) in enumerate(grid_coords):
            if x + w < 0 or y + h < 0 or x > paper_w or y > paper_h:
                return []

        return grid_coords

    def draw_shape(self, template, shape, coords):
        """Draw X or O shape onto the bird's eye view image."""
        if template is None or shape is None:
            return template

        x, y, w, h = map(int, coords)
        if w <= 0 or h <= 0:
            return template

        color = (0, 0, 255)  # Red
        thickness = 2
        margin_x = max(2, int(w * 0.15))
        margin_y = max(2, int(h * 0.15))

        if shape == 'O':
            center = (x + w // 2, y + h // 2)
            radius = min(w // 2 - margin_x, h // 2 - margin_y)
            if radius > 0:
                cv2.circle(template, center, radius, color, thickness)
        elif shape == 'X':
            if w > 2 * margin_x and h > 2 * margin_y:
                pt1_tl_br = (x + margin_x, y + margin_y)
                pt2_tl_br = (x + w - margin_x, y + h - margin_y)
                pt1_tr_bl = (x + w - margin_x, y + margin_y)
                pt2_tr_bl = (x + margin_x, y + h - margin_y)
                cv2.line(template, pt1_tl_br, pt2_tl_br, color, thickness)
                cv2.line(template, pt1_tr_bl, pt2_tr_bl, color, thickness)

        return template

    async def send_command_to_esp32(self, servo1_angle, servo2_angle, servo3_angle, enable_signal, pickup_flag):
        """Send servo command to ESP32 via WebSocket"""
        if self.esp32_client is None:
            print("[ESP32] No ESP32 client available, skipping command")
            return False
            
        try:
            s1 = int(servo1_angle)
            s2 = int(servo2_angle)
            s3 = int(servo3_angle)
            enable = int(enable_signal)
            pickup = int(pickup_flag)
            
            command = f"{s1},{s2},{s3},{enable},{pickup}"
            print(f"[ESP32] Sending command: {command}")
            
            await self.esp32_client.send_json({
                "action": "command",
                "command": command
            })
            return True
        except Exception as e:
            print(f"[ESP32] Error sending command: {e}")
            return False
    
    async def move_robot_arm(self, position):
        """Move robot arm to play a piece at the specified position"""
        if self.esp32_client is None:
            print("[ESP32] No ESP32 client available, skipping robot movement")
            return False
            
        pickup_delay = 2.0
        home_delay = 2.0
        play_delay = 2.0
        
        # Get pickup position
        pickup_key = f'pickup{self.next_pickup_index}'
        print(f"[ROBOT] Planning sequence using pickup position: {pickup_key}")
        self.next_pickup_index = (self.next_pickup_index + 1) % self.NUM_PICKUP_POSITIONS
        
        # Check if position is valid
        required_keys = [pickup_key, 'home', position]
        for key in required_keys:
            if key not in self.angle_map:
                print(f"[ERROR] Robot angle map missing required key: '{key}'. Aborting planned sequence.")
                return False
                
        pickup_angles = self.angle_map[pickup_key]
        home_angles = self.angle_map['home']
        play_angles = self.angle_map[position]
        
        try:
            # Step 1: Move to pickup position
            print(f"[ROBOT PLAN] 1. Move to {pickup_key} (Enable={self.ENABLE_ACTIVE}, Pickup={self.PICKUP_TRUE})...")
            await self.send_command_to_esp32(*pickup_angles, self.ENABLE_ACTIVE, self.PICKUP_TRUE)
            await asyncio.sleep(pickup_delay)
            
            # Step 2: Move to home position
            print(f"[ROBOT PLAN] 2. Move to Home (Enable={self.ENABLE_ACTIVE}, Pickup={self.PICKUP_FALSE})...")
            await self.send_command_to_esp32(*home_angles, self.ENABLE_ACTIVE, self.PICKUP_FALSE)
            await asyncio.sleep(home_delay)
            
            # Step 3: Move to play position
            print(f"[ROBOT PLAN] 3. Move to Play position {position} (Enable={self.ENABLE_ACTIVE}, Pickup={self.PICKUP_FALSE})...")
            await self.send_command_to_esp32(*play_angles, self.ENABLE_ACTIVE, self.PICKUP_TRUE)
            await asyncio.sleep(play_delay)
            
            # Step 4: Return to home
            print(f"[ROBOT PLAN] 4. Returning to Home (Enable={self.ENABLE_INACTIVE}, Pickup={self.PICKUP_FALSE})...")
            await self.send_command_to_esp32(*home_angles, self.ENABLE_INACTIVE, self.PICKUP_FALSE)
            await asyncio.sleep(home_delay)
            
            print(f"[ROBOT] Planned sequence for cell {position} complete.")
            return True
        except Exception as e:
            print(f"[ERROR] Unexpected error during robot sequence planning/execution: {e}")
            # Try emergency return home
            try:
                print("[ROBOT] Attempting emergency return home...")
                await self.send_command_to_esp32(*home_angles, self.ENABLE_INACTIVE, self.PICKUP_FALSE)
            except Exception as emergency_e:
                print(f"[ERROR] Failed emergency return: {emergency_e}")
            return False

    async def process_frame(self, frame_bytes):
        # Send the switch command on the first frame if not sent already
        if not self.switch_command_sent and self.esp32_client is not None:
            await self._send_switch_command()
            
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"status": "error", "message": "Invalid frame"}

        # Apply zoom if configured
        if self.zoom > 1.0:
            frame = self.zoom_frame(frame, self.zoom)
            
        # Paper detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_gray = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh_paper_detect = cv2.threshold(blurred_gray, self.paper_detection_threshold, 255, cv2.THRESH_BINARY)
        paper, corners = self.find_sheet_paper(frame, thresh_paper_detect, add_margin=True)
        
        # Draw corners on processed frame
        vis_frame = frame.copy()
        bird_view_display = None
        status_text = "Paper not detected"
        board_status = "waiting"
        debug_msg = ""
        
        if corners is not None:
            for c in corners:
                cv2.circle(vis_frame, tuple(map(int, c)), 5, (0, 255, 0), -1)
                
        if paper is not None and corners is not None:
            status_text = "Paper detected"
            # Process bird's eye view for grid and shapes
            paper_gray = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
            paper_gray_blurred = cv2.GaussianBlur(paper_gray, (5, 5), 0)
            _, paper_thresh = cv2.threshold(paper_gray_blurred, self.grid_detection_threshold, 255, cv2.THRESH_BINARY_INV)
            grid = self.get_board_template(paper_thresh)
            
            paper_display = paper.copy()
            bird_view_display = paper_display
            
            if not grid:
                status_text = "Grid not detected"
            else:
                status_text = "Board detected"
                # Draw grid cells and any existing pieces
                for i, (x, y, w, h) in enumerate(grid):
                    xi, yi, wi, hi = map(int, (x, y, w, h))
                    cv2.rectangle(paper_display, (xi, yi), (xi + wi, yi + hi), (0, 255, 0), 1)
                    if i in self.history:
                        shape = self.history[i]['shape']
                        paper_display = self.draw_shape(paper_display, shape, (xi, yi, wi, hi))
                
                bird_view_display = paper_display
                
                # Check for moves at specified interval
                current_time = time.time()
                if current_time - self.last_check_time >= self.check_interval:
                    debug_msg = f"Checking for moves (interval: {self.check_interval}s)"
                    self.last_check_time = current_time
                    
                    # Detect player's move
                    human_player = 'X'
                    ai_player = 'O'
                    current_state_detection = [None] * 9
                    available_indices = [i for i in range(9) if i not in self.history]
                    
                    if available_indices and paper_thresh is not None:
                        paper_h, paper_w = paper_thresh.shape[:2]
                        
                        # Check each empty cell for a potential move
                        for i in available_indices:
                            if not grid: 
                                continue
                            x, y, w, h = grid[i]
                            xi, yi, wi, hi = map(int, (x, y, w, h))
                            
                            # Ensure coordinates are within bounds
                            x1_clip, y1_clip = max(0, xi), max(0, yi)
                            x2_clip = min(paper_w, xi + wi)
                            y2_clip = min(paper_h, yi + hi)
                            clipped_w, clipped_h = x2_clip - x1_clip, y2_clip - y1_clip
                            
                            if clipped_w <= 5 or clipped_h <= 5:
                                current_state_detection[i] = None
                                continue
                                
                            # Extract cell and detect shape
                            cell = paper_thresh[y1_clip:y2_clip, x1_clip:x2_clip]
                            shape_in_cell = self.find_shape(cell)
                            current_state_detection[i] = shape_in_cell
                        
                        # Find new player moves
                        comparison_state = [self.history.get(i, {}).get('shape') for i in range(9)]
                        for i in available_indices:
                            comparison_state[i] = current_state_detection[i]
                        
                        new_move_index = -1
                        for i in range(9):
                            if comparison_state[i] == human_player and self.previous_state[i] != human_player:
                                if i in available_indices:
                                    debug_msg += f" | Move at {i}: {comparison_state[i]}"
                                    new_move_index = i
                                    break
                        
                        # Process player move if found
                        if new_move_index != -1:
                            debug_msg += f" | Player X at {new_move_index}"
                            self.move_detected_in_last_cycle = True
                            self.history[new_move_index] = {'shape': human_player, 'bbox': grid[new_move_index]}
                            self.board.make_move(new_move_index, human_player)
                            
                            # Draw the player's move
                            xi, yi, wi, hi = map(int, grid[new_move_index])
                            bird_view_display = self.draw_shape(bird_view_display, human_player, (xi, yi, wi, hi))
                            
                            # Update state tracking
                            self.previous_state = [self.history.get(i, {}).get('shape') for i in range(9)]
                            
                            # Check if game is complete
                            if self.board.complete():
                                debug_msg += " | Game over!"
                                board_status = "complete"
                            else:
                                # Computer's turn
                                debug_msg += " | Computer thinking..."
                                computer_move = determine(self.board, ai_player)
                                
                                if computer_move is not None and computer_move in self.board.available_moves():
                                    debug_msg += f" | Computer O at {computer_move}"
                                    self.board.make_move(computer_move, ai_player)
                                    self.history[computer_move] = {'shape': ai_player, 'bbox': grid[computer_move]}
                                    
                                    # Draw computer's move
                                    xi, yi, wi, hi = map(int, grid[computer_move])
                                    bird_view_display = self.draw_shape(bird_view_display, ai_player, (xi, yi, wi, hi))
                                    
                                    # Send move to ESP32 if available
                                    asyncio.create_task(self.move_robot_arm(computer_move))
                                    
                                    # Update state tracking
                                    self.previous_state = [self.history.get(i, {}).get('shape') for i in range(9)]
                                    
                                    if self.board.complete():
                                        debug_msg += " | Game over!"
                                        board_status = "complete"
                        else:
                            if self.move_detected_in_last_cycle:
                                self.move_detected_in_last_cycle = False
                                debug_msg += " | Waiting for next move"
                            
                            current_confirmed_state = [self.history.get(i, {}).get('shape') for i in range(9)]
                            if current_confirmed_state != self.previous_state:
                                self.previous_state = current_confirmed_state
                
        # Encode frames for frontend
        _, vis_jpg = cv2.imencode('.jpg', vis_frame)
        vis_b64 = base64.b64encode(vis_jpg).decode("utf-8")
        
        # Create bird view image if available
        bird_view_b64 = None
        if bird_view_display is not None:
            try:
                # Add status text to bird view
                cv2.putText(bird_view_display, status_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                _, bird_jpg = cv2.imencode('.jpg', bird_view_display)
                bird_view_b64 = base64.b64encode(bird_jpg).decode("utf-8")
            except Exception as e:
                print(f"Error encoding bird view: {e}")
        
        # Return all frame data and game state
        return {
            "status": "ok",
            "processed_frame": vis_b64,
            "bird_view_frame": bird_view_b64,
            "game_state": {
                "board": self.board.squares,
                "paper_detected": paper is not None,
                "status_text": status_text,
                "board_status": board_status,
                "debug": debug_msg,
                "winner": self.board.winner() if self.board.complete() else None
            }
        }
