import cv2
import numpy as np
import base64
import time # Added
import serial # Added
import concurrent.futures # Kept, though not strictly used in this merged single-session logic

# --- NEW: Configuration ---
# Define flexible color range for GREEN cups in HSV (Changed from white)
green_lower = np.array([30, 40, 80])      # Lower HSV for green (Initial Value threshold)
green_upper = np.array([90, 255, 255])    # Upper HSV for green

# Define color range for the BLUE ball (Changed from green)
ball_lower = np.array([100, 100, 100])    # Lower HSV for blue
ball_upper = np.array([130, 255, 255])    # Upper HSV for blue

# NEW: Adaptive green detection parameters
adaptive_green_threshold_low_initial = 80  # Initial low brightness threshold for green Value
adaptive_learning_rate = 0.01      # How quickly we adjust to lighting changes

# NEW: Motion tracking parameters
motion_threshold = 10  # pixels
motion_timeout = 10     # seconds (adjust as needed for API context)

# NEW: Serial Port Configuration (adjust COM port and baud rate)
ESP32_SERIAL_PORT = 'COM3'
ESP32_BAUD_RATE = 9600

# --- Helper functions for tracker creation (Handles OpenCV version differences) ---
def create_tracker():
    # Try common ways to create CSRT tracker based on OpenCV version
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    # Fallback or older versions (Example, adjust if using KCF, etc.)
    # if hasattr(cv2, "TrackerKCF_create"):
    #     return cv2.TrackerKCF_create()
    raise RuntimeError("Compatible tracker (like CSRT) not available in your OpenCV installation.")

def create_multitracker():
    if hasattr(cv2, "MultiTracker_create"):
        return cv2.MultiTracker_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "MultiTracker_create"):
        return cv2.legacy.MultiTracker_create()
    raise RuntimeError("MultiTracker not available in your OpenCV installation.")


class GameSession:
    def __init__(self):
        # --- Original State Variables (adapted) ---
        self.ball_position = None
        self.ball_under_cup = None # Index of cup ball is likely under
        self.last_nearest_cup = None # Index of last cup ball was nearest to when visible
        self.multi_tracker = None
        # self.cups = None # Replaced by tracked_ellipses for storing detected/tracked shapes

        # --- NEW: State Variables from the updated script ---
        self.green_lower = list(green_lower) # Make it a list to modify Value component
        self.green_upper = list(green_upper)
        self.ball_lower = list(ball_lower)
        self.ball_upper = list(ball_upper)

        self.adaptive_green_threshold_low = float(adaptive_green_threshold_low_initial) # Current adaptive threshold

        self.game_paused = False
        self.missing_cup_index = None # Index of the cup that went out of frame

        # Motion tracking state
        self.last_motion_times = [time.time()] * 3 # Initialize with current time for 3 cups
        self.last_cup_centers = [None] * 3 # Store last known center for each tracked cup
        self.last_moved_cup_idx = None # Track the index of the cup that moved most recently

        # Game end state
        self.game_ended = False
        # self.last_wanted_cup_idx = None # Index of cup to reveal (derived from last_moved_cup_idx when game ends)
        self.last_ball_under_cup_idx = None # Track the definite cup index holding the ball just before ending/hiding

        # Ball visibility tracking for hidden ball logic
        self.last_ball_visible = True # Was the ball visible in the previous frame?
        self.last_moved_cup_before_hidden = None # Which cup moved just before ball vanished?

        # Arm control state
        self.arm_command_sent = False # Has the command to lift the cup been sent?
        self.esp32_serial = None
        try:
            self.esp32_serial = serial.Serial(ESP32_SERIAL_PORT, ESP32_BAUD_RATE, timeout=1)
            print(f"Successfully connected to ESP32 on {ESP32_SERIAL_PORT}")
            time.sleep(2) # Allow time for serial connection to establish
        except serial.SerialException as e:
            print(f"ERROR: Could not open serial port {ESP32_SERIAL_PORT}: {e}")
            print("Robotic arm control will be disabled.")
            self.esp32_serial = None

        self.frame_count = 0 # Simple counter for periodic actions if needed
        self.tracked_ellipses = {} # Store data about tracked objects {tracker_idx: ellipse_data}

    # --- NEW: Method to reset the game state ---
    def reset_game(self):
        print("Resetting game state...")
        self.ball_position = None
        self.ball_under_cup = None
        self.last_nearest_cup = None
        self.multi_tracker = None
        self.tracked_ellipses = {}

        self.adaptive_green_threshold_low = float(adaptive_green_threshold_low_initial)
        self.green_lower[2] = int(self.adaptive_green_threshold_low) # Reset adaptive part

        self.game_paused = False
        self.missing_cup_index = None

        self.last_motion_times = [time.time()] * 3
        self.last_cup_centers = [None] * 3
        self.last_moved_cup_idx = None

        self.game_ended = False
        self.last_ball_under_cup_idx = None

        self.last_ball_visible = True
        self.last_moved_cup_before_hidden = None

        self.arm_command_sent = False
        # Note: Serial connection is not reset here, assumes it persists.

        self.frame_count = 0
        print("Game state reset.")


    # --- NEW: Method for adaptive thresholding ---
    def update_green_threshold(self, frame):
        """Dynamically adjust green detection Value based on overall brightness"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_brightness = np.mean(gray)
        # Heuristic: link threshold to ~40% of brightness, adjust multiplier if needed
        target_threshold = current_brightness * 0.4
        self.adaptive_green_threshold_low = (1 - adaptive_learning_rate) * self.adaptive_green_threshold_low + \
                                       adaptive_learning_rate * target_threshold
        # Clamp threshold to reasonable bounds
        self.adaptive_green_threshold_low = max(40, min(150, self.adaptive_green_threshold_low))
        self.green_lower[2] = int(self.adaptive_green_threshold_low) # Update the lower bound Value

    # --- UPDATED: Method to detect GREEN cups using Ellipses ---
    def detect_cups(self, frame):
        """Detect green cup tops using adaptive ellipse detection."""
        self.update_green_threshold(frame) # Adapt threshold first

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Use the instance's potentially adapted color range
        mask = cv2.inRange(hsv, np.array(self.green_lower), np.array(self.green_upper))

        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cups = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter out small contours based on area
                if len(contour) >= 5:  # Minimum points needed for ellipse fitting
                    try: # Add try-except block as fitEllipse can sometimes fail
                        ellipse = cv2.fitEllipse(contour)
                        # Ellipse format: ((center_x, center_y), (minor_axis, major_axis), angle)
                        (center, axes, angle) = ellipse
                        major_axis = max(axes)
                        minor_axis = min(axes)

                        # Filter based on aspect ratio (closer to 1 is more circular)
                        # Allow some tolerance for perspective distortion
                        if major_axis > 0 and 0.5 < (minor_axis / major_axis) < 1.5:
                             # Filter based on solidity (ratio of contour area to convex hull area)
                             hull = cv2.convexHull(contour)
                             hull_area = cv2.contourArea(hull)
                             if hull_area > 0:
                                 solidity = float(area) / hull_area
                                 if solidity > 0.8:  # Reject highly irregular shapes
                                     cups.append(ellipse)
                    except Exception as e:
                         print(f"Warning: fitEllipse failed for a contour. Error: {e}")
                         continue # Skip this contour

        # Sort detected cups by x-coordinate
        cups = sorted(cups, key=lambda x: x[0][0])
        return cups # Return list of ellipses

    # --- UPDATED: Method to detect BLUE ball ---
    def detect_ball(self, frame):
        """Detect ball position using color detection."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Use the instance's ball color range
        mask = cv2.inRange(hsv, np.array(self.ball_lower), np.array(self.ball_upper))
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
            # Ensure detected ball is large enough
            if radius > 5:
                return (int(x), int(y))
        return None

    # --- UPDATED: Method to check ball location relative to cups ---
    def check_ball_under_cup(self, ball_pos, current_cup_ellipses):
        """Check if ball is under a cup based on position or last known state."""
        # current_cup_ellipses: list of ((cx, cy), (ax1, ax2), angle) for currently tracked cups

        # If ball is not visible
        if ball_pos is None:
            # If the ball *just* disappeared in this frame...
            if self.last_ball_visible:
                # ...record which cup moved last *before* it vanished
                self.last_moved_cup_before_hidden = self.last_moved_cup_idx
                self.last_ball_visible = False # Mark ball as hidden now
            # Return the index of the cup that moved just before the ball hid
            # This assumes the ball goes under the cup that was last moved
            # print(f"Debug: Ball hidden. Last moved cup before hidden: {self.last_moved_cup_before_hidden}")
            return self.last_moved_cup_before_hidden
        else:
            # Ball is visible, reset the hidden state tracking
            self.last_ball_visible = True
            # self.last_moved_cup_before_hidden = None # Reset this only when ball reappears maybe?

        # Ball is visible, find the nearest cup
        min_distance = float("inf")
        nearest_cup_idx = None
        for i, ellipse_data in enumerate(current_cup_ellipses):
            if ellipse_data: # Ensure ellipse data exists for this index
                 center = ellipse_data[0] # Center (x, y)
                 distance = np.linalg.norm(np.array(ball_pos) - np.array(center))
                 if distance < min_distance:
                     min_distance = distance
                     nearest_cup_idx = i # Store the *index* in the current_cup_ellipses list

        # Update the last nearest cup index (even if not under it)
        self.last_nearest_cup = nearest_cup_idx

        # Check if the ball is physically close enough to the center of the nearest cup
        if nearest_cup_idx is not None and current_cup_ellipses[nearest_cup_idx]:
            # Use ellipse's major axis as a reference for size
            major_axis = max(current_cup_ellipses[nearest_cup_idx][1]) / 2
            # Check if distance is less than ~1.2 times the major radius (tolerance)
            if min_distance < major_axis * 1.2:
                self.ball_under_cup = nearest_cup_idx # Update the state
                # print(f"Debug: Ball VISIBLE under cup index {nearest_cup_idx}")
                return nearest_cup_idx # Return the index of the cup it's under

        # If ball is visible but not close enough to any cup, it's not under one
        self.ball_under_cup = None # Explicitly set to None if not under any cup
        # print(f"Debug: Ball VISIBLE but not under any cup.")
        return None

    # --- NEW: Method to check if cup is within frame ---
    def check_cup_in_frame(self, box, frame_width, frame_height):
        """Check if a cup (represented by bbox) is within frame boundaries."""
        x, y, w, h = box
        center_x = x + w / 2
        center_y = y + h / 2
        # Check if center is within frame (add a small margin if needed)
        margin = 10 # pixels
        return (center_x > margin and center_y > margin and
                center_x < frame_width - margin and center_y < frame_height - margin)

    # --- NEW: Method to send commands to ESP32 ---
    def send_arm_command(self, command_str):
        """Sends a command string to the connected ESP32."""
        if self.esp32_serial and self.esp32_serial.is_open:
            try:
                print(f"Sending to ESP32: {command_str.strip()}")
                self.esp32_serial.write(command_str.encode())
                # Optional: Wait for a response or add delay
                # response = self.esp32_serial.readline().decode().strip()
                # print(f"ESP32 Response: {response}")
                time.sleep(0.1) # Small delay after sending
                return True
            except serial.SerialException as e:
                print(f"ERROR: Failed to send command to ESP32: {e}")
                # Optionally try to reconnect or disable arm
                # self.esp32_serial.close()
                # self.esp32_serial = None
                return False
            except Exception as e:
                print(f"ERROR: Unexpected error during serial communication: {e}")
                return False
        else:
            # print("Warning: ESP32 serial port not available. Cannot send command.")
            return False

    # --- MAIN PROCESSING METHOD ---
    def process_frame(self, frame_bytes):
        self.frame_count += 1
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"status": "error", "message": "Invalid frame data"}

        # --- Resize frame (New dimensions) ---
        frame = cv2.resize(frame, (400, 300))
        frame_height, frame_width = frame.shape[:2]
        raw_frame = frame.copy() # Keep a copy for the raw output

        # --- Status dictionary to build response ---
        response = {
            "status": "initializing",
            "message": "Waiting for system setup...",
            "game_state": "initializing", # NEW state: initializing, running, paused, ended
            "ball_position": None,
            "ball_under_cup_index": None, # Index relative to tracked order
            "ball_under_cup_name": None, # Left/Middle/Right
            "cup_ellipses": [], # List of current ellipse data ((cx,cy),(ax1,ax2),angle)
            "cup_names": {}, # Mapping: initial index -> current name (L/M/R)
            "active_cup_count": 0,
            "adaptive_threshold": int(self.adaptive_green_threshold_low), # NEW
            "raw_frame": None,
            "processed_frame": None,
            "arm_command_sent_flag": self.arm_command_sent, # NEW
            "wanted_cup_lr_index": None # L=0, M=1, R=2 index of cup with ball *after* game ends
        }

        # --- Initialize Tracker ---
        if self.multi_tracker is None and not self.game_ended:
            response["status"] = "detecting_cups"
            response["message"] = "Attempting to detect 3 green cups..."
            response["game_state"] = "detecting"
            detected_cups = self.detect_cups(frame) # Use new detection method

            if len(detected_cups) == 3:
                self.multi_tracker = create_multitracker()
                self.tracked_ellipses = {} # Reset tracked data
                self.last_cup_centers = [None] * 3 # Reset centers
                self.last_motion_times = [time.time()] * 3 # Reset motion timers
                for i, ellipse in enumerate(detected_cups):
                    center, axes, angle = ellipse
                    # Get bounding box for the tracker
                    bbox = cv2.boundingRect(cv2.ellipse2Poly(
                        (int(center[0]), int(center[1])),
                        (int(axes[0]/2), int(axes[1]/2)),
                        int(angle), 0, 360, 1
                    ))
                    tracker = create_tracker()
                    self.multi_tracker.add(tracker, frame, bbox)
                    self.tracked_ellipses[i] = ellipse # Store initial ellipse data by index
                    self.last_cup_centers[i] = center # Store initial center
                print(f"{len(detected_cups)} cups detected and trackers initialized!")
                response["status"] = "tracking"
                response["message"] = "Trackers initialized."
                response["game_state"] = "running"
            else:
                # Draw detected cups even if not 3 yet
                 for ellipse in detected_cups:
                      cv2.ellipse(frame, ellipse, (0, 165, 255), 2) # Orange color for detection phase
                 response["message"] = f"Waiting for 3 cups ({len(detected_cups)} detected)..."
                 # Encode and return frame without further processing
                 _, raw_jpg = cv2.imencode('.jpg', raw_frame)
                 _, processed_jpg = cv2.imencode('.jpg', frame) # Show detected shapes
                 response["raw_frame"] = base64.b64encode(raw_jpg).decode("utf-8")
                 response["processed_frame"] = base64.b64encode(processed_jpg).decode("utf-8")
                 response["active_cup_count"] = len(detected_cups)
                 return response

        # --- If tracker exists, update and process ---
        if self.multi_tracker is not None:
            success, boxes = self.multi_tracker.update(frame)

            if not success:
                 # Handle tracker failure - maybe attempt re-detection or mark as error
                 print("Warning: MultiTracker update failed.")
                 response["status"] = "error"
                 response["message"] = "Cup tracking failed. Consider resetting."
                 # Optionally try to reset: self.reset_game()
                 # Keep showing the last known state or raw frame?
                 # For now, just return error state with raw frame
                 _, raw_jpg = cv2.imencode('.jpg', raw_frame)
                 response["raw_frame"] = base64.b64encode(raw_jpg).decode("utf-8")
                 response["processed_frame"] = response["raw_frame"] # Show raw if tracking fails
                 return response

            response["status"] = "tracking" # Default status if tracking is active
            response["game_state"] = "running" # Default state

            # --- Pause Logic ---
            current_cup_indices_in_frame = []
            if not self.game_ended: # Only check for pause if game is running
                cups_in_frame_flags = [False] * len(boxes)
                for i, box in enumerate(boxes):
                    if self.check_cup_in_frame(box, frame_width, frame_height):
                         cups_in_frame_flags[i] = True
                         current_cup_indices_in_frame.append(i)

                if not self.game_paused:
                    if not all(cups_in_frame_flags):
                        try:
                             self.missing_cup_index = cups_in_frame_flags.index(False)
                             self.game_paused = True
                             print(f"Game paused: Cup {self.missing_cup_index} is out of frame")
                        except ValueError:
                             print("Error: Could not find missing cup index despite not all being in frame.")
                             # Should not happen if not all are True
                elif self.game_paused: # If already paused, check if game can resume
                    if all(cups_in_frame_flags):
                        print("Game resumed: All cups are back in frame")
                        self.game_paused = False
                        self.missing_cup_index = None
                    else:
                         # Remain paused, find current missing index if it changed (unlikely)
                         try:
                              self.missing_cup_index = cups_in_frame_flags.index(False)
                         except ValueError:
                              pass # Keep the old index if error finding new one

            if self.game_paused:
                 response["game_state"] = "paused"
                 response["message"] = f"Game Paused - Cup {self.missing_cup_index} needs to return to view."

            # --- Ball Detection ---
            self.ball_position = self.detect_ball(frame)
            response["ball_position"] = self.ball_position

            # --- Process Tracked Cups (Motion, Drawing, Naming) ---
            current_ellipses = [None] * len(boxes) # Store ellipses derived from current boxes
            valid_boxes_indices = [] # Track indices of cups currently processed (not paused)

            now = time.time()
            for i, box in enumerate(boxes):
                 # If paused, skip processing the missing cup entirely
                 if self.game_paused and i == self.missing_cup_index:
                     continue
                 valid_boxes_indices.append(i) # This cup is active

                 x, y, w, h = [int(v) for v in box]
                 center = (int(x + w / 2), int(y + h / 2))
                 # Approximate axes from bounding box (less accurate than fitEllipse)
                 axes = (int(w / 2), int(h / 2))
                 angle = 0 # Cannot determine angle from bbox alone
                 current_ellipses[i] = (center, axes, angle) # Store current representation

                 # --- Motion Detection ---
                 if not self.game_paused and not self.game_ended:
                     if self.last_cup_centers[i] is not None:
                         dist = np.linalg.norm(np.array(center) - np.array(self.last_cup_centers[i]))
                         if dist > motion_threshold:
                             self.last_motion_times[i] = now # Update motion time
                             self.last_moved_cup_idx = i # Record which cup moved last
                             # print(f"Cup {i} moved.")
                     self.last_cup_centers[i] = center # Update last known center

                 # --- Drawing Cups (only if game not ended) ---
                 if not self.game_ended:
                     # Draw tracked ellipse approximation
                     cv2.ellipse(frame, center, axes, angle, 0, 360, (255, 0, 255), 2) # Magenta for tracked cups
                     # Add basic index text (names added later)
                     # cv2.putText(frame, f"Cup {i}", (center[0] - 20, center[1] - max(axes) - 10),
                     #              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            response["cup_ellipses"] = [e for e in current_ellipses if e is not None] # Return only valid ellipses
            response["active_cup_count"] = len(valid_boxes_indices)

            # --- Dynamic Cup Naming (Left, Middle, Right) ---
            cup_positions_for_naming = []
            if len(valid_boxes_indices) > 0: # Check if there are any active cups
                for i in valid_boxes_indices:
                    box = boxes[i]
                    center_x = int(box[0] + box[2] / 2)
                    cup_positions_for_naming.append((center_x, i)) # (x_position, original_index)

            # Sort by x-position to determine Left, Middle, Right
            cup_positions_sorted = sorted(cup_positions_for_naming, key=lambda tup: tup[0])

            # Create mapping from original index to L/M/R name
            cup_names_map = {}
            cup_idx_to_lr_idx = {} # Map original index to 0 (L), 1 (M), 2 (R)
            name_list = ["Left Cup", "Middle Cup", "Right Cup"]
            for lr_idx, (x_pos, original_idx) in enumerate(cup_positions_sorted):
                 if lr_idx < len(name_list): # Handle cases with < 3 cups temporarily
                     cup_names_map[original_idx] = name_list[lr_idx]
                     cup_idx_to_lr_idx[original_idx] = lr_idx
            response["cup_names"] = cup_names_map

            # --- Determine Ball Location ---
            # Pass only the ellipses of currently valid cups
            valid_ellipses_for_check = [current_ellipses[i] for i in valid_boxes_indices]
            current_ball_cup_original_idx = self.check_ball_under_cup(self.ball_position, valid_ellipses_for_check)

            # Translate the index from the 'valid_ellipses_for_check' list back to the original tracker index
            actual_ball_cup_idx = None
            if current_ball_cup_original_idx is not None and current_ball_cup_original_idx < len(valid_boxes_indices):
                 actual_ball_cup_idx = valid_boxes_indices[current_ball_cup_original_idx]
                 self.last_ball_under_cup_idx = actual_ball_cup_idx # Store the confirmed index if ball is under it
                 response["ball_under_cup_index"] = actual_ball_cup_idx
                 response["ball_under_cup_name"] = cup_names_map.get(actual_ball_cup_idx, "Unknown")
            elif self.ball_position is None and self.last_moved_cup_before_hidden is not None:
                 # If ball is hidden, use the tracked hidden state index
                 actual_ball_cup_idx = self.last_moved_cup_before_hidden
                 response["ball_under_cup_index"] = actual_ball_cup_idx
                 response["ball_under_cup_name"] = cup_names_map.get(actual_ball_cup_idx, "Unknown (Hidden)")
                 # Highlight the cup where the ball is presumed hidden
                 if actual_ball_cup_idx in valid_boxes_indices: # Make sure the cup is still tracked
                     ellipse_to_highlight = current_ellipses[actual_ball_cup_idx]
                     if ellipse_to_highlight:
                         center, axes, angle = ellipse_to_highlight
                         cv2.ellipse(frame, center, axes, angle, 0, 360, (0, 255, 255), 3) # Yellow highlight
                         name = cup_names_map.get(actual_ball_cup_idx, f"Cup {actual_ball_cup_idx}")
                         cv2.putText(frame, f"{name} (Ball!)", (center[0] - 60, center[1] - max(axes) - 30),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


            # --- Drawing Ball (if visible) and Names/Highlights ---
            if self.ball_position:
                cv2.circle(frame, self.ball_position, 10, (0, 0, 255), 2) # Red circle for ball
                cv2.putText(frame, "Ball", (self.ball_position[0] - 20, self.ball_position[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Draw names on cups (if game not ended)
            if not self.game_ended:
                 for original_idx, name in cup_names_map.items():
                     if original_idx in valid_boxes_indices: # Check if cup is active
                         ellipse_data = current_ellipses[original_idx]
                         if ellipse_data:
                             center, axes, angle = ellipse_data
                             # Highlight if ball is under this cup (and visible)
                             if self.ball_position is not None and actual_ball_cup_idx == original_idx:
                                 cv2.ellipse(frame, center, axes, angle, 0, 360, (0, 255, 255), 3) # Yellow highlight
                                 cv2.putText(frame, f"{name} (Ball!)", (center[0] - 60, center[1] - max(axes) - 30),
                                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                             else:
                                 # Just draw name if ball not under it or hidden
                                 cv2.putText(frame, name, (center[0] - 40, center[1] - max(axes) - 10),
                                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


            # --- Game End Check ---
            if not self.game_paused and not self.game_ended and len(self.last_motion_times) > 0 :
                 # Check if *all* cups haven't moved for the timeout period
                 if all(now - t > motion_timeout for t in self.last_motion_times):
                     self.game_ended = True
                     print(f"Game ended: No cup has moved for {motion_timeout} seconds")
                     # last_wanted_cup_idx is determined when sending arm command based on final ball location


            # --- Game Ended State Logic & Arm Control ---
            if self.game_ended:
                 response["game_state"] = "ended"
                 response["message"] = "Game Ended - No motion detected."

                 # Final determination of the cup with the ball
                 final_ball_cup_idx = None
                 if self.ball_position is None:
                     # If ball hidden, use the last index recorded when it was confirmed under a cup
                     # or the index of the cup moved just before it hid
                     final_ball_cup_idx = self.last_ball_under_cup_idx if self.last_ball_under_cup_idx is not None else self.last_moved_cup_before_hidden
                 else:
                     # If ball is visible, double-check its location now
                     final_check_idx = self.check_ball_under_cup(self.ball_position, valid_ellipses_for_check)
                     if final_check_idx is not None and final_check_idx < len(valid_boxes_indices):
                         final_ball_cup_idx = valid_boxes_indices[final_check_idx]

                 response["ball_under_cup_index"] = final_ball_cup_idx

                 # Determine Left/Middle/Right index (0, 1, 2) of the cup with the ball
                 wanted_cup_lr = -1 # Default to invalid
                 if final_ball_cup_idx is not None:
                     wanted_cup_lr = cup_idx_to_lr_idx.get(final_ball_cup_idx, -1)
                     response["ball_under_cup_name"] = cup_names_map.get(final_ball_cup_idx, "Unknown")
                     response["wanted_cup_lr_index"] = wanted_cup_lr

                 # --- Robotic Arm Control ---
                 if wanted_cup_lr in [0, 1, 2] and not self.arm_command_sent and self.esp32_serial:
                     print(f"Game Ended. Ball is under '{cup_names_map.get(final_ball_cup_idx)}' (L/M/R index: {wanted_cup_lr}). Sending command...")
                     # Define arm target positions (adjust these coordinates for your setup)
                     # Format: "X,Y,Z,MagnetState" (MagnetState: 0=Off, 1=On)
                     # Using placeholder coordinates - REPLACE WITH YOUR ACTUAL VALUES
                     if wanted_cup_lr == 0: # Left Cup
                         cup_pos = "150,140,155" # Position above/at the cup
                         lift_pos = "150,140,80"  # Position to lift the cup to
                     elif wanted_cup_lr == 1: # Middle Cup
                         cup_pos = "145,85,130"  # Adjusted Z slightly higher maybe
                         lift_pos = "145,85,90"
                     elif wanted_cup_lr == 2: # Right Cup
                         cup_pos = "150,35,155"
                         lift_pos = "150,35,100"
                     else:
                         cup_pos = None
                         lift_pos = None

                     if cup_pos and lift_pos:
                         # Sequence of arm movements
                         commands = [
                             f"{cup_pos},0\n", # 1. Go to cup position (magnet off)
                             f"{cup_pos},1\n", # 2. Activate magnet at cup position
                             f"{lift_pos},1\n", # 3. Go to lifting position (magnet on)
                             # Optional: Hold lifted position for a moment
                             # f"{lift_pos},1\n",
                             # time.sleep(1.0)
                             f"{cup_pos},1\n", # 4. Return to cup position (magnet on)
                             f"{cup_pos},0\n", # 5. Turn off magnet at cup position
                             # Optional: Move away slightly before homing
                             # f"{lift_pos},0\n"
                             "90,90,90,0\n" # 6. Return to a defined home/safe position (magnet off) - Adjust home coords
                         ]

                         arm_sequence_success = True
                         for cmd in commands:
                             if not self.send_arm_command(cmd):
                                 arm_sequence_success = False
                                 break # Stop sequence if one command fails
                             time.sleep(2.0) # Wait for arm to execute move (adjust delay)

                         if arm_sequence_success:
                              self.arm_command_sent = True # Mark command as sent successfully
                              response["arm_command_sent_flag"] = True
                              response["message"] += " Arm sequence initiated."
                              print("Arm command sequence sent.")
                         else:
                              response["message"] += " Arm sequence failed."
                              print("Arm command sequence failed.")
                     else:
                         print("Error: Could not determine arm positions for the target cup.")
                         response["message"] += " Could not determine arm positions."

                 # --- Drawing for Ended State ---
                 # Draw final cup names
                 for original_idx, name in cup_names_map.items():
                     if original_idx in valid_boxes_indices:
                         ellipse_data = current_ellipses[original_idx]
                         if ellipse_data:
                             center, axes, angle = ellipse_data
                             cv2.putText(frame, name, (center[0] - 40, center[1] - max(axes) - 10),
                                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2) # Cyan names when ended

                 # Highlight the cup where the ball finally is
                 if final_ball_cup_idx is not None and final_ball_cup_idx in valid_boxes_indices:
                     ellipse_data = current_ellipses[final_ball_cup_idx]
                     if ellipse_data:
                         center, axes, angle = ellipse_data
                         cv2.ellipse(frame, center, axes, angle, 0, 360, (0, 255, 0), 3) # Bright green highlight
                         name = cup_names_map.get(final_ball_cup_idx, f"Cup {final_ball_cup_idx}")
                         cv2.putText(frame, f"{name} (BALL HERE!)", (center[0] - 60, center[1] - max(axes) - 30),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                 # Add "Game Ended" text overlay
                 cv2.putText(frame, "GAME ENDED", (frame_width // 2 - 100, frame_height - 30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)


            # --- Drawing for Paused State ---
            if self.game_paused:
                cv2.putText(frame, "GAME PAUSED", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                missing_name = cup_names_map.get(self.missing_cup_index, f"Index {self.missing_cup_index}")
                cv2.putText(frame, f"{missing_name} missing", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # --- Display Adaptive Threshold ---
            cv2.putText(frame, f"Green V Thr: {self.green_lower[2]}",
                        (frame_width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


        # --- Encode Frames and Finalize Response ---
        try:
            _, raw_jpg = cv2.imencode('.jpg', raw_frame)
            _, processed_jpg = cv2.imencode('.jpg', frame)
            response["raw_frame"] = base64.b64encode(raw_jpg).decode("utf-8")
            response["processed_frame"] = base64.b64encode(processed_jpg).decode("utf-8")
        except Exception as e:
            print(f"Error encoding frames: {e}")
            response["status"] = "error"
            response["message"] = "Error encoding output frames."
            # Avoid sending potentially huge raw byte arrays if encoding fails
            response["raw_frame"] = None
            response["processed_frame"] = None

        return response

    def __del__(self):
        # Cleanup serial port when the object is destroyed
        if self.esp32_serial and self.esp32_serial.is_open:
            print("Closing serial port.")
            self.esp32_serial.close()

# Example usage (within a FastAPI endpoint context):
# session = GameSession() # Create a session instance
# @app.post("/process")
# async def process_video_frame(file: UploadFile = File(...)):
#     frame_bytes = await file.read()
#     result = session.process_frame(frame_bytes)
#     return JSONResponse(content=result)
#
# @app.post("/reset")
# async def reset_session():
#     session.reset_game() # Call the reset method
#     return {"message": "Game session reset"}