"""
Game of Tic Tac Toe using OpenCV to play against computer,
with support for IP camera streams, optional robot arm control,
zoom, automatic move detection, multiple pickup positions, and enable signal.
"""

import os
import sys
import cv2
import argparse
import numpy as np
import time
import serial # Required for robot arm control

# Check if Keras is installed, otherwise use tensorflow.keras
try:
    from keras.models import load_model
except ImportError:
    try:
        from tensorflow.keras.models import load_model
    except ImportError:
        print("[ERROR] Keras or TensorFlow not found. Please install one.")
        sys.exit(1)


# Assuming utils and alphabeta are in the same directory or accessible via PYTHONPATH
try:
    from utils import imutils
    from utils import detections
    from alphabeta import Tic, get_enemy, determine
except ImportError as e:
    print(f"[ERROR] Failed to import required modules (utils, alphabeta): {e}")
    print("Ensure utils/imutils.py, utils/detections.py, and alphabeta.py are accessible.")
    sys.exit(1)


# --- Global Variables ---
# Model will be loaded in main()
model = None
# Index for cycling through pickup positions for the robot arm
next_pickup_index = 0
NUM_PICKUP_POSITIONS = 4 # Define how many pickup positions you have


# --- Helper Functions ---

def zoom_frame(frame, zoom=2.0):
    """
    Zoom into the center of the frame by the specified factor.

    Args:
        frame: The input frame to zoom
        zoom: The zoom factor (default: 2.0)

    Returns:
        Zoomed frame with the same dimensions as the input, or original frame if zoom <= 1.
    """
    if frame is None or zoom <= 1.0:
        return frame

    height, width = frame.shape[:2]
    if height == 0 or width == 0:
        return frame

    # Calculate center and size of the ROI
    center_x, center_y = width // 2, height // 2

    # Calculate ROI dimensions
    roi_width = int(width / zoom)
    roi_height = int(height / zoom)

    # Ensure ROI dimensions are at least 1
    roi_width = max(1, roi_width)
    roi_height = max(1, roi_height)

    # Calculate top left corner of ROI
    x1 = center_x - roi_width // 2
    y1 = center_y - roi_height // 2

    # Ensure ROI is within frame bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    # Calculate x2, y2 based on clipped x1, y1 and roi dimensions
    x2 = min(width, x1 + roi_width)
    y2 = min(height, y1 + roi_height)

    # Ensure x1,y1 are still valid after x2,y2 clipping (in case roi is larger than frame)
    x1 = max(0, x2 - roi_width)
    y1 = max(0, y2 - roi_height)


    # Extract ROI
    try:
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            print("[WARN] Zoom resulted in empty ROI, returning original frame.")
            return frame
        # Resize ROI to original frame size
        return cv2.resize(roi, (width, height), interpolation=cv2.INTER_LINEAR)
    except Exception as e:
        print(f"[ERROR] Error during zoom/resize: {e}")
        print(f"Frame shape: {frame.shape}, zoom: {zoom}, roi coords: ({x1},{y1}) to ({x2},{y2})")
        return frame


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Play Tic Tac Toe using OpenCV and an optional robot arm.")

    parser.add_argument('camera_source', type=str,
                        help='Camera index (e.g., 0) or IP camera stream URL (e.g., "http://192.168.1.5:4747/video")')
    parser.add_argument('--model', '-m', type=str, default='data/model.h5',
                        help='Path to the Keras model file (.h5) to detect Xs and Os')
    parser.add_argument('--serial-port', type=str, default=None, # Default to None (no robot arm)
                        help='Serial port for ESP32 robot arm (e.g., COM3, /dev/ttyUSB0). If not provided, robot arm functions are skipped.')
    parser.add_argument('--baud-rate', type=int, default=9600,
                        help='Baud rate for serial communication (default: 9600)')
    parser.add_argument('--zoom', type=float, default=1.0, # Default to no zoom
                        help='Initial zoom factor (e.g., 2.0). Must be >= 1.0. (default: 1.0)')
    parser.add_argument('--no-zoom', action='store_true',
                        help='Explicitly disable zoom feature, overrides --zoom.')
    parser.add_argument('--check-interval', type=float, default=3.0,
                        help='Interval in seconds to check for player moves (default: 3.0)')
    parser.add_argument('--no-margin', action='store_true',
                        help='Disable adding margin when detecting paper.')


    return parser.parse_args(argv)


def find_sheet_paper(frame, thresh, add_margin=True):
    """Detect the coords of the sheet of paper the game will be played on"""
    if frame is None or thresh is None:
        return None, None

    stats = detections.find_corners(thresh)

    # Need 1 point for center of image + 4 points for corners
    if stats is None or len(stats) < 5:
        # print("[DEBUG] Not enough corners detected by Harris.") # Less verbose
        return None, None

    # First point is center of coordinate system (from find_corners), so ignore it
    corners = stats[1:, :2]
    corners = imutils.order_points(corners) # Ensure consistent order (tl, tr, br, bl)
    if corners is None:
         print("[WARN] Failed to order corner points.")
         return None, None

    # Get bird view of sheet of paper
    paper = imutils.four_point_transform(frame, corners)

    if paper is None or paper.size == 0:
         # print("[DEBUG] Failed to get bird's eye view transform.") # Less verbose
         return None, None

    if add_margin:
        # Add margin only if paper dimensions allow it
        h, w = paper.shape[:2]
        margin = 10
        if h > 2 * margin and w > 2 * margin:
            paper = paper[margin:-margin, margin:-margin]
        # else: # No need to warn if margin can't be added
            # print("[DEBUG] Paper dimensions too small for margin, skipping.")

    if paper is None or paper.size == 0:
        print("[WARN] Paper became invalid after applying margin.")
        return None, None

    return paper, corners


def find_shape(cell):
    """Classify the shape in a cell (X, O, or None) using the loaded Keras model."""
    global model
    if cell is None or cell.size == 0 or model is None:
        return None

    # Ensure the cell is grayscale (single channel) before resizing
    if len(cell.shape) == 3 and cell.shape[2] == 3:
         gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    elif len(cell.shape) == 2:
         gray_cell = cell
    else:
        print(f"[WARN] Unexpected cell shape for find_shape: {cell.shape}. Skipping.")
        return None

    # Further check if the cell contains significant non-zero pixels
    # This helps filter out empty/noise cells before prediction
    # Adjust the threshold (e.g., 0.05) as needed
    if cv2.countNonZero(gray_cell) < (gray_cell.size * 0.05):
         return None # Treat as empty if very few white pixels (assuming THRESH_BINARY_INV)


    mapper = {0: None, 1: 'X', 2: 'O'}
    try:
        processed_cell = detections.preprocess_input(gray_cell) # Use the grayscale cell
        prediction = model.predict(processed_cell, verbose=0) # Set verbose=0 to silence Keras prediction logs
        idx = np.argmax(prediction)
        confidence = prediction[0][idx]

        # --- Confidence Threshold ---
        # Adjust this threshold based on model performance
        confidence_threshold = 0.80
        if confidence < confidence_threshold:
            # print(f"[DEBUG] Shape prediction confidence {confidence:.2f} below threshold {confidence_threshold} for index {idx}.")
            return None # Not confident enough
        # --------------------------

        predicted_shape = mapper[idx]
        return predicted_shape

    except Exception as e:
        print(f"[ERROR] Error during shape prediction: {e}")
        return None


def get_board_template(thresh):
    """
    Detects the 3x3 grid structure from the thresholded paper view.
    Returns a list of 9 tuples, each representing (x, y, w, h) for a cell,
    ordered from top-left to bottom-right. Returns empty list on failure.
    """
    if thresh is None or thresh.size == 0:
        return []
    # Find grid's center cell first using contour detection
    middle_center = detections.contoured_bbox(thresh) # from detections.py
    if middle_center is None:
        return [] # Return empty list to indicate failure

    center_x, center_y, width, height = middle_center

    if width <= 5 or height <= 5: # Increased minimum size check
        print(f"[WARN] Invalid dimensions detected for center cell: w={width}, h={height}")
        return []

    gap = int(max(width, height) * 0.05) # Small gap, e.g., 5% of size
    eff_w = width + gap
    eff_h = height + gap

    left = center_x - eff_w
    right = center_x + eff_w
    top = center_y - eff_h
    bottom = center_y + eff_h

    grid_coords = [
        (left, top, width, height),         (center_x, top, width, height),     (right, top, width, height),
        (left, center_y, width, height),    (center_x, center_y, width, height),(right, center_y, width, height),
        (left, bottom, width, height),      (center_x, bottom, width, height),  (right, bottom, width, height)
    ]

    paper_h, paper_w = thresh.shape[:2]
    for i, (x, y, w, h) in enumerate(grid_coords):
        if x + w < 0 or y + h < 0 or x > paper_w or y > paper_h:
            print(f"[WARN] Calculated grid coordinates for cell {i} seem invalid relative to paper size. Grid detection likely failed.")
            return []

    return grid_coords


def draw_shape(template, shape, coords):
    """Draws the detected 'X' or 'O' shape onto the bird's eye view image."""
    if template is None or shape is None:
        return template

    x, y, w, h = map(int, coords) # Ensure integer coordinates

    if w <= 0 or h <= 0:
        return template

    color = (0, 0, 255) # Red color for drawing shapes
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

# --- Robot Arm Functions ---

def send_servo_commands(ser, servo1_angle, servo2_angle, servo3_angle, enable_signal, pickup_flag):
    """
    Constructs and prints the robot command string for debugging, then attempts
    to send it via serial if the port is available.
    Format: servo1,servo2,servo3,enable,pickup_flag\n

    Args:
        ser (serial.Serial): The active serial connection object (can be None).
        servo1_angle (int): Angle for servo 1.
        servo2_angle (int): Angle for servo 2.
        servo3_angle (int): Angle for servo 3.
        enable_signal (int): Enable signal (typically 0 or 1).
        pickup_flag (int): Flag indicating pickup phase (1 if picking up, 0 otherwise).

    Returns:
        bool: True if the command was successfully sent, False otherwise
              (or if the serial port was unavailable).
    """
    command_sent = False # Flag to track if write was successful
    try:
        # Ensure angles and flags are integers
        s1 = int(servo1_angle)
        s2 = int(servo2_angle)
        s3 = int(servo3_angle)
        enable = int(enable_signal)
        pickup = int(pickup_flag)

        # Validate flags
        if enable not in [0, 1]:
            print(f"[WARN] Invalid enable signal value ({enable}). Defaulting to 0.")
            enable = 0
        if pickup not in [0, 1]:
            print(f"[WARN] Invalid pickup flag value ({pickup}). Defaulting to 0.")
            pickup = 0

        # --- Construct the command string ---
        command = f"{s1},{s2},{s3},{enable},{pickup}\n" # Added pickup_flag

        # --- Always print the command for debugging ---
        print(f"[DEBUG ROBOT CMD] {command.strip()}") # Print regardless of 'ser' status

        # --- Attempt to send only if serial port is valid ---
        if ser is not None and ser.is_open:
            ser.write(command.encode('utf-8'))
            command_sent = True # Mark as sent
            time.sleep(0.1) # Small delay after successful write

        return command_sent # Return True only if write succeeded

    except serial.SerialException as e:
        print(f"[ERROR] Failed during serial write attempt: {e}")
        return False
    except ValueError as e:
        print(f"[ERROR] Invalid angle/flag value for servo command: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred during serial command processing: {e}")
        return False

def move_robot_arm(ser, position):
    """
    Calculates and prints robot sequence commands for debugging, then executes
    the sequence using send_servo_commands (which handles actual sending if ser is valid).
    Includes pickup_flag (1 for initial move to pickup, 0 otherwise).
    Cycles through NUM_PICKUP_POSITIONS pickup locations.
    Sends an 'enable' signal (1 during active move, 0 during final return home).

    Args:
        ser (serial.Serial): The active serial connection object (can be None).
        position (int): The target cell index on the game board (0-8) to place the piece.
    """
    global next_pickup_index
    global NUM_PICKUP_POSITIONS

    # --- === IMPORTANT: CALIBRATE THESE VALUES === ---
    angle_map = {
        # Play Positions (Cells 0-8) - EXAMPLE VALUES
        0: [150, 110, 170], 1: [150, 90, 170], 2: [150, 70, 170],
        3: [125, 115, 135], 4: [115, 90, 135], 5: [125, 65, 135],
        6: [100, 120, 120], 7: [90, 90, 120], 8: [100, 57, 120],
        # Special Positions
        'home':   [180, 90, 0],
        # Pickup Positions (Define 4)
        'pickup3': [155, 170, 150], 'pickup2': [125, 165, 130],
        'pickup1': [155, 180, 150], 'pickup0': [125, 180, 130],
    }
    # --- ========================================= ---

    # --- Define Delays (Adjust based on your robot's speed) ---
    pickup_delay = 2.0
    home_delay   = 2.0
    play_delay   = 2.0
    gripper_delay= 0.6
    # --- ========================================= ---

    # --- Gripper Control Function (Placeholder - Adapt if needed) ---
    def control_gripper(action):
        print(f"[DEBUG GRIPPER] {action}")
        if ser is not None and ser.is_open:
             # Send actual gripper command here if needed
             pass
        time.sleep(gripper_delay)

    # --- Sequence Logic ---
    pickup_key = f'pickup{next_pickup_index}'
    print(f"[ROBOT] Planning sequence using pickup position: {pickup_key}")
    next_pickup_index = (next_pickup_index + 1) % NUM_PICKUP_POSITIONS

    required_keys = [pickup_key, 'home', position]
    for key in required_keys:
        if key not in angle_map:
            print(f"[ERROR] Robot angle map missing required key: '{key}'. Aborting planned sequence.")
            return

    pickup_angles = angle_map[pickup_key]
    home_angles = angle_map['home']
    play_angles = angle_map[position]

    ENABLE_ACTIVE = 1
    ENABLE_INACTIVE = 0
    PICKUP_TRUE = 1  # Indicate pickup phase
    PICKUP_FALSE = 0 # Indicate non-pickup phase

    # --- Plan and Print/Execute Pickup -> Home -> Play -> Home Sequence ---
    try:
        print("-" * 20)
        print(f"[ROBOT] Planning sequence for cell {position}")

        # 1. Move to Pickup location (Active Phase, Pickup True)
        print(f"[ROBOT PLAN] 1. Move to {pickup_key} (Enable={ENABLE_ACTIVE}, Pickup={PICKUP_TRUE})...")
        command_ok = send_servo_commands(ser, *pickup_angles, ENABLE_ACTIVE, PICKUP_TRUE) # <<< pickup=1
        time.sleep(pickup_delay)
        # 1a. Close Gripper (Grab piece)
        control_gripper('close')
        if not command_ok and ser is not None:
             print("[ROBOT] Aborting sequence due to send failure.")
             return

        # 2. Move to Home position (with piece) (Active Phase, Pickup False)
        print(f"[ROBOT PLAN] 2. Move to Home (Enable={ENABLE_ACTIVE}, Pickup={PICKUP_FALSE})...")
        command_ok = send_servo_commands(ser, *home_angles, ENABLE_ACTIVE, PICKUP_FALSE) # <<< pickup=0
        time.sleep(home_delay)
        if not command_ok and ser is not None:
             print("[ROBOT] Aborting sequence due to send failure.")
             control_gripper('open')
             return

        # 3. Move to Play location (target cell) (Active Phase, Pickup False)
        print(f"[ROBOT PLAN] 3. Move to Play position {position} (Enable={ENABLE_ACTIVE}, Pickup={PICKUP_FALSE})...")
        command_ok = send_servo_commands(ser, *play_angles, ENABLE_ACTIVE, PICKUP_FALSE) # <<< pickup=0
        time.sleep(play_delay)
        # 3a. Open Gripper (Release piece - End of active phase)
        control_gripper('open')
        if not command_ok and ser is not None:
             print("[ROBOT] Aborting sequence due to send failure.")
             print("[ROBOT PLAN] Attempting return home after play failure...")
             send_servo_commands(ser, *home_angles, ENABLE_INACTIVE, PICKUP_FALSE) # <<< pickup=0
             time.sleep(home_delay)
             return

        # 4. Return to Home position (empty gripper) (Inactive Phase, Pickup False)
        print(f"[ROBOT PLAN] 4. Returning to Home (Enable={ENABLE_INACTIVE}, Pickup={PICKUP_FALSE})...")
        command_ok = send_servo_commands(ser, *home_angles, ENABLE_INACTIVE, PICKUP_FALSE) # <<< pickup=0
        time.sleep(home_delay)

        print(f"[ROBOT] Planned sequence for cell {position} complete.")
        print("-" * 20)

    except Exception as e:
        print(f"[ERROR] Unexpected error during robot sequence planning/execution: {e}")
        if ser is not None:
            try:
                print("[ROBOT] Attempting emergency return home and gripper open...")
                control_gripper('open')
                send_servo_commands(ser, *angle_map['home'], ENABLE_INACTIVE, PICKUP_FALSE) # <<< pickup=0
                time.sleep(home_delay)
            except Exception as emergency_e:
                print(f"[ERROR] Failed emergency return: {emergency_e}")

# --- Main Game Logic ---

def play(vcap, ser, add_paper_margin, zoom_enabled=True, zoom_factor=1.0, check_interval=3.0):
    """Runs the main Tic Tac Toe game loop."""
    board = Tic() # Initialize the game state representation
    history = {}  # Stores confirmed moves {cell_index: {'shape': 'X'/'O', 'bbox': (x,y,w,h)}}

    previous_state = [None] * 9 # State at the *last* check interval completion
    last_check_time = time.time()
    move_detected_in_last_cycle = False

    print("[INFO] Game started. Make your move (X) on the board.")
    print("[INFO] Press 'z' to toggle zoom, 'q' to quit.")

    paper_detection_threshold = 170
    grid_detection_threshold = 170

    while True:
        ret, frame_original = vcap.read()
        if not ret:
            print('[ERROR] Could not read frame from camera source. Exiting.')
            time.sleep(2)
            break

        # --- Frame Processing ---
        frame = zoom_frame(frame_original, zoom_factor) if zoom_enabled else frame_original.copy()
        if frame is None:
            print("[WARN] Zoom failed, using original frame.")
            frame = frame_original.copy()
            if frame is None:
                 print("[ERROR] Original frame is also None. Exiting.")
                 break

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print('[INFO] Quitting game.')
            break
        if key == ord('z'):
            zoom_enabled = not zoom_enabled
            zoom_factor_str = f"{zoom_factor}x" if zoom_enabled else "Off"
            print(f"[INFO] Zoom {'enabled' if zoom_enabled else 'disabled'} ({zoom_factor_str})")

        # --- Paper Detection ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_gray = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh_paper_detect = cv2.threshold(blurred_gray, paper_detection_threshold, 255, cv2.THRESH_BINARY)
        paper, corners = find_sheet_paper(frame, thresh_paper_detect, add_margin=add_paper_margin)

        # --- Visualization & Status ---
        display_frame = frame.copy()
        bird_view_display = np.zeros((300, 300, 3), dtype=np.uint8)
        status_text = ""

        if paper is None or corners is None:
            status_text = "Paper not detected"
        else:
            for c in corners:
                cv2.circle(display_frame, tuple(map(int, c)), 5, (0, 255, 0), -1)

            # --- Grid and Shape Detection ---
            paper_gray = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
            paper_gray_blurred = cv2.GaussianBlur(paper_gray, (5, 5), 0)
            _, paper_thresh = cv2.threshold(paper_gray_blurred, grid_detection_threshold, 255, cv2.THRESH_BINARY_INV)
            grid = get_board_template(paper_thresh)
            paper_display = paper.copy()

            if not grid:
                 status_text = "Grid not detected"
            else:
                status_text = "Board Detected"
                for i, (x, y, w, h) in enumerate(grid):
                     xi, yi, wi, hi = map(int, (x, y, w, h))
                     cv2.rectangle(paper_display, (xi, yi), (xi + wi, yi + hi), (0, 255, 0), 1)
                     if i in history:
                        shape = history[i]['shape']
                        paper_display = draw_shape(paper_display, shape, (xi, yi, wi, hi))
                bird_view_display = paper_display

                # --- Automatic Move Detection ---
                current_time = time.time()
                if current_time - last_check_time >= check_interval:
                    print(f"[INFO] Checking for player moves... (Interval: {check_interval}s)")
                    last_check_time = current_time
                    human_player = 'X'
                    ai_player = 'O'
                    current_state_detection = [None] * 9
                    available_indices = [i for i in range(9) if i not in history]
                    paper_h, paper_w = paper_thresh.shape[:2]

                    for i in available_indices:
                        if not grid: continue
                        x, y, w, h = grid[i]
                        xi, yi, wi, hi = map(int, (x, y, w, h))
                        x1_clip, y1_clip = max(0, xi), max(0, yi)
                        x2_clip = min(paper_w, xi + wi)
                        y2_clip = min(paper_h, yi + hi)
                        clipped_w, clipped_h = x2_clip - x1_clip, y2_clip - y1_clip

                        if clipped_w <= 5 or clipped_h <= 5:
                            current_state_detection[i] = None
                            continue

                        cell = paper_thresh[y1_clip : y2_clip, x1_clip : x2_clip]
                        shape_in_cell = find_shape(cell)
                        current_state_detection[i] = shape_in_cell

                    comparison_state = [history.get(i, {}).get('shape') for i in range(9)]
                    for i in available_indices:
                        comparison_state[i] = current_state_detection[i]

                    new_move_index = -1
                    for i in range(9):
                        if comparison_state[i] == human_player and previous_state[i] != human_player:
                            if i in available_indices:
                                print(f"[DEBUG] Potential move detected at index {i}: current={comparison_state[i]}, previous={previous_state[i]}")
                                new_move_index = i
                                break

                    if new_move_index != -1:
                        print(f"[INFO] Player (X) move detected in cell {new_move_index}!")
                        move_detected_in_last_cycle = True
                        history[new_move_index] = {'shape': human_player, 'bbox': grid[new_move_index]}
                        board.make_move(new_move_index, human_player)
                        xi, yi, wi, hi = map(int, grid[new_move_index])
                        bird_view_display = draw_shape(bird_view_display, human_player, (xi, yi, wi, hi))
                        previous_state = [history.get(i, {}).get('shape') for i in range(9)] # Update confirmed state

                        if board.complete():
                            print("[INFO] Player (X) wins or Draw!")
                            break

                        # --- Computer's Turn ---
                        print("[INFO] Computer (O) is thinking...")
                        computer_move = determine(board, ai_player)

                        if computer_move is not None and computer_move in board.available_moves():
                            print(f"[INFO] Computer chooses cell {computer_move}")
                            board.make_move(computer_move, ai_player)
                            history[computer_move] = {'shape': ai_player, 'bbox': grid[computer_move]}
                            xi, yi, wi, hi = map(int, grid[computer_move])
                            bird_view_display = draw_shape(bird_view_display, ai_player, (xi, yi, wi, hi))

                            # --- Trigger Robot Arm Sequence ---
                            move_robot_arm(ser, computer_move) # This now handles the full sequence

                            previous_state = [history.get(i, {}).get('shape') for i in range(9)] # Update confirmed state again

                            if board.complete():
                                print("[INFO] Computer (O) wins or Draw!")
                                break
                        else:
                            print(f"[WARN] AI determined move {computer_move}, but it's not available or None. Board state:\n{board}")
                            previous_state = [history.get(i, {}).get('shape') for i in range(9)] # Update state anyway

                    else: # No valid player move detected
                        if move_detected_in_last_cycle:
                             print("[INFO] Waiting for next player move...")
                             move_detected_in_last_cycle = False
                        current_confirmed_state = [history.get(i, {}).get('shape') for i in range(9)]
                        if current_confirmed_state != previous_state:
                            previous_state = current_confirmed_state

        # --- Display ---
        cv2.putText(display_frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('Tic Tac Toe - Camera View', display_frame)
        cv2.imshow('Tic Tac Toe - Bird\'s Eye View', bird_view_display)

    # --- Game Over ---
    winner = board.winner()
    result_text = f"Winner: {winner}" if winner else "It's a Draw!"
    print(f"\n[INFO] Game Over! {result_text}")

    final_display = bird_view_display if 'bird_view_display' in locals() and bird_view_display is not None else np.zeros((300, 300, 3), dtype=np.uint8)
    final_h, final_w = final_display.shape[:2]
    text_y = final_h - 20 if final_h > 40 else 20
    cv2.putText(final_display, result_text, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow('Tic Tac Toe - Bird\'s Eye View', final_display)
    print("[INFO] Press any key in the OpenCV window to exit.")
    cv2.waitKey(0)

    # --- Cleanup ---
    if vcap.isOpened(): vcap.release()
    if ser and ser.is_open: ser.close(); print("[INFO] Serial port closed.")
    cv2.destroyAllWindows()
    return winner

# --- Main Execution ---

def main(args):
    """Loads resources, initializes connections, and starts the game."""
    global model

    # --- Load Model ---
    if not os.path.exists(args.model):
        print(f"[ERROR] Model file not found at: {args.model}")
        sys.exit(1)
    try:
        model = load_model(args.model)
        print(f"[INFO] Keras model loaded successfully from: {args.model}")
    except Exception as e:
        print(f"[ERROR] Failed to load Keras model: {e}")
        sys.exit(1)

    # --- Initialize Camera ---
    source = args.camera_source
    vcap = None
    if source.isdigit():
        vcap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
        if not vcap.isOpened(): vcap = cv2.VideoCapture(int(source))
    else:
        if not source.startswith(('http://', 'https://', 'rtsp://')):
            source = 'http://' + source
            print(f"[INFO] Added http:// prefix to camera source: {source}")
        vcap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        print("[INFO] Waiting briefly for network stream to connect...")
        time.sleep(2.0)

    if not vcap or not vcap.isOpened():
        print(f"[ERROR] Could not open video source: {args.camera_source}")
        sys.exit(1)
    print(f"[INFO] Video source opened successfully: {args.camera_source}")

    # --- Initialize Serial ---
    ser = None
    if args.serial_port:
        try:
            ser = serial.Serial(args.serial_port, args.baud_rate, timeout=1)
            print(f"[INFO] Opening serial port {args.serial_port}...")
            time.sleep(2.5)
            if ser.is_open:
                print(f"[INFO] Serial port {args.serial_port} opened successfully at {args.baud_rate} baud.")
            else:
                 print(f"[WARN] Serial port {args.serial_port} found but failed to open. Continuing without robot control.")
                 ser = None
        except serial.SerialException as e:
            print(f"[WARN] Could not open serial port {args.serial_port}: {e}. Continuing without robot control.")
            ser = None
        except Exception as e:
             print(f"[ERROR] An unexpected error occurred during serial initialization: {e}")
             ser = None
    else:
        print("[INFO] No serial port specified. Robot arm control will be disabled.")

    # --- Configure Game Settings ---
    zoom_setting = args.zoom if not args.no_zoom else 1.0
    if zoom_setting < 1.0:
        print("[WARN] Zoom factor must be >= 1.0. Setting zoom to 1.0 (disabled).")
        zoom_setting = 1.0
    enable_zoom = zoom_setting > 1.0
    add_paper_margin_setting = not args.no_margin

    # --- Start Game ---
    try:
        winner = play(vcap, ser,
                      add_paper_margin=add_paper_margin_setting,
                      zoom_enabled=enable_zoom,
                      zoom_factor=zoom_setting,
                      check_interval=args.check_interval)
        print(f"\n--- FINAL RESULT ---")
        print(f"Winner is: {winner if winner else 'Draw'}")
        print("--------------------\n")
    except Exception as e:
        print(f"\n[FATAL ERROR] An unhandled exception occurred during the game loop:")
        import traceback
        traceback.print_exc()
    finally:
        print("[INFO] Cleaning up resources...")
        if vcap and vcap.isOpened(): vcap.release(); print("[INFO] Camera released.")
        if ser and ser.is_open: ser.close(); print("[INFO] Serial port closed.")
        cv2.destroyAllWindows(); print("[INFO] OpenCV windows closed.")

    sys.exit(0)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)