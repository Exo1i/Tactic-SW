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
import serial

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Use tensorflow.keras everywhere for compatibility
try:
    from tensorflow.keras.models import load_model
except ImportError:
    print("[ERROR] TensorFlow not found. Please install it.")
    sys.exit(1)

try:
    from utils import imutils
    from utils import detections
    from alphabeta import Tic, get_enemy, determine
except ImportError as e:
    print(f"[ERROR] Failed to import required modules (utils, alphabeta): {e}")
    print("Ensure utils/imutils.py, utils/detections.py, and alphabeta.py are accessible.")
    sys.exit(1)

# --- Global Variables ---
model = None
next_pickup_index = 0
NUM_PICKUP_POSITIONS = 4

# --- Helper Functions ---

def zoom_frame(frame, zoom=2.0):
    if frame is None or zoom <= 1.0:
        return frame
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2
    roi_width = max(1, int(width / zoom))
    roi_height = max(1, int(height / zoom))
    x1 = max(0, center_x - roi_width // 2)
    y1 = max(0, center_y - roi_height // 2)
    x2 = min(width, x1 + roi_width)
    y2 = min(height, y1 + roi_height)
    x1 = max(0, x2 - roi_width)
    y1 = max(0, y2 - roi_height)
    try:
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            print("[WARN] Zoom resulted in empty ROI, returning original frame.")
            return frame
        return cv2.resize(roi, (width, height), interpolation=cv2.INTER_LINEAR)
    except Exception as e:
        print(f"[ERROR] Error during zoom/resize: {e}")
        return frame

def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Play Tic Tac Toe using OpenCV and an optional robot arm.")
    parser.add_argument('camera_source', type=str, help='Camera index (e.g., 0) or IP camera stream URL')
    parser.add_argument('--model', '-m', type=str, default='data/model.h5', help='Path to the Keras model file (.h5)')
    parser.add_argument('--serial-port', type=str, default=None, help='Serial port for ESP32 robot arm')
    parser.add_argument('--baud-rate', type=int, default=9600, help='Baud rate for serial communication')
    parser.add_argument('--zoom', type=float, default=1.0, help='Initial zoom factor (default: 1.0)')
    parser.add_argument('--no-zoom', action='store_true', help='Disable zoom feature')
    parser.add_argument('--check-interval', type=float, default=3.0, help='Interval in seconds to check for player moves')
    parser.add_argument('--no-margin', action='store_true', help='Disable adding margin when detecting paper')
    return parser.parse_args(argv)

def find_sheet_paper(frame, thresh, add_margin=True):
    if frame is None or thresh is None:
        return None, None
    stats = detections.find_corners(thresh)
    if stats is None or len(stats) < 5:
        return None, None
    corners = stats[1:, :2]
    corners = imutils.order_points(corners)
    if corners is None:
        print("[WARN] Failed to order corner points.")
        return None, None
    paper = imutils.four_point_transform(frame, corners)
    if paper is None or paper.size == 0:
        return None, None
    if add_margin:
        h, w = paper.shape[:2]
        margin = 10
        if h > 2 * margin and w > 2 * margin:
            paper = paper[margin:-margin, margin:-margin]
    if paper is None or paper.size == 0:
        print("[WARN] Paper became invalid after applying margin.")
        return None, None
    return paper, corners

def find_shape(cell):
    global model
    if cell is None or cell.size == 0 or model is None:
        return None
    if len(cell.shape) == 3 and cell.shape[2] == 3:
        gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    elif len(cell.shape) == 2:
        gray_cell = cell
    else:
        print(f"[WARN] Unexpected cell shape for find_shape: {cell.shape}. Skipping.")
        return None
    if cv2.countNonZero(gray_cell) < (gray_cell.size * 0.05):
        return None
    mapper = {0: None, 1: 'X', 2: 'O'}
    try:
        processed_cell = detections.preprocess_input(gray_cell)
        prediction = model.predict(processed_cell, verbose=0)
        idx = np.argmax(prediction)
        confidence = prediction[0][idx]
        confidence_threshold = 0.80
        if confidence < confidence_threshold:
            return None
        return mapper[idx]
    except Exception as e:
        print(f"[ERROR] Error during shape prediction: {e}")
        return None

def get_board_template(thresh):
    if thresh is None or thresh.size == 0:
        return []
    middle_center = detections.contoured_bbox(thresh)
    if middle_center is None:
        return []
    center_x, center_y, width, height = middle_center
    if width <= 5 or height <= 5:
        print(f"[WARN] Invalid dimensions detected for center cell: w={width}, h={height}")
        return []
    gap = int(max(width, height) * 0.05)
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
    if template is None or shape is None:
        return template
    x, y, w, h = map(int, coords)
    if w <= 0 or h <= 0:
        return template
    color = (0, 0, 255)
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

def send_servo_commands(ser, servo1_angle, servo2_angle, servo3_angle, enable_signal, pickup_flag):
    command_sent = False
    try:
        s1 = int(servo1_angle)
        s2 = int(servo2_angle)
        s3 = int(servo3_angle)
        enable = int(enable_signal)
        pickup = int(pickup_flag)
        if enable not in [0, 1]:
            print(f"[WARN] Invalid enable signal value ({enable}). Defaulting to 0.")
            enable = 0
        if pickup not in [0, 1]:
            print(f"[WARN] Invalid pickup flag value ({pickup}). Defaulting to 0.")
            pickup = 0
        command = f"{s1},{s2},{s3},{enable},{pickup}\n"
        print(f"[DEBUG ROBOT CMD] {command.strip()}")
        if ser is not None and ser.is_open:
            ser.write(command.encode('utf-8'))
            command_sent = True
            time.sleep(0.1)
        return command_sent
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
    global next_pickup_index
    global NUM_PICKUP_POSITIONS
    angle_map = {
        0: [150, 110, 170], 1: [150, 90, 170], 2: [150, 70, 170],
        3: [125, 115, 135], 4: [115, 90, 135], 5: [125, 65, 135],
        6: [100, 120, 120], 7: [90, 90, 120], 8: [100, 57, 120],
        'home':   [180, 90, 0],
        'pickup3': [155, 170, 150], 'pickup2': [125, 165, 130],
        'pickup1': [155, 180, 150], 'pickup0': [125, 180, 130],
    }
    pickup_delay = 2.0
    home_delay = 2.0
    play_delay = 2.0
    gripper_delay = 0.6
    def control_gripper(action):
        print(f"[DEBUG GRIPPER] {action}")
        if ser is not None and ser.is_open:
            pass
        time.sleep(gripper_delay)
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
    PICKUP_TRUE = 1
    PICKUP_FALSE = 0
    try:
        print("-" * 20)
        print(f"[ROBOT] Planning sequence for cell {position}")
        print(f"[ROBOT PLAN] 1. Move to {pickup_key} (Enable={ENABLE_ACTIVE}, Pickup={PICKUP_TRUE})...")
        command_ok = send_servo_commands(ser, *pickup_angles, ENABLE_ACTIVE, PICKUP_TRUE)
        time.sleep(pickup_delay)
        control_gripper('close')
        if not command_ok and ser is not None:
            print("[ROBOT] Aborting sequence due to send failure.")
            return
        print(f"[ROBOT PLAN] 2. Move to Home (Enable={ENABLE_ACTIVE}, Pickup={PICKUP_FALSE})...")
        command_ok = send_servo_commands(ser, *home_angles, ENABLE_ACTIVE, PICKUP_FALSE)
        time.sleep(home_delay)
        if not command_ok and ser is not None:
            print("[ROBOT] Aborting sequence due to send failure.")
            control_gripper('open')
            return
        print(f"[ROBOT PLAN] 3. Move to Play position {position} (Enable={ENABLE_ACTIVE}, Pickup={PICKUP_FALSE})...")
        command_ok = send_servo_commands(ser, *play_angles, ENABLE_ACTIVE, PICKUP_FALSE)
        time.sleep(play_delay)
        control_gripper('open')
        if not command_ok and ser is not None:
            print("[ROBOT] Aborting sequence due to send failure.")
            print("[ROBOT PLAN] Attempting return home after play failure...")
            send_servo_commands(ser, *home_angles, ENABLE_INACTIVE, PICKUP_FALSE)
            time.sleep(home_delay)
            return
        print(f"[ROBOT PLAN] 4. Returning to Home (Enable={ENABLE_INACTIVE}, Pickup={PICKUP_FALSE})...")
        command_ok = send_servo_commands(ser, *home_angles, ENABLE_INACTIVE, PICKUP_FALSE)
        time.sleep(home_delay)
        print(f"[ROBOT] Planned sequence for cell {position} complete.")
        print("-" * 20)
    except Exception as e:
        print(f"[ERROR] Unexpected error during robot sequence planning/execution: {e}")
        if ser is not None:
            try:
                print("[ROBOT] Attempting emergency return home and gripper open...")
                control_gripper('open')
                send_servo_commands(ser, *angle_map['home'], ENABLE_INACTIVE, PICKUP_FALSE)
                time.sleep(home_delay)
            except Exception as emergency_e:
                print(f"[ERROR] Failed emergency return: {emergency_e}")

def play(vcap, ser, add_paper_margin, zoom_enabled=True, zoom_factor=1.0, check_interval=3.0):
    board = Tic()
    history = {}
    previous_state = [None] * 9
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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_gray = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh_paper_detect = cv2.threshold(blurred_gray, paper_detection_threshold, 255, cv2.THRESH_BINARY)
        paper, corners = find_sheet_paper(frame, thresh_paper_detect, add_margin=add_paper_margin)
        display_frame = frame.copy()
        bird_view_display = np.zeros((300, 300, 3), dtype=np.uint8)
        status_text = ""
        if paper is None or corners is None:
            status_text = "Paper not detected"
        else:
            for c in corners:
                cv2.circle(display_frame, tuple(map(int, c)), 5, (0, 255, 0), -1)
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
                        previous_state = [history.get(i, {}).get('shape') for i in range(9)]
                        if board.complete():
                            print("[INFO] Player (X) wins or Draw!")
                            break
                        print("[INFO] Computer (O) is thinking...")
                        computer_move = determine(board, ai_player)
                        if computer_move is not None and computer_move in board.available_moves():
                            print(f"[INFO] Computer chooses cell {computer_move}")
                            board.make_move(computer_move, ai_player)
                            history[computer_move] = {'shape': ai_player, 'bbox': grid[computer_move]}
                            xi, yi, wi, hi = map(int, grid[computer_move])
                            bird_view_display = draw_shape(bird_view_display, ai_player, (xi, yi, wi, hi))
                            move_robot_arm(ser, computer_move)
                            previous_state = [history.get(i, {}).get('shape') for i in range(9)]
                            if board.complete():
                                print("[INFO] Computer (O) wins or Draw!")
                                break
                        else:
                            print(f"[WARN] AI determined move {computer_move}, but it's not available or None. Board state:\n{board}")
                            previous_state = [history.get(i, {}).get('shape') for i in range(9)]
                    else:
                        if move_detected_in_last_cycle:
                            print("[INFO] Waiting for next player move...")
                            move_detected_in_last_cycle = False
                        current_confirmed_state = [history.get(i, {}).get('shape') for i in range(9)]
                        if current_confirmed_state != previous_state:
                            previous_state = current_confirmed_state
        cv2.putText(display_frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('Tic Tac Toe - Camera View', display_frame)
        cv2.imshow('Tic Tac Toe - Bird\'s Eye View', bird_view_display)
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
    if vcap.isOpened(): vcap.release()
    if ser and ser.is_open: ser.close(); print("[INFO] Serial port closed.")
    cv2.destroyAllWindows()
    return winner

def main(args):
    global model
    if not os.path.exists(args.model):
        print(f"[ERROR] Model file not found at: {args.model}")
        sys.exit(1)
    try:
        model = load_model(args.model)
        print(f"[INFO] Keras model loaded successfully from: {args.model}")
    except Exception as e:
        print(f"[ERROR] Failed to load Keras model: {e}")
        sys.exit(1)
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
    zoom_setting = args.zoom if not args.no_zoom else 1.0
    if zoom_setting < 1.0:
        print("[WARN] Zoom factor must be >= 1.0. Setting zoom to 1.0 (disabled).")
        zoom_setting = 1.0
    enable_zoom = zoom_setting > 1.0
    add_paper_margin_setting = not args.no_margin
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