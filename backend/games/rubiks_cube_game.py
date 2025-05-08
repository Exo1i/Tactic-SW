# File: games/rubiks_cube_game_reimplemented.py
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

class RubiksCubeGame: # Renamed from RubiksCubeSolver
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # --- State Variables (for frontend compatibility) ---
        self.mode = "idle" # "idle", "calibrating", "scanning", "solving", "scrambling", "error"
        self.status_message = "Ready"
        self.error_message = None
        self.calibration_step = 0 # Current color index being calibrated
        self.current_scan_idx = 0 # Current U-face scan index (0-11)
        self.current_solve_move_index = 0 # For displaying progress
        self.total_solve_moves = 0      # For displaying progress
        self.solution = None            # Stores the solution string
        self.serial_connection = None   # Will hold the serial.Serial object
        
        # --- Internal CV/Logic State ---
        self.last_valid_grid = None     # (pad_x, pad_y, grid_size) for drawing/sampling
        self.last_processed_frame = None # Stores the raw cv2 frame (after resize)
        self.u_scans = [[] for _ in range(12)] # Stores the 9 colors of U face for each scan
        self.last_scan_time = time.time()
        self.last_motor_move_time = time.time() # Initialize to now
        self.stability_counter = 0
        self.prev_contour = None        # For stability check
        self.prev_x = 0                 # For stability check
        self.prev_y = 0                 # For stability check
        self.prev_face_colors = None    # To detect duplicate scans

        # --- Constants (from RubiksCubeSolver) ---
        self.WINDOW_SIZE = (640, 480) # Internal processing size, display can be different
        self.SCAN_COOLDOWN = self.config.get('scan_cooldown', 0.5)
        self.MOTOR_STABILIZATION_TIME = self.config.get('motor_stabilization_time', 0.5)
        self.STABILITY_THRESHOLD = self.config.get('stability_threshold', 3)
        self.MIN_CONTOUR_AREA = self.config.get('min_contour_area', 5000)
        self.MAX_CONTOUR_AREA = self.config.get('max_contour_area', 50000)
        self.COLOR_NAMES = ["W", "R", "G", "Y", "O", "B"] # Standard for calibration UI

        # Rotation sequence (from RubiksCubeSolver)
        self.rotation_sequence = [
            "", "R L'", "B F'", "R L'", "B F'", "R L'", "B F'", "R L'", "B F'", "R L'", "B F'", "R L'"
        ]

        # Color ranges (from RubiksCubeSolver, or loaded)
        self.color_ranges = { # Default from RubiksCubeSolver
            "W": (np.array([0, 0, 180]), np.array([180, 50, 255])),
            "R": (np.array([0, 120, 100]), np.array([10, 255, 255])),
            "G": (np.array([40, 100, 100]), np.array([80, 255, 255])),
            "Y": (np.array([25, 120, 120]), np.array([35, 255, 255])),
            "O": (np.array([10, 120, 120]), np.array([20, 255, 255])),
            "B": (np.array([90, 100, 100]), np.array([120, 255, 255]))
        }
        self._load_color_ranges() # Load overrides if file exists

        # Initialize serial communication
        self.serial_port = self.config.get('serial_port', '/dev/ttyACM0')
        self.serial_baudrate = self.config.get('serial_baudrate', 9600)
        self._init_serial(self.serial_port, self.serial_baudrate)


    def _init_serial(self, serial_port, baud_rate):
        """Initialize serial communication with Arduino. (Adapted from RubiksCubeSolver)"""
        try:
            if self.serial_connection and self.serial_connection.is_open:
                self.serial_connection.close()
            self.serial_connection = serial.Serial(serial_port, baud_rate, timeout=1)
            time.sleep(2) # Wait for Arduino to reset
            print(f"Serial connection established on {serial_port}.")
            self.status_message = f"Arduino connected on {serial_port}"
        except Exception as e:
            print(f"Error opening serial port {serial_port}: {e}")
            self.serial_connection = None
            self.error_message = f"Serial Error: {e}"
            self.status_message = f"Error connecting to Arduino on {serial_port}"
            # Do not exit, let the game run in a degraded state or allow reconnection attempts.

    def _send_arduino_command(self, cmd: str, wait_for_ack: bool = True, timeout_seconds: int = 10) -> bool:
        """
        Send command to Arduino and optionally wait for acknowledgment.
        (Adapted from RubiksCubeSolver's _send_arduino_command and merged with RubiksCubeGame's robustness)
        """
        if not self.serial_connection or not self.serial_connection.is_open:
            self.error_message = "Serial connection not available."
            self.status_message = "Serial disconnected. Cannot send command."
            print(self.error_message)
            return False

        try:
            print(f"Sending to Arduino: {cmd}")
            self.serial_connection.write(f"{cmd}\n".encode())
            self.serial_connection.flush()

            if not wait_for_ack:
                return True

            start_time = time.time()
            # Adjust timeout for long solution strings
            is_long_cmd = len(cmd.split()) > 10 
            current_timeout = 30 if is_long_cmd else timeout_seconds

            while time.time() - start_time < current_timeout:
                if self.serial_connection.in_waiting:
                    response = self.serial_connection.readline().decode().strip()
                    print(f"Arduino response: {response}")
                    if "completed" in response.lower() or "executed" in response.lower() or "received" in response.lower():
                        print("Command execution acknowledged/completed by Arduino.")
                        if self.mode == "solving" and is_long_cmd:
                            self.mode = "idle"
                            self.status_message = "Solution completed."
                            self.current_solve_move_index = self.total_solve_moves # Mark as done
                        elif self.mode == "scrambling" and is_long_cmd:
                             self.mode = "idle"
                             self.status_message = "Scramble completed."
                             self.current_solve_move_index = self.total_solve_moves
                        return True
                    elif "error" in response.lower():
                        self.error_message = f"Arduino Error: {response}"
                        print(self.error_message)
                        return False
                time.sleep(0.05)
            
            self.error_message = f"Timeout waiting for Arduino response to '{cmd[:30]}...'"
            print(self.error_message)
            return False
        except Exception as e:
            self.error_message = f"Error sending command '{cmd[:30]}...': {e}"
            print(self.error_message)
            return False

    def _save_color_ranges(self, filename="color_ranges.json"): # From RubiksCubeSolver
        serializable_ranges = {
            color: (lower.tolist(), upper.tolist()) for color, (lower, upper) in self.color_ranges.items()
        }
        try:
            with open(filename, 'w') as f:
                json.dump(serializable_ranges, f, indent=4)
            print(f"Color ranges saved to {filename}")
            self.status_message = "Color calibration saved."
        except Exception as e:
            print(f"Error saving color ranges to {filename}: {e}")
            self.error_message = f"Failed to save color ranges: {e}"

    def _load_color_ranges(self, filename="color_ranges.json"): # From RubiksCubeSolver
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    serializable_ranges = json.load(f)
                
                loaded_ranges = {}
                for color, ranges_data in serializable_ranges.items():
                    if isinstance(ranges_data, list) and len(ranges_data) == 2: # Old format or single range as list
                         loaded_ranges[color] = (np.array(ranges_data[0]), np.array(ranges_data[1]))
                    elif isinstance(ranges_data, tuple) and len(ranges_data) == 2: # Single range
                         loaded_ranges[color] = (np.array(ranges_data[0]), np.array(ranges_data[1]))
                    # RubiksCubeSolver code did not have multi-range for 'R', if you add it, handle here.
                    else:
                        print(f"Warning: Malformed range data for color {color} in {filename}. Skipping.")
                        continue
                self.color_ranges.update(loaded_ranges)
                print(f"Color ranges loaded from {filename}")
            except Exception as e:
                print(f"Error loading color ranges from {filename}: {e}. Using defaults.")
        else:
            print("No saved color ranges found. Using defaults from RubiksCubeSolver.")

    # --- Core Logic Methods (Adapted from RubiksCubeSolver) ---
    def _detect_color(self, roi): # Directly from RubiksCubeSolver
        if roi is None or roi.size == 0 or len(roi.shape) != 3 or roi.shape[2] != 3:
            # print("Debug: Invalid ROI for color detection")
            return "U" # Unknown or fallback

        if roi.dtype != np.uint8:
            roi = np.uint8(roi)
        
        roi_blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        
        h, w = roi_blurred.shape[:2]
        y_start, y_end = h // 4, 3 * h // 4
        x_start, x_end = w // 4, 3 * w // 4

        if y_start >= y_end or x_start >= x_end:
            center_roi = roi_blurred
        else:
            center_roi = roi_blurred[y_start:y_end, x_start:x_end]

        if center_roi.size == 0: return "U"

        hsv_roi = cv2.cvtColor(center_roi, cv2.COLOR_BGR2HSV)
        
        color_matches_percent = {}
        for color_name, (lower, upper) in self.color_ranges.items():
            mask = cv2.inRange(hsv_roi, lower, upper)
            match_percentage = (cv2.countNonZero(mask) / (center_roi.shape[0] * center_roi.shape[1])) * 100
            color_matches_percent[color_name] = match_percentage
            
        range_best_color = max(color_matches_percent, key=color_matches_percent.get)
        range_best_match_percent = color_matches_percent[range_best_color]

        pixels = hsv_roi.reshape((-1, 3))
        if not pixels.size: return "U"
        pixel_list = [tuple(p) for p in pixels]
        most_common_hsv_tuple = Counter(pixel_list).most_common(1)[0][0]
        
        dominant_hsv_match_color = None
        min_hsv_distance = float('inf')
        
        for color_name, (lower, upper) in self.color_ranges.items():
            middle_hsv = (lower + upper) / 2
            h_dist = min(abs(most_common_hsv_tuple[0] - middle_hsv[0]), 180 - abs(most_common_hsv_tuple[0] - middle_hsv[0]))
            s_dist = abs(most_common_hsv_tuple[1] - middle_hsv[1])
            v_dist = abs(most_common_hsv_tuple[2] - middle_hsv[2])
            distance = (0.1 * h_dist + 0.3 * s_dist + 0.6 * v_dist) if color_name == "W" else (0.7 * h_dist + 0.2 * s_dist + 0.1 * v_dist)
            if distance < min_hsv_distance:
                min_hsv_distance = distance
                dominant_hsv_match_color = color_name
        
        # print(f"ROI: Range={range_best_color}({range_best_match_percent:.1f}%), Dominant={dominant_hsv_match_color}({min_hsv_distance:.1f})") # Keep for debug

        if range_best_match_percent > 20: return range_best_color
        if dominant_hsv_match_color and min_hsv_distance < 30: return dominant_hsv_match_color
        
        avg_hsv_roi = np.mean(hsv_roi, axis=(0,1))
        avg_hsv_match_color = None
        min_avg_hsv_distance = float('inf')
        for color_name, (lower, upper) in self.color_ranges.items():
            middle_hsv = (lower + upper) / 2
            h_dist = min(abs(avg_hsv_roi[0] - middle_hsv[0]), 180 - abs(avg_hsv_roi[0] - middle_hsv[0]))
            s_dist = abs(avg_hsv_roi[1] - middle_hsv[1])
            v_dist = abs(avg_hsv_roi[2] - middle_hsv[2])
            distance = (0.1 * h_dist + 0.3 * s_dist + 0.6 * v_dist) if color_name == "W" else (0.7 * h_dist + 0.2 * s_dist + 0.1 * v_dist)
            if distance < min_avg_hsv_distance:
                min_avg_hsv_distance = distance
                avg_hsv_match_color = color_name
        
        if avg_hsv_match_color and min_avg_hsv_distance < 30: return avg_hsv_match_color
        
        # print(f"Defaulting to W for ROI due to low confidence.") # Debug
        return "W" # Fallback, as per RubiksCubeSolver

    def _validate_cube(self, cube, order_name): # From RubiksCubeSolver
        if len(cube) != 54:
            raise ValueError(f"{order_name} must be 54 characters, got {len(cube)}")
        counts = Counter(cube)
        if len(counts) != 6 or any(count != 9 for count in counts.values()):
            raise ValueError(f"{order_name} invalid: {counts} (need 9 of each of 6 colors)")

    def _remap_colors_to_kociemba(self, cube_frblud): # From RubiksCubeSolver
        self._validate_cube(cube_frblud, "FRBLUD_for_remap")
        # Centers in FRBLUD: F:4, R:13, B:22, L:31, U:40, D:49
        centers_frblud_order = [cube_frblud[i] for i in [4, 13, 22, 31, 40, 49]]
        
        # Kociemba mapping: color of U-center -> 'U', R-center -> 'R', etc.
        # This is what RubiksCubeSolver's mapping achieved.
        # Centers physical colors: U_phys=centers_frblud_order[4], R_phys=centers_frblud_order[1], F_phys=centers_frblud_order[0]...
        color_map = {
            centers_frblud_order[4]: 'U', # Physical color of U-face's center becomes 'U'
            centers_frblud_order[1]: 'R', # Physical color of R-face's center becomes 'R'
            centers_frblud_order[0]: 'F', # Physical color of F-face's center becomes 'F'
            centers_frblud_order[5]: 'D', # Physical color of D-face's center becomes 'D'
            centers_frblud_order[3]: 'L', # Physical color of L-face's center becomes 'L'
            centers_frblud_order[2]: 'B'  # Physical color of B-face's center becomes 'B'
        }
        if len(set(color_map.keys())) != 6:
             raise ValueError(f"Center colors are not unique for Kociemba mapping: {centers_frblud_order}")
        return color_map, ''.join(color_map[c] for c in cube_frblud)

    def _remap_cube_to_kociemba_order(self, cube_frblud_mapped_colors): # From RubiksCubeSolver, renamed
        # Input is FRBLUD string but with Kociemba face characters ('U', 'R', 'F'...)
        # Output is URFDLB order string
        F_face = cube_frblud_mapped_colors[0:9]
        R_face = cube_frblud_mapped_colors[9:18]
        B_face = cube_frblud_mapped_colors[18:27]
        L_face = cube_frblud_mapped_colors[27:36]
        U_face = cube_frblud_mapped_colors[36:45]
        D_face = cube_frblud_mapped_colors[45:54]
        return U_face + R_face + F_face + D_face + L_face + B_face

    # _get_solved_state from RubiksCubeSolver is not strictly needed if kociemba.solve uses default target.

    def _is_cube_solved(self, cube_frblud_state): # From RubiksCubeSolver
        if len(cube_frblud_state) != 54: return False
        for i in range(0, 54, 9):
            face = cube_frblud_state[i:i+9]
            if not all(sticker == face[4] for sticker in face):
                return False
        return True

    def _simplify_cube_moves(self, moves_str): # From RubiksCubeSolver
        if not moves_str.strip(): return ""
        moves = moves_str.strip().split()
        if not moves: return ""

        def move_value(move):
            if move.endswith("2"): return 2
            elif move.endswith("'"): return -1 # Or 3 for mod 4
            return 1
        
        def value_to_move(face, value_mod_4):
            if value_mod_4 == 0: return None
            if value_mod_4 == 1: return face
            if value_mod_4 == 2: return face + "2"
            if value_mod_4 == 3: return face + "'"
            return None # Should not happen

        # Simpler pass from RubiksCubeSolver (doesn't use face_groups)
        i = 0
        simplified = []
        while i < len(moves):
            current_face = moves[i][0]
            current_net_value = 0 # Accumulate net effect (1, 2, or -1/3)
            
            start_idx = i
            while i < len(moves) and moves[i][0] == current_face:
                current_net_value += move_value(moves[i])
                i += 1
            
            # Convert net_value (e.g. R R R = 3 or R R R' = 2) to final move (R', R2)
            # Apply modulo 4: 0 means no net move, 1 means single, 2 means double, 3 means prime
            final_val_mod_4 = current_net_value % 4
            move_char = value_to_move(current_face, final_val_mod_4)
            if move_char:
                simplified.append(move_char)
        return " ".join(simplified) if simplified else "No moves"


    def _solve_cube_frblud(self, cube_frblud_phys_colors): # From RubiksCubeSolver
        self.error_message = None
        try:
            if self._is_cube_solved(cube_frblud_phys_colors):
                print("Cube is already solved!")
                self.status_message = "Cube is already solved!"
                return "" 
            
            # Remap to Kociemba characters and order
            color_map_phys_to_koc_char, cube_frblud_koc_chars = self._remap_colors_to_kociemba(cube_frblud_phys_colors)
            scrambled_kociemba_ordered_str = self._remap_cube_to_kociemba_order(cube_frblud_koc_chars)
            
            self._validate_cube(scrambled_kociemba_ordered_str, "Scrambled Kociemba for solver")
            
            print(f"Kociemba input string: {scrambled_kociemba_ordered_str}")
            solution = kociemba.solve(scrambled_kociemba_ordered_str) # Default target is solved state
            print(f"Raw Kociemba solution: {solution}")
            
            # U-move replacements (from RubiksCubeSolver)
            u_replacement = "R L F2 B2 R' L' D R L F2 B2 R' L'"
            u_prime_replacement = "R L F2 B2 R' L' D' R L F2 B2 R' L'"
            u2_replacement = "R L F2 B2 R' L' D2 R L F2 B2 R' L'" # D2 can be D D
            
            moves = solution.split()
            modified_solution_list = []
            for move in moves:
                if move == "U": modified_solution_list.extend(u_replacement.split())
                elif move == "U'": modified_solution_list.extend(u_prime_replacement.split())
                elif move == "U2": modified_solution_list.extend(u2_replacement.split())
                else: modified_solution_list.append(move)
            
            final_solution_after_u_replace = " ".join(modified_solution_list)
            optimized_solution = self._simplify_cube_moves(final_solution_after_u_replace)
            
            print(f"Solution after U-replacement: {final_solution_after_u_replace}")
            print(f"Optimized solution: {optimized_solution} (Length: {len(optimized_solution.split()) if optimized_solution else 0})")
            
            return optimized_solution
        
        except ValueError as ve: # Kociemba specific error
            self.error_message = f"Kociemba solver error: {str(ve)}. Cube state likely invalid."
            print(self.error_message)
            return None
        except Exception as e:
            self.error_message = f"Error solving cube: {str(e)}"
            print(self.error_message)
            return None

    def _construct_cube_from_u_scans(self, u_scans_list): # From RubiksCubeSolver
        # This mapping is CRITICAL and specific to the physical scanner + rotation_sequence
        if not all(u_scans_list[i] and len(u_scans_list[i]) == 9 for i in range(12)):
             self.error_message = "Incomplete u_scans data for construction."
             print(self.error_message)
             return None

        cube_state_list = ['-'] * 54 # Initialize with placeholder

        # Fixed centers (as per RubiksCubeSolver's construct method)
        # These define the "canonical" orientation of the FRBLUD string we are building
        cube_state_list[4] = 'B'   # F center (Blue)
        cube_state_list[13] = 'O'  # R center (Orange)
        cube_state_list[22] = 'G'  # B center (Green)
        cube_state_list[31] = 'R'  # L center (Red)
        cube_state_list[40] = 'W'  # U center (White)
        cube_state_list[49] = 'Y'  # D center (Yellow)
        
        # Scan 0 (U face)
        cube_state_list[36:45] = u_scans_list[0] 
        
        # The rest of the mappings are taken directly from RubiksCubeSolver's _construct_cube_from_u_scans
        s = u_scans_list[1]; cube_state_list[0]=s[0]; cube_state_list[2]=s[2]; cube_state_list[3]=s[3]; cube_state_list[5]=s[5]; cube_state_list[6]=s[6]; cube_state_list[8]=s[8]
        s = u_scans_list[2]; cube_state_list[9]=s[0]; cube_state_list[10]=s[1]; cube_state_list[11]=s[2]; cube_state_list[15]=s[6]; cube_state_list[16]=s[7]; cube_state_list[17]=s[8]
        s = u_scans_list[3]; cube_state_list[47]=s[0]; cube_state_list[53]=s[2]; cube_state_list[1]=s[3]; cube_state_list[7]=s[5]; cube_state_list[45]=s[6]; cube_state_list[51]=s[8]
        s = u_scans_list[4]; cube_state_list[24]=s[0]; cube_state_list[12]=s[1]; cube_state_list[18]=s[2]; cube_state_list[26]=s[6]; cube_state_list[14]=s[7]; cube_state_list[20]=s[8]
        s = u_scans_list[5]; cube_state_list[33]=s[0]; cube_state_list[27]=s[2]; cube_state_list[50]=s[3]; cube_state_list[48]=s[5]; cube_state_list[35]=s[6]; cube_state_list[29]=s[8]
        s = u_scans_list[6]; cube_state_list[36]=s[0]; cube_state_list[46]=s[1]; cube_state_list[38]=s[2]; cube_state_list[42]=s[6]; cube_state_list[52]=s[7]; cube_state_list[44]=s[8]
        s = u_scans_list[7]; cube_state_list[21]=s[3]; cube_state_list[23]=s[5]
        s = u_scans_list[8]; cube_state_list[34]=s[1]; cube_state_list[28]=s[7]
        s = u_scans_list[9]; cube_state_list[25]=s[3]; cube_state_list[19]=s[5]
        s = u_scans_list[10]; cube_state_list[30]=s[1]; cube_state_list[32]=s[7]
        s = u_scans_list[11]; cube_state_list[39]=s[3]; cube_state_list[41]=s[5]
        
        final_str = "".join(cube_state_list)
        if '-' in final_str:
            self.error_message = f"Cube construction incomplete, {final_str.count('-')} placeholders remain."
            print(self.error_message)
            return None
        return final_str

    def _generate_scramble(self, moves=20): # From RubiksCubeSolver
        basic_moves = ['F', 'B', 'R', 'L', 'D'] # No U moves by default
        modifiers = ['', '\'', '2']
        scramble_list = []
        last_face = None
        for _ in range(moves):
            available_moves = [m for m in basic_moves if m != last_face]
            face = random.choice(available_moves)
            modifier = random.choice(modifiers)
            scramble_list.append(face + modifier)
            last_face = face
        return ' '.join(scramble_list)
    
    def _print_cube_state_visual(self, cube_frblud_str: str): # Adapted from RubiksCubeSolver's _print_full_cube_state
        if len(cube_frblud_str) != 54:
            print("Invalid cube string length for visual printing.")
            return
        
        print(f"\nConstructed Cube State (FRBLUD order): {cube_frblud_str}")
        print("Visual representation (U on top, F in front):")
        
        U = cube_frblud_str[36:45]; L = cube_frblud_str[27:36]; F = cube_frblud_str[0:9]
        R = cube_frblud_str[9:18]; B = cube_frblud_str[18:27]; D = cube_frblud_str[45:54]

        def pl(face_str, start_idx): return " ".join(face_str[start_idx : start_idx+3])
        print(f"        {pl(U,0)}\n        {pl(U,3)}\n        {pl(U,6)}")
        print(f"{pl(L,0)} {pl(F,0)} {pl(R,0)} {pl(B,0)}")
        print(f"{pl(L,3)} {pl(F,3)} {pl(R,3)} {pl(B,3)}")
        print(f"{pl(L,6)} {pl(F,6)} {pl(R,6)} {pl(B,6)}")
        print(f"        {pl(D,0)}\n        {pl(D,3)}\n        {pl(D,6)}")
        print("-" * 30)

    # --- Public Methods for FastAPI Interaction ---
    def process_frame(self, frame_bytes: bytes) -> Dict[str, Any]:
        """Main frame processing, called by WebSocket, returns state for frontend."""
        try:
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Failed to decode frame")

            # Resize to internal processing size (e.g., 640x480)
            frame = cv2.resize(frame, self.WINDOW_SIZE, interpolation=cv2.INTER_AREA)
            self.last_processed_frame = frame.copy() # Store for calibration/manual scan
            
            display_overlay_frame = frame.copy() # For drawing on

            if self.mode == "calibrating":
                # Draw calibration UI elements
                grid_size = int(min(self.WINDOW_SIZE) * 0.4)
                grid_cell_size = grid_size // 3
                pad_x, pad_y = 20, 50
                center_y_start = pad_y + grid_cell_size; center_y_end = pad_y + 2 * grid_cell_size
                center_x_start = pad_x + grid_cell_size; center_x_end = pad_x + 2 * grid_cell_size
                cv2.rectangle(display_overlay_frame, (center_x_start, center_y_start), (center_x_end, center_y_end), (0, 255, 0), 2)
                current_calib_color_name = self.COLOR_NAMES[self.calibration_step] if self.calibration_step < len(self.COLOR_NAMES) else "Done"
                instr = f"Show {current_calib_color_name}, press Capture" if current_calib_color_name != "Done" else "Calib Done. Save/Reset."
                cv2.putText(display_overlay_frame, instr, (pad_x, pad_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                self.last_valid_grid = (pad_x, pad_y, grid_size) # For calibrate_color method

            elif self.mode == "scanning":
                display_overlay_frame = self._perform_scan_cv_and_logic(frame, display_overlay_frame)
            
            elif self.mode in ["solving", "scrambling"]:
                mode_text = "Solving" if self.mode == "solving" else "Scrambling"
                prog_text = f"{mode_text} ({self.current_solve_move_index}/{self.total_solve_moves})"
                cv2.putText(display_overlay_frame, prog_text, (10, self.WINDOW_SIZE[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            
            else: # idle, error
                if self.last_valid_grid: # Show last grid in idle
                    px,py,gs = self.last_valid_grid
                    gcs = gs//3
                    cv2.rectangle(display_overlay_frame, (px,py), (px+gs, py+gs), (0,255,0),1)
                    for i in range(1,3):
                        cv2.line(display_overlay_frame, (px+i*gcs, py), (px+i*gcs, py+gs), (0,255,0),1)
                        cv2.line(display_overlay_frame, (px, py+i*gcs), (px+gs, py+i*gcs), (0,255,0),1)

            # Resize for sending to frontend (which expects 320x240 usually)
            frontend_display_frame = cv2.resize(display_overlay_frame, (320, 240), interpolation=cv2.INTER_AREA)
            _, buffer = cv2.imencode('.jpg', frontend_display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')

            return self.get_state(processed_frame_b64=frame_b64)

        except Exception as e:
            print(f"Error in process_frame: {e}")
            self.mode = "error"
            self.error_message = f"Frame processing error: {e}"
            # Send black frame on error
            error_img = np.zeros((240,320,3), dtype=np.uint8)
            cv2.putText(error_img, "Backend Error", (50,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
            _,err_buf = cv2.imencode('.jpg', error_img)
            err_b64 = base64.b64encode(err_buf).decode('utf-8')
            return self.get_state(processed_frame_b64=err_b64)

    def _perform_scan_cv_and_logic(self, raw_frame, display_frame_to_draw_on):
        """Internal CV and logic for scanning mode. Modifies display_frame_to_draw_on."""
        hsv = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2HSV)
        combined_mask = np.zeros((raw_frame.shape[0], raw_frame.shape[1]), dtype=np.uint8)
        for color, (lower, upper) in self.color_ranges.items():
            combined_mask |= cv2.inRange(hsv, lower, upper)
        
        kernel = np.ones((5,5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cube_detected = False
        best_contour_details = None # (contour, x,y,w,h, pad_x, pad_y, grid_size)

        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            area = cv2.contourArea(contour)
            if self.MIN_CONTOUR_AREA < area < self.MAX_CONTOUR_AREA:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                if len(approx) >= 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = float(w) / h
                    if 0.7 < aspect_ratio < 1.4: # Square-ish
                        grid_s = min(w,h)
                        cx, cy = x + w//2, y + h//2
                        pad_x, pad_y = cx - grid_s//2, cy - grid_s//2
                        best_contour_details = (contour, x,y,w,h, pad_x, pad_y, grid_s)
                        cube_detected = True
                        break
        
        status_text_display = f"Scan {self.current_scan_idx + 1}/12"

        if cube_detected and best_contour_details:
            contour, x,y,w,h, pad_x, pad_y, grid_size = best_contour_details
            self.last_valid_grid = (pad_x, pad_y, grid_size) # Update for display/sampling
            
            cv2.drawContours(display_frame_to_draw_on, [contour], -1, (0,255,0), 2, cv2.LINE_AA)
            gcs = grid_size // 3
            for i in range(1,3):
                cv2.line(display_frame_to_draw_on, (pad_x+i*gcs, pad_y), (pad_x+i*gcs, pad_y+grid_size), (0,255,0),1)
                cv2.line(display_frame_to_draw_on, (pad_x, pad_y+i*gcs), (pad_x+grid_size, pad_y+i*gcs), (0,255,0),1)
            cv2.rectangle(display_frame_to_draw_on, (pad_x,pad_y),(pad_x+grid_size, pad_y+grid_size),(0,255,0),2)

            if self.prev_contour is not None:
                shape_diff = cv2.matchShapes(contour, self.prev_contour, cv2.CONTOURS_MATCH_I1, 0.0)
                pos_diff = abs(x - self.prev_x) + abs(y - self.prev_y)
                if shape_diff < 0.3 and pos_diff < 20: self.stability_counter +=1
                else: self.stability_counter = 0
            self.prev_contour, self.prev_x, self.prev_y = contour, x, y
            status_text_display += f" - Stab: {self.stability_counter}/{self.STABILITY_THRESHOLD}"

            curr_time = time.time()
            motor_ready = (curr_time - self.last_motor_move_time) >= self.MOTOR_STABILIZATION_TIME
            cooldown_passed = (curr_time - self.last_scan_time) >= self.SCAN_COOLDOWN

            if self.stability_counter >= self.STABILITY_THRESHOLD and motor_ready and cooldown_passed and self.current_scan_idx < 12:
                face_colors = []
                all_colors_detected = True
                for r_idx in range(3):
                    for c_idx in range(3):
                        ystart, yend = pad_y + r_idx*gcs, pad_y + (r_idx+1)*gcs
                        xstart, xend = pad_x + c_idx*gcs, pad_x + (c_idx+1)*gcs
                        cell_pad = gcs // 8
                        roi = raw_frame[ystart+cell_pad : yend-cell_pad, xstart+cell_pad : xend-cell_pad]
                        color = self._detect_color(roi)
                        if color == "U": # Unknown
                            all_colors_detected = False; break
                        face_colors.append(color)
                        cv2.putText(display_frame_to_draw_on, color, (xstart+gcs//4, ystart+gcs//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),2,cv2.LINE_AA)
                    if not all_colors_detected: break
                
                if all_colors_detected and len(face_colors) == 9:
                    if self.prev_face_colors is None or face_colors != self.prev_face_colors:
                        self.u_scans[self.current_scan_idx] = face_colors
                        self.status_message = f"Scan {self.current_scan_idx+1} captured: {face_colors}"
                        print(self.status_message)
                        self.current_scan_idx += 1
                        self.prev_face_colors = face_colors
                        self.last_scan_time = curr_time
                        self.stability_counter = 0 # Reset stability after good scan

                        if self.current_scan_idx < 12:
                            next_move = self.rotation_sequence[self.current_scan_idx] # Move for *next* scan
                            if next_move:
                                self.status_message = f"Rotating for scan {self.current_scan_idx + 1}..."
                                if self._send_arduino_command(next_move):
                                    self.last_motor_move_time = time.time()
                                else:
                                    self.mode = "error"; self.error_message = f"Failed Arduino move: {next_move}"
                        else: # All 12 scans done
                            self.status_message = "All scans done. Processing..."
                            self._process_scans_and_solve() 
                    else:
                        self.status_message = "Duplicate scan. Hold steady or adjust."
                elif not all_colors_detected:
                     self.status_message = "Could not identify all 9 cell colors."
        else: # Cube not detected
            self.stability_counter = 0
            self.prev_contour = None
            status_text_display = f"Position cube for scan {self.current_scan_idx + 1}/12"
            if self.mode == "scanning": self.status_message = status_text_display
        
        cv2.putText(display_frame_to_draw_on, status_text_display, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),2,cv2.LINE_AA)
        return display_frame_to_draw_on

    def _process_scans_and_solve(self):
        """Called after all 12 scans are complete."""
        print("All scans completed! Sending final orienting move B F'...")
        self.status_message = "Scans complete. Orienting for solve..."
        if not self._send_arduino_command("B F'"): # Final orientation move from RubiksCubeSolver
            self.mode = "error"
            self.error_message = "Failed to execute final B F' post-scan."
            self.status_message = self.error_message
            return
        
        time.sleep(self.MOTOR_STABILIZATION_TIME) # Wait for B F'
        self.status_message = "Constructing cube state..."
        
        cube_frblud_str = self._construct_cube_from_u_scans(self.u_scans)
        if cube_frblud_str:
            self._print_cube_state_visual(cube_frblud_str)
            self.status_message = "Solving cube..."
            solution_str = self._solve_cube_frblud(cube_frblud_str)
            
            if solution_str is not None: # Can be "" if solved
                self.solution = solution_str
                self.total_solve_moves = len(solution_str.split()) if solution_str else 0
                self.current_solve_move_index = 0
                if not solution_str: # Already solved
                    self.mode = "idle"
                    self.status_message = "Cube is already solved!"
                else:
                    self.mode = "solving"
                    self.status_message = f"Solution found ({self.total_solve_moves} moves). Executing..."
                    if not self._send_arduino_command(f"SOLUTION:{solution_str}", timeout_seconds=60): # Use longer timeout for solution
                        self.mode = "error"
                        self.error_message = self.error_message or "Failed to send solution to Arduino."
                        self.status_message = self.error_message
                    # If successful, _send_arduino_command should set mode to idle for long solution
            else: # Solver error
                self.mode = "error"
                # self.error_message already set by _solve_cube_frblud
                self.status_message = self.error_message or "Solver returned an error."
        else: # Construction error
            self.mode = "error"
            # self.error_message already set by _construct_cube_from_u_scans
            self.status_message = self.error_message or "Failed to construct cube state from scans."

    def start_calibration(self):
        if self.mode not in ["idle", "error"]:
            self.status_message = f"Cannot start calibration from {self.mode}."
            return
        self.mode = "calibrating"
        self.calibration_step = 0
        self.error_message = None
        self.solution = None
        self.status_message = f"Calibration: Show {self.COLOR_NAMES[0]} center."
        print("Calibration mode started.")

    def capture_calibration_color(self): # Renamed from calibrate_color for clarity
        if self.mode != "calibrating" or self.calibration_step >= len(self.COLOR_NAMES):
            self.status_message = "Not in calibration or already done."
            return
        if not self.last_valid_grid or self.last_processed_frame is None:
            self.status_message = "Position cube in green box first."
            self.error_message = "No grid/frame for calibration sample."
            return

        try:
            pad_x, pad_y, grid_size = self.last_valid_grid
            gcs = grid_size // 3
            roi = self.last_processed_frame[pad_y+gcs : pad_y+2*gcs, pad_x+gcs : pad_x+2*gcs]
            if roi.size == 0: raise ValueError("Calibration ROI empty")

            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            # Using median for robustness, as in merged code
            h_median = np.median(hsv_roi[:,:,0]); s_median = np.median(hsv_roi[:,:,1]); v_median = np.median(hsv_roi[:,:,2])
            avg_hsv = np.array([h_median, s_median, v_median])
            
            current_color = self.COLOR_NAMES[self.calibration_step]
            # Dynamic ranges from RubiksCubeSolver's _calibrate_colors
            h_range = 5 if current_color != "W" else 90 
            s_range, v_range = 40, 40 # From RubiksCubeSolver
            
            lower = np.array([max(0, avg_hsv[0]-h_range), max(0,avg_hsv[1]-s_range), max(0,avg_hsv[2]-v_range)])
            upper = np.array([min(180, avg_hsv[0]+h_range), min(255,avg_hsv[1]+s_range), min(255,avg_hsv[2]+v_range)])
            
            # RubiksCubeSolver didn't have special handling for White during calibration beyond h_range
            # If you want the stricter White from the *merged* code:
            if current_color == "W":
                 lower[1] = 0; # Min Sat
                 lower[2] = max(150, avg_hsv[2] - v_range) # Min Brightness
                 upper[0] = 180 # Max Hue (any hue for white)
            
            self.color_ranges[current_color] = (lower, upper)
            print(f"Calibrated {current_color}: HSV_median={avg_hsv}, Range L={lower}, U={upper}")
            
            self.calibration_step += 1
            if self.calibration_step >= len(self.COLOR_NAMES):
                self.status_message = "All colors calibrated. Click Save."
            else:
                self.status_message = f"Calibrated {current_color}. Show {self.COLOR_NAMES[self.calibration_step]}."
            self.error_message = None
        except Exception as e:
            self.error_message = f"Calibration sample error: {e}"
            self.status_message = "Error during calibration capture."
            print(self.error_message)

    def save_calibration(self):
        if self.mode != "calibrating": 
            self.status_message = "Not in calibration mode."; return
        if self.calibration_step < len(self.COLOR_NAMES):
            self.status_message = "Calibration incomplete."; 
            self.error_message = "Calibrate all colors first."; return
        self._save_color_ranges() # Sets its own status_message
        self.mode = "idle"

    def reset_calibration(self):
        # Reset to RubiksCubeSolver's defaults
        self.color_ranges = {
            "W": (np.array([0, 0, 180]), np.array([180, 50, 255])),
            "R": (np.array([0, 120, 100]), np.array([10, 255, 255])),
            "G": (np.array([40, 100, 100]), np.array([80, 255, 255])),
            "Y": (np.array([25, 120, 120]), np.array([35, 255, 255])),
            "O": (np.array([10, 120, 120]), np.array([20, 255, 255])),
            "B": (np.array([90, 100, 100]), np.array([120, 255, 255]))
        }
        self.mode = "idle"
        self.calibration_step = 0
        self.status_message = "Calibration reset to defaults."
        self.error_message = None
        print("Color calibration reset to RubiksCubeSolver defaults.")

    def start_solve(self): # Corresponds to "/start_solve"
        if self.mode not in ["idle", "error"]:
            self.status_message = f"Cannot start solve from {self.mode}."
            return
        if not self.serial_connection or not self.serial_connection.is_open:
            self.mode = "error"; self.error_message = "Serial disconnected."
            self.status_message = self.error_message; return

        self.mode = "scanning"
        self.current_scan_idx = 0
        self.u_scans = [[] for _ in range(12)]
        self.prev_face_colors = None; self.stability_counter = 0; self.prev_contour = None
        self.last_scan_time = time.time(); self.last_motor_move_time = time.time()
        self.error_message = None; self.solution = None
        self.status_message = "Scanning: Position cube for first scan (U face)."
        print("Scanning mode started for solving.")

    def start_scramble(self): # Corresponds to "/start_scramble"
        if self.mode not in ["idle", "error"]:
            self.status_message = f"Cannot start scramble from {self.mode}."
            return
        if not self.serial_connection or not self.serial_connection.is_open:
            self.mode = "error"; self.error_message = "Serial disconnected."
            self.status_message = self.error_message; return
        
        try:
            scramble_str = self._generate_scramble()
            self.mode = "scrambling"
            self.status_message = f"Executing scramble: {scramble_str[:30]}..."
            self.error_message = None; self.solution = None
            self.total_solve_moves = len(scramble_str.split())
            self.current_solve_move_index = 0
            
            print(f"Sending scramble to Arduino: {scramble_str}")
            if not self._send_arduino_command(scramble_str, timeout_seconds=30): # Give more time for scramble
                self.mode = "error"
                self.error_message = self.error_message or "Failed to execute scramble."
                self.status_message = self.error_message
            # _send_arduino_command should set mode to idle if it was a long cmd
            elif self.mode == "scrambling": # If not reset by send_arduino_command
                self.mode = "idle"
                self.status_message = "Scramble completed."
        except Exception as e:
            self.mode = "error"; self.error_message = f"Scramble error: {e}"
            self.status_message = self.error_message; print(self.error_message)

    def stop_and_reset(self): # Corresponds to "/stop_and_reset"
        print("Stop and reset requested.")
        if self.serial_connection and self.serial_connection.is_open:
            try: self._send_arduino_command("STOP", wait_for_ack=False) # Try to stop motors quickly
            except: pass
        
        self.mode = "idle"
        self.status_message = "Operation stopped. Ready."
        self.error_message = None; self.solution = None
        self.calibration_step = 0; self.current_scan_idx = 0
        self.current_solve_move_index = 0; self.total_solve_moves = 0
        self.u_scans = [[] for _ in range(12)]; self.prev_face_colors = None
        self.prev_contour = None
        print("Game state reset to idle.")
        
    # capture_scan is not explicitly used by frontend, auto-scan is preferred
    # If manual scan is needed, it can be added similarly to capture_calibration_color

    def get_state(self, processed_frame_b64: Optional[str] = None) -> Dict[str, Any]:
        """Returns the current state, compatible with frontend."""
        return {
            "mode": self.mode,
            "status_message": self.status_message,
            "error_message": self.error_message,
            "solution": self.solution,
            "serial_connected": self.serial_connection is not None and self.serial_connection.is_open,
            "calibration_step": self.calibration_step if self.mode == "calibrating" else None,
            "current_color": self.COLOR_NAMES[self.calibration_step] if self.mode == "calibrating" and self.calibration_step < len(self.COLOR_NAMES) else None,
            "scan_index": self.current_scan_idx if self.mode == "scanning" else None,
            "solve_move_index": self.current_solve_move_index if self.mode in ["solving", "scrambling"] else 0,
            "total_solve_moves": self.total_solve_moves if self.mode in ["solving", "scrambling"] else 0,
            "processed_frame": processed_frame_b64 # This comes from process_frame()
        }

    def cleanup(self):
        print("Cleaning up RubiksCubeGame session...")
        if self.serial_connection and self.serial_connection.is_open:
            try:
                self._send_arduino_command("STOP", wait_for_ack=False) # Attempt to stop motors
                self.serial_connection.close()
                print("Serial connection closed.")
            except Exception as e:
                print(f"Exception during serial cleanup: {e}")
        self.serial_connection = None