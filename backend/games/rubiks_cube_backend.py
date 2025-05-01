# rubiks_cube_backend.py
import base64
import random
import threading
import time
from collections import Counter
from contextlib import asynccontextmanager  # For lifespan

import cv2
import kociemba
import numpy as np
import serial
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# --- Constants and Initial Configuration ---
WINDOW_SIZE = (640, 480)  # For resizing frames before processing
SCAN_COOLDOWN = 0.5
MOTOR_STABILIZATION_TIME = 0.7
STABILITY_THRESHOLD = 5
MIN_CONTOUR_AREA = 4000
MAX_CONTOUR_AREA = 60000
COLOR_RANGE_FILE = "color_ranges.json"
SERIAL_PORT = 'COM7'  # <<< ADJUST THIS TO YOUR ARDUINO PORT >>>
SERIAL_BAUDRATE = 9600
TEMP_SCAN_DIR = "cube_scans_backend"
MOVE_EXECUTION_DELAY = 0.05
MOVE_ACK_TIMEOUT = 5

# Rotation sequence for scanning
rotation_sequence = [
    "", "R L'", "B F'", "R L'", "B F'", "R L'", "B F'", "R L'", "B F'", "R L'", "B F'", "R L'"
]

# Default HSV color ranges
default_color_ranges = {
    "W": (np.array([0, 0, 140]), np.array([180, 70, 255])),
    "R": (np.array([0, 100, 80]), np.array([10, 255, 255])),
    "G": (np.array([35, 80, 80]), np.array([85, 255, 255])),
    "Y": (np.array([20, 100, 100]), np.array([35, 255, 255])),
    "O": (np.array([5, 100, 100]), np.array([20, 255, 255])),
    "B": (np.array([85, 80, 80]), np.array([130, 255, 255]))
}
COLOR_NAMES = ["W", "R", "G", "Y", "O", "B"]


# --- State Management ---
class SolverState:
    def __init__(self):
        self.mode = "idle"
        self.status_message = "Initializing..."
        self.color_ranges = default_color_ranges.copy()
        self.calibration_step = 0
        self.calibrated_ranges_temp = {}
        self.u_scans = [[] for _ in range(12)]
        self.current_scan_idx = 0
        self.last_scan_time = time.time()
        self.last_motor_move_time = 0
        self.stability_counter = 0
        self.prev_contour_details = None  # (x, y, contour)
        self.last_valid_grid = None  # (pad_x, pad_y, grid_size)
        self.last_processed_frame = None  # numpy array
        self.solution = None
        self.current_solve_move_index = 0
        self.total_solve_moves = 0
        self.error_message = None
        self.serial_connection = None
        self.serial_lock = threading.Lock()
        self.stop_requested = False


state = SolverState()


# --- Utility Functions ---
# === These were missing in the previous combined block ===

def save_color_ranges(color_ranges, filename=COLOR_RANGE_FILE):
    """Save color ranges to a JSON file."""
    serializable_ranges = {
        color: (lower.tolist(), upper.tolist()) for color, (lower, upper) in color_ranges.items()
    }
    try:
        with open(filename, 'w') as f:
            json.dump(serializable_ranges, f, indent=4)
        print(f"Color ranges saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving color ranges: {e}")
        return False


def load_color_ranges(filename=COLOR_RANGE_FILE):
    """Load color ranges from a JSON file."""
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                serializable_ranges = json.load(f)
            loaded_ranges = {
                color: (np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
                for color, (lower, upper) in serializable_ranges.items()
            }
            print(f"Color ranges loaded from {filename}")
            return loaded_ranges
        except Exception as e:
            print(f"Error loading color ranges: {e}")
            return None
    print(f"Color range file not found: {filename}")
    return None


def detect_color(roi, color_ranges):
    """Detect the dominant color in an ROI."""
    if roi is None or roi.shape[0] < 5 or roi.shape[1] < 5: return "?"  # Indicate error
    try:
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    except cv2.error as e:
        print(f"ROI shape for cvtColor: {roi.shape}, error: {e}")
        return "?"

    best_match = "?"
    max_match_percentage = -1

    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv_roi, lower, upper)
        mask_area = mask.shape[0] * mask.shape[1]
        if mask_area == 0: continue
        match_percentage = (cv2.countNonZero(mask) / mask_area) * 100

        if match_percentage > max_match_percentage:
            max_match_percentage = match_percentage
            best_match = color

    # Fallback strategy if match percentage is very low
    if max_match_percentage < 3:
        avg_hsv = np.mean(hsv_roi, axis=(0, 1))
        min_dist = float('inf')
        closest_color = "?"
        for color, (lower, upper) in color_ranges.items():
            center_hsv = (lower.astype(float) + upper.astype(float)) / 2.0
            h_diff = min(abs(avg_hsv[0] - center_hsv[0]), 180 - abs(avg_hsv[0] - center_hsv[0]))
            s_diff = abs(avg_hsv[1] - center_hsv[1])
            v_diff = abs(avg_hsv[2] - center_hsv[2])
            dist = (h_diff * 0.6 + s_diff * 0.3 + v_diff * 0.1) if color != "W" else (
                    h_diff * 0.1 + s_diff * 0.3 + v_diff * 0.6)
            if dist < min_dist:
                min_dist = dist
                closest_color = color
        return closest_color
    return best_match


def validate_cube(cube, order_name):
    """Validate the cube string structure and colors."""
    if not isinstance(cube, str) or len(cube) != 54:
        raise ValueError(f"{order_name} must be a string of 54 characters. Got: {len(cube)}")
    counts = Counter(cube)
    valid_colors = set(default_color_ranges.keys())
    if '?' in counts:
        raise ValueError(f"{order_name} contains undetermined colors ('?').")
    if len(counts) != 6:
        raise ValueError(f"{order_name} has {len(counts)} unique colors ({counts.keys()}), expected 6.")
    for color, count in counts.items():
        if color not in valid_colors:
            raise ValueError(f"{order_name} contains invalid color '{color}'.")
        if count != 9:
            raise ValueError(f"{order_name} invalid counts: {counts} (expected 9 of each).")
    # If all checks pass, validation successful
    print(f"{order_name} validation passed.")


def remap_colors_to_kociemba(cube_frblud):
    """Map physical colors to Kociemba face letters based on centers."""
    # Kociemba convention: URFDLB
    # Our convention: FRBLUD (indices 0-8 F, 9-17 R, 18-26 B, 27-35 L, 36-44 U, 45-53 D)
    # Centers: F=4, R=13, B=22, L=31, U=40, D=49
    try:
        validate_cube(cube_frblud, "Input FRBLUD for remap")  # Validate before accessing indices
    except ValueError as e:
        print(f"Validation failed during remap: {e}")
        raise

    centers = {
        'F': cube_frblud[4], 'R': cube_frblud[13], 'B': cube_frblud[22],
        'L': cube_frblud[31], 'U': cube_frblud[40], 'D': cube_frblud[49]
    }
    # Map the *physical* color on the center sticker to the Kociemba face name
    color_map = {
        centers['U']: 'U', centers['R']: 'R', centers['F']: 'F',
        centers['D']: 'D', centers['L']: 'L', centers['B']: 'B'
    }
    if len(color_map) != 6:
        # This indicates duplicate center colors, which shouldn't happen on a valid cube
        raise ValueError(f"Center color ambiguity or duplication detected. Centers found: {centers}")

    remapped_list = [color_map.get(c) for c in cube_frblud]
    if None in remapped_list:  # Check if any color wasn't in the map (shouldn't happen if validation passed)
        raise ValueError("Failed to map all colors using center map.")

    return color_map, "".join(remapped_list)


def remap_cube_to_kociemba(cube_frblud_remapped):
    """Rearrange FRBLUD string (with URFDLB letters) into Kociemba order URFDLB."""
    try:
        if len(cube_frblud_remapped) != 54:
            raise ValueError("Input string must be 54 characters for Kociemba remapping.")
        F = cube_frblud_remapped[0:9]
        R = cube_frblud_remapped[9:18]
        B = cube_frblud_remapped[18:27]
        L = cube_frblud_remapped[27:36]
        U = cube_frblud_remapped[36:45]
        D = cube_frblud_remapped[45:54]
        # Kociemba order: U, R, F, D, L, B
        kociemba_string = U + R + F + D + L + B
        return kociemba_string
    except IndexError:
        # This should not happen if length check passes, but good practice
        raise ValueError("Internal error during Kociemba string construction (IndexError).")


def get_solved_state(cube_frblud, color_map):
    """Get the solved state string in FRBLUD order (used by original code, maybe useful)."""
    centers = [cube_frblud[i] for i in [4, 13, 22, 31, 40, 49]]
    return ''.join(c * 9 for c in centers)


def is_cube_solved(cube_state):
    """Check if a cube state string represents a solved cube."""
    if not isinstance(cube_state, str) or len(cube_state) != 54:
        return False
    for i in range(0, 54, 9):  # Iterate through faces
        face = cube_state[i:i + 9]
        center_sticker = face[4]
        if not all(sticker == center_sticker for sticker in face):
            return False
    return True


def simplify_cube_moves(moves_str):
    """Simplify a sequence of cube moves."""
    moves = moves_str.strip().split()
    if not moves: return ""  # Handle empty input

    def move_value(move):
        if not move: return 0
        if move.endswith("2"): return 2
        if move.endswith("'"): return -1
        return 1

    def value_to_move(face, value):
        value %= 4
        if value == 0: return None
        if value == 1: return face
        if value == 2: return face + "2"
        if value == 3: return face + "'"

    face_groups = [['L', 'R'], ['F', 'B'], ['U', 'D']]

    # First pass: Combine consecutive same-face moves
    simplified, i = [], 0
    while i < len(moves):
        if not moves[i] or len(moves[i]) == 0: i += 1; continue
        current_face = moves[i][0]
        current_value = move_value(moves[i])
        j = i + 1
        while j < len(moves) and moves[j] and len(moves[j]) > 0 and moves[j][0] == current_face:
            current_value += move_value(moves[j])
            j += 1
        move = value_to_move(current_face, current_value)
        if move: simplified.append(move)
        i = j

    # Second pass: Combine opposite face groups (optional but can shorten)
    final_simplified, i = [], 0
    while i < len(simplified):
        if not simplified[i]: i += 1; continue
        current_face = simplified[i][0]
        face_group = None
        for group in face_groups:
            if current_face in group: face_group = group; break
        if face_group:
            counts = {face: 0 for face in face_group}
            j = i
            while j < len(simplified) and simplified[j] and simplified[j][0] in face_group:
                face = simplified[j][0];
                counts[face] += move_value(simplified[j]);
                j += 1
            for face in face_group:
                move = value_to_move(face, counts[face])
                if move: final_simplified.append(move)
            i = j
        else:  # Should not happen for standard moves
            final_simplified.append(simplified[i]);
            i += 1

    result = " ".join(final_simplified)
    return result if final_simplified else ""


def solve_cube_frblud(cube_frblud):
    """Solves the cube given FRBLUD string, returns optimized move sequence."""
    try:
        if is_cube_solved(cube_frblud):
            print("Cube is already solved!")
            return ""

        print("Validating input cube state...")
        validate_cube(cube_frblud, "Input FRBLUD for solving")

        print("Remapping colors to Kociemba standard...")
        color_map, cube_frblud_remapped = remap_colors_to_kociemba(cube_frblud)

        print("Rearranging to Kociemba face order...")
        scrambled_kociemba = remap_cube_to_kociemba(cube_frblud_remapped)
        print(f"Final Kociemba input string: {scrambled_kociemba}")
        validate_cube(scrambled_kociemba, "Remapped Kociemba String")  # Validate final string

        print("Calling Kociemba solver...")
        solution = kociemba.solve(scrambled_kociemba)
        print(f"Kociemba Raw Solution: {solution}")

        # U-move replacements
        u_replacement = "R L F2 B2 R' L' D R L F2 B2 R' L'"
        u_prime_replacement = "R L F2 B2 R' L' D' R L F2 B2 R' L'"
        u2_replacement = "R L F2 B2 R' L' D2 R L F2 B2 R' L'"

        moves = solution.split()
        modified_solution_list = []
        for move in moves:
            if move == "U":
                modified_solution_list.append(u_replacement)
            elif move == "U'":
                modified_solution_list.append(u_prime_replacement)
            elif move == "U2":
                modified_solution_list.append(u2_replacement)
            elif move in ["R", "L", "F", "B", "D", "R'", "L'", "F'", "B'", "D'", "R2", "L2", "F2", "B2",
                          "D2"]:  # Check valid moves
                modified_solution_list.append(move)
            elif move:  # Log unexpected move from kociemba
                print(f"Warning: Ignoring unexpected move from Kociemba: '{move}'")

        final_solution_with_u_expanded = " ".join(modified_solution_list)
        print(f"Solution with U-moves expanded: {final_solution_with_u_expanded}")

        print("Simplifying expanded solution...")
        optimized_solution = simplify_cube_moves(final_solution_with_u_expanded)
        print(
            f"Final Optimized Solution ({len(optimized_solution.split()) if optimized_solution else 0} moves): {optimized_solution}")
        return optimized_solution

    except ValueError as e:
        print(f"Error solving cube (ValueError): {str(e)}")
        raise  # Re-raise the specific error to be caught by caller
    except Exception as e:
        if "Error" in str(e):  # Catch Kociemba's specific errors
            print(f"Kociemba library error: {e}")
            raise ValueError(f"Kociemba solver failed: {e}")
        else:
            print(f"Unexpected error solving cube: {type(e).__name__}: {str(e)}")
            # Optional: include traceback here for debugging
            # import traceback; traceback.print_exc()
            raise ValueError(f"Unexpected Solver Error: {e}")


def print_full_cube_state(cube_state):
    """Prints a visual representation of the cube state (FRBLUD)."""
    print("\nFull cube state (FRBLUD):")
    print("".join(cube_state))
    # Add visual print if needed, or remove if just for debug
    # print("\nVisual representation:") ... etc ...


def construct_cube_from_u_scans(u_scans):
    """Construct the full FRBLUD cube state string from 12 U face scans."""
    if any(not scan or len(scan) != 9 for scan in u_scans):
        raise ValueError("Incomplete or invalid scan data length.")
    if any('?' in scan for scan in u_scans):
        raise ValueError("Scan data contains undetermined colors ('?').")

    cube_state = ['?'] * 54
    # --- Assign centers based on *expected* standard color scheme ---
    # W=U (idx 40), Y=D (idx 49), B=F (idx 4), G=B (idx 22), R=L (idx 31), O=R (idx 13)
    cube_state[40] = 'W';
    cube_state[49] = 'Y';
    cube_state[4] = 'B';
    cube_state[22] = 'G';
    cube_state[31] = 'R';
    cube_state[13] = 'O';
    print(
        f"Assigned centers: { {4: 'F', 13: 'R', 22: 'B', 31: 'L', 40: 'U', 49: 'D'} } -> { {4: cube_state[4], 13: cube_state[13], 22: cube_state[22], 31: cube_state[31], 40: cube_state[40], 49: cube_state[49]} }")

    # --- Fill scanned pieces based on the fixed mapping ---
    try:
        # Scan 0: U face directly
        cube_state[36:45] = u_scans[0]

        # Scan 1 (after R L') - Pieces moved to U face view
        cube_state[0] = u_scans[1][0];
        cube_state[2] = u_scans[1][2];
        cube_state[3] = u_scans[1][3]
        cube_state[5] = u_scans[1][5];
        cube_state[6] = u_scans[1][6];
        cube_state[8] = u_scans[1][8]

        # Scan 2 (after B F')
        cube_state[9] = u_scans[2][0];
        cube_state[10] = u_scans[2][1];
        cube_state[11] = u_scans[2][2]
        cube_state[15] = u_scans[2][6];
        cube_state[16] = u_scans[2][7];
        cube_state[17] = u_scans[2][8]

        # Scan 3 (after R L')
        cube_state[47] = u_scans[3][0];
        cube_state[53] = u_scans[3][2];
        cube_state[1] = u_scans[3][3]
        cube_state[7] = u_scans[3][5];
        cube_state[45] = u_scans[3][6];
        cube_state[51] = u_scans[3][8]

        # Scan 4 (after B F')
        cube_state[24] = u_scans[4][0];
        cube_state[12] = u_scans[4][1];
        cube_state[18] = u_scans[4][2]
        cube_state[26] = u_scans[4][6];
        cube_state[14] = u_scans[4][7];
        cube_state[20] = u_scans[4][8]

        # Scan 5 (after R L')
        cube_state[33] = u_scans[5][0];
        cube_state[27] = u_scans[5][2];
        cube_state[50] = u_scans[5][3]
        cube_state[48] = u_scans[5][5];
        cube_state[35] = u_scans[5][6];
        cube_state[29] = u_scans[5][8]

        # Scan 6 (after B F')
        # Note: Original code assigned u_scans[6][0] to cube_state[36], which was already assigned by scan 0.
        # This implies scan 6 might overwrite a U face piece. Let's assume original mapping is correct for now.
        cube_state[36] = u_scans[6][0];
        cube_state[46] = u_scans[6][1];
        cube_state[38] = u_scans[6][2]
        cube_state[42] = u_scans[6][6];
        cube_state[52] = u_scans[6][7];
        cube_state[44] = u_scans[6][8]
        # If cube_state[36] is U center (should be index 40), then this mapping needs review based on rotations.
        # Assuming index 36 is UBL corner's U-sticker after moves.

        # Scan 7 (after R L')
        cube_state[21] = u_scans[7][3];
        cube_state[23] = u_scans[7][5]

        # Scan 8 (after B F')
        cube_state[34] = u_scans[8][1];
        cube_state[28] = u_scans[8][7]

        # Scan 9 (after R L')
        cube_state[25] = u_scans[9][3];
        cube_state[19] = u_scans[9][5]

        # Scan 10 (after B F')
        cube_state[30] = u_scans[10][1];
        cube_state[32] = u_scans[10][7]

        # Scan 11 (after R L')
        cube_state[39] = u_scans[11][3];
        cube_state[41] = u_scans[11][5]

    except IndexError as e:
        # This error indicates a problem with the scan data array structure
        raise ValueError(f"Error accessing scan data during construction: {e}. Ensure all 12 scans have 9 colors.")

    final_state = "".join(cube_state)
    # Final check for completeness
    if '?' in final_state:
        missing_indices = [i for i, char in enumerate(final_state) if char == '?']
        print(f"Error: Cube state construction resulted in missing pieces at indices: {missing_indices}")
        print(f"Constructed state: {final_state}")
        raise ValueError(f"Cube construction failed, missing pieces at indices: {missing_indices}")

    print("Cube state constructed successfully.")
    # print_full_cube_state(final_state) # Optional debug print
    return final_state


def generate_scramble(moves=20):
    """Generate a random scramble sequence (avoids U moves)."""
    basic_moves = ['F', 'B', 'R', 'L', 'D']
    modifiers = ['', '\'', '2']
    scramble = []
    last_face = None
    for _ in range(moves):
        available_moves = [m for m in basic_moves if m != last_face]
        if not available_moves: available_moves = basic_moves  # Should not happen
        face = random.choice(available_moves)
        modifier = random.choice(modifiers)
        scramble.append(face + modifier)
        last_face = face
    return ' '.join(scramble)


# === End of Missing Utility Functions ===


# --- Serial Communication ---
def init_serial():
    # (Keep function from previous version)
    global state
    try:
        if state.serial_connection and state.serial_connection.is_open:
            state.serial_connection.close();
            print("Closed existing serial connection.")
        state.serial_connection = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=1)
        time.sleep(2)
        if state.serial_connection.in_waiting:
            initial_data = state.serial_connection.read_all().decode(errors='ignore')
            print(f"Discarded Arduino startup data: {initial_data.strip()}")
        print(f"Serial connection established on {SERIAL_PORT}")
        return True
    except serial.SerialException as e:
        state.mode = "error";
        state.error_message = f"Serial Error: {e}"
        state.status_message = f"Error connecting to Arduino on {SERIAL_PORT}."
        print(state.status_message);
        state.serial_connection = None
        return False
    except Exception as e:
        state.mode = "error";
        state.error_message = f"Serial Init Error: {e}"
        state.status_message = f"Unexpected error initializing serial port."
        print(state.status_message);
        state.serial_connection = None
        return False


def send_arduino_command(cmd, wait_for_ack=True, ack_timeout=MOVE_ACK_TIMEOUT):
    """Sends a command, optionally waits for ack."""
    # (Keep function from previous version)
    global state
    if state.stop_requested: print("Stop requested, skipping command:", cmd); return False
    if state.serial_connection is None or not state.serial_connection.is_open:
        print(f"Serial port not available for command: {cmd}")
        state.status_message = "Error: Serial port disconnected.";
        if state.mode != "error": state.mode = "error"; state.error_message = "Serial port disconnected."
        return False

    print(f"Sending to Arduino: {cmd}")
    try:
        with state.serial_lock:
            state.serial_connection.reset_input_buffer()
            state.serial_connection.write(f"{cmd}\n".encode('ascii'))
            state.serial_connection.flush()
            if not wait_for_ack: time.sleep(0.05); return True

            start_time = time.time();
            response_buffer = ""
            while time.time() - start_time < ack_timeout:
                if state.serial_connection.in_waiting:
                    try:
                        byte_data = state.serial_connection.readline()
                        line = byte_data.decode('ascii', errors='ignore').strip()
                        response_buffer += line + "\n"
                        if line:
                            print(f"Arduino response: {line}")
                            ack_words = ["completed", "done", "ready", "ok", cmd.lower()]
                            if any(kw in line.lower() for kw in ack_words):
                                print(f"Command '{cmd}' acknowledged.");
                                state.last_motor_move_time = time.time();
                                return True
                            error_words = ["error", "fail", "invalid"]
                            if any(kw in line.lower() for kw in error_words):
                                print(f"Arduino reported error for '{cmd}': {line}")
                                state.status_message = f"Error from Arduino: {line}"
                                state.mode = "error";
                                state.error_message = f"Arduino error: {line}";
                                return False
                    except serial.SerialException as serial_err:
                        print(f"Serial error during read: {serial_err}");
                        state.mode = "error";
                        state.error_message = f"Serial read error: {serial_err}"
                        state.serial_connection = None;
                        return False
                    except UnicodeDecodeError:
                        pass  # Ignore decode errors for now
                time.sleep(0.02)

            print(f"Timeout waiting for ack for command: {cmd}. Buffer: {response_buffer.strip()}");
            state.status_message = f"Timeout waiting for Arduino response to '{cmd}'"
            state.mode = "error";
            state.error_message = state.status_message;
            return False
    except serial.SerialException as e:
        print(f"Serial communication error during send: {e}");
        state.status_message = f"Serial Error: {e}";
        state.mode = "error"
        state.error_message = f"Serial communication error: {e}";
        state.serial_connection = None;
        return False
    except Exception as e:
        print(f"Error sending command '{cmd}': {e}");
        state.status_message = f"Error sending command: {e}"
        state.mode = "error";
        state.error_message = f"Error sending command: {e}";
        return False


# --- Background Task Execution ---
def execute_move_sequence(sequence: str, initial_status: str, mode_on_completion: str = "idle"):
    """ Executes a sequence of moves (solve/scramble) in a background thread. """
    # (Keep function from previous version)
    global state
    moves = sequence.strip().split()  # Ensure trimmed and split
    if not moves: print(
        "execute_move_sequence called with empty sequence."); state.status_message = "No moves to execute."; state.mode = mode_on_completion; return

    total_moves = len(moves);
    state.total_solve_moves = total_moves;
    state.current_solve_move_index = 0;
    state.stop_requested = False
    print(f"Starting execution of {total_moves} moves: {sequence}")

    for i, move in enumerate(moves):
        if state.stop_requested:
            state.status_message = f"Execution stopped by user at move {i + 1}/{total_moves}.";
            state.mode = "idle"
            print(state.status_message);
            state.solution = None;
            state.current_solve_move_index = 0;
            state.total_solve_moves = 0;
            return

        state.current_solve_move_index = i + 1;
        state.status_message = f"{initial_status} ({i + 1}/{total_moves}): Executing {move}"
        success = send_arduino_command(move, wait_for_ack=True)
        if not success:
            print(f"Execution failed at move {i + 1}: {move}");
            state.solution = None;
            return  # Error state set by send_arduino_command
        time.sleep(MOVE_EXECUTION_DELAY)

    state.status_message = f"{initial_status.split(':')[0]} completed ({total_moves} moves).";
    state.mode = mode_on_completion
    state.solution = None;
    state.current_solve_move_index = 0;
    state.total_solve_moves = 0;
    print("Move sequence execution finished.")


# --- FastAPI Setup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # (Keep lifespan manager from previous version)
    global state
    print("FastAPI starting up...")
    loaded_ranges = load_color_ranges()
    if loaded_ranges:
        state.color_ranges.update(
            loaded_ranges);
        state.status_message = "Color ranges loaded. Initializing Serial...";
        print(
            "Using previously calibrated color ranges.")
    else:
        state.status_message = "Default colors loaded. Calibration needed. Initializing Serial...";
        print(
            "No saved color ranges found. Using defaults.")
    if not init_serial():
        print("Serial initialization failed. Hardware control disabled.")
    else:
        state.status_message = "Serial OK. Idle."
    if not os.path.exists(TEMP_SCAN_DIR):
        try:
            os.makedirs(TEMP_SCAN_DIR)
        except OSError as e:
            print(f"Error creating temp scan directory '{TEMP_SCAN_DIR}': {e}")
    print("Startup complete.")
    yield
    print("FastAPI shutting down...");
    state.stop_requested = True;
    await asyncio.sleep(0.1)
    if state.serial_connection and state.serial_connection.is_open: state.serial_connection.close(); print(
        "Serial port closed.")
    print("Shutdown complete.")


app = FastAPI(lifespan=lifespan)

origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])


# --- WebSocket Endpoint ---
@app.websocket("/ws/rubiks")
async def websocket_endpoint(websocket: WebSocket):
    # (Keep WebSocket endpoint function from previous version - it uses the helper functions)
    await websocket.accept()
    global state
    client_ip = websocket.client.host if websocket.client else "Unknown"
    print(f"WebSocket connection established from {client_ip}.")
    await send_status_update(websocket)
    try:
        while True:
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None: continue
            try:
                frame = cv2.resize(frame, WINDOW_SIZE, interpolation=cv2.INTER_AREA)
            except cv2.error as e:
                print(f"Error resizing frame: {e}");
                continue
            processed_frame = frame.copy()
            current_mode = state.mode;
            current_time = time.time()
            if current_mode == "calibrating":
                processed_frame = process_frame_for_calibration(frame, processed_frame)
            elif current_mode == "scanning":
                processed_frame = await process_frame_for_scanning(frame, processed_frame, current_time, websocket)
            elif current_mode == "solving" or current_mode == "scrambling":
                processed_frame, _, _, _ = draw_detection_overlay(frame, processed_frame)
                progress_text = f"{current_mode.capitalize()} ({state.current_solve_move_index}/{state.total_solve_moves})"
                cv2.putText(processed_frame, progress_text, (11, WINDOW_SIZE[1] - 19), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 0), 2)
                cv2.putText(processed_frame, progress_text, (10, WINDOW_SIZE[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 1)
            elif current_mode == "idle" or current_mode == "error":
                processed_frame, _, _, _ = draw_detection_overlay(frame, processed_frame)

            state.last_processed_frame = processed_frame.copy()
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
            _, buffer = cv2.imencode('.jpg', processed_frame, encode_param)
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            await send_status_update(websocket, frame_b64)
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        print(f"WebSocket connection closed by client {client_ip}.")
    except Exception as e:
        print(f"WebSocket Error ({client_ip}): {type(e).__name__}: {e}")
        if state.mode != "error": state.mode = "error"; state.error_message = f"WebSocket error: {e}"; state.status_message = "Internal server error."
        try:
            await send_status_update(websocket)
        except Exception:
            pass


# --- Helper functions for WebSocket frame processing ---
async def send_status_update(websocket: WebSocket, frame_b64=None):
    """Sends the current state to the client via WebSocket."""
    # (Keep function from previous version)
    global state
    data_to_send = {
        "mode": state.mode, "status_message": state.status_message,
        "calibration_step": state.calibration_step if state.mode == "calibrating" else None,
        "current_color": COLOR_NAMES[
            state.calibration_step] if state.mode == "calibrating" and state.calibration_step < len(
            COLOR_NAMES) else None,
        "scan_index": state.current_scan_idx if state.mode == "scanning" else None,
        "solve_move_index": state.current_solve_move_index if state.mode in ["solving", "scrambling"] else 0,
        "total_solve_moves": state.total_solve_moves if state.mode in ["solving", "scrambling"] else 0,
        "solution": state.solution if state.solution else None,
        "error_message": state.error_message if state.mode == "error" else None,
        "serial_connected": state.serial_connection is not None and state.serial_connection.is_open
    }
    if frame_b64: data_to_send["processed_frame"] = frame_b64
    try:
        await websocket.send_json(data_to_send)
    except Exception as e:
        if not isinstance(e, WebSocketDisconnect): print(f"Failed to send status update: {e}")


def draw_detection_overlay(frame, display_frame):
    """Finds the cube face and draws overlay. Returns display_frame, detected, contour, pos."""
    # (Keep function from previous version)
    global state
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    combined_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    for color, (lower, upper) in state.color_ranges.items():
        combined_mask |= cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_contour, best_score, cube_detected, best_pos = None, float('inf'), False, None
    for contour in contours:
        area = cv2.contourArea(contour)
        if MIN_CONTOUR_AREA < area < MAX_CONTOUR_AREA:
            peri = cv2.arcLength(contour, True);
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(approx);
                aspect_ratio = float(w) / h if h > 0 else 0
                squareness_score = abs(1 - aspect_ratio)
                if 0.7 < aspect_ratio < 1.3 and squareness_score < best_score:
                    best_score = squareness_score;
                    best_contour = contour;
                    best_pos = (x, y);
                    cube_detected = True
    if cube_detected and best_contour is not None and best_pos is not None:
        x, y, w, h = cv2.boundingRect(best_contour);
        grid_size = min(w, h)
        center_x, center_y = x + w // 2, y + h // 2;
        pad_x, pad_y = center_x - grid_size // 2, center_y - grid_size // 2
        state.last_valid_grid = (pad_x, pad_y, grid_size)
        cv2.drawContours(display_frame, [best_contour], -1, (0, 255, 0), 2)
        cv2.rectangle(display_frame, (pad_x, pad_y), (pad_x + grid_size, pad_y + grid_size), (0, 255, 0), 2)
        grid_cell_size = grid_size // 3
        for i in range(1, 3):
            cv2.line(display_frame, (pad_x + i * grid_cell_size, pad_y),
                     (pad_x + i * grid_cell_size, pad_y + grid_size), (0, 255, 0), 1)
            cv2.line(display_frame, (pad_x, pad_y + i * grid_cell_size),
                     (pad_x + grid_size, pad_y + i * grid_cell_size), (0, 255, 0), 1)
        return display_frame, True, best_contour, best_pos
    state.last_valid_grid = None;
    return display_frame, False, None, None


def process_frame_for_calibration(frame, display_frame):
    """Draws calibration target on detected top-left grid cell."""
    # (Keep function from previous version)
    global state
    display_frame, cube_detected, _, _ = draw_detection_overlay(frame, display_frame)
    if state.calibration_step >= len(
            COLOR_NAMES): state.status_message = "Calibration complete. Press 'Save Calibration'."; return display_frame
    current_color_name = COLOR_NAMES[state.calibration_step]
    if cube_detected and state.last_valid_grid:
        pad_x, pad_y, grid_size = state.last_valid_grid;
        grid_cell_size = grid_size // 3
        target_x1, target_y1 = pad_x, pad_y;
        target_x2, target_y2 = pad_x + grid_cell_size, pad_y + grid_cell_size
        cv2.rectangle(display_frame, (target_x1, target_y1), (target_x2, target_y2), (255, 255, 0), 3)  # Cyan box
        state.status_message = f"Aim '{current_color_name}' at TOP-LEFT cell. Press 'Capture Color'."
        cv2.putText(display_frame, f"Sample {current_color_name}", (target_x1, target_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 0), 1)
    else:
        state.status_message = f"Show '{current_color_name}'. Position cube clearly."
    return display_frame


async def process_frame_for_scanning(frame, display_frame, current_time, websocket):
    """Handles automatic scanning, rotation triggers, and initiates solve."""
    # (Keep function from previous version - it contains the main scan loop logic)
    global state
    display_frame, cube_detected, contour, pos = draw_detection_overlay(frame, display_frame)
    time_since_last_scan = current_time - state.last_scan_time;
    time_since_last_move = current_time - state.last_motor_move_time
    is_stable = False
    if cube_detected and contour is not None and pos is not None:
        x, y = pos
        if state.prev_contour_details is not None:
            prev_x, prev_y, _ = state.prev_contour_details
            position_diff = abs(x - prev_x) + abs(y - prev_y)
            if position_diff < 15:
                state.stability_counter += 1
            else:
                state.stability_counter = 0
        else:
            state.stability_counter = 0
        state.prev_contour_details = (x, y, contour)
        if state.stability_counter >= STABILITY_THRESHOLD: is_stable = True
    else:
        state.stability_counter = 0;
        state.prev_contour_details = None

    if state.current_scan_idx >= 12: state.status_message = "All scans done. Processing..."; return display_frame
    scan_msg = f"Scan {state.current_scan_idx + 1}/12"
    if not cube_detected:
        state.status_message = f"{scan_msg}: Position cube in view."
    elif time_since_last_move < MOTOR_STABILIZATION_TIME:
        state.status_message = f"{scan_msg}: Motor stabilizing ({(MOTOR_STABILIZATION_TIME - time_since_last_move):.1f}s)"
    elif not is_stable:
        state.status_message = f"{scan_msg}: Hold steady ({state.stability_counter}/{STABILITY_THRESHOLD})"
    elif time_since_last_scan < SCAN_COOLDOWN:
        state.status_message = f"{scan_msg}: Scan cooldown ({(SCAN_COOLDOWN - time_since_last_scan):.1f}s)"
    else:
        state.status_message = f"{scan_msg}: Stable. Ready."

    if (
            cube_detected and is_stable and state.last_valid_grid and time_since_last_scan >= SCAN_COOLDOWN and time_since_last_move >= MOTOR_STABILIZATION_TIME):
        print(f"Attempting capture for scan {state.current_scan_idx + 1}")
        pad_x, pad_y, grid_size = state.last_valid_grid;
        grid_cell_size = grid_size // 3;
        face_colors = [];
        capture_success = True
        for i in range(3):
            for j in range(3):
                y_start, y_end = pad_y + i * grid_cell_size, pad_y + (i + 1) * grid_cell_size
                x_start, x_end = pad_x + j * grid_cell_size, pad_x + (j + 1) * grid_cell_size
                padding = grid_cell_size // 6
                roi_y_start = max(0, y_start + padding);
                roi_y_end = min(frame.shape[0], y_end - padding)
                roi_x_start = max(0, x_start + padding);
                roi_x_end = min(frame.shape[1], x_end - padding)
                if roi_y_start >= roi_y_end or roi_x_start >= roi_x_end:
                    roi = None;
                    color = '?'  # Handle invalid ROI
                else:
                    roi = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end];
                    color = detect_color(roi,
                                         state.color_ranges)

                if color == '?': print(f"Warning: Empty or invalid ROI at cell ({i},{j})"); capture_success = False
                face_colors.append(color)
                # Draw detected color during scan
                cv2.putText(display_frame, color, (x_start + grid_cell_size // 3, y_start + grid_cell_size // 2 + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(display_frame, color, (x_start + grid_cell_size // 3, y_start + grid_cell_size // 2 + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if not capture_success:
            state.status_message = f"Scan {state.current_scan_idx + 1} failed (ROI error). Retry positioning.";
            print(f"Scan {state.current_scan_idx + 1} failed. Colors: {face_colors}")
            state.last_scan_time = current_time;
            state.stability_counter = 0;
            await send_status_update(websocket);
            return display_frame

        print(f"Scan {state.current_scan_idx + 1} detected colors: {face_colors}");
        state.u_scans[state.current_scan_idx] = face_colors
        scan_img_path = os.path.join(TEMP_SCAN_DIR, f"scan_{state.current_scan_idx + 1}.jpg");
        cv2.imwrite(scan_img_path, display_frame)
        current_scan_index_captured = state.current_scan_idx;
        state.current_scan_idx += 1;
        state.last_scan_time = current_time;
        state.stability_counter = 0

        if state.current_scan_idx < 12:
            next_move = rotation_sequence[state.current_scan_idx]
            state.status_message = f"Scan {current_scan_index_captured + 1} captured. Rotating: {next_move}";
            await send_status_update(websocket)
            success = send_arduino_command(next_move, wait_for_ack=True, ack_timeout=10)
            if not success: state.status_message = f"Error sending move '{next_move}' to Arduino."; state.mode = "error"; state.error_message = state.status_message; await send_status_update(
                websocket); return display_frame
        else:  # All scans done
            state.status_message = "All scans complete. Constructing & Solving...";
            await send_status_update(websocket)
            try:
                final_orient_move = "B F'";
                print(f"Performing final orientation move: {final_orient_move}")
                send_arduino_command(final_orient_move, wait_for_ack=True, ack_timeout=10);
                await asyncio.sleep(MOTOR_STABILIZATION_TIME)
                cube_state_str = construct_cube_from_u_scans(state.u_scans);
                state.status_message = "Cube state constructed. Generating solution...";
                await send_status_update(websocket)
                solution_str = solve_cube_frblud(cube_state_str)
                if solution_str == "":
                    state.status_message = "Cube is already solved!";
                    state.mode = "idle";
                    state.solution = "Already Solved"
                elif solution_str:
                    state.solution = solution_str;
                    state.mode = "solving";
                    state.status_message = f"Solution found ({len(solution_str.split())} moves). Starting execution..."
                    print(f"Starting execution thread for: {solution_str}")
                    thread = threading.Thread(target=execute_move_sequence, args=(solution_str, "Solving:", "idle"),
                                              daemon=True);
                    thread.start()
                else:
                    state.status_message = "Failed to generate a solution.";
                    state.mode = "error";
                    state.error_message = state.status_message
            except ValueError as e:
                print(
                    f"Error during solve process initiation: {e}");
                state.status_message = f"Error: {e}";
                state.mode = "error";
                state.error_message = str(
                    e);
                state.solution = None
            except Exception as e:
                print(
                    f"Unexpected error during solve initiation: {e}");
                state.status_message = "Unexpected error during solving setup.";
                state.mode = "error";
                state.error_message = str(
                    e);
                state.solution = None
            await send_status_update(websocket)
    return display_frame


# --- HTTP Endpoints ---


# --- Main execution using uvicorn command line ---
# Example command: uvicorn rubiks_cube_backend:app --host 0.0.0.0 --port 8000 --reload
# main.py (or rubiks_cube_backend.py)
# Requires FastAPI, HTTPException, asyncio, threading, os, json, numpy, cv2 etc. imported
# Requires the 'state' object (instance of SolverState) to be defined globally
# Requires helper functions like init_serial, send_arduino_command, load/save_color_ranges,
# COLOR_NAMES, default_color_ranges, solve_cube_frblud, generate_scramble,
# execute_move_sequence, etc., to be defined elsewhere in the file.

from fastapi import HTTPException
import asyncio
import json
import numpy as np
import os


# Assuming other necessary imports and the 'state' object exist

# --- HTTP Endpoints ---

@app.post("/start_calibration")
async def start_calibration():
    """Initiates the color calibration process."""
    global state
    # Check if the system is in a state where calibration can start
    if state.mode not in ["idle", "error"]:
        raise HTTPException(status_code=409, detail=f"System busy ({state.mode}). Cannot start calibration.")

    # Optional: Warn if serial isn't connected, but allow calibration anyway
    if state.serial_connection is None:
        print("Warning: Starting calibration without serial connection.")

    # Signal any running background task (like solving) to stop
    state.stop_requested = True
    await asyncio.sleep(0.1)  # Give thread a moment to potentially stop

    # Reset calibration state
    state.mode = "calibrating"
    state.calibration_step = 0
    state.calibrated_ranges_temp = {}  # Clear previous temporary calibration data
    state.status_message = f"Calibration started. Show '{COLOR_NAMES[0]}' face, aim at highlighted TOP-LEFT cell."
    state.error_message = None
    state.solution = None  # Clear any previous solution
    print("Calibration started via API.")
    # The WebSocket will send the updated state ('calibrating') to the frontend
    return {"message": "Calibration started."}


@app.post("/capture_calibration_color")
async def capture_calibration_color():
    """Captures the color from the detected top-left grid cell for the current calibration step."""
    global state
    if state.mode != "calibrating":
        raise HTTPException(status_code=400, detail="Not in calibration mode.")
    if state.calibration_step >= len(COLOR_NAMES):
        raise HTTPException(status_code=400, detail="Calibration already completed for all colors.")

    # Use the last frame processed by the websocket loop for consistency
    frame_to_process = state.last_processed_frame
    if frame_to_process is None:
        raise HTTPException(status_code=400,
                            detail="No frame available for capture (backend hasn't processed one recently).")

    # Check if a valid grid was detected in that frame
    if state.last_valid_grid is None:
        raise HTTPException(status_code=400,
                            detail="Cube face not detected clearly. Position cube and ensure detection before capturing.")

    current_color_name = COLOR_NAMES[state.calibration_step]

    # Calculate ROI for the top-left cell using the stored grid info
    try:
        pad_x, pad_y, grid_size = state.last_valid_grid
        grid_cell_size = grid_size // 3
        y_start, y_end = pad_y, pad_y + grid_cell_size
        x_start, x_end = pad_x, pad_x + grid_cell_size

        # Add padding within the cell for sampling
        padding = grid_cell_size // 8  # Use smaller padding for calibration sample
        roi_y_start = max(0, y_start + padding)
        roi_y_end = min(frame_to_process.shape[0], y_end - padding)
        roi_x_start = max(0, x_start + padding)
        roi_x_end = min(frame_to_process.shape[1], x_end - padding)

        if roi_y_start >= roi_y_end or roi_x_start >= roi_x_end:
            raise ValueError("Invalid ROI calculated.")  # Internal error if calculation fails

        roi = frame_to_process[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

        if roi.size == 0:
            raise ValueError("Empty ROI captured.")

    except (TypeError, ValueError, IndexError) as e:  # Catch issues with grid data or ROI extraction
        print(f"Error calculating/extracting ROI for calibration: {e}")
        raise HTTPException(status_code=400, detail=f"Could not capture ROI from top-left cell: {e}")

    # --- Continue with HSV analysis ---
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    avg_hsv = np.mean(hsv_roi, axis=(0, 1))
    h_range = 10 if current_color_name != "W" else 30
    s_range = 70;
    v_range = 70
    lower = np.array([max(0, avg_hsv[0] - h_range), max(30, avg_hsv[1] - s_range), max(30, avg_hsv[2] - v_range)])
    upper = np.array([min(179, avg_hsv[0] + h_range), min(255, avg_hsv[1] + s_range), min(255, avg_hsv[2] + v_range)])

    # Basic Hue wrap handling (can be improved)
    h_mean = avg_hsv[0]
    if h_mean < h_range:
        lower[0] = 0
    elif h_mean > (180 - h_range):
        upper[0] = 179
    else:
        lower[0] = max(0, h_mean - h_range);
        upper[0] = min(179, h_mean + h_range)

    state.calibrated_ranges_temp[current_color_name] = (lower.astype(np.uint8), upper.astype(np.uint8))
    print(
        f"Calibrated {current_color_name} from TopLeft: Avg HSV={avg_hsv}, Range={lower.tolist()} to {upper.tolist()}")

    # --- Update state ---
    state.calibration_step += 1
    if state.calibration_step < len(COLOR_NAMES):
        state.status_message = f"'{current_color_name}' captured. Show '{COLOR_NAMES[state.calibration_step]}' at TOP-LEFT cell."
    else:
        state.status_message = "All colors captured. Press 'Save Calibration'."

    # The WebSocket will send the updated state ('calibrating', new step, new message)
    return {"message": f"Color {current_color_name} captured.",
            "next_color": COLOR_NAMES[state.calibration_step] if state.calibration_step < len(COLOR_NAMES) else None}


@app.post("/save_calibration")
async def save_calibration():
    """Saves the temporarily calibrated colors and returns to idle mode."""
    global state
    if state.mode != "calibrating":
        raise HTTPException(status_code=400, detail="Not in calibration mode.")
    if state.calibration_step < len(COLOR_NAMES):
        raise HTTPException(status_code=400, detail="Calibration not complete. Capture all colors first.")

    # Update the main color ranges with the temporary ones
    state.color_ranges.update(state.calibrated_ranges_temp)

    if save_color_ranges(state.color_ranges):  # Attempt to save to file
        state.status_message = "Calibration saved successfully. Idle."
        state.mode = "idle"
        state.calibrated_ranges_temp = {}  # Clear temp data
        state.calibration_step = 0  # Reset step
        print("Calibration saved via API.")
        # WebSocket sends the new 'idle' state
        return {"message": "Calibration saved successfully."}
    else:
        # Keep state as 'calibrating' or move to 'error'? Let's move to error.
        state.status_message = "Error saving calibration data to file."
        state.mode = "error"
        state.error_message = "Failed to write calibration file. Check permissions."
        # WebSocket sends the new 'error' state
        raise HTTPException(status_code=500, detail="Failed to save calibration file.")


@app.post("/reset_calibration")
async def reset_calibration():
    """Resets color calibration to defaults and returns to idle mode."""
    global state
    print("Reset calibration requested via API.")
    # Stop any potentially running task
    state.stop_requested = True
    await asyncio.sleep(0.1)

    # Reset state variables
    state.color_ranges = default_color_ranges.copy()  # Load defaults
    state.calibrated_ranges_temp = {}
    state.calibration_step = 0
    state.mode = "idle"  # Go back to idle
    state.status_message = "Calibration reset to defaults. Idle."
    state.error_message = None
    state.solution = None

    # Optionally delete the saved file
    if os.path.exists(COLOR_RANGE_FILE):
        try:
            os.remove(COLOR_RANGE_FILE)
            print(f"Deleted saved color range file: {COLOR_RANGE_FILE}")
        except OSError as e:
            print(f"Warning: Error deleting color range file {COLOR_RANGE_FILE}: {e}")

    # WebSocket sends the new 'idle' state
    return {"message": "Calibration reset to defaults."}


@app.post("/start_solve")
async def start_solve():
    """Initiates the cube solving process by starting the scan sequence."""
    global state
    if state.serial_connection is None:
        raise HTTPException(status_code=503, detail="Cannot start solve: Arduino not connected.")
    if state.mode not in ["idle", "error"]:
        raise HTTPException(status_code=409, detail=f"System busy ({state.mode}). Cannot start solve.")

    # Stop previous task
    state.stop_requested = True
    await asyncio.sleep(0.1)

    # Reset all relevant scanning/solving state variables
    state.mode = "scanning"
    state.u_scans = [[] for _ in range(12)]
    state.current_scan_idx = 0
    state.last_scan_time = time.time()
    state.last_motor_move_time = time.time()  # Assume motors are settled initially
    state.stability_counter = 0
    state.prev_contour_details = None
    state.last_valid_grid = None
    state.solution = None
    state.error_message = None
    state.current_solve_move_index = 0
    state.total_solve_moves = 0
    state.status_message = "Starting scan 1/12. Position cube in view."
    print("Solve process initiated (scanning) via API.")
    # WebSocket sends the new 'scanning' state
    return {"message": "Scanning process initiated."}


@app.post("/stop_and_reset")
async def stop_and_reset():
    """ Stops any ongoing scanning, solving, or scrambling and resets the system to idle mode. """
    global state
    print("Stop and reset requested via API.")
    state.stop_requested = True  # Signal background threads to stop

    await asyncio.sleep(0.2)  # Allow thread a moment to react

    # Force state reset to idle
    current_mode_before_reset = state.mode  # For logging
    state.mode = "idle"
    state.status_message = f"Operation ({current_mode_before_reset}) stopped. Reset to Idle."
    # Clear all process-related state
    state.u_scans = [[] for _ in range(12)]
    state.current_scan_idx = 0
    state.stability_counter = 0
    state.prev_contour_details = None
    state.solution = None
    state.error_message = None
    state.current_solve_move_index = 0
    state.total_solve_moves = 0
    print("System reset to idle state.")

    # Optionally send a STOP command to Arduino (non-blocking)
    if state.serial_connection and state.serial_connection.is_open:
        print("Sending STOP command to Arduino.")
        # Use a thread to send non-blocking or make send_arduino_command async
        threading.Thread(target=send_arduino_command, args=("STOP", False),
                         daemon=True).start()  # Send 'STOP', don't wait for ack

    # WebSocket will send the new 'idle' state
    return {"message": "Operation stopped and system reset."}


@app.post("/start_scramble")
async def start_scramble():
    """Generates a scramble sequence and starts executing it on the Arduino."""
    global state
    if state.serial_connection is None:
        raise HTTPException(status_code=503, detail="Cannot start scramble: Arduino not connected.")
    if state.mode not in ["idle", "error"]:
        raise HTTPException(status_code=409, detail=f"System busy ({state.mode}). Cannot start scramble.")

    # Stop previous task
    state.stop_requested = True
    await asyncio.sleep(0.1)

    # Set up for scrambling
    state.mode = "scrambling"
    state.error_message = None
    state.solution = None  # Scramble sequence isn't stored as 'solution'
    state.current_solve_move_index = 0
    state.total_solve_moves = 0

    scramble_sequence = generate_scramble(moves=random.randint(18, 22))  # Random length
    num_moves = len(scramble_sequence.split())
    state.status_message = f"Generated Scramble ({num_moves} moves). Executing..."
    print(f"Starting scramble thread for: {scramble_sequence}")

    # Start execution in background thread
    # The 'execute_move_sequence' function updates state progress vars
    thread = threading.Thread(target=execute_move_sequence,
                              args=(scramble_sequence, "Scrambling:", "idle"),  # Go idle after scramble
                              daemon=True)  # Allow app exit even if thread stuck
    thread.start()

    # Return immediately, WebSocket updates status
    return {"message": f"Scramble initiated ({num_moves} moves)."}


# --- Optional GET Endpoints for Status/Info ---

@app.get("/status")
async def get_status():
    """Returns a snapshot of the current system state (less real-time than WebSocket)."""
    global state
    return {
        "mode": state.mode,
        "status_message": state.status_message,
        "calibration_step": state.calibration_step if state.mode == "calibrating" else None,
        "current_color": COLOR_NAMES[
            state.calibration_step] if state.mode == "calibrating" and state.calibration_step < len(
            COLOR_NAMES) else None,
        "scan_index": state.current_scan_idx if state.mode == "scanning" else None,
        "solve_move_index": state.current_solve_move_index,
        "total_solve_moves": state.total_solve_moves,
        "solution": state.solution,  # May contain the solution string if just generated
        "error_message": state.error_message,
        "serial_connected": state.serial_connection is not None and state.serial_connection.is_open
    }


@app.get("/color_ranges")
async def get_color_ranges_http():
    """Returns the currently active color ranges."""
    global state
    # Convert numpy arrays for JSON serialization
    serializable_ranges = {
        color: (lower.tolist(), upper.tolist())
        for color, (lower, upper) in state.color_ranges.items()
    }
    return serializable_ranges


if __name__ == "__main__":
    uvicorn.run("rubiks_cube_backend:app", host="0.0.0.0", port=8000, reload=True)
