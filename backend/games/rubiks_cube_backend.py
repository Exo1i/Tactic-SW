# import argparse # Not needed for FastAPI version
import asyncio
import base64
import json
import logging
import os
import random
import time
from collections import Counter
from contextlib import asynccontextmanager
from enum import Enum

import cv2
import kociemba
import numpy as np
import serial
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Logging Setup ---
# Increased logging level for debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants and Configuration ---
# WINDOW_SIZE = (640, 480) # Not used directly for frame capture, frontend sends size
SCAN_COOLDOWN = 0.5
MOTOR_STABILIZATION_TIME = 0.5
STABILITY_THRESHOLD = 3
MIN_CONTOUR_AREA = 500  # Adjust based on camera distance/resolution
MAX_CONTOUR_AREA = 60000  # Adjust based on camera distance/resolution
COLOR_NAMES = ["W", "R", "G", "Y", "O", "B"]
COLOR_RANGES_FILE = "color_ranges.json"
SERIAL_PORT = '/dev/ttyACM0'  # Adjusted for typical Linux Arduino port, change if needed
SERIAL_BAUDRATE = 9600
SERIAL_TIMEOUT = 1  # For initial connection read/write
SERIAL_ACK_TIMEOUT = 10.0  # Timeout for waiting for command acknowledgment
TEMP_SCAN_DIR = "cube_scans"

# Rotation sequence for scanning (ensure this matches Arduino logic if applicable)
rotation_sequence = [
    "",  # Scan 1: Initial U face
    "R L'",  # Scan 2 -> U becomes F
    "B F'",  # Scan 3 -> U becomes R
    "R L'",  # Scan 4 -> U becomes B  <- Check logic vs Original, order might differ
    "B F'",  # Scan 5 -> U becomes L
    "R L'",  # Scan 6 -> U becomes D
    # Sequence needs verification against physical setup AND construct_cube_from_u_scans
    # The original script's sequence/construction logic might be flawed.
    # A simpler, more robust method involves scanning each of the 6 faces directly
    # by rotating the cube to present F, R, B, L, U, D to the camera.
    # Using only U scans requires precise knowledge of intermediate orientations.
    # Placeholder - needs verification:
    "B F'",  # Scan 7
    "R L'",  # Scan 8
    "B F'",  # Scan 9
    "R L'",  # Scan 10
    "B F'",  # Scan 11
    "R L'"  # Scan 12
]

# --- Global State (Shared across requests/connections - use Locks if modifying concurrently) ---
ser: serial.Serial | None = None
color_ranges = {}  # Loaded at startup/calibration
active_connection: WebSocket | None = None
solver_task: asyncio.Task | None = None  # To manage background tasks like solve/scramble
app_state = {}  # Populated in lifespan manager


# --- State Management Enums and Models ---
class SolverMode(str, Enum):
    IDLE = "idle"
    CONNECTING = "connecting"
    CALIBRATING = "calibrating"
    SCANNING = "scanning"
    SOLVING = "solving"
    SCRAMBLING = "scrambling"
    ERROR = "error"
    STOPPING = "stopping"


class SolverStatus(BaseModel):
    mode: SolverMode = SolverMode.CONNECTING
    status_message: str = "Initializing..."
    error_message: str | None = None
    solution: str | None = None
    serial_connected: bool = False
    processed_frame: str | None = None  # Base64 encoded JPEG
    calibration_step: int | None = None
    current_color: str | None = None
    scan_index: int | None = None
    solve_move_index: int = 0
    total_solve_moves: int = 0


# --- Helper Functions (Color Ranges, Cube Logic - Adapted/Verified) ---

def save_color_ranges(ranges, filename=COLOR_RANGES_FILE):
    """Save color ranges to a JSON file."""
    serializable_ranges = {}
    try:
        for color, (lower, upper) in ranges.items():
            lower_list = lower.tolist() if isinstance(lower, np.ndarray) else lower
            upper_list = upper.tolist() if isinstance(upper, np.ndarray) else upper
            serializable_ranges[color] = (lower_list, upper_list)

        with open(filename, 'w') as f:
            json.dump(serializable_ranges, f, indent=4)
        logger.info(f"Color ranges saved to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving color ranges: {e}")
        return False


def load_color_ranges(filename=COLOR_RANGES_FILE):
    """Load color ranges from a JSON file or return defaults."""
    default_ranges = {
        "W": ([0, 0, 150], [180, 60, 255]),  # White - High Value, Low Saturation
        "R": ([0, 100, 100], [10, 255, 255]),  # Red (low Hue range) - May need second range near 170-180
        "G": ([35, 80, 80], [85, 255, 255]),  # Green
        "Y": ([20, 100, 100], [35, 255, 255]),  # Yellow
        "O": ([5, 100, 100], [20, 255, 255]),  # Orange
        "B": ([85, 80, 80], [130, 255, 255])  # Blue
    }
    loaded_successfully = False
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                serializable_ranges = json.load(f)
            loaded_ranges = {
                color: (np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
                for color, (lower, upper) in serializable_ranges.items()
                if len(lower) == 3 and len(upper) == 3  # Basic validation
            }
            # Ensure all required colors are present, add defaults if missing
            final_ranges = {}
            for color in COLOR_NAMES:
                if color in loaded_ranges:
                    final_ranges[color] = loaded_ranges[color]
                else:
                    logger.warning(f"Color '{color}' not found in {filename} or invalid format, using default.")
                    final_ranges[color] = (np.array(default_ranges[color][0], dtype=np.uint8),
                                           np.array(default_ranges[color][1], dtype=np.uint8))

            logger.info(f"Color ranges loaded from {filename}")
            return final_ranges

        except Exception as e:
            logger.error(f"Error loading or parsing {filename}: {e}. Using defaults.")
            # Fallback to defaults on error
            return {k: (np.array(v[0], dtype=np.uint8), np.array(v[1], dtype=np.uint8)) for k, v in
                    default_ranges.items()}
    else:
        logger.warning(f"{filename} not found. Using default color ranges.")
        return {k: (np.array(v[0], dtype=np.uint8), np.array(v[1], dtype=np.uint8)) for k, v in default_ranges.items()}


# --- Color Detection (Simplified, relies on good ranges) ---
def detect_color(roi, current_color_ranges):
    """Detect the dominant color in ROI based on max pixel count within ranges."""
    if not current_color_ranges:
        logger.error("Color ranges not loaded for detect_color")
        return "?"  # Indicate error

    if roi is None or roi.size == 0 or len(roi.shape) != 3 or roi.shape[2] != 3:
        logger.warning(f"Invalid ROI provided to detect_color. Shape: {roi.shape if roi is not None else 'None'}")
        return "?"

    if roi.dtype != np.uint8: roi = np.uint8(roi)

    h, w = roi.shape[:2]
    # Use a smaller central part of the ROI to avoid edge noise/blurring
    y_start, y_end = h // 4, 3 * h // 4
    x_start, x_end = w // 4, 3 * w // 4
    if y_start >= y_end or x_start >= x_end:  # Check if center ROI is valid
        logger.warning("Center ROI calculation resulted in zero size, using full ROI.")
        center_roi = roi
    else:
        center_roi = roi[y_start:y_end, x_start:x_end]

    if center_roi.size == 0: return "?"

    hsv_roi = cv2.cvtColor(center_roi, cv2.COLOR_BGR2HSV)

    max_pixels = -1
    detected_color = "?"  # Default if no match

    for color, (lower, upper) in current_color_ranges.items():
        mask = cv2.inRange(hsv_roi, lower, upper)
        pixel_count = cv2.countNonZero(mask)

        if pixel_count > max_pixels:
            max_pixels = pixel_count
            detected_color = color

    # Optional: Add a confidence threshold (e.g., percentage of ROI pixels)
    # roi_area = center_roi.shape[0] * center_roi.shape[1]
    # confidence = max_pixels / roi_area if roi_area > 0 else 0
    # if confidence < 0.1: # 10% threshold
    #     logger.debug(f"Low confidence ({confidence:.2f}) for detected color {detected_color}")
    #     # Could return '?' or the best guess

    # logger.debug(f"Detected Color: {detected_color} (Pixels: {max_pixels})")
    return detected_color


# --- Cube Logic (Validate, Remap - Keep as is, ensure validation used) ---
def validate_cube(cube, order_name):
    if not isinstance(cube, str) or len(cube) != 54:
        raise ValueError(f"{order_name} must be a 54 character string, got {type(cube)} len {len(cube)}")
    counts = Counter(c for c in cube if c != '-')  # Count non-placeholder chars
    valid_colors = set(COLOR_NAMES)
    actual_colors = {c for c in cube if c in valid_colors}

    # Allow placeholders during construction, but final validation needs 9 of 6
    if '-' not in cube:
        if len(actual_colors) != 6 or any(count != 9 for color, count in counts.items()):
            raise ValueError(f"{order_name} invalid final counts: {counts} (need 9 of each of 6 colors)")
    elif len(actual_colors) > 6:
        raise ValueError(f"{order_name} invalid: Too many colors detected {counts}")


def remap_colors_to_kociemba(cube_frblud):
    # Assumes FRBLUD order: F=0-8, R=9-17, B=18-26, L=27-35, U=36-44, D=45-53
    logger.debug(f"Remapping FRBLUD: {cube_frblud}")
    validate_cube(cube_frblud, "Input FRBLUD for Remap")  # Validate before proceeding
    centers = {
        'F': cube_frblud[4], 'R': cube_frblud[13], 'B': cube_frblud[22],
        'L': cube_frblud[31], 'U': cube_frblud[40], 'D': cube_frblud[49]
    }
    # Check for placeholder centers
    if '-' in centers.values():
        raise ValueError(f"Cannot remap: Cube state has placeholder center colors: {centers}")
    # Check if all centers are unique colors from COLOR_NAMES
    center_colors = set(centers.values())
    if len(center_colors) != 6 or not center_colors.issubset(set(COLOR_NAMES)):
        raise ValueError(f"Invalid cube state: Center colors are not 6 unique standard colors: {centers}")

    # Kociemba expects URFDLB physical face mapping
    # Map the *color* found at the physical center to the Kociemba face name
    kociemba_map = {
        centers['U']: 'U', centers['R']: 'R', centers['F']: 'F',
        centers['D']: 'D', centers['L']: 'L', centers['B']: 'B'
    }
    # Check if mapping is valid (should be if centers are unique)
    if len(kociemba_map) != 6:
        raise ValueError("Internal error creating Kociemba color map.")

    # Remap the input string sticker by sticker
    try:
        remapped_cube = ''.join(kociemba_map[c] for c in cube_frblud)
    except KeyError as e:
        raise ValueError(
            f"Invalid sticker color '{e}' found in cube string during remapping. Centers: {centers}, String: {cube_frblud}")

    logger.debug(f"Kociemba map: {kociemba_map}, Remapped string: {remapped_cube}")
    return kociemba_map, remapped_cube  # Return map and remapped string


def remap_cube_to_kociemba_face_order(cube_frblud_remapped):
    """Input is already mapped to URFDLB letters, just needs face order changed."""
    # FRBLUD slices -> URFDLB concatenation for Kociemba solver
    up = cube_frblud_remapped[36:45]
    right = cube_frblud_remapped[9:18]
    front = cube_frblud_remapped[0:9]
    down = cube_frblud_remapped[45:54]
    left = cube_frblud_remapped[27:36]
    back = cube_frblud_remapped[18:27]
    kociemba_ordered_string = up + right + front + down + left + back
    logger.debug(f"Reordered to URFDLB for Kociemba: {kociemba_ordered_string}")
    validate_cube(kociemba_ordered_string, "Kociemba Input (URFDLB)")  # Validate final string
    return kociemba_ordered_string


def is_cube_solved(cube_state_frblud):
    """Check if the cube is already solved (FRBLUD order)."""
    if not isinstance(cube_state_frblud, str) or len(cube_state_frblud) != 54: return False
    for i in range(0, 54, 9):
        face = cube_state_frblud[i:i + 9]
        center_color = face[4]
        if center_color == '-': return False  # Not solved if placeholders exist
        if not all(sticker == center_color for sticker in face):
            return False
    return True


# --- Move Simplification (Keep as is) ---
def simplify_cube_moves(moves_str):
    # ... (Keep the existing simplification logic) ...
    moves = moves_str.strip().split()
    if not moves: return ""
    # Simple cancellation pass
    simplified = []
    i = 0
    while i < len(moves):
        move = moves[i]
        if not move: i += 1; continue  # Skip empty strings if any
        face = move[0]
        mod = move[1:] if len(move) > 1 else ''

        if i + 1 < len(moves) and moves[i + 1] and moves[i + 1][0] == face:
            next_move = moves[i + 1]
            next_mod = next_move[1:] if len(next_move) > 1 else ''
            val1 = 1 if mod == '' else -1 if mod == "'" else 2
            val2 = 1 if next_mod == '' else -1 if next_mod == "'" else 2
            net_val = (val1 + val2) % 4
            if net_val == 0:
                i += 2;
                continue  # Cancel out
            elif net_val == 1:
                simplified.append(face)
            elif net_val == 2:
                simplified.append(face + '2')
            elif net_val == 3:
                simplified.append(face + "'")
            i += 2
        else:
            simplified.append(move)
            i += 1
    # Repeat simplification until no changes
    current_moves = simplified
    last_len = -1
    while len(current_moves) != last_len:
        last_len = len(current_moves)
        simplified_pass = []
        i = 0
        temp_moves = current_moves  # Work on a copy
        while i < len(temp_moves):
            move = temp_moves[i]
            if not move: i += 1; continue
            face = move[0]
            mod = move[1:] if len(move) > 1 else ''
            if i + 1 < len(temp_moves) and temp_moves[i + 1] and temp_moves[i + 1][0] == face:
                next_move = temp_moves[i + 1]
                next_mod = next_move[1:] if len(next_move) > 1 else ''
                val1 = 1 if mod == '' else -1 if mod == "'" else 2
                val2 = 1 if next_mod == '' else -1 if next_mod == "'" else 2
                net_val = (val1 + val2) % 4
                if net_val == 0:
                    i += 2;
                    continue
                elif net_val == 1:
                    simplified_pass.append(face)
                elif net_val == 2:
                    simplified_pass.append(face + '2')
                elif net_val == 3:
                    simplified_pass.append(face + "'")
                i += 2
            else:
                simplified_pass.append(move)
                i += 1
        current_moves = simplified_pass

    final_str = " ".join(current_moves)
    logger.debug(f"Simplified moves: '{moves_str}' -> '{final_str}'")
    return final_str


# --- Cube Solving with U-move Replacement ---
def solve_cube_frblud(cube_frblud):
    try:
        logger.info(f"Attempting to solve FRBLUD: {cube_frblud}")
        # Ensure final validation before solving
        validate_cube(cube_frblud, "Final FRBLUD for Solving")

        if is_cube_solved(cube_frblud):
            logger.info("Cube is already solved!")
            return ""  # No moves needed

        kociemba_map, cube_frblud_remapped = remap_colors_to_kociemba(cube_frblud)
        scrambled_kociemba_ordered = remap_cube_to_kociemba_face_order(cube_frblud_remapped)

        logger.info(f"Solving Kociemba Input (URFDLB): {scrambled_kociemba_ordered}")
        # Kociemba solve function implicitly solves towards the standard solved state UUU... RRR... etc.
        raw_solution = kociemba.solve(scrambled_kociemba_ordered)
        logger.info(f"Kociemba raw solution: {raw_solution} (Length: {len(raw_solution.split())})")

        # Replace U moves (assuming hardware cannot perform U)
        # ** Verify these replacements are correct for your hardware **
        u_replacement = "R L F2 B2 R' L' D R L F2 B2 R' L'"  # From original script
        u_prime_replacement = "R L F2 B2 R' L' D' R L F2 B2 R' L'"  # From original script
        u2_replacement = "R L F2 B2 R' L' D2 R L F2 B2 R' L'"  # From original script

        moves = raw_solution.split()
        modified_solution_moves = []
        for move in moves:
            if move == "U":
                modified_solution_moves.extend(u_replacement.split())
            elif move == "U'":
                modified_solution_moves.extend(u_prime_replacement.split())
            elif move == "U2":
                modified_solution_moves.extend(u2_replacement.split())
            else:
                modified_solution_moves.append(move)

        final_solution_str = " ".join(m for m in modified_solution_moves if m)  # Ensure no empty strings
        logger.info(
            f"Solution after U-move replacement (Length: {len(final_solution_str.split())}): {final_solution_str}")

        simplified_solution = simplify_cube_moves(final_solution_str)
        logger.info(f"Simplified solution (Length: {len(simplified_solution.split())}): {simplified_solution}")

        return simplified_solution

    except ValueError as ve:
        logger.error(f"Validation Error solving cube: {ve}")
        raise  # Re-raise validation errors
    except Exception as e:
        # Catch potential Kociemba errors (e.g., invalid state)
        logger.error(f"Error during Kociemba solve: {type(e).__name__}: {e}")
        raise RuntimeError(f"Kociemba solving failed: {e}") from e


# --- Cube State Construction (Needs Verification) ---
# !! Critical: This mapping depends *heavily* on the physical rotations
# performed by the rotation_sequence AND the camera's viewpoint.
# Double-check this logic against the actual hardware behavior.
def construct_cube_from_u_scans(u_scans):
    """Construct the full cube state (FRBLUD order) from 12 U face scans."""
    logger.info("Constructing cube state from U-scans...")
    if len(u_scans) != 12 or any(s is None or len(s) != 9 for s in u_scans):
        logger.error(
            f"Invalid scan data: Need 12 scans of 9 stickers each. Got lengths: {[len(s) if s else 0 for s in u_scans]}")
        raise ValueError("Invalid scan data: Need 12 scans of 9 stickers each.")

    # Assume standard color scheme centers determined by first few scans
    # Centers: U=W, F=B, R=O, B=G, L=R, D=Y (adjust if your cube differs)
    cube_state = ['-'] * 54

    try:
        # Scan 0: Initial U face
        cube_state[36:45] = u_scans[0]  # U face (0-8 -> 36-44)
        cube_state[40] = 'W'  # U center

        # The following assignments are based on the *original script's logic*.
        # This needs careful validation against the actual rotation sequence.
        # If the sequence R L', B F', etc., results in different faces being
        # presented as 'U' to the camera, these indices will be wrong.

        # Scan 1: After R L' (Original F face becomes U?)
        scan1 = u_scans[1]
        cube_state[4] = 'B'  # F center
        # Map scan1 stickers to F face (0-8), excluding center
        indices_f = [0, 1, 2, 3, 5, 6, 7, 8]
        scan_indices = [0, 1, 2, 3, 5, 6, 7, 8]  # Corresponding indices in scan1
        for i_scan, i_cube in zip(scan_indices, indices_f): cube_state[i_cube] = scan1[i_scan]

        # Scan 2: After B F' (Original R face becomes U?)
        scan2 = u_scans[2]
        cube_state[13] = 'O'  # R center
        indices_r = [9, 10, 11, 12, 14, 15, 16, 17]
        for i_scan, i_cube in zip(scan_indices, indices_r): cube_state[i_cube] = scan2[i_scan]

        # Scan 3: After R L' (Original B face becomes U?)
        scan3 = u_scans[3]
        cube_state[22] = 'G'  # B center
        indices_b = [18, 19, 20, 21, 23, 24, 25, 26]
        for i_scan, i_cube in zip(scan_indices, indices_b): cube_state[i_cube] = scan3[i_scan]

        # Scan 4: After B F' (Original L face becomes U?)
        scan4 = u_scans[4]
        cube_state[31] = 'R'  # L center
        indices_l = [27, 28, 29, 30, 32, 33, 34, 35]
        for i_scan, i_cube in zip(scan_indices, indices_l): cube_state[i_cube] = scan4[i_scan]

        # Scan 5: After R L' (Original D face becomes U?)
        scan5 = u_scans[5]
        cube_state[49] = 'Y'  # D center
        indices_d = [45, 46, 47, 48, 50, 51, 52, 53]
        for i_scan, i_cube in zip(scan_indices, indices_d): cube_state[i_cube] = scan5[i_scan]

        # --- PROBLEM: Scans 6-11 ---
        # The original script used these scans to fill in individual missing stickers.
        # This indicates the first 6 scans DON'T capture everything cleanly, OR
        # the mapping is complex due to rotations.
        # This section is highly suspect and prone to errors.
        # A robust scanner would ideally scan each of the 6 faces directly.
        # For now, replicating the original logic structure, but it needs testing.
        if any('-' in s for s in u_scans[6:]): logger.warning(
            "Placeholders found in scans 6-11, construction may fail.")

        try:
            # These indices seem arbitrary without knowing the exact orientation shifts.
            # Example: cube_state[0] = u_scans[1][0] -> This was already set above? Redundant/Conflict?
            # It seems the original script overwrites previous assignments. Let's comment out the F,R,B,L,D bulk assignments above
            # and *only* use the specific index assignments from the original script.

            cube_state = ['-'] * 54  # Re-initialize
            # Set centers first
            cube_state[40] = 'W';
            cube_state[4] = 'B';
            cube_state[13] = 'O';
            cube_state[22] = 'G';
            cube_state[31] = 'R';
            cube_state[49] = 'Y';

            # Scan 0 (U face)
            cube_state[36:40] = u_scans[0][0:4];
            cube_state[41:45] = u_scans[0][5:9]

            # Scan 1
            cube_state[0] = u_scans[1][0];
            cube_state[2] = u_scans[1][2]
            cube_state[3] = u_scans[1][3];
            cube_state[5] = u_scans[1][5]
            cube_state[6] = u_scans[1][6];
            cube_state[8] = u_scans[1][8]

            # Scan 2
            cube_state[9] = u_scans[2][0];
            cube_state[10] = u_scans[2][1]
            cube_state[11] = u_scans[2][2];
            cube_state[15] = u_scans[2][6]
            cube_state[16] = u_scans[2][7];
            cube_state[17] = u_scans[2][8]

            # Scan 3
            cube_state[47] = u_scans[3][0];
            cube_state[53] = u_scans[3][2]
            cube_state[1] = u_scans[3][3];
            cube_state[7] = u_scans[3][5]
            cube_state[45] = u_scans[3][6];
            cube_state[51] = u_scans[3][8]

            # Scan 4
            cube_state[24] = u_scans[4][0];
            cube_state[12] = u_scans[4][1]
            cube_state[18] = u_scans[4][2];
            cube_state[26] = u_scans[4][6]
            cube_state[14] = u_scans[4][7];
            cube_state[20] = u_scans[4][8]

            # Scan 5
            cube_state[33] = u_scans[5][0];
            cube_state[27] = u_scans[5][2]
            cube_state[50] = u_scans[5][3];
            cube_state[48] = u_scans[5][5]
            cube_state[35] = u_scans[5][6];
            cube_state[29] = u_scans[5][8]

            # Scan 6
            # cube_state[36] = u_scans[6][0] # UBL corner - already set by scan 0?
            cube_state[46] = u_scans[6][1];
            cube_state[38] = u_scans[6][2]
            cube_state[42] = u_scans[6][6];
            cube_state[52] = u_scans[6][7]
            cube_state[44] = u_scans[6][8]

            # Scan 7
            cube_state[21] = u_scans[7][3];
            cube_state[23] = u_scans[7][5]

            # Scan 8
            cube_state[34] = u_scans[8][1];
            cube_state[28] = u_scans[8][7]

            # Scan 9
            cube_state[25] = u_scans[9][3];
            cube_state[19] = u_scans[9][5]

            # Scan 10
            cube_state[30] = u_scans[10][1];
            cube_state[32] = u_scans[10][7]

            # Scan 11
            cube_state[39] = u_scans[11][3];
            cube_state[41] = u_scans[11][5]

        except IndexError as e:
            logger.error(
                f"IndexError during cube construction using specific indices: {e}. Scan data might be incomplete.")
            raise ValueError(f"Failed to construct cube state due to missing scan data: {e}")
        except TypeError as e:
            logger.error(f"TypeError during cube construction: {e}. A scan might be None.")
            raise ValueError(f"Failed to construct cube state due to None scan: {e}")

        final_state = "".join(cube_state)
        placeholders = final_state.count('-')
        if placeholders > 0:
            logger.warning(f"Cube state constructed with {placeholders} placeholders: {final_state}")
            # Don't raise error here, let validation catch it if needed later
            # raise ValueError(f"Failed to construct complete cube state. Missing: {placeholders}")
        logger.info(f"Constructed FRBLUD: {final_state}")
        # Final validation before returning
        validate_cube(final_state, "Constructed FRBLUD")
        return final_state

    except Exception as e:
        logger.error(f"Unexpected error in construct_cube_from_u_scans: {e}", exc_info=True)
        raise ValueError(f"Failed to construct cube state: {e}")


# --- Scramble Generation ---
def generate_scramble(moves=20):
    """Generate a random scramble sequence (excluding U moves)."""
    basic_moves = ['F', 'B', 'R', 'L', 'D']  # Hardware limitations?
    modifiers = ['', "'", '2']
    scramble = []
    last_face = None
    for _ in range(moves):
        available_moves = [move for move in basic_moves if move != last_face]
        if not available_moves: available_moves = basic_moves  # Prevent getting stuck if only one move type left
        face = random.choice(available_moves)
        modifier = random.choice(modifiers)
        scramble.append(face + modifier)
        last_face = face
    return ' '.join(scramble)


# --- Serial Communication ---
serial_lock = asyncio.Lock()  # Lock for sending commands and waiting for ack
serial_read_buffer = ""  # Buffer for incomplete lines from Arduino
serial_ack_event = asyncio.Event()  # Event to signal ack received
last_serial_ack_message = ""  # Store the ack message


async def serial_reader_task():
    """Dedicated task to continuously read from serial port."""
    global ser, serial_read_buffer, serial_ack_event, last_serial_ack_message, app_state
    logger.info("Serial reader task started.")
    while True:
        if ser is None or not ser.is_open:
            # logger.debug("Serial port not available in reader task.")
            await asyncio.sleep(1)  # Wait longer if serial not connected
            continue

        try:
            if ser.in_waiting > 0:
                try:
                    # Read all available bytes, decode cautiously
                    data_bytes = ser.read(ser.in_waiting)
                    data = data_bytes.decode(errors='ignore')
                    # logger.debug(f"Serial READ raw: {data!r}") # Very verbose
                    serial_read_buffer += data
                except serial.SerialException as read_err:
                    logger.error(f"Serial read error: {read_err}")
                    await update_status(serial_connected=False, error_message=f"Serial read error: {read_err}")
                    ser.close();
                    ser = None  # Assume port is dead
                    continue  # Restart loop to wait/check for reconnect
                except Exception as decode_err:
                    logger.error(f"Error decoding serial data: {decode_err}")
                    # Don't close port, just clear buffer maybe?
                    serial_read_buffer = ""  # Clear potentially corrupt buffer
                    continue

                # Process complete lines from buffer
                while '\n' in serial_read_buffer:
                    line, serial_read_buffer = serial_read_buffer.split('\n', 1)
                    line = line.strip()  # Remove leading/trailing whitespace and \r
                    if line:
                        logger.info(f"Serial READ line: {line}")
                        # Check if it's an acknowledgment message
                        if "completed" in line.lower() or "executed" in line.lower() or "error" in line.lower():
                            logger.debug(f"Detected ACK message: {line}")
                            last_serial_ack_message = line
                            serial_ack_event.set()  # Signal acknowledgment received
                        # Handle other potential messages from Arduino if needed
                        # elif "status:" in line.lower(): parse_status(line)
                        # elif "ready" in line.lower(): await update_status(arduino_ready=True)

            else:
                # No data waiting, sleep briefly to yield control
                await asyncio.sleep(0.02)  # 20ms sleep

        except Exception as e:
            # Catch unexpected errors in the reader loop
            logger.error(f"Unexpected error in serial_reader_task: {e}", exc_info=True)
            # Decide how to handle: stop? retry? For robustness, try to continue.
            await asyncio.sleep(1)  # Longer sleep after an error

    logger.info("Serial reader task finished.")  # Should not happen normally


async def send_arduino_command(cmd: str, wait_for_ack=True, timeout=SERIAL_ACK_TIMEOUT):
    """Send command to Arduino and optionally wait for acknowledgment."""
    global ser, serial_ack_event, last_serial_ack_message
    if ser is None or not ser.is_open:
        logger.error("Serial port not open. Cannot send command.")
        await update_status(serial_connected=False, error_message="Serial port not connected.")
        return False

    async with serial_lock:  # Ensure only one command send/wait happens at a time
        try:
            logger.info(f"Serial SEND -> : {cmd}")
            serial_ack_event.clear()  # Reset event before sending
            last_serial_ack_message = ""  # Clear previous ack message
            ser.write(f"{cmd}\n".encode())
            await asyncio.sleep(0.05)  # Allow time for data to be sent

            if wait_for_ack:
                logger.debug(f"Waiting for ack for '{cmd}' (timeout: {timeout}s)...")
                try:
                    await asyncio.wait_for(serial_ack_event.wait(), timeout=timeout)
                    # Event was set - ack received
                    logger.info(f"Serial RECV <- : Ack received for '{cmd}': {last_serial_ack_message}")
                    if "error" in last_serial_ack_message.lower():
                        logger.error(f"Arduino reported error for command '{cmd}': {last_serial_ack_message}")
                        raise RuntimeError(f"Arduino error: {last_serial_ack_message}")
                    return True
                except asyncio.TimeoutError:
                    logger.error(f"Timeout ({timeout}s) waiting for Arduino ack for command: {cmd}")
                    raise TimeoutError(f"Timeout waiting for Arduino ack: {cmd}")
                except Exception as wait_err:  # Catch errors during wait
                    logger.error(f"Error waiting for ack: {wait_err}")
                    raise RuntimeError(f"Error waiting for ack: {wait_err}")
            else:
                logger.debug("Command sent, not waiting for ack.")
                return True  # Command sent

        except serial.SerialException as e:
            logger.error(f"Serial write error for command '{cmd}': {e}")
            await update_status(serial_connected=False, error_message=f"Serial write error: {e}")
            if ser: ser.close(); ser = None  # Assume port died
            return False
        except (TimeoutError, RuntimeError) as e:
            # Errors raised from ack waiting or Arduino error report
            await update_status(error_message=str(e), mode=SolverMode.ERROR)
            return False  # Indicate failure
        except Exception as e:
            logger.error(f"Unexpected error sending command '{cmd}': {e}")
            await update_status(error_message=f"Serial communication error: {e}", mode=SolverMode.ERROR)
            return False
    # Return False if lock couldn't be acquired? Should not happen with current logic.
    return False  # Fallback


# --- FastAPI Application Setup ---

# Lifespan Manager for Startup/Shutdown Events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup Logic
    global ser, color_ranges, app_state, solver_task
    logger.info("Application startup via lifespan manager...")
    app_state = SolverStatus().dict()  # Initialize state
    app_state["mode"] = SolverMode.CONNECTING  # Start in connecting state
    app_state["status_message"] = "Backend initializing..."

    # Load color ranges
    color_ranges = load_color_ranges(COLOR_RANGES_FILE)
    app_state["color_ranges_loaded"] = True  # Indicate ranges are ready
    logger.info(f"Loaded {len(color_ranges)} color ranges.")

    # Initialize Serial Port
    serial_port_to_try = SERIAL_PORT
    try:
        logger.info(f"Attempting to open serial port {serial_port_to_try}...")
        ser = serial.Serial(serial_port_to_try, SERIAL_BAUDRATE, timeout=SERIAL_TIMEOUT)
        # Use non-blocking mode? Might interact better with asyncio reader task
        # ser.nonblocking() # Or set timeout=0 after opening? Check pyserial docs.
        await asyncio.sleep(2)  # Wait for Arduino bootloader/initialization
        if ser.is_open:
            ser.flushInput()  # Clear any stale data from previous runs
            ser.flushOutput()
            logger.info(f"Serial port {serial_port_to_try} opened successfully.")
            app_state["serial_connected"] = True
            app_state["status_message"] = "Serial connected. Ready."
            # Start the dedicated serial reader task
            asyncio.create_task(serial_reader_task())
            logger.info("Serial reader task scheduled.")
        else:
            # This case should not happen if Serial() constructor succeeded without error
            raise serial.SerialException(f"Port {serial_port_to_try} failed to open despite no exception.")

    except serial.SerialException as e:
        logger.error(f"Failed to open serial port {serial_port_to_try}: {e}")
        app_state["serial_connected"] = False
        app_state["status_message"] = f"Serial Error: {e}"
        app_state[
            "error_message"] = f"Could not connect to Arduino on {serial_port_to_try}. Check connection and port name."
        ser = None
    except Exception as e:
        logger.error(f"Unexpected error during serial initialization: {e}")
        app_state["serial_connected"] = False
        app_state["status_message"] = f"Unexpected Serial Init Error: {e}"
        ser = None

    app_state["mode"] = SolverMode.IDLE  # Set to IDLE after init attempts
    if not app_state["serial_connected"]:
        app_state["status_message"] = "Ready (No Serial Connection)."
    else:
        app_state["status_message"] = "Ready."

    logger.info(f"Startup complete. Initial state: {app_state['mode']}, Serial: {app_state['serial_connected']}")

    yield  # Application runs here

    # Shutdown Logic
    logger.info("Application shutdown via lifespan manager...")
    if solver_task and not solver_task.done():
        logger.info("Cancelling active solver task...")
        solver_task.cancel()
        try:
            await solver_task
        except asyncio.CancelledError:
            logger.info("Solver task cancelled successfully.")
        except Exception as e:
            logger.error(f"Error during background task cancellation: {e}")

    if active_connection:
        logger.info("Closing active WebSocket connection...")
        try:
            await active_connection.close(code=1001, reason="Server shutting down")
        except Exception:
            pass  # Ignore errors during shutdown close
        active_connection = None

    if ser and ser.is_open:
        logger.info(f"Closing serial port {ser.name}.")
        ser.close()
        ser = None  # Ensure global var is cleared

    # Stop reader task? Difficult as it runs indefinitely. Relies on daemon=True or other mechanism.

    logger.info("Shutdown complete.")


# Create FastAPI app instance with lifespan manager
app = FastAPI(title="Rubik's Cube Solver Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Global Task Cancellation ---
async def cancel_solver_task():
    global solver_task
    if solver_task and not solver_task.done():
        logger.info("Cancelling existing solver task...")
        solver_task.cancel()
        try:
            await solver_task
        except asyncio.CancelledError:
            logger.info("Solver task cancelled successfully.")
        except Exception as e:
            logger.error(f"Error awaiting cancelled task: {e}")
        solver_task = None
        # Return state to IDLE after cancellation? Maybe better done by caller.
    else:
        logger.info("No active solver task to cancel or task already done.")


# --- Status Update Function ---
async def update_status(**kwargs):
    """Update global state and notify client if connected."""
    global app_state, active_connection
    updated_keys = list(kwargs.keys())
    logger.debug(f"Updating state with keys: {updated_keys}")
    # Update the state dictionary (ensure thread-safety if tasks modify state concurrently)
    # For simple updates from single tasks/requests, direct modification is ok here.
    for key, value in kwargs.items():
        if key in app_state or key in SolverStatus.__fields__:  # Check against model too
            app_state[key] = value
        else:
            logger.warning(f"Attempted to update unknown state key: {key}")

    if active_connection:
        try:
            # Create status object from current state for sending
            status_payload = SolverStatus(**app_state).dict()

            # Don't send raw frame bytes in status
            # status_payload.pop("last_frame_for_capture", None)
            # Send the prepared payload
            await active_connection.send_json(status_payload)
            logger.debug(f"Sent status update keys: {list(status_payload.keys())}")
        except WebSocketDisconnect:
            logger.warning("Client disconnected during status update send.")
            active_connection = None
        except Exception as e:
            logger.error(f"Error sending status update: {e}")


# --- OpenCV Frame Processing ---
def process_frame_for_client(frame_bytes: bytes):
    """Decodes frame, runs CV, encodes result"""
    global app_state, color_ranges  # Needs access to current color ranges
    try:
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            logger.error("Failed to decode frame")
            return None, None

        display = frame.copy()
        detected_colors_on_face = None  # Colors detected in the grid for this frame
        contour_details = {"detected": False, "x": 0, "y": 0, "w": 0, "h": 0}

        # --- Add CV processing based on current mode ---
        current_mode = app_state.get("mode")
        active_ranges = color_ranges  # Use the globally loaded/calibrated ranges

        # Only run CV if needed (Idle, Calibrating, Scanning) and ranges are loaded
        if current_mode in [SolverMode.IDLE, SolverMode.CALIBRATING, SolverMode.SCANNING] and active_ranges:
            # Basic contour finding
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

            for color, (lower, upper) in active_ranges.items():
                combined_mask |= cv2.inRange(hsv, lower, upper)

            kernel = np.ones((5, 5), np.uint8)  # Adjust kernel size if needed
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            # Optional: Add MORPH_OPEN to remove small noise speckles
            # combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            best_contour = None
            best_score = float('inf')

            for contour in contours:
                area = cv2.contourArea(contour)
                if MIN_CONTOUR_AREA < area < MAX_CONTOUR_AREA:
                    peri = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                    if len(approx) >= 4:  # Could check for exactly 4 for stricter squareness
                        x, y, w, h = cv2.boundingRect(approx)
                        if w > 0 and h > 0:
                            aspect_ratio = float(w) / h
                            if 0.75 < aspect_ratio < 2.1:  # Slightly stricter aspect ratio
                                # Could add solidity check: area / cv2.contourArea(cv2.convexHull(contour))
                                squareness_score = abs(1 - aspect_ratio)
                                if squareness_score < best_score:
                                    best_score = squareness_score
                                    best_contour = approx
                                    contour_details = {"detected": True, "x": x, "y": y, "w": w, "h": h}

            # --- Draw grid and detect colors if contour found ---
            if contour_details["detected"] and best_contour is not None:
                x, y, w, h = contour_details["x"], contour_details["y"], contour_details["w"], contour_details["h"]
                cv2.drawContours(display, [best_contour], -1, (0, 255, 0), 2)

                grid_size = min(w, h)
                pad_x = x + (w - grid_size) // 2
                pad_y = y + (h - grid_size) // 2
                grid_cell_size = grid_size // 3

                # Draw grid lines
                for i in range(1, 3):
                    cv2.line(display, (pad_x + i * grid_cell_size, pad_y),
                             (pad_x + i * grid_cell_size, pad_y + grid_size), (0, 255, 0), 1)
                    cv2.line(display, (pad_x, pad_y + i * grid_cell_size),
                             (pad_x + grid_size, pad_y + i * grid_cell_size), (0, 255, 0), 1)
                cv2.rectangle(display, (pad_x, pad_y), (pad_x + grid_size, pad_y + grid_size), (0, 255, 0), 2)

                # --- Color Detection within Grid (Only if Scanning) ---
                if current_mode == SolverMode.SCANNING:
                    detected_colors_on_face = []
                    for i in range(3):
                        for j in range(3):
                            y_start = pad_y + i * grid_cell_size
                            y_end = pad_y + (i + 1) * grid_cell_size
                            x_start = pad_x + j * grid_cell_size
                            x_end = pad_x + (j + 1) * grid_cell_size

                            # Extract ROI slightly inset from grid lines
                            padding = grid_cell_size // 8  # Adjust padding
                            roi_y_start = max(0, y_start + padding)
                            roi_y_end = min(frame.shape[0], y_end - padding)
                            roi_x_start = max(0, x_start + padding)
                            roi_x_end = min(frame.shape[1], x_end - padding)

                            roi = None
                            if roi_y_end > roi_y_start and roi_x_end > roi_x_start:
                                roi = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

                            color = detect_color(roi, active_ranges)  # Handles invalid ROI
                            detected_colors_on_face.append(color)

                            # Draw detected color on display frame
                            text_x = x_start + grid_cell_size // 4
                            text_y = y_start + grid_cell_size // 2 + 10
                            cv2.putText(display, color, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (255, 255, 255), 3, cv2.LINE_AA)
                            cv2.putText(display, color, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1,
                                        cv2.LINE_AA)

                # --- Calibration specific drawing ---
                elif current_mode == SolverMode.CALIBRATING:
                    # Highlight the center cell (1,1)
                    center_y_start = pad_y + grid_cell_size
                    center_y_end = pad_y + 2 * grid_cell_size
                    center_x_start = pad_x + grid_cell_size
                    center_x_end = pad_x + 2 * grid_cell_size
                    cv2.rectangle(display, (center_x_start, center_y_start),
                                  (center_x_end, center_y_end), (0, 0, 255), 3)  # Red box target

        # Encode the processed frame to JPEG bytes, then Base64
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]  # Adjust quality (70-80 is usually good)
        is_success, buffer = cv2.imencode('.jpg', display, encode_param)
        if not is_success:
            logger.error("Failed to encode processed frame to JPEG.")
            return None, None

        processed_frame_b64 = base64.b64encode(buffer).decode('utf-8')

        # Return processed frame, detected colors (if scanning), and contour info
        return processed_frame_b64, detected_colors_on_face  # Contour info not directly returned, used internally

    except Exception as e:
        logger.error(f"Error processing frame: {e}", exc_info=True)
        return None, None


# --- Background Task Implementations ---

# Calibration state variables (managed within app_state now)
# calibration_samples = {} -> Store samples within app_state if needed, or just recalc ranges
# temp_color_ranges = {} -> Store directly into global color_ranges after validation/saving

async def calibration_task():
    """Background task for guiding user through calibration steps."""
    global app_state, color_ranges
    logger.info("Starting Calibration Task")
    # Initialize calibration specific state
    app_state["calibration_step"] = 0
    app_state["calibration_samples"] = {c: [] for c in COLOR_NAMES}  # Store samples per color
    app_state["temp_color_ranges"] = color_ranges.copy()  # Start with current/loaded ranges

    while app_state["mode"] == SolverMode.CALIBRATING:
        step = app_state["calibration_step"]

        if step >= len(COLOR_NAMES):
            await update_status(status_message=f"Calibration sampling complete. Review and Save.", current_color=None)
            # Wait here until save or reset is called via API
            await asyncio.sleep(0.5)
            continue  # Loop without changing step

        current_color_to_capture = COLOR_NAMES[step]
        await update_status(
            status_message=f"Show center of '{current_color_to_capture}' face and click Capture",
            current_color=current_color_to_capture
        )

        # Wait for the capture command (handled by the API endpoint)
        # The API endpoint will modify app_state["calibration_step"] if successful.
        last_known_step = step
        while app_state["mode"] == SolverMode.CALIBRATING and app_state["calibration_step"] == last_known_step:
            await asyncio.sleep(0.1)  # Wait for step change or mode change

        if app_state["mode"] != SolverMode.CALIBRATING:
            logger.info("Calibration mode ended externally.")
            break  # Exit loop if mode changed

        # Step was advanced by capture endpoint or possibly reset. Loop continues.

    logger.info("Calibration Task finished.")
    # Clean up calibration state if task ends unexpectedly? Handled by reset/save endpoints.


# In rubiks_cube_backend.py

async def capture_calibration_color_logic():
    """Logic called by the API endpoint to capture color sample."""
    global app_state, color_ranges  # Need global color_ranges for fallback/initial state
    logger.info("--- Entering capture_calibration_color_logic ---")

    if app_state.get("mode") != SolverMode.CALIBRATING:
        logger.warning("Capture called when not in calibration mode.")
        await update_status(error_message="Not in calibration mode.")
        return

    step = app_state.get("calibration_step")
    if step is None or step >= len(COLOR_NAMES):
        logger.warning(f"Capture called when calibration step {step} is invalid or complete.")
        await update_status(error_message="Cannot capture color now (step invalid/complete).")
        return

    current_color = COLOR_NAMES[step]
    logger.info(f"Attempting to capture color for: {current_color} (step {step})")

    last_frame_data = app_state.get("last_frame_for_capture")
    if not last_frame_data:
        state_copy = {k: v for k, v in app_state.items() if k != "last_frame_for_capture"}
        logger.error(
            f">>> CRITICAL: No frame data found in app_state['last_frame_for_capture']. Current state (excluding frame): {state_copy}")
        await update_status(error_message="Could not get camera frame for capture.")
        return
    logger.info(f">>> Found frame data in app_state. Length: {len(last_frame_data)} bytes.")

    try:
        logger.debug("Attempting to decode frame bytes...")
        np_arr = np.frombuffer(last_frame_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            logger.error(">>> CRITICAL: Failed to decode frame stored in app_state.")
            await update_status(error_message="Frame decoding error during capture.")
            return
        frame_h, frame_w = frame.shape[:2]
        if frame_h == 0 or frame_w == 0:
            logger.error(">>> CRITICAL: Decoded frame has zero height or width.")
            await update_status(error_message="Decoded frame is empty.")
            return
        logger.info(f">>> Successfully decoded frame for capture. Shape: {frame.shape}")
    except Exception as e:
        logger.error(f">>> CRITICAL: Exception during frame decoding: {e}", exc_info=True)
        await update_status(error_message=f"Frame decoding exception: {e}")
        return

    # --- Find Contour (with detailed logging) ---
    logger.debug("Running contour detection for capture...")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    # Use *temporary* ranges if they exist, otherwise global ones for finding the cube itself
    active_ranges = app_state.get("temp_color_ranges", color_ranges)
    if not active_ranges:
        logger.error("No color ranges available for capture detection!");
        await update_status(error_message="Color ranges missing.")
        return

    for c, (lower, upper) in active_ranges.items():
        # Ensure ranges are numpy arrays before use
        if not isinstance(lower, np.ndarray): lower = np.array(lower, dtype=np.uint8)
        if not isinstance(upper, np.ndarray): upper = np.array(upper, dtype=np.uint8)
        # Add try-except for inRange just in case ranges are bad
        try:
            combined_mask |= cv2.inRange(hsv, lower, upper)
        except cv2.error as cv_err:
            logger.error(f"OpenCV error during inRange for color {c} (L={lower}, U={upper}): {cv_err}")
            # Optionally skip this color or stop entirely
            continue  # Skip this range if it causes an error

    kernel = np.ones((5, 5), np.uint8);
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logger.debug(f"Found {len(contours)} contours initially.")

    best_contour = None;
    best_score = float('inf')
    contour_details = {"detected": False}  # Reset details for this run
    suitable_contour_found_in_loop = False  # Flag to see if any contour passes checks

    # --- Start Contour Check Loop ---
    for i, contour in enumerate(contours):  # Iterate with index
        logger.debug(f"--- Checking Contour #{i} ---")
        area = cv2.contourArea(contour)
        logger.debug(f"[Contour Check #{i}] Area: {area:.1f} (Min: {MIN_CONTOUR_AREA}, Max: {MAX_CONTOUR_AREA})")

        # --- Condition 1: Area ---
        if MIN_CONTOUR_AREA < area < MAX_CONTOUR_AREA:
            peri = cv2.arcLength(contour, True)
            # Epsilon for approxPolyDP - 0.02 is standard, maybe try 0.03 or 0.04 if corners are rounded
            epsilon = 0.02 * peri
            approx = cv2.approxPolyDP(contour, epsilon, True)
            approx_len = len(approx)
            logger.debug(f"[Contour Check #{i}] Approx Points: {approx_len}")

            # --- Condition 2: Polygon Points ---
            if approx_len >= 4:
                x_c, y_c, w_c, h_c = cv2.boundingRect(approx)
                # Check bounding rect validity
                if w_c > 0 and h_c > 0:
                    aspect_ratio = float(w_c) / h_c
                    logger.debug(f"[Contour Check #{i}] Aspect Ratio: {aspect_ratio:.2f} (Range: 0.75-1.25)")

                    # --- Condition 3: Aspect Ratio ---
                    if 0.75 < aspect_ratio < 2.5:  # Adjust range if needed
                        squareness_score = abs(1 - aspect_ratio)
                        logger.debug(
                            f"[Contour Check #{i}] Passed all checks! Score: {squareness_score:.3f}. Comparing to best_score: {best_score:.3f}")

                        # --- Condition 4: Best Score (Implicit) ---
                        if squareness_score < best_score:
                            logger.debug(f"[Contour Check #{i}] New best contour found!")
                            best_score = squareness_score
                            best_contour = approx  # Store the polygon points
                            contour_details = {"detected": True, "x": x_c, "y": y_c, "w": w_c, "h": h_c}
                            suitable_contour_found_in_loop = True  # Mark that at least one passed
                        else:
                            logger.debug(
                                f"[Contour Check #{i}] Contour passed checks, but score {squareness_score:.3f} is not better than best_score {best_score:.3f}")
                            suitable_contour_found_in_loop = True  # Still mark that one passed
                    else:
                        logger.debug(f"[Contour Check #{i}] Failed: Aspect Ratio")
                else:
                    logger.debug(f"[Contour Check #{i}] Failed: Zero width/height bounding rect")
            else:
                logger.debug(f"[Contour Check #{i}] Failed: Not enough approx points (need >= 4)")
        else:
            logger.debug(f"[Contour Check #{i}] Failed: Area")
        logger.debug(f"--- End Check Contour #{i} ---")
    # --- End Contour Check Loop ---

    # Now check if contour_details indicates final success
    if not contour_details["detected"]:
        # Log slightly different message depending on whether *any* contour passed checks but wasn't the best, or if none passed at all
        if suitable_contour_found_in_loop:
            logger.warning(
                "Suitable contours found, but final selection failed? (Should not happen if best_score initialized correctly)")
        else:
            logger.warning("No suitable cube contour found during capture after checking all contours.")
        await update_status(error_message="Hold cube steady in center / Check lighting.")
        return
    # If we reach here, contour_details["detected"] is True and best_contour is set
    logger.info(">>> Found suitable contour for capture.")

    # --- Calculate ROI and Capture HSV (using simplified single-sample range logic) ---
    try:
        x, y, w, h = contour_details["x"], contour_details["y"], contour_details["w"], contour_details["h"]
        grid_size = min(w, h)
        grid_cell_size = grid_size // 3
        # Calculate grid origin (top-left) relative to the contour's bounding box
        pad_x = x + (w - grid_size) // 2
        pad_y = y + (h - grid_size) // 2

        # Calculate coordinates for the center cell (1, 1) relative to the image origin
        center_y_start = pad_y + grid_cell_size
        center_y_end = pad_y + 2 * grid_cell_size
        center_x_start = pad_x + grid_cell_size
        center_x_end = pad_x + 2 * grid_cell_size

        # Safety check boundaries against frame dimensions
        if not (0 <= center_y_start < center_y_end <= frame_h and \
                0 <= center_x_start < center_x_end <= frame_w):
            logger.error(
                f">>> CRITICAL: Calculated center ROI outside frame boundaries. Frame:({frame_w},{frame_h}), ROI:[{center_x_start}:{center_x_end}, {center_y_start}:{center_y_end}]")
            await update_status(error_message="Internal error: Bad ROI calculation.")
            return

        # Extract the ROI
        roi = frame[center_y_start:center_y_end, center_x_start:center_x_end]
        if roi.size == 0:
            logger.error("Extracted ROI is empty.")
            await update_status(error_message="Could not extract center ROI.")
            return
        logger.info(f"Extracted ROI for HSV. Shape: {roi.shape}")

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        avg_hsv = np.mean(hsv_roi, axis=(0, 1))
        logger.info(
            f"Captured avg HSV for {current_color}: [H={avg_hsv[0]:.1f}, S={avg_hsv[1]:.1f}, V={avg_hsv[2]:.1f}]")

        # --- Use the captured average HSV to define a new range (Single Sample Method) ---
        # ** TUNABLE PARAMETERS **
        h_range = 15 if current_color != "W" else 40
        s_range = 70
        v_range = 70
        min_s_non_white = 40
        min_v_non_white = 40
        min_v_white = 100

        avg_h, avg_s, avg_v = avg_hsv[0], avg_hsv[1], avg_hsv[2]

        # Calculate lower bounds
        lower_h = avg_h - h_range
        if current_color == 'W':
            lower_s = avg_s - s_range
            lower_v = max(min_v_white, avg_v - v_range)
        else:
            lower_s = max(min_s_non_white, avg_s - s_range)
            lower_v = max(min_v_non_white, avg_v - v_range)

        # Calculate upper bounds
        upper_h = avg_h + h_range
        upper_s = avg_s + s_range
        upper_v = avg_v + v_range

        # Handle Red Hue Wrap-around potential simplistically
        if current_color == 'R':
            if avg_h < h_range:
                logger.debug(f"Adjusting Red lower hue bound (was {lower_h:.1f}) due to proximity to 0.")
                lower_h = 0
            elif avg_h > 180 - h_range:
                logger.debug(f"Adjusting Red upper hue bound (was {upper_h:.1f}) due to proximity to 180.")
                upper_h = 180

        # Clamp Hue, Saturation, Value
        final_lower_h = np.clip(int(lower_h), 0, 180)  # OpenCV HSV Hue is 0-179
        final_upper_h = np.clip(int(upper_h), 0, 180)
        final_lower_s = np.clip(int(lower_s), 0, 255)
        final_upper_s = np.clip(int(upper_s), 0, 255)
        final_lower_v = np.clip(int(lower_v), 0, 255)
        final_upper_v = np.clip(int(upper_v), 0, 255)

        # Ensure lower H <= upper H after clamping
        if final_lower_h > final_upper_h:
            # This indicates the range likely needs to wrap around 0/180.
            # A simple fix is to swap them, effectively creating a very narrow range,
            # OR use 0-180 as a failsafe (very broad).
            # Let's log and use the broad approach for now.
            logger.warning(
                f"Hue Range Inverted for {current_color}: L={final_lower_h} > U={final_upper_h}. AvgH={avg_h:.1f}. Using 0-180 failsafe.")
            final_lower_h = 0
            final_upper_h = 180

        # Create final numpy arrays
        new_lower = np.array([final_lower_h, final_lower_s, final_lower_v], dtype=np.uint8)
        new_upper = np.array([final_upper_h, final_upper_s, final_upper_v], dtype=np.uint8)

        # Final check: ensure lower <= upper element-wise
        # Note: np.clip already handles S, V. This mainly affects Hue if the failsafe above wasn't triggered.
        new_lower = np.minimum(new_lower, new_upper)
        new_upper = np.maximum(new_lower, new_upper)

        # Update temporary ranges directly in app_state
        if "temp_color_ranges" not in app_state:
            app_state["temp_color_ranges"] = {k: v for k, v in color_ranges.items()}
            logger.info("Initialized temp_color_ranges from global ranges.")

        app_state["temp_color_ranges"][current_color] = (new_lower, new_upper)
        logger.info(f"Captured and set temp range for {current_color}: L={new_lower.tolist()} U={new_upper.tolist()}")

    except Exception as e:
        logger.error(f"Error during ROI/HSV calculation or range update: {e}", exc_info=True)
        await update_status(error_message=f"Processing error: {e}")
        return  # Stop processing on error

    # --- Advance step ---
    next_step = step + 1
    app_state["calibration_step"] = next_step  # Update state directly

    # Prepare status message for the next step or completion
    status_msg_next = f"Captured '{current_color}'."
    next_color_prompt = None
    if next_step < len(COLOR_NAMES):
        next_color_prompt = COLOR_NAMES[next_step]
        status_msg_next += f" Show center of '{next_color_prompt}' face and click Capture"
    else:
        status_msg_next += f" Sampling complete. Review and Save."

    # Log the update payload before sending
    update_payload = {
        "calibration_step": next_step,
        "current_color": next_color_prompt,
        "status_message": status_msg_next
    }
    logger.info(f"Preparing final status update: {update_payload}")
    await update_status(**update_payload)

    logger.info(f"Advanced calibration step to {next_step}")
    logger.info("--- Exiting capture_calibration_color_logic ---")


# --- Scan and Solve Task ---
async def scan_and_solve_task():
    """Background task for scanning all faces and solving."""
    global app_state, color_ranges  # Use global ranges as fallback/base
    logger.info("Starting Scan and Solve Task")

    # State specific to this task run
    u_scans = [None] * 12
    current_scan_idx = 0
    stability_counter = 0
    last_successful_scan_time = 0
    last_motor_move_time = time.time()  # Assume motors might have just moved
    last_stable_detected_face = None  # Store the last stable detected face colors

    # Ensure temp scan directory exists
    if not os.path.exists(TEMP_SCAN_DIR):
        try:
            os.makedirs(TEMP_SCAN_DIR)
        except OSError as e:
            logger.error(f"Failed to create temp scan dir {TEMP_SCAN_DIR}: {e}")  # Non-fatal

    # Get current ranges (might be default or calibrated)
    active_color_ranges = color_ranges

    while current_scan_idx < 12 and app_state["mode"] == SolverMode.SCANNING:
        # Update frontend with current scan index
        await update_status(scan_index=current_scan_idx)

        # --- Wait for conditions to be met for scanning ---
        status_msg = f"Scan {current_scan_idx + 1}/12: "
        ready_to_scan = False
        current_time = time.time()
        time_since_last_move = current_time - last_motor_move_time
        time_since_last_scan = current_time - last_successful_scan_time

        # Get latest detection results from app_state (updated by WebSocket handler)
        latest_face_colors = app_state.get("latest_detected_face_colors")  # List of 9 colors or None
        is_cube_detected = app_state.get("latest_cube_detected", False)

        if not is_cube_detected or not latest_face_colors:
            status_msg += "Position cube..."
            stability_counter = 0
            last_stable_detected_face = None
        elif "?" in latest_face_colors:
            status_msg += "Detection uncertain..."
            stability_counter = 0
            last_stable_detected_face = None
        elif time_since_last_move < MOTOR_STABILIZATION_TIME:
            status_msg += f"Stabilizing ({MOTOR_STABILIZATION_TIME - time_since_last_move:.1f}s)"
            stability_counter = 0
            last_stable_detected_face = None  # Reset face during stabilization
        elif time_since_last_scan < SCAN_COOLDOWN:
            status_msg += f"Cooldown ({SCAN_COOLDOWN - time_since_last_scan:.1f}s)"
            # Don't reset counter during cooldown
        else:
            # Cube detected, valid colors, post-stabilization/cooldown
            if latest_face_colors == last_stable_detected_face:
                stability_counter += 1
                status_msg += f"Stable ({stability_counter}/{STABILITY_THRESHOLD})"
            else:
                # New face detected or first stable detection
                stability_counter = 1
                last_stable_detected_face = latest_face_colors  # Store the new potential face
                status_msg += "Checking stability (1)"

            if stability_counter >= STABILITY_THRESHOLD:
                ready_to_scan = True
                status_msg += " Ready."

        await update_status(status_message=status_msg)

        # --- Perform Scan ---
        if ready_to_scan:
            captured_face = last_stable_detected_face
            logger.info(f"Scan {current_scan_idx + 1}: Capturing stable face {captured_face}")
            u_scans[current_scan_idx] = captured_face

            # --- Save scan image (optional, uses last captured frame) ---
            last_frame_data = app_state.get("last_frame_for_capture")
            if last_frame_data:
                try:
                    np_arr = np.frombuffer(last_frame_data, np.uint8)
                    scan_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if scan_image is not None:
                        # Optionally add detected colors overlay onto the saved image
                        # (Requires grid info from process_frame or re-detection)
                        filename = os.path.join(TEMP_SCAN_DIR, f"scan_{current_scan_idx + 1}.jpg")
                        cv2.imwrite(filename, scan_image)
                        logger.info(f"Saved scan image to {filename}")
                except Exception as e:
                    logger.error(f"Failed to save scan image: {e}")

            # --- Advance Scan Index and Reset Stability ---
            current_scan_idx += 1
            last_successful_scan_time = time.time()
            stability_counter = 0
            last_stable_detected_face = None
            app_state["latest_detected_face_colors"] = None  # Clear state until next detection

            # --- Send Motor Command for Next Rotation ---
            if current_scan_idx < 12:
                try:
                    move_cmd = rotation_sequence[current_scan_idx]
                    logger.info(f"Scan {current_scan_idx} complete. Sending move for next scan: '{move_cmd}'")
                    if move_cmd:
                        await update_status(status_message=f"Rotating for scan {current_scan_idx + 1} ({move_cmd})...")
                        success = await send_arduino_command(move_cmd, wait_for_ack=True)  # Use default timeout
                        if not success:
                            logger.error(f"Failed to execute move: {move_cmd}. Aborting scan.")
                            # Error status already set by send_arduino_command
                            return  # Exit the task
                        last_motor_move_time = time.time()  # Record time after move completes
                    else:
                        logger.info("No move needed for next scan.")
                        last_motor_move_time = time.time()  # Still update time
                except IndexError:
                    logger.error(f"Rotation sequence index out of bounds for scan {current_scan_idx + 1}. Aborting.")
                    await update_status(mode=SolverMode.ERROR,
                                        error_message="Internal error: Invalid rotation sequence.")
                    return
            else:
                logger.info("All 12 scans completed.")
                # Optional: Send final rotation if needed by hardware setup?
                # await send_arduino_command("SOME_RESET_MOVE")
                last_motor_move_time = time.time()
                break  # Exit the scanning loop

        await asyncio.sleep(0.05)  # Brief sleep to prevent busy loop

    # --- Post-Scanning ---
    if app_state["mode"] != SolverMode.SCANNING:
        logger.info("Scanning task cancelled or mode changed.")
        return  # Exit if mode changed externally

    if current_scan_idx == 12:
        # All scans supposedly completed
        logger.info("Proceeding to cube construction and solving...")
        try:
            await update_status(status_message="Constructing cube state...")
            # Ensure all scans are valid before construction
            if any(s is None for s in u_scans):
                missing_scans = [i + 1 for i, s in enumerate(u_scans) if s is None]
                raise ValueError(f"Cannot construct cube: Missing scan data for scans {missing_scans}")

            full_cube_state = construct_cube_from_u_scans(u_scans)
            # print_full_cube_state(full_cube_state) # Log visual state if needed

            await update_status(mode=SolverMode.SOLVING, status_message="Solving cube...")
            solution = solve_cube_frblud(full_cube_state)  # Can raise errors

            if solution == "":
                await update_status(mode=SolverMode.IDLE, status_message="Cube is already solved!", solution=solution)
            elif solution is not None:
                await update_status(status_message="Solution found! Executing...", solution=solution)
                logger.info(f"Executing solution: {solution}")
                moves = solution.split()
                total_moves = len(moves)
                await update_status(total_solve_moves=total_moves, solve_move_index=0)

                # --- Execute Solution Moves ---
                for i, move in enumerate(moves):
                    if app_state["mode"] != SolverMode.SOLVING:
                        logger.warning("Solving task cancelled during execution.")
                        # Don't change state back here, let /stop_and_reset handle it
                        return

                    logger.info(f"Executing move {i + 1}/{total_moves}: {move}")
                    await update_status(solve_move_index=i + 1,
                                        status_message=f"Executing: {move} ({i + 1}/{total_moves})")
                    success = await send_arduino_command(move, wait_for_ack=True)
                    if not success:
                        # Error status already set by send_arduino_command
                        return
                    await asyncio.sleep(0.1)  # Small delay between moves?

                # --- Solved ---
                await update_status(mode=SolverMode.IDLE, status_message="Cube solved successfully!",
                                    solve_move_index=total_moves)
                logger.info("Solution execution completed.")

            # else: # solution is None case handled by exception from solve_cube_frblud

        except (ValueError, RuntimeError) as e:  # Catch construction or solving errors
            logger.error(f"Error during cube construction or solving: {e}")
            await update_status(mode=SolverMode.ERROR, error_message=f"Solve Error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error post-scanning: {e}", exc_info=True)
            await update_status(mode=SolverMode.ERROR, error_message=f"Unexpected error: {e}")
    else:
        # Loop finished but not all scans were done (likely cancelled)
        logger.warning("Scan loop exited before completing 12 scans.")
        if app_state["mode"] == SolverMode.SCANNING:  # If not cancelled by user
            await update_status(mode=SolverMode.IDLE, status_message="Scanning stopped prematurely.")


# --- Scramble Task ---
async def scramble_task():
    """Background task for scrambling the cube."""
    global app_state
    logger.info("Starting Scramble Task")
    try:
        scramble_sequence = generate_scramble(moves=20)
        logger.info(f"Generated scramble: {scramble_sequence}")
        await update_status(status_message=f"Executing scramble...", solution=scramble_sequence)  # Show sequence

        moves = scramble_sequence.split()
        total_moves = len(moves)
        await update_status(total_solve_moves=total_moves, solve_move_index=0)  # Use solve progress fields

        for i, move in enumerate(moves):
            if app_state["mode"] != SolverMode.SCRAMBLING:
                logger.warning("Scrambling task cancelled.")
                # Let /stop_and_reset handle state change
                return

            logger.info(f"Executing scramble move {i + 1}/{total_moves}: {move}")
            await update_status(solve_move_index=i + 1, status_message=f"Scrambling: {move} ({i + 1}/{total_moves})")
            success = await send_arduino_command(move, wait_for_ack=True)
            if not success:  # Error status set by send_arduino_command
                return
            await asyncio.sleep(0.1)  # Small delay?

        logger.info("Scramble execution completed.")
        await update_status(mode=SolverMode.IDLE, status_message="Scramble complete.", solution=None,
                            total_solve_moves=0, solve_move_index=0)

    except Exception as e:
        logger.error(f"Error during scramble task: {e}", exc_info=True)
        await update_status(mode=SolverMode.ERROR, error_message=f"Scramble failed: {e}")


# --- WebSocket Endpoint ---
@app.websocket("/ws/rubiks")
async def websocket_endpoint(websocket: WebSocket):
    global active_connection, app_state
    try:
        await websocket.accept()
    except Exception as e:
        logger.error(f"WebSocket accept failed: {e}");
        return

    if active_connection:
        logger.warning("New client connected, closing previous connection.")
        try:
            await active_connection.close(code=1012, reason="New connection established")
        except Exception as e:
            logger.warning(f"Error closing previous WS connection: {e}")
    active_connection = websocket
    logger.info("WebSocket client connected. Sending initial state.")
    await update_status()  # Send current state immediately

    # --- Stability & Detection State (Local to handler) ---
    stability_counter = 0
    last_processed_colors = None  # Store the list of 9 detected colors
    frame_counter = 0

    try:
        while True:
            logger.debug(f"[WS Handler Loop #{frame_counter}] Waiting for message...")
            message_bytes = None
            try:
                message_bytes = await asyncio.wait_for(websocket.receive_bytes(), timeout=10.0)  # 10s timeout
                if message_bytes:
                    logger.debug(f"[WS Handler Loop #{frame_counter}] Received {len(message_bytes)} bytes")
                else:
                    logger.warning(f"[WS Handler Loop #{frame_counter}] Received empty message.");
                    continue
            except asyncio.TimeoutError:
                logger.warning(f"[WS Handler Loop #{frame_counter}] WS receive timed out.")
                try:
                    await update_status()  # Send heartbeat status
                except Exception:
                    logger.error("[WS Handler Loop] WS seems dead after timeout.");
                    break
                continue  # Continue waiting
            except WebSocketDisconnect:
                logger.info(f"[WS Handler Loop #{frame_counter}] Client disconnected during receive.");
                break
            except Exception as e:
                logger.error(f"[WS Handler Loop #{frame_counter}] Error during receive: {e}");
                break

            # --- Store & Process Frame ---
            app_state["last_frame_for_capture"] = message_bytes  # Store unconditionally for capture/save
            logger.debug(f"[WS Handler Loop #{frame_counter}] Stored frame in app_state")

            processed_frame_b64, detected_colors = process_frame_for_client(message_bytes)
            # logger.debug(f"[WS Handler Loop #{frame_counter}] Frame processed. Has B64: {processed_frame_b64 is not None}, Colors: {detected_colors}")

            # --- Update State for Background Tasks ---
            current_mode = app_state.get("mode")
            is_cube_currently_detected = app_state.get("latest_cube_detected", False)  # Get status from contour check

            if current_mode == SolverMode.SCANNING:
                # Update stability based on *this frame's* detection result
                app_state[
                    "latest_cube_detected"] = detected_colors is not None and "?" not in detected_colors  # Cube detected if colors are valid
                app_state["latest_detected_face_colors"] = detected_colors  # Store colors (or None)

                # We don't track stability counter here anymore, the scan_task does based on state changes.

            elif current_mode == SolverMode.CALIBRATING:
                # Update if cube is detected for UI feedback maybe?
                app_state["latest_cube_detected"] = detected_colors is not None  # Or use contour info if needed

            # --- Send Update Back to Client ---
            update_payload = {}
            if processed_frame_b64:
                update_payload["processed_frame"] = processed_frame_b64
            # Send other state? Usually updated by tasks/API calls. Send processed frame mainly.
            if update_payload:
                await update_status(**update_payload)

            frame_counter += 1
            # Optional sleep if frontend sends frames too rapidly
            # await asyncio.sleep(0.01)

    # --- Loop Exit Handling ---
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected (caught after loop).")
    except Exception as e:
        logger.error(f"WebSocket handler loop terminated by error: {e}", exc_info=True)
    finally:
        logger.info(f"WebSocket connection handler finished after processing {frame_counter} frames.")
        if active_connection == websocket: active_connection = None; logger.info("Cleared active_connection.")
        # Clear frame? Or keep last one? Let's clear it.
        # if "last_frame_for_capture" in app_state: app_state["last_frame_for_capture"] = None


# --- HTTP API Endpoints ---

@app.post("/start_calibration", status_code=200)
async def start_calibration():
    global solver_task, app_state
    if app_state.get("mode", SolverMode.IDLE) != SolverMode.IDLE:
        raise HTTPException(status_code=409, detail=f"Busy: {app_state.get('mode')}")
    await cancel_solver_task()
    logger.info("HTTP POST /start_calibration")
    # Reset calibration state before starting
    app_state["calibration_step"] = 0
    app_state["calibration_samples"] = {c: [] for c in COLOR_NAMES}
    app_state["temp_color_ranges"] = color_ranges.copy()  # Reset temp to current global
    await update_status(mode=SolverMode.CALIBRATING, status_message="Starting calibration...", error_message=None,
                        solution=None, calibration_step=0, current_color=COLOR_NAMES[0])
    solver_task = asyncio.create_task(calibration_task())
    return {"message": "Calibration mode started."}


@app.post("/capture_calibration_color", status_code=200)
async def capture_color():
    global app_state
    if app_state.get("mode") != SolverMode.CALIBRATING:
        raise HTTPException(status_code=409, detail="Not in calibration mode.")
    logger.info("HTTP POST /capture_calibration_color")
    # Run synchronously as it should be quick
    await capture_calibration_color_logic()
    return {"message": "Capture attempt processed."}


@app.post("/save_calibration", status_code=200)
async def save_calibration():
    global app_state, color_ranges  # Need to update global color_ranges
    if app_state.get("mode") != SolverMode.CALIBRATING:
        raise HTTPException(status_code=409, detail="Not in calibration mode.")

    step = app_state.get("calibration_step")
    if step is None or step < len(COLOR_NAMES):
        raise HTTPException(status_code=400,
                            detail=f"Calibration sampling not complete. On step {step}/{len(COLOR_NAMES)}.")

    logger.info("HTTP POST /save_calibration")
    temp_ranges = app_state.get("temp_color_ranges")
    if not temp_ranges or len(temp_ranges) != len(COLOR_NAMES):
        raise HTTPException(status_code=400, detail="Incomplete calibration data.")

    # Save the temporary ranges
    if save_color_ranges(temp_ranges, filename=COLOR_RANGES_FILE):
        color_ranges = temp_ranges.copy()  # Update runtime ranges **IMPORTANT**
        # Clean up calibration state
        app_state.pop("calibration_samples", None)
        app_state.pop("temp_color_ranges", None)
        await update_status(mode=SolverMode.IDLE, status_message="Calibration saved successfully.",
                            calibration_step=None, current_color=None)
        return {"message": "Calibration saved successfully."}
    else:
        await update_status(error_message="Failed to save calibration data to file.")
        raise HTTPException(status_code=500, detail="Failed to save calibration data.")


@app.post("/reset_calibration", status_code=200)
async def reset_calibration():
    global app_state, color_ranges  # Need to update global color_ranges
    logger.info("HTTP POST /reset_calibration")

    if app_state.get("mode") == SolverMode.CALIBRATING:
        await cancel_solver_task()  # Stop calibration task if running

    # Reload from file or defaults
    color_ranges = load_color_ranges(COLOR_RANGES_FILE)
    # Clear temporary calibration state
    app_state.pop("calibration_samples", None)
    app_state.pop("temp_color_ranges", None)
    app_state.pop("calibration_step", None)
    app_state.pop("current_color", None)
    await update_status(mode=SolverMode.IDLE, status_message="Calibration reset to saved/default values.",
                        error_message=None)
    return {"message": "Calibration reset."}


@app.post("/start_solve", status_code=200)
async def start_solve():
    global solver_task, app_state
    if not app_state.get("serial_connected"):
        raise HTTPException(status_code=400, detail="Serial port not connected.")
    if app_state.get("mode", SolverMode.IDLE) != SolverMode.IDLE:
        raise HTTPException(status_code=409, detail=f"Busy: {app_state.get('mode')}")
    await cancel_solver_task()
    logger.info("HTTP POST /start_solve")
    # Reset scan/solve state variables before starting
    app_state["scan_index"] = 0
    app_state["solve_move_index"] = 0
    app_state["total_solve_moves"] = 0
    app_state["latest_cube_detected"] = False
    app_state["latest_detected_face_colors"] = None
    app_state["solution"] = None
    await update_status(mode=SolverMode.SCANNING, status_message="Starting cube scan...", error_message=None)
    solver_task = asyncio.create_task(scan_and_solve_task())
    return {"message": "Scan and solve process initiated."}


@app.post("/start_scramble", status_code=200)
async def start_scramble():
    global solver_task, app_state
    if not app_state.get("serial_connected"):
        raise HTTPException(status_code=400, detail="Serial port not connected.")
    if app_state.get("mode", SolverMode.IDLE) != SolverMode.IDLE:
        raise HTTPException(status_code=409, detail=f"Busy: {app_state.get('mode')}")
    await cancel_solver_task()
    logger.info("HTTP POST /start_scramble")
    # Reset relevant state
    app_state["solve_move_index"] = 0
    app_state["total_solve_moves"] = 0
    app_state["solution"] = None
    await update_status(mode=SolverMode.SCRAMBLING, status_message="Starting scramble...", error_message=None)
    solver_task = asyncio.create_task(scramble_task())
    return {"message": "Scramble process initiated."}


@app.post("/stop_and_reset", status_code=200)
async def stop_and_reset():
    global solver_task, app_state
    logger.info("HTTP POST /stop_and_reset")
    current_mode = app_state.get("mode")
    logger.info(f"Current mode before reset: {current_mode}")

    if current_mode != SolverMode.IDLE and current_mode != SolverMode.CONNECTING:
        await update_status(mode=SolverMode.STOPPING, status_message="Stopping current operation...")
        await cancel_solver_task()  # Cancel background task first
        # Optionally send a stop command to Arduino? Depends on Arduino firmware.
        # If Arduino waits for ack, stopping mid-move is hard without a specific STOP command.
        # logger.info("Sending STOP command to Arduino (if implemented)...")
        # await send_arduino_command("STOP", wait_for_ack=False) # Example

    # Reset state variables to default IDLE values
    default_status = SolverStatus(
        mode=SolverMode.IDLE,
        serial_connected=app_state.get("serial_connected", False),  # Keep serial status
        status_message="Operation stopped. Ready." if current_mode != SolverMode.IDLE else "Ready.",
    ).dict()

    # Preserve certain state if needed (like serial_connected)
    serial_status = app_state.get("serial_connected", False)
    # Update global state dictionary
    app_state.clear()
    app_state.update(default_status)
    app_state["serial_connected"] = serial_status  # Restore serial status
    # Ensure color ranges are still loaded
    app_state["color_ranges_loaded"] = True if color_ranges else False

    await update_status()  # Send the reset state to client
    logger.info(f"Reset complete. State set to IDLE. Serial: {serial_status}")

    return {"message": "Reset to idle state."}


# --- Main Entry Point (for running with uvicorn) ---
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Uvicorn server...")
    uvicorn.run("rubiks_cube_backend:app", host="0.0.0.0", port=8000, log_level="info", reload=True)
