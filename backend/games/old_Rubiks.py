import cv2
import numpy as np
from collections import Counter
import kociemba
import os
import serial
import time
import json
import random
import argparse


class RubiksCubeSolver:
    def __init__(self, stream_url="http://192.168.187.31:4747/video", serial_port='COM7', baud_rate=9600):
        # Constants
        self.WINDOW_SIZE = (300, 300)
        self.SCAN_COOLDOWN = 0.5
        self.MOTOR_STABILIZATION_TIME = 0.5
        self.STABILITY_THRESHOLD = 2
        self.MIN_CONTOUR_AREA = 6000
        self.MAX_CONTOUR_AREA = 20000

        # Rotation sequence for scanning
        self.rotation_sequence = [
            "", "R L'", "B F'", "R L'", "B F'", "R L'", "B F'", "R L'", "B F'", "R L'", "B F'", "R L'"
        ]

        # Initialize color ranges with default values (lists of ranges)
        self.color_ranges = {
            "W": [(np.array([0, 0, 180]), np.array([180, 50, 255]))],  # White
            "R": [(np.array([0, 120, 100]), np.array([10, 255, 255])),  # Red low hue
                  (np.array([170, 120, 100]), np.array([180, 255, 255]))],  # Red high hue
            "G": [(np.array([40, 100, 100]), np.array([80, 255, 255]))],  # Green
            "Y": [(np.array([25, 120, 120]), np.array([35, 255, 255]))],  # Yellow
            "O": [(np.array([10, 120, 120]), np.array([20, 255, 255]))],  # Orange
            "B": [(np.array([90, 100, 100]), np.array([120, 255, 255]))]  # Blue
        }

        # Initialize serial communication
        self.ser = None
        self._init_serial(serial_port, baud_rate)

        # Initialize video capture
        self.cap = None
        self._init_video_capture(stream_url)

        # Load color ranges
        self._load_color_ranges()

        # Store the current frame for corner detection
        self.frame = None

    def _init_serial(self, serial_port, baud_rate):
        """Initialize serial communication with Arduino."""
        try:
            self.ser = serial.Serial(serial_port, baud_rate, timeout=1)
            time.sleep(2)
            print("Serial connection established.")
        except Exception as e:
            print(f"Error opening serial port: {e}")
            exit()

    def _init_video_capture(self, stream_url):
        """Initialize video capture from stream or default camera."""
        self.cap = cv2.VideoCapture(stream_url)
        if not self.cap.isOpened():
            print("Error: Could not open video stream. Trying default camera...")
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Error: Could not open default camera.")
                self.ser.close()
                exit()

        # Optimize video capture settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Clear buffered frames
        for _ in range(5):
            self.cap.grab()

    def _save_color_ranges(self, filename="color_ranges.json"):
        """Save color ranges to a JSON file."""
        serializable_ranges = {
            color: [(lower.tolist(), upper.tolist()) for lower, upper in ranges]
            for color, ranges in self.color_ranges.items()
        }
        with open(filename, 'w') as f:
            json.dump(serializable_ranges, f, indent=4)
        print(f"Color ranges saved to {filename}")

    def _load_color_ranges(self, filename="color_ranges.json"):
        """Load color ranges from a JSON file."""
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    serializable_ranges = json.load(f)
                self.color_ranges.update({
                    color: [(np.array(lower), np.array(upper)) for lower, upper in ranges]
                    for color, ranges in serializable_ranges.items()
                })
                print(f"Color ranges loaded from {filename}")
                print("Using previously calibrated color ranges.")
            except Exception as e:
                print(f"Error loading color ranges: {e}")
                print("Using default color ranges.")
        else:
            print("No saved color ranges found. Using default color ranges.")

    def _calibrate_colors(self, color_names=["W", "R", "G", "Y", "O", "B"]):
        """Calibrate HSV ranges by sampling the center of each face."""
        print("\nStarting calibration. Show each face's center when prompted.")
        print("Press 'c' to capture, 'q' to quit early.")

        calibrated_ranges = {}
        grid_size = int(min(self.WINDOW_SIZE) * 0.4)
        grid_cell_size = grid_size // 3
        pad_x, pad_y = 20, 50

        for color in color_names:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame during calibration")
                    return None

                frame = cv2.resize(frame, self.WINDOW_SIZE, interpolation=cv2.INTER_AREA)
                display = frame.copy()

                center_y_start = pad_y + grid_cell_size
                center_y_end = pad_y + 2 * grid_cell_size
                center_x_start = pad_x + grid_cell_size
                center_x_end = pad_x + 2 * grid_cell_size
                cv2.rectangle(display, (center_x_start, center_y_start),
                              (center_x_end, center_y_end), (0, 255, 0), 2)

                instruction = f"Show {color} center (e.g., White=W, Red=R), then press 'c'"
                cv2.putText(display, instruction, (pad_x, pad_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Calibration", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    roi = frame[center_y_start:center_y_end, center_x_start:center_x_end]
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    avg_hsv = np.mean(hsv_roi, axis=(0, 1))

                    h_range = 5 if color != "W" else 90  # Tighter hue range for non-white
                    s_range = 40
                    v_range = 40
                    lower = np.array([max(0, avg_hsv[0] - h_range),
                                      max(0, avg_hsv[1] - s_range),
                                      max(0, avg_hsv[2] - v_range)])
                    upper = np.array([min(180, avg_hsv[0] + h_range),
                                      min(255, avg_hsv[1] + s_range),
                                      min(255, avg_hsv[2] + v_range)])

                    calibrated_ranges[color] = [(lower, upper)]
                    print(f"Calibrated {color}: HSV={avg_hsv}, Range={lower} to {upper}")
                    break
                elif key == ord('q'):
                    print("Calibration aborted.")
                    return None

        cv2.destroyWindow("Calibration")
        return calibrated_ranges

    def _detect_color(self, roi):
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

    def _validate_cube(self, cube, order_name):
        """Validate the cube state."""
        if len(cube) != 54:
            raise ValueError(f"{order_name} must be 54 characters")
        counts = Counter(cube)
        if len(counts) != 6 or any(count != 9 for count in counts.values()):
            raise ValueError(f"{order_name} invalid: {counts} (need 9 of each of 6 colors)")

    def _remap_colors_to_kociemba(self, cube_frblud):
        """Remap cube colors to Kociemba notation."""
        self._validate_cube(cube_frblud, "FRBLUD")
        centers = [cube_frblud[i] for i in [4, 13, 22, 31, 40, 49]]
        color_map = {
            centers[4]: 'U', centers[1]: 'R', centers[0]: 'F',
            centers[5]: 'D', centers[3]: 'L', centers[2]: 'B'
        }
        return color_map, ''.join(color_map[c] for c in cube_frblud)

    def _remap_cube_to_kociemba(self, cube_frblud_remapped):
        """Remap cube faces to Kociemba order."""
        front, right, back, left, up, down = [cube_frblud_remapped[i:i + 9] for i in range(0, 54, 9)]
        return up + right + front + down + left + back

    def _get_solved_state(self, cube_frblud, color_map):
        """Generate the solved cube state."""
        centers = [cube_frblud[i] for i in [4, 13, 22, 31, 40, 49]]
        return ''.join(c * 9 for c in centers)

    def _is_cube_solved(self, cube_state):
        """Check if the cube is already solved."""
        for i in range(0, 54, 9):
            face = cube_state[i:i + 9]
            if not all(sticker == face[4] for sticker in face):
                return False
        return True

    def _simplify_cube_moves(self, moves_str):
        """Simplify a sequence of cube moves."""
        moves = moves_str.strip().split()

        def move_value(move):
            if move.endswith("2"):
                return 2
            elif move.endswith("'"):
                return -1
            return 1

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

        face_groups = [['L', 'R'], ['F', 'B'], ['U', 'D']]

        # First pass: Combine consecutive moves of the same face
        i = 0
        simplified = []
        while i < len(moves):
            current_face = moves[i][0]
            current_value = move_value(moves[i])

            j = i + 1
            while j < len(moves) and moves[j][0] == current_face:
                current_value += move_value(moves[j])
                j += 1

            move = value_to_move(current_face, current_value)
            if move:
                simplified.append(move)

            i = j

        # Second pass: Combine moves by face groups
        final_simplified = []
        i = 0
        while i < len(simplified):
            current_face = simplified[i][0]

            face_group = None
            for group in face_groups:
                if current_face in group:
                    face_group = group
                    break

            if face_group:
                counts = {face: 0 for face in face_group}
                j = i

                while j < len(simplified) and simplified[j][0] in face_group:
                    face = simplified[j][0]
                    counts[face] += move_value(simplified[j])
                    j += 1

                for face in face_group:
                    move = value_to_move(face, counts[face])
                    if move:
                        final_simplified.append(move)

                i = j
            else:
                final_simplified.append(simplified[i])
                i += 1

        return " ".join(final_simplified) if final_simplified else "No moves"

    def _solve_cube_frblud(self, cube_frblud):
        """Solve the cube given its FRBLUD state."""
        try:
            if self._is_cube_solved(cube_frblud):
                print("\nCube is already solved! No moves needed.")
                return ""

            color_map, cube_frblud_remapped = self._remap_colors_to_kociemba(cube_frblud)
            scrambled_kociemba = self._remap_cube_to_kociemba(cube_frblud_remapped)
            solved_frblud = self._get_solved_state(cube_frblud, color_map)
            _, solved_frblud_remapped = self._remap_colors_to_kociemba(solved_frblud)
            solved_kociemba = self._remap_cube_to_kociemba(solved_frblud_remapped)
            self._validate_cube(scrambled_kociemba, "Scrambled Kociemba")
            self._validate_cube(solved_kociemba, "Solved Kociemba")
            solution = kociemba.solve(scrambled_kociemba, solved_kociemba)

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
            optimized_solution = self._simplify_cube_moves(final_solution)
            print("\nOriginal solution length:", len(final_solution.split()))
            print("Optimized solution length:", len(optimized_solution.split()))

            return optimized_solution

        except Exception as e:
            print(f"Error solving cube: {str(e)}")
            return None

    def _print_full_cube_state(self, cube_state):
        """Print the full cube state in a readable format."""
        print("\nFull cube state (Front, Right, Back, Left, Up, Down):")
        print("".join(cube_state))
        print("\nVisual representation:")
        idx = [0, 9, 18, 27, 36, 45]
        for i in range(3):
            start = idx[4] + i * 3
            print("        " + " ".join(cube_state[start:start + 3]))
        for i in range(3):
            line = ""
            for face_start in idx[:4]:
                start = face_start + i * 3
                line += " ".join(cube_state[start:start + 3]) + " | "
            print(line[:-3])
        for i in range(3):
            start = idx[5] + i * 3
            print("        " + " ".join(cube_state[start:start + 3]))

    def _construct_cube_from_u_scans(self, u_scans):
        """Construct the full cube state from 12 U face scans."""
        cube_state = [''] * 54
        cube_state[4] = 'B'  # F center
        cube_state[13] = 'O'  # R center
        cube_state[22] = 'G'  # B center
        cube_state[31] = 'R'  # L center
        cube_state[40] = 'W'  # U center
        cube_state[49] = 'Y'  # D center

        cube_state[36:45] = u_scans[0]
        for i in range(54):
            if not cube_state[i]:
                cube_state[i] = '-'

        cube_state[0] = u_scans[1][0]
        cube_state[2] = u_scans[1][2]
        cube_state[3] = u_scans[1][3]
        cube_state[5] = u_scans[1][5]
        cube_state[6] = u_scans[1][6]
        cube_state[8] = u_scans[1][8]

        cube_state[9] = u_scans[2][0]
        cube_state[10] = u_scans[2][1]
        cube_state[11] = u_scans[2][2]
        cube_state[15] = u_scans[2][6]
        cube_state[16] = u_scans[2][7]
        cube_state[17] = u_scans[2][8]

        cube_state[47] = u_scans[3][0]
        cube_state[53] = u_scans[3][2]
        cube_state[1] = u_scans[3][3]
        cube_state[7] = u_scans[3][5]
        cube_state[45] = u_scans[3][6]
        cube_state[51] = u_scans[3][8]

        cube_state[24] = u_scans[4][0]
        cube_state[12] = u_scans[4][1]
        cube_state[18] = u_scans[4][2]
        cube_state[26] = u_scans[4][6]
        cube_state[14] = u_scans[4][7]
        cube_state[20] = u_scans[4][8]

        cube_state[33] = u_scans[5][0]
        cube_state[27] = u_scans[5][2]
        cube_state[50] = u_scans[5][3]
        cube_state[48] = u_scans[5][5]
        cube_state[35] = u_scans[5][6]
        cube_state[29] = u_scans[5][8]

        cube_state[36] = u_scans[6][0]
        cube_state[46] = u_scans[6][1]
        cube_state[38] = u_scans[6][2]
        cube_state[42] = u_scans[6][6]
        cube_state[52] = u_scans[6][7]
        cube_state[44] = u_scans[6][8]

        cube_state[21] = u_scans[7][3]
        cube_state[23] = u_scans[7][5]

        cube_state[34] = u_scans[8][1]
        cube_state[28] = u_scans[8][7]

        cube_state[25] = u_scans[9][3]
        cube_state[19] = u_scans[9][5]

        cube_state[30] = u_scans[10][1]
        cube_state[32] = u_scans[10][7]

        cube_state[39] = u_scans[11][3]
        cube_state[41] = u_scans[11][5]

        return ''.join(cube_state)

    def _generate_scramble(self, moves=20):
        """Generate a random scramble sequence."""
        basic_moves = ['F', 'B', 'R', 'L', 'D']
        modifiers = ['', '\'', '2']
        scramble = []
        last_face = None

        for _ in range(moves):
            available_moves = [move for move in basic_moves if move != last_face]
            face = random.choice(available_moves)
            modifier = random.choice(modifiers)
            scramble.append(face + modifier)
            last_face = face

        return ' '.join(scramble)

    def _send_arduino_command(self, cmd, wait_time=None):
        """Send command to Arduino and wait for acknowledgment."""
        self.ser.write(f"{cmd}\n".encode())
        while True:
            if self.ser.in_waiting:
                response = self.ser.readline().decode("latin1").strip()
                if "completed" in response.lower() or "executed" in response.lower():
                    break
        if wait_time:
            time.sleep(wait_time)

    def _send_compound_move(self, move):
        """Send a compound move command to Arduino."""
        if move:
            self._send_arduino_command(move)

    def _close_all_scan_windows(self, scan_windows):
        """Close all scan windows."""
        for window in scan_windows:
            cv2.destroyWindow(window)
        scan_windows.clear()

    def _show_menu(self):
        """Display the main menu and get user choice."""
        print("\n=== Rubik's Cube Solver Menu ===")
        print("1. Calibrate Colors")
        print("2. Solve Cube")
        print("3. Scramble Cube")
        print("4. Exit")
        while True:
            try:
                choice = input("\nEnter your choice (1-4): ")
                if choice in ['1', '2', '3', '4']:
                    return int(choice)
                print("Invalid choice. Please enter a number between 1 and 4.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    def calibrate(self):
        """Run the calibration process."""
        print("\n=== Color Calibration Mode ===")
        print("Starting color calibration...")
        new_ranges = self._calibrate_colors()
        if new_ranges:
            self.color_ranges.update(new_ranges)
            self._save_color_ranges()
            print("Color calibration completed and saved.")
        else:
            print("Calibration aborted.")
        input("\nPress Enter to return to main menu...")

    def solve(self):
        """Run the cube solving process with improved boundary checking."""
        print("\n=== Solve Mode ===")
        print("Controls:")
        print("- Press 'r' to reset scan process")
        print("- Press 'q' to return to main menu")
        print("\nHold the cube steady in frame for automatic scanning.")

        temp_dir = "cube_scans"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        u_scans = [[] for _ in range(12)]
        current_scan_idx = 0
        prev_face_colors = None
        scan_windows = []
        last_scan_time = time.time()
        last_motor_move_time = 0
        stability_counter = 0
        prev_contour = None
        last_valid_grid = None

        cv2.namedWindow("Rubik's Cube Scanner", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Rubik's Cube Scanner", self.WINDOW_SIZE[0], self.WINDOW_SIZE[1])

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Store the original frame for corner detection
            self.frame = frame.copy()

            frame = cv2.resize(frame, self.WINDOW_SIZE, interpolation=cv2.INTER_AREA)
            display = frame.copy()

            # Use the improved contour detection method
            best_contour = self._find_best_cube_contour(frame)
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
                            last_valid_grid = (warped_image, grid_size, grid_cell_size)

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

                    if prev_contour is not None:
                        try:
                            shape_diff = cv2.matchShapes(best_contour, prev_contour, 1, 0.0)
                            # Calculate position difference (center point difference)
                            M1 = cv2.moments(best_contour)
                            M2 = cv2.moments(prev_contour)
                            if M1["m00"] != 0 and M2["m00"] != 0:
                                cx1 = int(M1["m10"] / M1["m00"])
                                cy1 = int(M1["m01"] / M1["m00"])
                                cx2 = int(M2["m10"] / M2["m00"])
                                cy2 = int(M2["m01"] / M2["m00"])
                                position_diff = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
                            else:
                                position_diff = 100  # Large value to indicate instability

                            if shape_diff < 0.2 and position_diff < 15:
                                stability_counter += 1
                            else:
                                stability_counter = 0
                        except:
                            stability_counter = 0
                    else:
                        stability_counter = 1

                    prev_contour = best_contour
                except Exception as e:
                    print(f"Error processing cube: {str(e)}")
                    stability_counter = 0
            else:
                stability_counter = 0
                prev_contour = None

            current_time = time.time()
            time_since_last_scan = current_time - last_scan_time
            time_since_last_move = current_time - last_motor_move_time

            if current_scan_idx < 12:
                if cube_detected:
                    status = f"Scan {current_scan_idx + 1}/12 - "
                    if time_since_last_move < self.MOTOR_STABILIZATION_TIME:
                        status += f"Waiting for cube to stabilize: {self.MOTOR_STABILIZATION_TIME - time_since_last_move:.1f}s"
                    elif time_since_last_scan < self.SCAN_COOLDOWN:
                        status += f"Cooldown: {self.SCAN_COOLDOWN - time_since_last_scan:.1f}s"
                    else:
                        status += f"Stability: {stability_counter}/{self.STABILITY_THRESHOLD}"
                else:
                    status = f"Position cube for scan {current_scan_idx + 1}/12"

                cv2.putText(display, status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Rubik's Cube Scanner", display)

            if (cube_detected and stability_counter >= self.STABILITY_THRESHOLD and
                    current_scan_idx < 12 and
                    time_since_last_scan >= self.SCAN_COOLDOWN and
                    time_since_last_move >= self.MOTOR_STABILIZATION_TIME):

                if last_valid_grid is not None:
                    warped_image, grid_size, grid_cell_size = last_valid_grid
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
                                color = self._detect_color(roi)
                                face_colors.append(color)

                        if len(face_colors) == 9:  # Ensure we have all 9 colors
                            if prev_face_colors is None or not all(
                                    a == b for a, b in zip(face_colors, prev_face_colors)):
                                u_scans[current_scan_idx] = face_colors
                                prev_face_colors = face_colors

                                if current_scan_idx < 11:
                                    self._send_compound_move(self.rotation_sequence[current_scan_idx + 1])
                                    last_motor_move_time = time.time()

                                window_name = f"U Face Scan #{current_scan_idx + 1}"
                                scan_windows.append(window_name)

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

                                cv2.imshow(window_name, scan_image)
                                cv2.imwrite(os.path.join(temp_dir, f"u_scan_{current_scan_idx + 1}_processed.jpg"),
                                            scan_image)

                                current_scan_idx += 1
                                if current_scan_idx == 12:
                                    print("\nAll scans completed! Processing solution...")
                                    self._send_compound_move("B F'")
                                    last_motor_move_time = time.time()

                                    cube_state = self._construct_cube_from_u_scans(u_scans)
                                    try:
                                        solution = self._solve_cube_frblud(cube_state)
                                        if solution:
                                            print(f"\nSolution: {solution}")
                                            self._send_arduino_command(f"SOLUTION:{solution}")
                                    except Exception as e:
                                        print(f"Failed to solve: {e}")
                                    print("\nPress 'q' to return to main menu.")

                                last_scan_time = current_time
                                stability_counter = 0
                            else:
                                print("Duplicate face detected, skipping")
                    except Exception as e:
                        print(f"Error processing grid cells: {str(e)}")

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self._close_all_scan_windows(scan_windows)
                cv2.destroyWindow("Rubik's Cube Scanner")
                return
            elif key == ord('r'):
                print("Resetting scan process...")
                self._close_all_scan_windows(scan_windows)
                u_scans = [[] for _ in range(12)]
                current_scan_idx = 0
                stability_counter = 0
                prev_contour = None
                prev_face_colors = None
                last_scan_time = time.time()
                print("Scan process reset. Ready for scan 1.")
    def scramble(self):
        """Run the cube scrambling process."""
        print("\n=== Scramble Mode ===")
        try:
            print("\nGenerating scramble sequence...")
            scramble = self._generate_scramble()
            print(f"Scramble sequence: {scramble}")
            print("Executing scramble...")
            self._send_arduino_command(scramble)
            print("Scramble completed!")
            time.sleep(1)
            print("Scramble completed successfully!")
        except Exception as e:
            print(f"Error during scramble: {e}")
        input("\nPress Enter to return to main menu...")

    def run(self):
        """Run the main program loop."""
        self._send_arduino_command("SWITCH:RUBIK\n")
        while True:
            choice = self._show_menu()

            if choice == 1:
                self.calibrate()
            elif choice == 2:
                self.solve()
            elif choice == 3:
                self.scramble()
            elif choice == 4:
                print("\nExiting program...")
                break

        # Cleanup
        self.cap.release()
        self.ser.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rubik\'s Cube Solver')
    parser.add_argument('--video', type=str, default="http://192.168.187.31:4747/video",
                        help='Video source URL or camera index')
    parser.add_argument('--port', type=str, default='COM7',
                        help='Serial port for Arduino')
    parser.add_argument('--baud', type=int, default=9600,
                        help='Baud rate for serial communication')

    args = parser.parse_args()

    solver = RubiksCubeSolver(args.video, args.port, args.baud)
    solver.run()