import cv2
import numpy as np
import json
import os
from collections import defaultdict

class ColorDetector:
    def __init__(self, game_config_file="game_color_ranges.json"):
        self.game_config_file = game_config_file

        # Default color ranges in HSV
        self.default_color_ranges = {
            "red": [
                {"lower": np.array([0, 100, 100]), "upper": np.array([10, 255, 255])},
                {"lower": np.array([160, 100, 100]), "upper": np.array([180, 255, 255])}
            ],
            "purple": [
                {"lower": np.array([125, 40, 80]), "upper": np.array([155, 255, 255])}
            ],
            "cyan": [
                {"lower": np.array([85, 100, 100]), "upper": np.array([105, 255, 255])}
            ],
            "lime_green": [
                {"lower": np.array([35, 100, 100]), "upper": np.array([55, 255, 255])}
            ]
        }

        # Initialize with default ranges for normal mode
        self.color_ranges = {color: list(ranges) for color, ranges in self.default_color_ranges.items()}

        # Game mode ranges (starts empty)
        self.game_color_ranges = {}

        # Load any saved configurations
        self.load_game_config()

        # Color display values (B,G,R)
        self.color_display = {
            "red": (0, 0, 255),
            "purple": (128, 0, 128),
            "cyan": (255, 255, 0),
            "lime_green": (50, 255, 50)
        }

        # For collecting training data
        self.training_data = defaultdict(list)
        self.training_mode = False
        self.current_training_color = None

        # Game mode flag
        self.game_mode = False

    def load_game_config(self):
        """Load game mode color ranges from game configuration file"""
        if os.path.exists(self.game_config_file):
            try:
                with open(self.game_config_file, 'r') as f:
                    loaded_ranges = json.load(f)

                self.game_color_ranges = {}
                for color, ranges in loaded_ranges.items():
                    self.game_color_ranges[color] = []
                    for range_data in ranges:
                        lower = np.array(range_data["lower"])
                        upper = np.array(range_data["upper"])
                        self.game_color_ranges[color].append({"lower": lower, "upper": upper})

                print(f"Loaded game color configurations from {self.game_config_file}")
            except Exception as e:
                print(f"Error loading game config file: {e}")
                print("Starting with empty game colors")

    def save_game_config(self):
        """Save game mode color ranges to game configuration file"""
        serializable_ranges = {}
        for color, ranges in self.game_color_ranges.items():
            serializable_ranges[color] = []
            for range_data in ranges:
                serializable_ranges[color].append({
                    "lower": range_data["lower"].tolist(),
                    "upper": range_data["upper"].tolist()
                })

        with open(self.game_config_file, 'w') as f:
            json.dump(serializable_ranges, f, indent=4)
        print(f"Saved game color configurations to {self.game_config_file}")

    def detect_colors(self, frame):
        """Detect configured colors in the frame"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        results = {}

        # Use game mode ranges if in game mode, otherwise use normal mode ranges
        color_ranges = self.game_color_ranges if self.game_mode else self.color_ranges

        for color_name, ranges in color_ranges.items():
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            for range_data in ranges:
                lower = range_data["lower"]
                upper = range_data["upper"]
                current_mask = cv2.inRange(hsv, lower, upper)
                mask = cv2.bitwise_or(mask, current_mask)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]  # Reduced threshold

            if contours:
                if self.game_mode:
                    # ✅ Return immediately if in game mode and a match is found
                    return {color_name: contours}
                else:
                    results[color_name] = contours

        return results

    def start_training(self, color_name):
        """Start training mode for a specific color"""
        if color_name not in self.default_color_ranges:
            print(f"Color {color_name} not in supported colors")
            return False

        self.training_mode = True
        self.current_training_color = color_name
        self.training_data[color_name] = []

        print(f"Training mode activated for {color_name}")
        print("Place the object with target color in frame and press 'S' to sample")
        print("Press 'F' to finish training")
        return True

    def add_training_sample(self, frame, error_margin=(30, 80, 80)):
        """Add a training sample for the current color with valid HSV ranges"""
        if not self.training_mode or not self.current_training_color:
            return False

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]

        # Use a 40x40 central area of the frame
        sample_area = hsv[h//2-20:h//2+20, w//2-20:w//2+20]

        # Get min/max HSV values
        min_val = np.min(sample_area, axis=(0, 1)).astype(int)
        max_val = np.max(sample_area, axis=(0, 1)).astype(int)

        # Expand range with error margin
        min_val[0] = max(0, min_val[0] - error_margin[0])      # H
        max_val[0] = min(179, max_val[0] + error_margin[0])    # H
        min_val[1] = max(0, min_val[1] - error_margin[1])      # S
        max_val[1] = min(255, max_val[1] + error_margin[1])    # S
        min_val[2] = max(0, min_val[2] - error_margin[2])      # V
        max_val[2] = min(255, max_val[2] + error_margin[2])    # V

        # Handle hue wraparound (like for red)
        if min_val[0] > max_val[0]:
            # Split into two ranges: one from min to 179, another from 0 to max
            range1 = {
                "lower": np.array([min_val[0], min_val[1], min_val[2]]),
                "upper": np.array([179,       max_val[1], max_val[2]])
            }
            range2 = {
                "lower": np.array([0,         min_val[1], min_val[2]]),
                "upper": np.array([max_val[0], max_val[1], max_val[2]])
            }
            self.training_data[self.current_training_color].extend([range1, range2])
            print("Added split hue range for wraparound:")
            print(f"  Range 1: {range1}")
            print(f"  Range 2: {range2}")
        else:
            new_range = {
                "lower": np.array(min_val),
                "upper": np.array(max_val)
            }
            self.training_data[self.current_training_color].append(new_range)
            print(f"Added single HSV range: {new_range}")

        return True

    def finish_training(self):
        """Complete training and update the color ranges"""
        if not self.training_mode or not self.current_training_color:
            return False

        if not self.training_data[self.current_training_color]:
            print("No samples were collected. Training cancelled.")
            self.training_mode = False
            self.current_training_color = None
            return False

        # In game mode, update game color ranges
        if self.current_training_color not in self.game_color_ranges:
            self.game_color_ranges[self.current_training_color] = []

        for new_range in self.training_data[self.current_training_color]:
            self.game_color_ranges[self.current_training_color].append(new_range)

        self.save_game_config()
        print(f"Added {len(self.training_data[self.current_training_color])} ranges to game color {self.current_training_color}")

        self.training_mode = False
        self.current_training_color = None
        return True

    def toggle_game_mode(self):
        """Toggle between regular mode and game mode"""
        self.game_mode = not self.game_mode
        mode_name = "Game Mode" if self.game_mode else "Normal Mode"
        print(f"Switched to {mode_name}")
        return self.game_mode

    def draw_results(self, frame, results):
        """Draw detection results on the frame"""
        result_frame = frame.copy()
        h, w = result_frame.shape[:2]

        # Draw training indicator if in training mode
        if self.training_mode:
            # Draw training sampling square
            cv2.rectangle(result_frame, (w//2-10, h//2-10), (w//2+10, h//2+10), (255, 102, 204), 2)
            mode_text = f"Training: {self.current_training_color}"
            cv2.putText(result_frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw game mode indicator
        elif self.game_mode:
            mode_text = "GAME MODE"
            cv2.putText(result_frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # Draw detected colors
        for color_name, contours in results.items():
            if color_name in self.color_display:
                display_color = self.color_display[color_name]
            else:
                # Generate color for unknown color names
                color_hash = hash(color_name) % 0xFFFFFF
                b = color_hash & 0xFF
                g = (color_hash >> 8) & 0xFF
                r = (color_hash >> 16) & 0xFF
                display_color = (b, g, r)
                self.color_display[color_name] = display_color

            for contour in contours:
                cv2.drawContours(result_frame, [contour], -1, display_color, 2)
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result_frame, (x, y), (x+w, y+h), display_color, 2)
                cv2.putText(result_frame, color_name, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, display_color, 2)

        return result_frame

def connect_to_droidcam():
    """Connect to DroidCam using WiFi IP and port"""
    ip = input("Enter the DroidCam IP address (e.g., 192.168.1.101): ")
    port = input("Enter the DroidCam port (e.g., 4747): ")
    camera_url = f"http://{ip}:{port}/video"
    print(f"Connecting to DroidCam at {camera_url}")
    cap = cv2.VideoCapture(camera_url)

    if not cap.isOpened():
        print("Failed to connect to DroidCam. Please check the IP and port.")
        return None

    return cap

def main():
    """Main function to run the color detection application"""
    detector = ColorDetector()
    cap = connect_to_droidcam()

    if cap is None:
        print("Exiting due to camera connection failure")
        return

    print("\nControls:")
    print("1-4: Start training for red, purple, cyan, lime_green")
    print("S: Sample color in training mode")
    print("F: Finish training")
    print("G: Toggle game mode")
    print("Q: Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        results = detector.detect_colors(frame)  # This now works correctly
        result_frame = detector.draw_results(frame, results)
        cv2.imshow("Color Detection", result_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
            color_map = {ord('1'): "red", ord('2'): "purple", ord('3'): "cyan", ord('4'): "lime_green"}
            detector.start_training(color_map[key])
        elif key == ord('s'):
            if detector.training_mode:
                detector.add_training_sample(frame)
        elif key == ord('f'):
            if detector.training_mode:
                detector.finish_training()
        elif key == ord('g'):
            detector.toggle_game_mode()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()