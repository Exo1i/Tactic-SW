import cv2
import numpy as np
import base64
import time
import queue  # Add for frame streaming
import threading
import asyncio
# import serial  # REMOVE: No longer needed

from utils.esp32_client import esp32_client  # ADD: Use shared ESP32 client

# --- Threaded VideoStream class (copied from target_shooter_game) ---
class VideoStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else (False, None)

    def release(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()

# --- Hardcoded IP camera URL ---
IPCAM_URL = "http://192.168.41.139:4747/video"  # <-- Change to your webcam's IP

# Define flexible color range for green cups in HSV (for black background)
green_lower = np.array([30, 40, 80])
green_upper = np.array([90, 255, 255])

# Define color range for the light blue ball (replace red)
ball_lower = np.array([85, 80, 120])    # Lower HSV for light blue
ball_upper = np.array([105, 255, 255])  # Upper HSV for light blue

adaptive_green_threshold_low = 80
adaptive_learning_rate = 0.01

# REMOVE: Serial initialization
# esp32_serial = serial.Serial('COM3', 9600, timeout=1)  # Change port if needed

def update_green_threshold(frame):
    global adaptive_green_threshold_low
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    current_brightness = np.mean(gray)
    target_threshold = current_brightness * 0.4
    adaptive_green_threshold_low = (1 - adaptive_learning_rate) * adaptive_green_threshold_low + \
                                   adaptive_learning_rate * target_threshold
    adaptive_green_threshold_low = max(40, min(150, adaptive_green_threshold_low))
    green_lower[2] = int(adaptive_green_threshold_low)
    green_upper[2] = 255

def create_tracker():
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    raise RuntimeError("CSRT tracker not available in your OpenCV installation.")

def create_multitracker():
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "MultiTracker_create"):
        return cv2.legacy.MultiTracker_create()
    if hasattr(cv2, "MultiTracker_create"):
        return cv2.MultiTracker_create()
    raise RuntimeError("MultiTracker not available in your OpenCV installation.")

class ShellGame:
    def __init__(self, esp32_client=None):
        self.esp32_client = esp32_client  # Store the ESP32 client if provided
        self.ball_position = None
        self.ball_under_cup = None
        self.last_nearest_cup = None
        self.multi_tracker = None
        self.cups = None
        self.last_motion_times = [time.time(), time.time(), time.time()]
        self.last_cup_centers = [None, None, None]
        self.motion_threshold = 10
        self.motion_timeout = 10
        self.game_paused = False
        self.missing_cup_index = None
        self.game_ended = False
        self.last_wanted_cup_idx = None
        self.last_moved_cup_idx = None
        self.last_ball_under_cup_idx = None
        self.stopped = False
        self.video_stream = VideoStream(IPCAM_URL)
        self.frame_queue = queue.Queue(maxsize=100)  # For HTTP streaming
        self.arm_command_sent = False
        self.last_sent_cup = None
        time.sleep(1)  # Give the IP camera time to initialize

    def stop(self):
        """Explicitly stop and release the camera."""
        if not self.stopped:
            self.stopped = True
            if self.video_stream is not None:
                self.video_stream.release()
                self.video_stream = None

    def get_ipcam_frame(self):
        if self.stopped:
            return None
        if not self.video_stream:
            self.video_stream = VideoStream(IPCAM_URL)
            time.sleep(1)
        ret, frame = self.video_stream.read()
        if not ret or frame is None or frame.size == 0:
            return None
        return frame

    def detect_cups(self, frame):
        update_green_threshold(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, green_lower, green_upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cups = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500 and len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                (center, axes, angle) = ellipse
                major_axis = max(axes)
                minor_axis = min(axes)
                if 0.5 < (minor_axis/major_axis) < 1.5:
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    if hull_area > 0:
                        solidity = float(area)/hull_area
                        if solidity > 0.8:
                            cups.append(ellipse)
        cups = sorted(cups, key=lambda x: x[0][0])[:3]
        return cups

    def detect_ball(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, ball_lower, ball_upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
            if radius > 5:
                return (int(x), int(y))
        return None

    def check_ball_under_cup(self, ball_pos, cup_ellipses):
        if not hasattr(self, "last_ball_visible"):
            self.last_ball_visible = True
        if not hasattr(self, "last_moved_cup_before_hidden"):
            self.last_moved_cup_before_hidden = None

        if ball_pos is None:
            if self.last_ball_visible:
                self.last_moved_cup_before_hidden = self.last_moved_cup_idx
                self.last_ball_visible = False
            return self.last_moved_cup_before_hidden
        else:
            self.last_ball_visible = True

        min_distance = float('inf')
        nearest_cup_idx = None
        for i, ellipse in enumerate(cup_ellipses):
            center = ellipse[0]
            distance = np.linalg.norm(np.array(ball_pos) - np.array(center))
            if distance < min_distance:
                min_distance = distance
                nearest_cup_idx = i

        self.last_nearest_cup = nearest_cup_idx

        if nearest_cup_idx is not None:
            major_axis = max(cup_ellipses[nearest_cup_idx][1]) / 2
            if min_distance < major_axis * 1.2:
                self.ball_under_cup = nearest_cup_idx
                return nearest_cup_idx

        return self.ball_under_cup

    def check_cup_in_frame(self, box, frame_width, frame_height):
        x, y, w, h = box
        return (x + w/2 > 0 and y + h/2 > 0 and 
                x + w/2 < frame_width and y + h/2 < frame_height)

    def process_frame(self, frame_bytes=None):
        # Use persistent cap object
        frame = self.get_ipcam_frame()
        if frame is None:
            return {"error": "Could not retrieve frame from IP camera"}
        frame = cv2.resize(frame, (640, 480))
        raw_frame = frame.copy()
        frame_height, frame_width = frame.shape[:2]

        if self.multi_tracker is None:
            cups = self.detect_cups(frame)
            if len(cups) == 3:
                self.cups = cups
                self.multi_tracker = create_multitracker()
                for ellipse in cups:
                    center, axes, angle = ellipse
                    bbox = cv2.boundingRect(cv2.ellipse2Poly(
                        (int(center[0]), int(center[1])),
                        (int(axes[0]/2), int(axes[1]/2)),
                        int(angle), 0, 360, 1
                    ))
                    tracker = create_tracker()
                    self.multi_tracker.add(tracker, frame, bbox)
            else:
                _, raw_jpg = cv2.imencode('.jpg', raw_frame)
                raw_b64 = base64.b64encode(raw_jpg).decode("utf-8")
                return {
                    "status": "waiting",
                    "message": "Detecting cups...",
                    "raw_frame": raw_b64,
                    "processed_frame": raw_b64,
                }

        success, boxes = self.multi_tracker.update(frame)
        if success and not self.game_paused:
            cups_in_frame = [self.check_cup_in_frame(box, frame_width, frame_height) for box in boxes]
            if not all(cups_in_frame):
                self.missing_cup_index = cups_in_frame.index(False)
                self.game_paused = True

        if self.game_paused:
            cups_in_frame = [self.check_cup_in_frame(box, frame_width, frame_height) for box in boxes]
            if all(cups_in_frame):
                self.game_paused = False
                self.missing_cup_index = None

        ball_position = self.detect_ball(frame)
        self.ball_position = ball_position

        if ball_position:
            cv2.circle(frame, ball_position, 10, (0, 0, 255), 2)
            cv2.putText(
                frame,
                "Ball",
                (ball_position[0] - 20, ball_position[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

        cup_ellipses = []
        valid_boxes = []
        for i, (x, y, w, h) in enumerate(boxes):
            if self.game_paused and i == self.missing_cup_index:
                continue
            center = (int(x + w/2), int(y + h/2))
            axes = (int(w/2), int(h/2))
            angle = 0
            cup_ellipses.append((center, axes, angle))
            valid_boxes.append((x, y, w, h))

        # --- Determine cup positions (left/middle/right) ---
        cup_positions = []
        for i, (x, y, w, h) in enumerate(valid_boxes):
            center_x = int(x + w/2)
            cup_positions.append((center_x, i))
        cup_positions_sorted = sorted(cup_positions, key=lambda tup: tup[0])
        cup_names = ["left", "middle", "right"]
        cup_idx_to_name = {}
        for idx, (_, i) in enumerate(cup_positions_sorted):
            cup_idx_to_name[i] = cup_names[idx]

        for i, (x, y, w, h) in enumerate(valid_boxes):
            center = (int(x + w/2), int(y + h/2))
            axes = (int(w/2), int(h/2))
            if self.last_cup_centers[i] is not None:
                dist = np.linalg.norm(np.array(center) - np.array(self.last_cup_centers[i]))
                if dist > self.motion_threshold:
                    self.last_motion_times[i] = time.time()
                    self.last_moved_cup_idx = i
            self.last_cup_centers[i] = center
            if not self.game_ended:
                cv2.ellipse(frame, center, axes, 0, 0, 360, (255, 0, 255), 2)
                cv2.putText(frame, f"Cup {i}", (center[0] - 20, center[1] - max(axes) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if not self.game_paused and not self.game_ended:
            current_ball_cup_idx = self.check_ball_under_cup(self.ball_position, cup_ellipses)
            if current_ball_cup_idx is not None:
                x, y, w, h = [int(v) for v in valid_boxes[current_ball_cup_idx]]
                center = (int(x + w/2), int(y + h/2))
                axes = (int(w/2), int(h/2))
                name = cup_idx_to_name.get(current_ball_cup_idx, f"Cup {current_ball_cup_idx}")
                cv2.putText(frame, f"{name} (Ball!)", (center[0] - 60, center[1] - max(axes) - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.ellipse(frame, center, axes, 0, 0, 360, (0, 255, 255), 3)
                print(name)  # <-- Print the cup name to the terminal

        if success and not self.game_paused:
            ball_under_cup_idx = self.check_ball_under_cup(ball_position, cup_ellipses)
            if ball_under_cup_idx is not None:
                self.last_ball_under_cup_idx = ball_under_cup_idx

        if success and not self.game_paused and not self.game_ended:
            now = time.time()
            if all(now - t > self.motion_timeout for t in self.last_motion_times):
                self.game_ended = True
                self.last_wanted_cup_idx = self.last_moved_cup_idx

        # --- SERIAL ARM CONTROL LOGIC (after game ends) ---
        if self.game_ended and not self.arm_command_sent:
            cup_positions = []
            for i, (x, y, w, h) in enumerate(boxes):
                center_x = int(x + w/2)
                cup_positions.append((center_x, i))
            cup_positions_sorted = sorted(cup_positions, key=lambda tup: tup[0])
            cup_names = ["Left Cup", "Middle Cup", "Right Cup"]
            cup_idx_to_name = {}
            for idx, (_, i) in enumerate(cup_positions_sorted):
                cup_idx_to_name[i] = cup_names[idx]
            idx_to_lr = {i: lr for lr, (_, i) in enumerate(cup_positions_sorted)}

            detected_cup_idx = None
            if ball_position is None:
                detected_cup_idx = self.last_ball_under_cup_idx
            else:
                detected_cup_idx = self.check_ball_under_cup(ball_position, cup_ellipses)

            if detected_cup_idx is not None:
                wantedCup = idx_to_lr.get(detected_cup_idx, -1)
            else:
                wantedCup = -1

            if wantedCup in [0, 1, 2]:
                # Define positions for each cup
                if wantedCup == 0:
                    cup_pos = "150,140,155"
                    lift_pos = "150,140,80"
                elif wantedCup == 1:
                    cup_pos = "105,85,130"
                    lift_pos = "145,85,90"
                elif wantedCup == 2:
                    cup_pos = "150,35,155"
                    lift_pos = "150,35,100"
                else:
                    cup_pos = None
                    lift_pos = None

                if cup_pos and lift_pos:
                    # ARM CONTROL: Use esp32_client instead of serial, run async in background
                    asyncio.create_task(self.send_arm_sequence(cup_pos, lift_pos))
                    self.arm_command_sent = True
                    self.last_sent_cup = wantedCup

        # Encode both frames to base64 JPEG
        _, raw_jpg = cv2.imencode('.jpg', raw_frame)
        _, processed_jpg = cv2.imencode('.jpg', frame)
        raw_b64 = base64.b64encode(raw_jpg).decode("utf-8")
        processed_b64 = base64.b64encode(processed_jpg).decode("utf-8")

        # --- STREAM: Put processed frame in queue for HTTP streaming ---
        try:
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put_nowait(processed_b64)
        except Exception as qerr:
            print(f"ShellGame frame queue error: {qerr}")

        resp = {
            "status": "ok" if not self.game_ended else "ended",
            "ball_position": ball_position,
            "ball_under_cup": self.last_ball_under_cup_idx,
            "cups": [dict(center=ellipse[0], axes=ellipse[1], angle=ellipse[2]) for ellipse in cup_ellipses],
            "raw_frame": raw_b64,
            "processed_frame": processed_b64,
        }
        return resp

    async def send_arm_sequence(self, cup_pos, lift_pos):
        print(cup_pos)
        print(lift_pos)
        # Helper to send the arm sequence using esp32_client (async)
        client = self.esp32_client if self.esp32_client is not None else esp32_client
        if not client or not client.connected:
            print("ESP32 client not connected, cannot send arm commands.")
            return
        await client.send_command(f"{cup_pos},0")
        await asyncio.sleep(2.0)
        await client.send_command(f"{cup_pos},1")
        await asyncio.sleep(2.0)
        await client.send_command(f"{lift_pos},1")
        await asyncio.sleep(2.0)
        await client.send_command(f"{cup_pos},1")
        await asyncio.sleep(2.0)
        await client.send_command(f"{cup_pos},0")
        await asyncio.sleep(2.0)
        await client.send_command(f"{lift_pos},0")
        await client.send_command("180,90,0,0")
        await asyncio.sleep(2.0)

    def get_stream_generator(self):
        """
        Yields multipart JPEG frames for HTTP streaming.
        """
        boundary = "frame"
        while not self.stopped:
            try:
                b64_frame = self.frame_queue.get(timeout=2)
                jpg_bytes = base64.b64decode(b64_frame)
                yield (
                    b"--%b\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: %d\r\n\r\n" % (boundary.encode(), len(jpg_bytes))
                )
                yield jpg_bytes
                yield b"\r\n"
            except queue.Empty:
                continue
            except Exception as e:
                print(f"ShellGame stream generator error: {e}")
                break
        print("ShellGame stream generator exiting.")

    def __del__(self):
        self.stop()

