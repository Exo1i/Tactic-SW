import cv2
import numpy as np
import base64
import time
import queue  # Add for frame streaming
import threading
import asyncio
# import serial  # REMOVE: No longer needed

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
IPCAM_URL = "http://192.168.49.1:4747/video"  # <-- Change to your webcam's IP

# Define flexible color range for green cups in HSV (for black background)
green_lower = np.array([30, 40, 80])
green_upper = np.array([90, 255, 255])

# --- Ball color: restricted red, avoid hand detection ---
# Red in HSV wraps around, so use two ranges
ball_lower1 = np.array([0, 140, 120])    # Lower HSV for red (low hue, high S/V)
ball_upper1 = np.array([10, 255, 255])
ball_lower2 = np.array([170, 140, 120])  # Upper HSV for red (high hue, high S/V)
ball_upper2 = np.array([180, 255, 255])

adaptive_green_threshold_low = 80
adaptive_learning_rate = 0.01

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
        self.motion_timeout = 10  # seconds before game ends due to inactivity
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
        self.switch_command_sent = False  # Add flag to track if switch command was sent
        self.pending_arm_sequence = None  # Store pending arm sequence
        time.sleep(1)  # Give the IP camera time to initialize

        # Schedule the switch command task during initialization
        if self.esp32_client is not None:
            try:
                # Try to get a running event loop or create one if needed
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    # No running event loop, create a new one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Schedule the switch command
                loop.create_task(self._send_switch_command())
                print("[ESP32] Scheduled switch command during initialization")
            except Exception as e:
                print(f"[ESP32] Failed to schedule switch command: {e}")

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
        # Red mask (combine two ranges)
        mask1 = cv2.inRange(hsv, ball_lower1, ball_upper1)
        mask2 = cv2.inRange(hsv, ball_lower2, ball_upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        # Morphological filtering to reduce noise (remove hand blobs)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=4)
        # Find contours and filter by area/circularity
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_circle = None
        best_score = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100 or area > 2500:
                continue  # Ignore too small/large (likely not the ball or hand)
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            if radius < 8 or radius > 40:
                continue  # Ignore unlikely radii
            # Circularity check
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity < 0.7:
                continue  # Not round enough
            # Optionally: restrict to lower part of frame (if ball is always on table)
            # if y < frame.shape[0] * 0.3:
            #     continue
            # Pick the most circular/likely candidate
            if circularity > best_score:
                best_score = circularity
                best_circle = (int(x), int(y))
        return best_circle

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
        if self.game_ended:
            # After game ends, skip all further processing and just return the final state/frames
            cup_ellipses = []
            valid_boxes = []
            for i, (x, y, w, h) in enumerate(boxes):
                center = (int(x + w/2), int(y + h/2))
                axes = (int(w/2), int(h/2))
                angle = 0
                cup_ellipses.append((center, axes, angle))
                valid_boxes.append((x, y, w, h))
            # Encode both frames to base64 JPEG
            _, raw_jpg = cv2.imencode('.jpg', raw_frame)
            _, processed_jpg = cv2.imencode('.jpg', frame)
            raw_b64 = base64.b64encode(raw_jpg).decode("utf-8")
            processed_b64 = base64.b64encode(processed_jpg).decode("utf-8")
            resp = {
                "status": "ended",
                "ball_position": self.ball_position,
                "ball_under_cup": self.last_ball_under_cup_idx,
                "cups": [dict(center=ellipse[0], axes=ellipse[1], angle=ellipse[2]) for ellipse in cup_ellipses],
                "raw_frame": raw_b64,
                "processed_frame": processed_b64,
                "cup_name_result": None,  # Will be set below if available
            }
            # Set cup_name_result if it was determined at end
            if hasattr(self, "latest_debug_state") and self.latest_debug_state.get("cup_name_result"):
                resp["cup_name_result"] = self.latest_debug_state["cup_name_result"]
            else:
                # Try to compute it one last time if not set
                cup_name_result = None
                cup_positions = []
                for i, (x, y, w, h) in enumerate(boxes):
                    center_x = int(x + w/2)
                    cup_positions.append((center_x, i))
                cup_positions_sorted = sorted(cup_positions, key=lambda tup: tup[0])
                cup_names = ["left", "middle", "right"]
                idx_to_lr = {i: lr for lr, (_, i) in enumerate(cup_positions_sorted)}
                detected_cup_idx = self.last_ball_under_cup_idx
                wantedCup = idx_to_lr.get(detected_cup_idx, -1) if detected_cup_idx is not None else -1
                if wantedCup in [0, 1, 2]:
                    cup_name_result = cup_names[wantedCup]
                resp["cup_name_result"] = cup_name_result
            self.latest_debug_state = {k: v for k, v in resp.items() if k != "processed_frame"}
            return resp

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
                    print(f"Cup {i} moved! Distance: {dist:.2f}px")
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
                # print(name)  # <-- Print the cup name to the terminal

        if success and not self.game_paused:
            ball_under_cup_idx = self.check_ball_under_cup(ball_position, cup_ellipses)
            if ball_under_cup_idx is not None:
                self.last_ball_under_cup_idx = ball_under_cup_idx

        if success and not self.game_paused and not self.game_ended:
            now = time.time()
            time_since_last_motion = [now - t for t in self.last_motion_times]
            
            # Add debug info to frame
            for i, elapsed in enumerate(time_since_last_motion):
                cup_label = f"Cup {i}: {elapsed:.1f}s"
                cv2.putText(frame, cup_label, (10, 30 + i * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Check if all cups haven't moved for longer than the timeout
            if all(elapsed > self.motion_timeout for elapsed in time_since_last_motion):
                print(f"Game ended: No cup has moved for {self.motion_timeout} seconds")
                for i, elapsed in enumerate(time_since_last_motion):
                    print(f"  Cup {i} last moved {elapsed:.2f} seconds ago")
                
                self.game_ended = True
                self.last_wanted_cup_idx = self.last_moved_cup_idx

        # --- SERIAL ARM CONTROL LOGIC (after game ends) ---
        cup_name_result = None  # <--- Add this
        if self.game_ended and not self.arm_command_sent:
            cup_positions = []
            for i, (x, y, w, h) in enumerate(boxes):
                center_x = int(x + w/2)
                cup_positions.append((center_x, i))
            cup_positions_sorted = sorted(cup_positions, key=lambda tup: tup[0])
            cup_names_full = ["Left Cup", "Middle Cup", "Right Cup"]
            cup_idx_to_name_full = {}
            for idx, (_, i) in enumerate(cup_positions_sorted):
                cup_idx_to_name_full[i] = cup_names_full[idx]
            idx_to_lr = {i: lr for lr, (_, i) in enumerate(cup_positions_sorted)}

            detected_cup_idx = None
            if ball_position is None:
                detected_cup_idx = self.last_ball_under_cup_idx
            else:
                detected_cup_idx = self.check_ball_under_cup(ball_position, cup_ellipses)

            if detected_cup_idx is not None:
                wantedCup = idx_to_lr.get(detected_cup_idx, -1)
                cup_name_result = cup_names[wantedCup] if wantedCup in [0, 1, 2] else None
            else:
                wantedCup = -1

            if wantedCup in [0, 1, 2]:
                # Define positions for each cup
                if wantedCup == 0:
                    cup_pos = "170,135,130"
                    lift_pos = "170,135,80"
                elif wantedCup == 1:
                    cup_pos = "150,90,110"
                    lift_pos = "180,90,80"
                elif wantedCup == 2:
                    cup_pos = "170,50,130"
                    lift_pos = "170,50,80"
                else:
                    cup_pos = None
                    lift_pos = None

                if cup_pos and lift_pos:
                    # Store the arm sequence coroutine instead of creating a task directly
                    self.pending_arm_sequence = (cup_pos, lift_pos)
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
            # --- Add cup_name_result to debug/game state ---
            "cup_name_result": cup_name_result if self.game_ended else None,
        }
        # --- Always update latest_debug_state here, so /shell-game/debug is correct ---
        self.latest_debug_state = {k: v for k, v in resp.items() if k != "processed_frame"}
        return resp

    async def send_arm_sequence(self, cup_pos, lift_pos):
        print(cup_pos)
        print(lift_pos)
        # Helper to send the arm sequence using esp32_client (async)
        client = self.esp32_client if self.esp32_client is not None else esp32_client
        if not client or not client.connected:
            print("ESP32 client not connected, cannot send arm commands.")
            return
        # Send properly formatted JSON commands instead of raw strings
        await client.send_json({
            "action": "command",
            "command": f"{cup_pos},0,0"
        })
        await asyncio.sleep(2.0)
        await client.send_json({
            "action": "command",
            "command": f"{cup_pos},1,0"
        })
        await asyncio.sleep(2.0)
        await client.send_json({
            "action": "command", 
            "command": f"{lift_pos},1,0"
        })
        await asyncio.sleep(2.0)
        await client.send_json({
            "action": "command",
            "command": f"{cup_pos},1,0"
        })
        await asyncio.sleep(2.0)
        await client.send_json({
            "action": "command",
            "command": f"{cup_pos},0,0"
        })
        await asyncio.sleep(2.0)
        await client.send_json({
            "action": "command",
            "command": f"{lift_pos},0,0"
        })
        await client.send_json({
            "action": "command",
            "command": "180,0,0,0,0"
        })
        await asyncio.sleep(2.0)

    async def send_livefeed_ws(self, websocket):
        """
        Continuously send the latest raw camera frame as a base64 string via WebSocket.
        """
        while not self.stopped:
            try:
                frame = self.get_ipcam_frame()
                if frame is not None:
                    frame = cv2.resize(frame, (640, 480))
                    _, raw_jpg = cv2.imencode('.jpg', frame)
                    raw_b64 = base64.b64encode(raw_jpg).decode("utf-8")
                    await websocket.send_json({"type": "livefeed", "payload": raw_b64})
                await asyncio.sleep(0.05)  # ~20 FPS
            except Exception as e:
                print(f"ShellGame livefeed WS error: {e}")
                break
        print("ShellGame livefeed WS loop exiting.")

    def get_stream_generator(self):
        """
        Yields multipart JPEG frames for HTTP streaming.
        Each time the frontend requests a frame (i.e., the MJPEG stream is open),
        this method grabs a fresh frame from the camera, processes it, and yields the result.
        Additionally, exposes debug/game state via a parallel HTTP endpoint.
        """
        boundary = "frame"
        while not self.stopped:
            try:
                result = self.process_frame()
                processed_b64 = result.get("processed_frame")
                
                # Check if there's a pending arm sequence to execute
                if self.pending_arm_sequence:
                    cup_pos, lift_pos = self.pending_arm_sequence
                    self.pending_arm_sequence = None
                    
                    # Run the arm sequence in a separate thread to avoid blocking the generator
                    loop = asyncio.new_event_loop()
                    
                    def run_arm_sequence():
                        asyncio.set_event_loop(loop)
                        try:
                            loop.run_until_complete(self.send_arm_sequence(cup_pos, lift_pos))
                        except Exception as e:
                            print(f"Error executing arm sequence: {e}")
                        finally:
                            loop.close()
                    
                    # Start the arm sequence in a separate thread
                    threading.Thread(target=run_arm_sequence, daemon=True).start()
                    
                if not processed_b64:
                    continue
                    
                jpg_bytes = base64.b64decode(processed_b64)
                yield (
                    b"--%b\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: %d\r\n\r\n" % (boundary.encode(), len(jpg_bytes))
                )
                yield jpg_bytes
                yield b"\r\n"
                
                # Store latest debug/game state for a parallel endpoint
                self.latest_debug_state = {k: v for k, v in result.items() if k != "processed_frame"}
            except Exception as e:
                print(f"ShellGame stream generator error: {e}")
                break
        print("ShellGame stream generator exiting.")

    def get_latest_debug_state(self):
        """Return the latest debug/game state (excluding processed_frame)."""
        return getattr(self, "latest_debug_state", {})

    def __del__(self):
        self.stop()

# Add this function to handle running a coroutine in a thread
def run_async_in_thread(coroutine):
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coroutine)
    finally:
        loop.close()