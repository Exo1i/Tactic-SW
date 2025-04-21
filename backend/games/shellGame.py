import cv2
import numpy as np
import base64
import concurrent.futures

# Define color range for white cups in HSV
white_lower = np.array([0, 0, 200])  # Lower HSV for white (low saturation, high value)
white_upper = np.array([180, 50, 255])  # Upper HSV for white

# Define color range for the green ball
ball_lower = np.array([35, 100, 100])  # Lower HSV for green
ball_upper = np.array([85, 255, 255])  # Upper HSV for green

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

class GameSession:
    def __init__(self):
        self.ball_position = None
        self.ball_under_cup = None
        self.last_nearest_cup = None
        self.multi_tracker = None
        self.cups = None

    def detect_cups(self, frame):
        """Detect white cups using color segmentation and contour filtering."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, white_lower, white_upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cups = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter out small contours
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                cups.append((center, radius))
        # Sort cups by x-coordinate and select the top 3
        cups = sorted(cups, key=lambda x: x[0][0])[:3]
        return cups

    def detect_ball(self, frame):
        """Detect ball position using color detection."""
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

    def check_ball_under_cup(self, ball_pos, cup_circles):
        """Check if ball is under a cup based on position or last known state."""
        if ball_pos is None:
            # If the ball disappears, assume it is under the last nearest cup
            return self.last_nearest_cup

        # Find the nearest cup to the ball
        min_distance = float("inf")
        nearest_cup_idx = None
        for i, (center, radius) in enumerate(cup_circles):
            distance = np.linalg.norm(np.array(ball_pos) - np.array(center))
            if distance < min_distance:
                min_distance = distance
                nearest_cup_idx = i

        # Update the last nearest cup
        self.last_nearest_cup = nearest_cup_idx

        # Check if the ball is under the nearest cup
        if min_distance < cup_circles[nearest_cup_idx][1]:
            self.ball_under_cup = nearest_cup_idx
            return nearest_cup_idx

        return self.ball_under_cup

    def process_frame(self, frame_bytes):
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "Invalid frame"}
        frame = cv2.resize(frame, (640, 480))
        raw_frame = frame.copy()

        # Detect cups and initialize tracker if not already done
        if self.multi_tracker is None:
            cups = self.detect_cups(frame)
            if len(cups) == 3:
                self.cups = cups
                self.multi_tracker = create_multitracker()
                for center, radius in cups:
                    x, y = center
                    bbox = (x - radius, y - radius, 2 * radius, 2 * radius)
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

        # Update cup positions
        success, boxes = self.multi_tracker.update(frame)
        ball_position = self.detect_ball(frame)
        self.ball_position = ball_position

        # Draw overlays on processed frame
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
        cup_circles = [
            ((int(x + w / 2), int(y + h / 2)), int((w + h) / 4))
            for (x, y, w, h) in boxes
        ]
        if success:
            for i, (x, y, w, h) in enumerate(boxes):
                center = (int(x + w / 2), int(y + h / 2))
                radius = int((w + h) / 4)
                cv2.circle(frame, center, radius, (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Cup {i}",
                    (center[0] - 20, center[1] - radius - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

        # Determine which cup has the ball
        ball_under_cup_idx = self.check_ball_under_cup(ball_position, cup_circles)
        if ball_under_cup_idx is not None and ball_position is None and len(boxes) > ball_under_cup_idx:
            x, y, w, h = [int(v) for v in boxes[ball_under_cup_idx]]
            center = (int(x + w / 2), int(y + h / 2))
            cv2.putText(
                frame,
                "Ball is here!",
                (center[0] - 50, center[1] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        # Encode both frames to base64 JPEG
        _, raw_jpg = cv2.imencode('.jpg', raw_frame)
        _, processed_jpg = cv2.imencode('.jpg', frame)
        raw_b64 = base64.b64encode(raw_jpg).decode("utf-8")
        processed_b64 = base64.b64encode(processed_jpg).decode("utf-8")

        resp = {
            "status": "ok",
            "ball_position": ball_position,
            "ball_under_cup": ball_under_cup_idx,
            "cups": [dict(center=center, radius=radius) for center, radius in cup_circles],
            "raw_frame": raw_b64,
            "processed_frame": processed_b64,
        }
        return resp
