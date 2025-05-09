import cv2
import numpy as np
from ultralytics import YOLO
import threading
import serial
import time
import math
import webcolors
from scipy.spatial import KDTree

# Set prediction confidence threshold (adjust as needed)
CONF_THRESHOLD = 0.7

# Depth estimation parameters
FOCAL_LENGTH = 580  # in pixels (adjust based on your camera)
BALLOON_WIDTH = 0.18  # in meters (e.g., 30 cm, adjust based on your balloon size)

TARGET_COLOR = "yellow"    # Target color wanted

IMAGE_WIDTH = 640 
IMAGE_HEIGHT = 480

X_CAMERA_FOV = 86  # in degrees
Y_CAMERA_FOV = 53  # in degrees

# Shooter offset from camera (cm)
LASER_OFFSET_CM_X = 4   # Example: 5cm to the right of the camera
LASER_OFFSET_CM_Y = 18  # 7cm below the camera

# PID-like control parameters (ADJUST THESE TO TUNE PERFORMANCE)
# Proportional gain - how quickly to move toward target (higher = faster but may overshoot)
KP_X = 0.05  # For pan control
KP_Y = 0.05  # For tilt control

# Current servo positions (start at center positions)
current_pan = 90
current_tilt = 90

# Centering tolerance - how close to center we need to be (in pixels)
CENTER_TOLERANCE = 10  # Reduced from 15 to 10 for stricter locking

# Max movement per iteration to prevent jerky motion
MAX_ANGLE_CHANGE = 5

depth = 150  # Initial depth estimate in cm

shot_balloons = []  # Stores (x, y) tuples of previously shot balloons

# Modify the global variable to store angles instead of coordinates
shot_angles = []  # Stores (pan, tilt) tuples of previously shot positions

# Add this constant at the top with other constants
INIT_PAN = 90
INIT_TILT = 90

def is_angle_already_shot(pan, tilt, threshold=5):

    """
    Check if we've already shot at these angles
    threshold: angle difference tolerance in degrees
    """
    for shot_pan, shot_tilt in shot_angles:
        if (abs(pan - shot_pan) < threshold and 
            abs(tilt - shot_tilt) < threshold):
            return True
    return False

# Check if balloon is already shot before
def is_balloon_already_shot(center_x, center_y, threshold=50):
    for shot_x, shot_y in shot_balloons:
        distance = math.sqrt((center_x - shot_x) ** 2 + (center_y - shot_y) ** 2)
        if distance < threshold:
            return True
    return False

def send_to_arduino(arduino, pan_angle, tilt_angle, timeout=50000):
    if arduino is None:
        print("Arduino not connected.")
        return False
    try:
        # Send data
        data_str = f"{pan_angle},{tilt_angle}\n"
        print(f"Sending to Arduino: {data_str.strip()}")
        arduino.write(data_str.encode('utf-8'))
        time.sleep(0.05)  # Allow Arduino to respond
        return True
    except Exception as e:
        print(f"Serial communication error: {e}")
        return False

# Add a function to wait for an "ACK" message from the Arduino
def wait_for_ack(arduino, timeout=40):
    """Wait for an ACK message from the Arduino."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if arduino.in_waiting > 0:
            response = arduino.readline().decode('utf-8').strip()
            if response == "ACK":
                print("ACK received from Arduino.")
                return True
    print("Timeout waiting for ACK from Arduino.")
    return False

# Calculate error from center of the frame
def calculate_error(target_x, target_y):
    center_x = IMAGE_WIDTH / 2
    center_y = IMAGE_HEIGHT / 2
    
    # Calculate pixel error
    error_x = target_x - center_x
    error_y = target_y - center_y
    
    return error_x, error_y

# Calculate new pan/tilt angles based on error
def calculate_new_angles(error_x, error_y):
    global current_pan, current_tilt
    
    # Calculate angle adjustments based on error and gain factors
    # Negative error_x means target is to the left, so decrease pan angle
    # Positive error_x means target is to the right, so increase pan angle
    pan_adjustment = -KP_X * error_x
    
    # Negative error_y means target is above center, so decrease tilt angle
    # Positive error_y means target is below center, so increase tilt angle
    tilt_adjustment = KP_Y * error_y
    
    # Limit the max adjustment per iteration for smooth movement
    pan_adjustment = max(-MAX_ANGLE_CHANGE, min(MAX_ANGLE_CHANGE, pan_adjustment))
    tilt_adjustment = max(-MAX_ANGLE_CHANGE, min(MAX_ANGLE_CHANGE, tilt_adjustment))
    
    # Calculate new angles
    new_pan = current_pan + pan_adjustment
    new_tilt = current_tilt + tilt_adjustment
    
    # Ensure angles stay within servo limits
    new_pan = max(0, min(180, new_pan))
    new_tilt = max(0, min(180, new_tilt))
    
    return new_pan, new_tilt

# Helper function to determine a basic color name from BGR values
def get_color_name(bgr):
    # Convert BGR to RGB for easier interpretation
    r, g, b = bgr[2], bgr[1], bgr[0]
    # Simple heuristic to determine the color name
    if r > 150 and g > 70 and g<150 and b > 70 and b<150:
        return "red"
    elif r < 140 and g > 150 and b < 170:
        return "green"
    elif r < 175 and g < 175 and b > 150:
        return "blue"
    elif r > 160 and g > 140 and  b < 100:
        return "yellow"
    elif r < 50 and g < 50 and b < 50:
        return "black"
    elif r > 200 and g > 200 and b > 200:
        return "white"
    else:
        # Return the RGB tuple if no basic color is matched
        return f"rgb({int(r)}, {int(g)}, {int(b)})"

# Check if target is centered within tolerance
def is_target_centered(error_x, error_y):
    return abs(error_x) < CENTER_TOLERANCE and abs(error_y) < CENTER_TOLERANCE

# Load your trained YOLO model \\
model = YOLO(r'/home/ahmed-fathy0_0/Documents/CMP27/Micro/Memory-Matching-Game-/docker-it/backend/games/TargetDetection/runs/detect/train/weights/best.pt')

# Arduino Serial Setup
try:
    arduino = serial.Serial('COM8', 9600, timeout=1)
    time.sleep(2)  # Wait for the connection to establish
    print("Arduino connected successfully.")
except Exception as e:
    print(f"Could not open serial port: {e}")
    arduino = None

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

# Open the IP camera stream
ip_camera_url = 'http://192.168.55.215:4747/video'  # Example for IP Webcam app
cap = VideoStream(ip_camera_url)

# Draw crosshair in the center of the frame
def draw_crosshair(frame, color=(0, 0, 255), size=20, thickness=2):
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2

    # Horizontal line
    cv2.line(frame, (center_x - size, center_y), (center_x + size, center_y), color, thickness)
    # Vertical line
    cv2.line(frame, (center_x, center_y - size), (center_x, center_y + size), color, thickness)

    # Calculate pixel offsets based on FOV and depth
    fov_rad_x = math.radians(X_CAMERA_FOV)
    fov_rad_y = math.radians(Y_CAMERA_FOV)

    # Calculate width and height at the target depth
    width_at_target = 2 * depth * math.tan(fov_rad_x / 2)
    height_at_target = 2 * depth * math.tan(fov_rad_y / 2)

    # Pixels per cm for X and Y axes
    pixels_per_cm_x = IMAGE_WIDTH / width_at_target
    pixels_per_cm_y = IMAGE_HEIGHT / height_at_target

    # Calculate shooter position offsets in pixels
    shooter_x_offset = int(LASER_OFFSET_CM_X * pixels_per_cm_x)
    shooter_y_offset = int(LASER_OFFSET_CM_Y * pixels_per_cm_y)

    # Adjust shooter position
    shooter_x = center_x + shooter_x_offset
    shooter_y = center_y + shooter_y_offset

    # Draw shooter position indicator
    cv2.circle(frame, (shooter_x, shooter_y), 5, (0, 255, 255), -1)
    cv2.line(frame, (shooter_x - 10, shooter_y), (shooter_x + 10, shooter_y), (0, 255, 255), 2)
    cv2.line(frame, (shooter_x, shooter_y - 10), (shooter_x, shooter_y + 10), (0, 255, 255), 2)

# Time tracking for movement cooldown
last_movement_time = 0
MOVEMENT_COOLDOWN = 0.2  # seconds between movements
send_to_arduino(arduino, 90,90);

NO_BALLOON_TIMEOUT = 10  # seconds
last_balloon_detected_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    current_time = time.time()
    balloon_detected = False  # Flag to track if we see any unshot balloons

    # Run inference on the current frame
    results = model.predict(source=frame, show=False)

    # Draw crosshair to show center of frame
    draw_crosshair(frame)

    # List to store detected balloon info for display
    balloons_info = []
    target_found = False
    best_target = None
    best_confidence = 0

    # Process each result
    for result in results:
        # Skip if there are no detected boxes
        if result.boxes is None or len(result.boxes) == 0:
            continue

        # Convert detection data to numpy arrays
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        cls_ids = result.boxes.cls.cpu().numpy()  # Class indices
        confs = result.boxes.conf.cpu().numpy()  # Confidence scores

        # Loop through each detected object
        for i, box in enumerate(boxes):
            # Apply confidence threshold filter
            if confs[i] < CONF_THRESHOLD:
                continue

            # Get label name
            label = model.names[int(cls_ids[i])]
            # Only process detections labeled as 'balloon'
            if label.lower() != 'balloon':
                continue

            # Convert bounding box coordinates to integers
            x1, y1, x2, y2 = box.astype(int)

            # Calculate the center of the bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Calculate angles for this balloon position BEFORE any movement
            error_x, error_y = calculate_error(center_x, center_y)
            target_pan, target_tilt = calculate_new_angles(error_x, error_y)

            # Skip this balloon immediately if we've already shot at similar angles
            if is_angle_already_shot(target_pan, target_tilt, threshold=15):
                # Draw red box around ignored balloon
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Already Shot", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                continue

            # If we get here, we've found an unshot balloon
            balloon_detected = True
            last_balloon_detected_time = current_time

            # Calculate pixel width of the balloon
            pixel_width = x2 - x1

            # Estimate depth: Z = (f * W) / w
            estimated_depth = (FOCAL_LENGTH * BALLOON_WIDTH) / pixel_width if pixel_width > 0 else 0.0
            estimated_depth *= 100  # Convert to cm
            
            # Update global depth if this is a good measurement
            if 30 < estimated_depth < 1000:  # Sanity check on depth
                depth = estimated_depth

            # Extract the region of interest (ROI)
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # Compute the average BGR color in the ROI
            mean_color = [int(round(x)) for x in cv2.mean(roi)[:3]]
            color_name = get_color_name(mean_color)

            print(color_name)

            if(color_name != TARGET_COLOR):
                continue

            # Prepare text info
            info_text = f"{label} ({color_name}) {confs[i]:.2f} Pos: ({center_x}, {center_y}) D:{estimated_depth:.1f}cm"
            
            # Store balloon info
            balloons_info.append({
                'box': (x1, y1, x2, y2),
                'color': color_name,
                'conf': confs[i],
                'pos': (center_x, center_y),
                'depth': estimated_depth
            })
            
            # Draw the bounding box
            color = (0, 255, 0)  # Default green box
            if confs[i] > best_confidence:
                best_confidence = confs[i]
                best_target = {
                    'pos': (center_x, center_y),
                    'box': (x1, y1, x2, y2),
                    'info': info_text
                }
                target_found = True
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, info_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Check for timeout
    if current_time - last_balloon_detected_time > NO_BALLOON_TIMEOUT:
        print(f"No unshot balloons detected for {NO_BALLOON_TIMEOUT} seconds. Terminating...")
        break

    # If we found a target balloon, track it
    current_time = time.time()
    if target_found and (current_time - last_movement_time) > MOVEMENT_COOLDOWN:
        last_movement_time = current_time
        
        center_x, center_y = best_target['pos']
        
        # Calculate error from center
        error_x, error_y = calculate_error(center_x, center_y)
        
        # Draw a line from center to target
        frame_center_x, frame_center_y = IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2
        cv2.line(frame, (frame_center_x, frame_center_y), (center_x, center_y), (255, 0, 0), 2)
        
        # Check if target is centered within tolerance
        if is_target_centered(error_x, error_y):
            # Target is centered - ready to shoot!
            cv2.putText(frame, "TARGET LOCKED!", (frame_center_x - 80, frame_center_y - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Check if we've already shot at these angles
            if not is_angle_already_shot(current_pan, current_tilt):
                print(f"FIRE! at angles: Pan={current_pan:.1f}, Tilt={current_tilt:.1f}")
                print(f"Estimated Depth: {estimated_depth:.1f} cm")
                
                # Send shoot command to Arduino
                if arduino is not None:
                    arduino.write("SHOOT\n".encode('utf-8'))
                    print("Shoot command sent to Arduino.")
                    
                    # Wait for ACK from Arduino
                    if wait_for_ack(arduino):
                        print("Balloon successfully shot.")
                        shot_angles.append((current_pan, current_tilt))
                        
                        # Return to initial position
                        print("Returning to initial position...")
                        current_pan = INIT_PAN
                        current_tilt = INIT_TILT
                        send_to_arduino(arduino, INIT_PAN, INIT_TILT)
                        time.sleep(RETURN_TO_CENTER_DELAY)  # Give time for the servos to move and system to stabilize
                        
                        # Reset the balloon detection timeout after movement
                        last_balloon_detected_time = time.time()
                        balloon_detected = False
                        
                    else:
                        print("Failed to receive ACK. Retrying...")
        else:
            # Calculate new pan/tilt angles to center the target
            new_pan, new_tilt = calculate_new_angles(error_x, error_y)
            
            # Only send to Arduino if values changed
            if abs(new_pan - current_pan) > 0.5 or abs(new_tilt - current_tilt) > 0.5:
                # Update current position
                current_pan = new_pan
                current_tilt = new_tilt
                
                # Send movement command to Arduino
                send_to_arduino(arduino, int(round(current_pan)), int(round(current_tilt)))
                
                # Display movement info
                cv2.putText(frame, f"Moving: Pan={int(current_pan)} Tilt={int(current_tilt)}", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(frame, f"Error: X={int(error_x)} Y={int(error_y)}", 
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Display current servo positions and control parameters
    cv2.putText(frame, f"Pan: {int(current_pan)} Tilt: {int(current_tilt)}", 
                (10, IMAGE_HEIGHT - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"Kp_X: {KP_X:.3f} Kp_Y: {KP_Y:.3f}", 
                (10, IMAGE_HEIGHT - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Show the final frame with overlays
    cv2.imshow('YOLO Balloon Tracking', frame)

    # Exit the loop on pressing 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    # Adjust KP values with keyboard (for tuning)
    elif key == ord('a'):  # Decrease KP_X
        KP_X = max(0.001, KP_X - 0.01)
    elif key == ord('d'):  # Increase KP_X
        KP_X += 0.01
    elif key == ord('w'):  # Increase KP_Y
        KP_Y += 0.01
    elif key == ord('s'):  # Decrease KP_Y
        KP_Y = max(0.001, KP_Y - 0.01)

cap.release()
cv2.destroyAllWindows()
if arduino is not None:
    arduino.close()