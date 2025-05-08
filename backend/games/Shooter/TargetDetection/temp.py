import cv2
import numpy as np
from ultralytics import YOLO
import threading
import serial
import time
import math
from scipy.spatial import KDTree

# VideoStream class for threaded camera reading
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
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)  # Add timeout to prevent hanging
        if hasattr(self, 'cap'):
            self.cap.release()

class GameSession:
    def __init__(self, config=None):
        # --- Default constants ---
        self.CONF_THRESHOLD = 0.7
        self.FOCAL_LENGTH = 580  # In pixels
        self.BALLOON_WIDTH = 0.18  # In meters
        self.TARGET_COLOR = "red"
        self.IMAGE_WIDTH = 640
        self.IMAGE_HEIGHT = 480
        self.X_CAMERA_FOV = 86  # Degrees
        self.Y_CAMERA_FOV = 53  # Degrees
        self.LASER_OFFSET_CM_X = 5
        self.LASER_OFFSET_CM_Y = 18
        self.KP_X = 0.05
        self.KP_Y = 0.05
        self.CENTER_TOLERANCE = 10  # Pixels
        self.MAX_ANGLE_CHANGE = 5   # Degrees
        self.INIT_PAN = 90
        self.INIT_TILT = 90
        
        # --- Runtime variables ---
        self.current_pan = self.INIT_PAN
        self.current_tilt = self.INIT_TILT
        self.depth = 150  # Initial depth estimate in cm
        self.shot_angles = [] 
        self.shot_balloons = []  # Keep this for compatibility
        self.last_movement_time = 0
        self.MOVEMENT_COOLDOWN = 0.2 # Seconds

        # --- Camera setup ---
        self.ip_camera_url = 'http://192.168.187.44:4747/video'  # Hard-coded IP camera URL 
        self.cap = None  # Will be initialized as VideoStream
        self.initialize_camera()
        
        # --- Local Preview ---
        self.enable_local_preview = False
        self.local_preview_window_name = "Backend Shooter Preview"

        if config and isinstance(config, dict):
            print(f"Initializing Shooter Game with config: {config}")
            self.TARGET_COLOR = config.get("color", self.TARGET_COLOR).lower()
            self.LASER_OFFSET_CM_X = float(config.get("shooterOffsetX", self.LASER_OFFSET_CM_X))
            self.LASER_OFFSET_CM_Y = float(config.get("shooterOffsetY", self.LASER_OFFSET_CM_Y))
            self.FOCAL_LENGTH = float(config.get("focalLength", self.FOCAL_LENGTH))
            self.KP_X = float(config.get("kp_x", self.KP_X)) 
            self.KP_Y = float(config.get("kp_y", self.KP_Y))
            self.enable_local_preview = config.get("enable_local_preview", False)
            print(f"Local backend preview: {'Enabled' if self.enable_local_preview else 'Disabled'}")
            
            # Check if custom camera URL is provided
            if "camera_url" in config:
                self.ip_camera_url = config["camera_url"]
                print(f"Using custom camera URL: {self.ip_camera_url}")
                # Re-initialize camera with new URL
                if self.cap:
                    self.cap.release()
                self.initialize_camera()
        
        try:
            model_path = r"D:\\projects\\micro project sw integration\\docker-it\\backend\\games\\Shooter\\TargetDetection\\runs\\detect\\train\\weights\\best.pt"
            self.model = YOLO(model_path)
            print("YOLO model loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model = None
            
        try:
            arduino_port = 'COM8'
            self.arduino = serial.Serial(arduino_port, 9600, timeout=1)
            time.sleep(2) 
            self.send_to_arduino(self.INIT_PAN, self.INIT_TILT) 
            print(f"Arduino connected successfully on {arduino_port} and initialized.")
        except Exception as e:
            print(f"Could not open serial port {arduino_port}: {e}")
            self.arduino = None

        if self.enable_local_preview:
            cv2.namedWindow(self.local_preview_window_name, cv2.WINDOW_NORMAL)
    
    def initialize_camera(self):
        """Initialize the camera connection using VideoStream"""
        try:
            print(f"Initializing camera with URL: {self.ip_camera_url}")
            self.cap = VideoStream(self.ip_camera_url)
            # Give it a moment to start capturing frames
            time.sleep(1)
            # Test if we can read a frame
            ret, _ = self.cap.read()
            if ret:
                print("Camera initialized successfully.")
            else:
                print("Warning: Camera returned empty frame on initialization.")
        except Exception as e:
            print(f"Error initializing camera: {e}")
            self.cap = None

    # --- All original helper functions as methods (with self) ---
    def is_angle_already_shot(self, pan, tilt, threshold=10):
        for shot_pan, shot_tilt in self.shot_angles:
            if (abs(pan - shot_pan) < threshold and abs(tilt - shot_tilt) < threshold):
                return True
        return False

    def is_balloon_already_shot(self, center_x, center_y, threshold=50):
        # Keep this for compatibility
        for shot_x, shot_y in self.shot_balloons:
            distance = math.sqrt((center_x - shot_x) ** 2 + (center_y - shot_y) ** 2)
            if distance < threshold:
                return True
        return False

    def send_to_arduino(self, pan_angle, tilt_angle):
        if self.arduino is None:
            return False
        try:
            data_str = f"{int(round(pan_angle))},{int(round(tilt_angle))}\n"
            print(f"Sending to Arduino: {data_str.strip()}")
            self.arduino.write(data_str.encode('utf-8'))
            time.sleep(0.05) 
            return True
        except Exception as e:
            print(f"Serial communication error: {e}")
            return False

    def wait_for_ack(self, timeout_seconds=5): 
        if self.arduino is None:
            return False
        start_time = time.time()
        buffer = ""
        while time.time() - start_time < timeout_seconds:
            if self.arduino.in_waiting > 0:
                buffer += self.arduino.read(self.arduino.in_waiting).decode('utf-8', errors='ignore')
                if "ACK" in buffer:
                    print("ACK received from Arduino.")
                    return True
            time.sleep(0.01) 
        print("Timeout waiting for ACK from Arduino.")
        return False

    def calculate_error(self, target_x, target_y):
        center_x = self.IMAGE_WIDTH / 2
        center_y = self.IMAGE_HEIGHT / 2
        error_x = target_x - center_x
        error_y = target_y - center_y
        return error_x, error_y

    def calculate_new_angles(self, error_x, error_y):
        pan_adjustment = -self.KP_X * error_x
        tilt_adjustment = self.KP_Y * error_y 
        pan_adjustment = max(-self.MAX_ANGLE_CHANGE, min(self.MAX_ANGLE_CHANGE, pan_adjustment))
        tilt_adjustment = max(-self.MAX_ANGLE_CHANGE, min(self.MAX_ANGLE_CHANGE, tilt_adjustment))
        new_pan = self.current_pan + pan_adjustment
        new_tilt = self.current_tilt + tilt_adjustment
        new_pan = max(0, min(180, new_pan)) 
        new_tilt = max(0, min(180, new_tilt))
        return new_pan, new_tilt

    def get_color_name(self, bgr):
        # Convert BGR to RGB for easier interpretation
        r, g, b = bgr[2], bgr[1], bgr[0]
        # Simple heuristic to determine the color name
        if r > 150 and g > 70 and g<120 and b > 70 and b<120:
            return "red"
        elif r > 140 and g > 150 and b < 150 and b<200:
            return "green"
        elif r < 100 and g < 160 and b > 170:
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

    def is_target_centered(self, error_x, error_y):
        return abs(error_x) < self.CENTER_TOLERANCE and abs(error_y) < self.CENTER_TOLERANCE

    def draw_crosshair(self, frame, color=(0, 0, 255), size=20, thickness=2):
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2

        # Horizontal line
        cv2.line(frame, (center_x - size, center_y), (center_x + size, center_y), color, thickness)
        # Vertical line
        cv2.line(frame, (center_x, center_y - size), (center_x, center_y + size), color, thickness)

        # Calculate pixel offsets based on FOV and depth
        fov_rad_x = math.radians(self.X_CAMERA_FOV)
        fov_rad_y = math.radians(self.Y_CAMERA_FOV)

        # Calculate width and height at the target depth
        width_at_target = 2 * self.depth * math.tan(fov_rad_x / 2)
        height_at_target = 2 * self.depth * math.tan(fov_rad_y / 2)

        # Pixels per cm for X and Y axes
        pixels_per_cm_x = self.IMAGE_WIDTH / width_at_target
        pixels_per_cm_y = self.IMAGE_HEIGHT / height_at_target

        # Calculate shooter position offsets in pixels
        shooter_x_offset = int(self.LASER_OFFSET_CM_X * pixels_per_cm_x)
        shooter_y_offset = int(self.LASER_OFFSET_CM_Y * pixels_per_cm_y)

        # Adjust shooter position
        shooter_x = center_x + shooter_x_offset
        shooter_y = center_y + shooter_y_offset

        # Draw shooter position indicator
        cv2.circle(frame, (shooter_x, shooter_y), 5, (0, 255, 255), -1)
        cv2.line(frame, (shooter_x - 10, shooter_y), (shooter_x + 10, shooter_y), (0, 255, 255), 2)
        cv2.line(frame, (shooter_x, shooter_y - 10), (shooter_x, shooter_y + 10), (0, 255, 255), 2)

    async def process_frame(self, frame_bytes=None):
        """
        Process a frame from either provided bytes or camera
        If frame_bytes is None, read from the camera instead
        """
        if frame_bytes is None:
            # No frame bytes provided, try to read from the camera
            if not self.cap:
                print("No camera available")
                return {"status": "error", "message": "No camera available"}
                
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("Failed to read frame from camera")
                return {"status": "error", "message": "Failed to read frame from camera"}
        else:
            # Use the provided frame bytes
            np_arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                return {"status": "error", "message": "Invalid frame data"}

        # The rest of the processing is identical whether the frame came from camera or bytes
        if self.model is None:
            self.draw_crosshair(frame)
            if self.enable_local_preview:
                cv2.imshow(self.local_preview_window_name, frame)
                cv2.waitKey(1)
            _, buffer = cv2.imencode('.jpg', frame)
            return {"status": "error", "message": "YOLO model not loaded", "frame": buffer.tobytes()}

        results = self.model.predict(source=frame, show=False)

        # Draw crosshair to show center of frame
        self.draw_crosshair(frame)

        # List to store detected balloon info for display
        balloons_info = []
        target_found = False
        best_target = None
        best_confidence = 0

        # Process detection results
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
                if confs[i] < self.CONF_THRESHOLD:
                    continue

                # Get label name
                label = self.model.names[int(cls_ids[i])]
                # Only process detections labeled as 'balloon'
                if label.lower() != 'balloon':
                    continue

                # Convert bounding box coordinates to integers
                x1, y1, x2, y2 = box.astype(int)

                # Calculate the center of the bounding box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Calculate pixel width of the balloon
                pixel_width = x2 - x1

                # Estimate depth: Z = (f * W) / w
                estimated_depth = (self.FOCAL_LENGTH * self.BALLOON_WIDTH) / pixel_width if pixel_width > 0 else 0.0
                estimated_depth *= 100  # Convert to cm
                
                # Update depth if this is a good measurement
                if 30 < estimated_depth < 1000:  # Sanity check on depth
                    self.depth = estimated_depth

                # Extract the region of interest (ROI)
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                # Compute the average BGR color in the ROI
                mean_color = [int(round(x)) for x in cv2.mean(roi)[:3]]
                color_name = self.get_color_name(mean_color)

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

        # Process target if found
        current_time = time.time()
        if target_found and (current_time - self.last_movement_time) > self.MOVEMENT_COOLDOWN:
            self.last_movement_time = current_time
            
            center_x, center_y = best_target['pos']
            
            # Calculate error from center
            error_x, error_y = self.calculate_error(center_x, center_y)
            
            # Draw a line from center to target
            frame_center_x, frame_center_y = self.IMAGE_WIDTH // 2, self.IMAGE_HEIGHT // 2
            cv2.line(frame, (frame_center_x, frame_center_y), (center_x, center_y), (255, 0, 0), 2)
            
            # Check if target is centered within tolerance
            if self.is_target_centered(error_x, error_y):
                # Target is centered - ready to shoot!
                cv2.putText(frame, "TARGET LOCKED!", (frame_center_x - 80, frame_center_y - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Check if we've already shot at these angles
                if not self.is_angle_already_shot(self.current_pan, self.current_tilt):
                    print(f"FIRE! at angles: Pan={self.current_pan:.1f}, Tilt={self.current_tilt:.1f}")
                    print(f"Estimated Depth: {self.depth:.1f} cm")
                    
                    # Send shoot command to Arduino
                    if self.arduino is not None:
                        self.arduino.write("SHOOT\n".encode('utf-8'))
                        print("Shoot command sent to Arduino.")
                        
                        # Wait for ACK from Arduino
                        if self.wait_for_ack():
                            print("Balloon successfully shot.")
                            self.shot_angles.append((self.current_pan, self.current_tilt))
                            
                            # Return to initial position
                            print("Returning to initial position...")
                            self.current_pan = self.INIT_PAN
                            self.current_tilt = self.INIT_TILT
                            self.send_to_arduino(self.INIT_PAN, self.INIT_TILT)
                            time.sleep(0.5)  # Give time for the servos to move
                        else:
                            print("Failed to receive ACK. Retrying...")
            else:
                # Calculate new pan/tilt angles to center the target
                new_pan, new_tilt = self.calculate_new_angles(error_x, error_y)
                
                # Only send to Arduino if values changed
                if abs(new_pan - self.current_pan) > 0.5 or abs(new_tilt - self.current_tilt) > 0.5:
                    # Update current position
                    self.current_pan = new_pan
                    self.current_tilt = new_tilt
                    
                    # Send movement command to Arduino
                    self.send_to_arduino(self.current_pan, self.current_tilt)
                    
                    # Display movement info
                    cv2.putText(frame, f"Moving: Pan={int(self.current_pan)} Tilt={int(self.current_tilt)}", 
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    cv2.putText(frame, f"Error: X={int(error_x)} Y={int(error_y)}", 
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Display current servo positions and control parameters
        cv2.putText(frame, f"Pan: {int(self.current_pan)} Tilt: {int(self.current_tilt)}", 
                    (10, self.IMAGE_HEIGHT - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Kp_X: {self.KP_X:.3f} Kp_Y: {self.KP_Y:.3f}", 
                    (10, self.IMAGE_HEIGHT - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Show local preview if enabled
        if self.enable_local_preview:
            cv2.imshow(self.local_preview_window_name, frame)
            cv2.waitKey(1)

        # Encode and return the processed frame
        _, buffer = cv2.imencode('.jpg', frame)
        return {"status": "ok", "frame": buffer.tobytes(), "balloons_info": balloons_info}

    def stop(self):
        """Clean up resources when stopping the game"""
        print("Stopping Shooter Game...")
        
        # Close the camera
        if self.cap:
            try:
                self.cap.release()
                print("Camera resources released.")
            except Exception as e:
                print(f"Error releasing camera: {e}")
        
        # Close Arduino connection
        if self.arduino is not None:
            try:
                # Return to initial position
                print("Returning servos to initial position.")
                self.send_to_arduino(self.INIT_PAN, self.INIT_TILT)
                time.sleep(0.5)  # Give time for the servos to move
                self.arduino.close()
                print("Arduino connection closed.")
            except Exception as e:
                print(f"Error during Arduino cleanup: {e}")
        
        # Close preview window if open
        if self.enable_local_preview:
            try:
                cv2.destroyWindow(self.local_preview_window_name)
                cv2.waitKey(1)  # Process window events
                print("Preview window closed.")
            except Exception as e:
                print(f"Error closing preview window: {e}")
        
        print("Shooter Game stopped successfully.")

    def cleanup(self):
        """Alias for stop() for compatibility"""
        self.stop()