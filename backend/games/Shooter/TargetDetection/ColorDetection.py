# This script detects circles with a certain color in a video stream from the camera.
# The script uses OpenCV to capture video from the camera and detect circles using the Hough Circle Transform.

import cv2
import numpy as np

# Open the camera
cap = cv2.VideoCapture(0)  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Detect edges using Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area > 500:  # Ignore small objects
            # Get a circle that encloses the contour
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            # Ensure the detected shape is roughly circular
            if 20 < radius < 200:  # Adjust size range as needed
                cv2.circle(frame, center, radius, (0, 255, 0), 3)

    # Show results
    cv2.imshow("Detected Balloons", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  

cap.release()
cv2.destroyAllWindows()
