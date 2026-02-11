#!/usr/bin/env python3
"""
Lunar Mission Planning System - Step-by-Step Build
STEP 5: Obstacle Detection

Detects obstacles using contour detection
"""

import cv2
import numpy as np
import sys

print("=" * 60)
print("STEP 5: Obstacle Detection")
print("=" * 60)
print("\nTIP: Point camera at objects to detect as obstacles")
print("=" * 60)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam!")
    sys.exit(1)

print("✓ Camera opened")

def preprocess_frame(frame):
    """Convert to grayscale and blur"""
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    return blurred


def detect_obstacles(gray_image, min_area=100):
    """
    Detect obstacles using adaptive thresholding and contours
    
    Args:
        gray_image: Grayscale image
        min_area: Minimum area to consider as obstacle
        
    Returns:
        List of obstacle bounding boxes [(x, y, w, h), ...]
    """
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )
    
    # Morphological operations to clean up
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by area and get bounding boxes
    obstacles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            obstacles.append((x, y, w, h))
    
    return obstacles, thresh


def draw_obstacles(image, obstacles):
    """
    Draw bounding boxes around obstacles
    
    Args:
        image: Image to draw on
        obstacles: List of bounding boxes
        
    Returns:
        Image with obstacles marked
    """
    result = image.copy()
    for i, (x, y, w, h) in enumerate(obstacles):
        # Draw bounding box
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
        # Label obstacle
        cv2.putText(result, f"Obs {i+1}", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return result


print("\nDetecting obstacles...")
print("Press 'q' to quit")
print("Press 'a' to adjust sensitivity")

min_area = 500

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process frame
    gray = preprocess_frame(frame)
    obstacles, thresh = detect_obstacles(gray, min_area=min_area)
    
    # Draw obstacles
    result = draw_obstacles(frame, obstacles)
    
    # Add info
    cv2.putText(result, f"Obstacles: {len(obstacles)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(result, f"Min area: {min_area}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(result, "Press 'a' to adjust", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Show threshold view in corner
    thresh_small = cv2.resize(thresh, (200, 150))
    thresh_bgr = cv2.cvtColor(thresh_small, cv2.COLOR_GRAY2BGR)
    result[10:160, result.shape[1]-210:result.shape[1]-10] = thresh_bgr
    cv2.putText(result, "Threshold", (result.shape[1]-200, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    cv2.imshow('Step 5: Obstacle Detection', result)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('a'):
        min_area = (min_area + 200) % 2000 + 100
        print(f"Min area: {min_area}")

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("STEP 5 COMPLETE!")
print("Obstacle detection works!")
print("Ready for Step 6!")
print("=" * 60)
