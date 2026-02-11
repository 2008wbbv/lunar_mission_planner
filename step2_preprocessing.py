#!/usr/bin/env python3
"""
Lunar Mission Planning System - Step-by-Step Build
STEP 2: Image Preprocessing

Converts to grayscale and applies blur to reduce noise
"""

import cv2
import numpy as np
import sys

print("=" * 60)
print("STEP 2: Image Preprocessing")
print("=" * 60)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam!")
    sys.exit(1)

print("✓ Camera opened")

def preprocess_frame(frame):
    """
    Preprocess frame for better feature detection
    
    Args:
        frame: BGR image from camera
        
    Returns:
        grayscale: Grayscale image
        blurred: Blurred grayscale image
    """
    # Convert to grayscale
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    
    return grayscale, blurred


print("\nProcessing frames...")
print("Press 'q' to quit")
print("You should see: Original | Grayscale | Blurred")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame
    gray, blurred = preprocess_frame(frame)
    
    # Convert grayscale back to BGR for display
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    blurred_bgr = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    
    # Stack images side by side
    combined = np.hstack([frame, gray_bgr, blurred_bgr])
    
    # Resize for display
    scale = 0.6
    display = cv2.resize(combined, None, fx=scale, fy=scale)
    
    # Add labels
    cv2.putText(display, "Original", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display, "Grayscale", (int(frame.shape[1]*scale) + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display, "Blurred", (int(frame.shape[1]*2*scale) + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Step 2: Preprocessing', display)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("STEP 2 COMPLETE!")
print("Preprocessing works. Ready for Step 3!")
print("=" * 60)
