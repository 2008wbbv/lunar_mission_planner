#!/usr/bin/env python3
"""
Lunar Mission Planning System - Step-by-Step Build
STEP 1: Camera Setup and Basic Frame Capture

Test this first to make sure your webcam works!
"""

import cv2
import numpy as np
import sys

print("=" * 60)
print("STEP 1: Testing Camera Capture")
print("=" * 60)

# Try to open webcam
print("\n[1] Attempting to open webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open webcam!")
    print("Try changing camera index (0, 1, 2, etc.)")
    sys.exit(1)

print("✓ Webcam opened successfully!")

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Get actual resolution
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"✓ Camera resolution: {width}x{height} @ {fps} FPS")

print("\n[2] Capturing test frame...")
ret, frame = cap.read()

if not ret:
    print("ERROR: Failed to capture frame!")
    cap.release()
    sys.exit(1)

print(f"✓ Frame captured! Shape: {frame.shape}")

# Display the frame
print("\n[3] Displaying frame...")
print("Press 'q' to quit, 's' to save a test image")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Add instructions on frame
    cv2.putText(frame, "Step 1: Camera Test", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Lunar Mission - Step 1: Camera Test', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\nQuitting...")
        break
    elif key == ord('s'):
        cv2.imwrite('test_capture.jpg', frame)
        print("✓ Saved test_capture.jpg")

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("STEP 1 COMPLETE!")
print("Camera is working. Ready for Step 2!")
print("=" * 60)
