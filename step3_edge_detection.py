#!/usr/bin/env python3
"""
Lunar Mission Planning System - Step-by-Step Build
STEP 3: Edge Detection

Uses Canny edge detection to find edges in the image
"""

import cv2
import numpy as np
import sys

print("=" * 60)
print("STEP 3: Edge Detection")
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


def detect_edges(gray_image, threshold1=50, threshold2=150):
    """
    Detect edges using Canny edge detection
    
    Args:
        gray_image: Grayscale image
        threshold1: Lower threshold for edge detection
        threshold2: Upper threshold for edge detection
        
    Returns:
        Binary edge image
    """
    edges = cv2.Canny(gray_image, threshold1, threshold2)
    return edges


print("\nDetecting edges...")
print("Press 'q' to quit")
print("Press '+' to increase sensitivity, '-' to decrease")
print("You should see edges of objects in your view")

# Adjustable thresholds
threshold1 = 50
threshold2 = 150

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess
    gray = preprocess_frame(frame)
    
    # Detect edges
    edges = detect_edges(gray, threshold1, threshold2)
    
    # Convert edges to BGR for display
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Combine original and edges
    combined = np.hstack([frame, edges_bgr])
    
    # Resize for display
    scale = 0.6
    display = cv2.resize(combined, None, fx=scale, fy=scale)
    
    # Add labels and info
    cv2.putText(display, "Original", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display, f"Edges (T1:{threshold1}, T2:{threshold2})",
                (int(frame.shape[1]*scale) + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display, "Press +/- to adjust", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    cv2.imshow('Step 3: Edge Detection', display)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+') or key == ord('='):
        threshold1 = min(threshold1 + 10, 200)
        threshold2 = min(threshold2 + 10, 250)
        print(f"Thresholds: {threshold1}, {threshold2}")
    elif key == ord('-') or key == ord('_'):
        threshold1 = max(threshold1 - 10, 10)
        threshold2 = max(threshold2 - 10, 20)
        print(f"Thresholds: {threshold1}, {threshold2}")

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("STEP 3 COMPLETE!")
print("Edge detection works. Ready for Step 4!")
print("=" * 60)
