#!/usr/bin/env python3
"""
Lunar Mission Planning System - Step-by-Step Build
STEP 4: Line Detection

Uses Hough Transform to detect lines (paths to follow)
"""

import cv2
import numpy as np
import sys

print("=" * 60)
print("STEP 4: Line Detection (Hough Transform)")
print("=" * 60)
print("\nTIP: Point camera at paper with drawn lines or edges")
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


def detect_edges(gray_image):
    """Detect edges using Canny"""
    edges = cv2.Canny(gray_image, 50, 150)
    return edges


def detect_lines(edges, threshold=100, min_line_length=50, max_line_gap=10):
    """
    Detect lines using Probabilistic Hough Transform
    
    Args:
        edges: Binary edge image
        threshold: Minimum votes for line detection
        min_line_length: Minimum length of line
        max_line_gap: Maximum gap between line segments
        
    Returns:
        List of lines as [(x1, y1, x2, y2), ...]
    """
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    
    if lines is None:
        return []
    
    # Convert to list of tuples
    line_list = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        line_list.append((x1, y1, x2, y2))
    
    return line_list


def draw_lines(image, lines, color=(0, 255, 0), thickness=2):
    """
    Draw lines on image
    
    Args:
        image: Image to draw on
        lines: List of lines
        color: Line color (B, G, R)
        thickness: Line thickness
    """
    result = image.copy()
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(result, (x1, y1), (x2, y2), color, thickness)
    return result


print("\nDetecting lines...")
print("Press 'q' to quit")
print("Press 't' to adjust threshold (fewer/more lines)")

threshold = 100

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process frame
    gray = preprocess_frame(frame)
    edges = detect_edges(gray)
    lines = detect_lines(edges, threshold=threshold)
    
    # Draw lines on original frame
    result = draw_lines(frame, lines, color=(0, 255, 0), thickness=2)
    
    # Add info
    cv2.putText(result, f"Lines detected: {len(lines)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(result, f"Threshold: {threshold}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(result, "Press 't' to cycle threshold", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    cv2.imshow('Step 4: Line Detection', result)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('t'):
        threshold = (threshold + 20) % 200 + 50
        print(f"Threshold: {threshold}")

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("STEP 4 COMPLETE!")
print(f"Detected lines successfully!")
print("Ready for Step 5!")
print("=" * 60)
