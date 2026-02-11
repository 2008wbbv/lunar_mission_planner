#!/usr/bin/env python3
"""
Lunar Mission Planning System - Step-by-Step Build
STEP 6: Path Planning with A*

Click to set start (green) and goal (red), then A* finds the path
"""

import cv2
import numpy as np
import sys
from collections import defaultdict
import heapq

print("=" * 60)
print("STEP 6: Path Planning (A* Algorithm)")
print("=" * 60)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam!")
    sys.exit(1)

print("✓ Camera opened")

# Global variables for mouse clicks
start_point = None
goal_point = None
current_frame = None


def preprocess_frame(frame):
    """Convert to grayscale and blur"""
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    return blurred


def detect_obstacles(gray_image, min_area=500):
    """Detect obstacles"""
    thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    obstacles = []
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            obstacles.append(contour)
    return obstacles


def create_obstacle_map(obstacles, shape, safety_margin=20):
    """
    Create binary obstacle map
    
    Args:
        obstacles: List of contours
        shape: Image shape (height, width)
        safety_margin: Pixels around obstacles to avoid
        
    Returns:
        Binary map where 1 = obstacle, 0 = free
    """
    obstacle_map = np.zeros(shape, dtype=np.uint8)
    
    for contour in obstacles:
        cv2.drawContours(obstacle_map, [contour], -1, 1, -1)
    
    # Dilate to add safety margin
    if safety_margin > 0:
        kernel = np.ones((safety_margin, safety_margin), np.uint8)
        obstacle_map = cv2.dilate(obstacle_map, kernel, iterations=1)
    
    return obstacle_map


def heuristic(a, b):
    """Euclidean distance heuristic for A*"""
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def a_star_pathfinding(start, goal, obstacle_map, step_size=10):
    """
    A* pathfinding algorithm
    
    Args:
        start: (x, y) start position
        goal: (x, y) goal position
        obstacle_map: Binary obstacle map
        step_size: Grid step size in pixels
        
    Returns:
        List of waypoints from start to goal, or None if no path
    """
    height, width = obstacle_map.shape
    
    # Check if start or goal is in obstacle
    if obstacle_map[start[1], start[0]] == 1:
        print("Start point is in obstacle!")
        return None
    if obstacle_map[goal[1], goal[0]] == 1:
        print("Goal point is in obstacle!")
        return None
    
    # A* data structures
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = defaultdict(lambda: float('inf'))
    g_score[start] = 0
    f_score = defaultdict(lambda: float('inf'))
    f_score[start] = heuristic(start, goal)
    
    # 8-directional movement
    directions = [
        (-step_size, 0), (step_size, 0), (0, -step_size), (0, step_size),
        (-step_size, -step_size), (step_size, -step_size),
        (-step_size, step_size), (step_size, step_size)
    ]
    
    iterations = 0
    max_iterations = 10000
    
    while open_set and iterations < max_iterations:
        iterations += 1
        current = heapq.heappop(open_set)[1]
        
        # Check if we reached the goal
        if heuristic(current, goal) < step_size:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            path.append(goal)  # Add actual goal
            print(f"Path found in {iterations} iterations!")
            return path
        
        # Explore neighbors
        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            
            # Check bounds
            if not (0 <= neighbor[0] < width and 0 <= neighbor[1] < height):
                continue
            
            # Check obstacle
            if obstacle_map[neighbor[1], neighbor[0]] == 1:
                continue
            
            # Calculate tentative g_score
            tentative_g = g_score[current] + heuristic(current, neighbor)
            
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    print("No path found!")
    return None


def draw_path(image, path, color=(0, 255, 0), thickness=3):
    """Draw path on image"""
    if path and len(path) > 1:
        for i in range(len(path) - 1):
            cv2.line(image, path[i], path[i+1], color, thickness)
    return image


def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks to set start and goal"""
    global start_point, goal_point
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if start_point is None:
            start_point = (x, y)
            print(f"Start point set: {start_point}")
        elif goal_point is None:
            goal_point = (x, y)
            print(f"Goal point set: {goal_point}")
        else:
            # Reset
            start_point = (x, y)
            goal_point = None
            print(f"Reset. New start point: {start_point}")


cv2.namedWindow('Step 6: Path Planning')
cv2.setMouseCallback('Step 6: Path Planning', mouse_callback)

print("\nClick on image:")
print("  1st click = Start point (green)")
print("  2nd click = Goal point (red)")
print("  3rd click = Reset")
print("\nPress 'q' to quit, 'r' to reset points")

path = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    current_frame = frame.copy()
    
    # Detect obstacles
    gray = preprocess_frame(frame)
    obstacles = detect_obstacles(gray)
    obstacle_map = create_obstacle_map(obstacles, gray.shape)
    
    # Draw obstacles
    for obs in obstacles:
        cv2.drawContours(current_frame, [obs], -1, (0, 0, 255), 2)
    
    # Compute path if both points are set
    if start_point and goal_point:
        path = a_star_pathfinding(start_point, goal_point, obstacle_map)
        if path:
            current_frame = draw_path(current_frame, path, (0, 255, 0), 3)
    
    # Draw start and goal
    if start_point:
        cv2.circle(current_frame, start_point, 10, (0, 255, 0), -1)
        cv2.putText(current_frame, "START", (start_point[0] + 15, start_point[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    if goal_point:
        cv2.circle(current_frame, goal_point, 10, (0, 0, 255), -1)
        cv2.putText(current_frame, "GOAL", (goal_point[0] + 15, goal_point[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Add instructions
    cv2.putText(current_frame, "Click: Start (green) -> Goal (red)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    if path:
        cv2.putText(current_frame, f"Path: {len(path)} waypoints", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imshow('Step 6: Path Planning', current_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        start_point = None
        goal_point = None
        path = None
        print("Points reset")

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("STEP 6 COMPLETE!")
print("A* pathfinding works!")
print("Ready to build the full system!")
print("=" * 60)
