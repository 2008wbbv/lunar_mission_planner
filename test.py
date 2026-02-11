#!/usr/bin/env python3
"""
Lunar Mission Planning System - Smart Box Navigation
Navigates AROUND closed boxes but can go INSIDE if goal requires it
"""

import cv2
import numpy as np
import sys
from collections import defaultdict
import heapq

print("=" * 60)
print("SMART BOX NAVIGATION - Advanced Path Planning")
print("=" * 60)
print("\nFeatures:")
print("- Detects closed shapes (boxes/obstacles)")
print("- Navigates AROUND obstacles by default")
print("- Goes INSIDE if start/goal requires it")
print("- Follows lines as paths")
print("=" * 60)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam!")
    sys.exit(1)

# Global variables
roi_box_start = None
roi_box_end = None
roi_complete = False
start_point = None
goal_point = None
drawing_box = False


def mouse_callback(event, x, y, flags, param):
    """Handle mouse events"""
    global roi_box_start, roi_box_end, roi_complete, start_point, goal_point, drawing_box
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if not roi_complete:
            roi_box_start = (x, y)
            drawing_box = True
        elif start_point is None:
            start_point = (x, y)
            print(f"Start: {start_point}")
        elif goal_point is None:
            goal_point = (x, y)
            print(f"Goal: {goal_point}")
        else:
            roi_complete = False
            roi_box_start = None
            roi_box_end = None
            start_point = None
            goal_point = None
            print("Reset!")
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing_box:
            roi_box_end = (x, y)
    
    elif event == cv2.EVENT_LBUTTONUP:
        if drawing_box:
            roi_box_end = (x, y)
            roi_complete = True
            drawing_box = False
            print(f"ROI set: {roi_box_start} to {roi_box_end}")


def detect_lines_and_contours(gray_image, box):
    """
    Detect both lines and closed contours (boxes) in the ROI
    
    Returns:
        lines: List of line segments
        obstacles: List of closed contours (boxes to avoid)
        obstacle_map: Binary map of obstacle areas
    """
    x1, y1 = box[0]
    x2, y2 = box[1]
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    
    # Extract ROI
    roi = gray_image[y1:y2, x1:x2]
    height, width = roi.shape
    
    # Preprocessing
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    
    # Adaptive threshold for obstacle detection
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours (potential obstacles/boxes)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours - identify closed shapes as obstacles
    obstacles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Filter by size and shape
        if area > 500 and area < (width * height * 0.8):  # Not too small, not whole image
            # Check if it's relatively rectangular/closed
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # Adjust contour to global coordinates
                adjusted_contour = contour + np.array([x1, y1])
                obstacles.append(adjusted_contour)
    
    # Create obstacle map
    obstacle_map = np.zeros(gray_image.shape, dtype=np.uint8)
    for obs in obstacles:
        cv2.drawContours(obstacle_map, [obs], -1, 255, -1)  # Fill the obstacle
    
    # Detect lines using Canny + Hough
    edges = cv2.Canny(blurred, 50, 150)
    
    # Remove obstacle edges from line detection
    roi_obstacle_map = obstacle_map[y1:y2, x1:x2]
    
    # Dilate obstacles slightly to avoid detecting their edges as paths
    kernel = np.ones((5, 5), np.uint8)
    dilated_obs = cv2.dilate(roi_obstacle_map, kernel, iterations=1)
    
    # Remove obstacle edges
    edges_clean = cv2.bitwise_and(edges, cv2.bitwise_not(dilated_obs))
    
    # Detect lines
    lines = cv2.HoughLinesP(edges_clean, rho=1, theta=np.pi/180,
                           threshold=40, minLineLength=20, maxLineGap=10)
    
    line_list = []
    if lines is not None:
        for line in lines:
            lx1, ly1, lx2, ly2 = line[0]
            # Convert to global coordinates
            line_list.append((lx1 + x1, ly1 + y1, lx2 + x1, ly2 + y1))
    
    return line_list, obstacles, obstacle_map


def point_in_obstacle(point, obstacles):
    """Check if a point is inside any obstacle"""
    for obs in obstacles:
        if cv2.pointPolygonTest(obs, point, False) >= 0:
            return True
    return False


def build_navigation_graph(lines, obstacles, start, goal, obstacle_map, img_shape):
    """
    Build navigation graph that routes around obstacles
    but allows paths inside if start/goal requires it
    """
    # Extract line endpoints as potential nodes
    points = []
    for line in lines:
        x1, y1, x2, y2 = line
        points.append((x1, y1))
        points.append((x2, y2))
    
    # Add start and goal
    points.append(start)
    points.append(goal)
    
    # Add corner points around obstacles for routing
    for obs in obstacles:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(obs)
        margin = 15  # Margin around obstacle
        
        # Add corner points around the obstacle
        corners = [
            (x - margin, y - margin),
            (x + w + margin, y - margin),
            (x - margin, y + h + margin),
            (x + w + margin, y + h + margin)
        ]
        
        for corner in corners:
            if 0 <= corner[0] < img_shape[1] and 0 <= corner[1] < img_shape[0]:
                points.append(corner)
    
    # Cluster nearby points
    tolerance = 15
    nodes = []
    used = [False] * len(points)
    
    for i, p1 in enumerate(points):
        if used[i]:
            continue
        
        cluster = [p1]
        used[i] = True
        
        for j, p2 in enumerate(points):
            if not used[j]:
                dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                if dist < tolerance:
                    cluster.append(p2)
                    used[j] = True
        
        avg_x = int(sum(p[0] for p in cluster) / len(cluster))
        avg_y = int(sum(p[1] for p in cluster) / len(cluster))
        nodes.append((avg_x, avg_y))
    
    # Build graph with intelligent edge connections
    graph = defaultdict(list)
    
    # Check if a line segment crosses an obstacle
    def line_crosses_obstacle(p1, p2, obstacle_map):
        """Check if line from p1 to p2 crosses an obstacle"""
        # Sample points along the line
        num_samples = int(np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2) / 5)
        num_samples = max(num_samples, 10)
        
        for i in range(num_samples + 1):
            t = i / num_samples
            x = int(p1[0] + t * (p2[0] - p1[0]))
            y = int(p1[1] + t * (p2[1] - p1[1]))
            
            if 0 <= y < obstacle_map.shape[0] and 0 <= x < obstacle_map.shape[1]:
                if obstacle_map[y, x] > 0:
                    # Check if BOTH endpoints are inside the same obstacle
                    start_inside = point_in_obstacle(p1, obstacles)
                    end_inside = point_in_obstacle(p2, obstacles)
                    
                    # Allow connection if both inside same obstacle (navigating within)
                    if start_inside and end_inside:
                        # Check they're in the SAME obstacle
                        for obs in obstacles:
                            if (cv2.pointPolygonTest(obs, p1, False) >= 0 and 
                                cv2.pointPolygonTest(obs, p2, False) >= 0):
                                return False  # Same obstacle, allow
                    
                    return True  # Crosses obstacle boundary
        
        return False
    
    # Connect nodes that don't cross obstacles
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if i >= j:
                continue
            
            # Check distance (don't connect very far nodes)
            dist = np.sqrt((node1[0]-node2[0])**2 + (node1[1]-node2[1])**2)
            if dist > 200:  # Max connection distance
                continue
            
            # Check if line crosses obstacle
            if not line_crosses_obstacle(node1, node2, obstacle_map):
                graph[node1].append(node2)
                graph[node2].append(node1)
    
    return graph, nodes


def a_star_pathfinding(start, goal, graph, nodes):
    """A* pathfinding on the navigation graph"""
    def find_nearest_node(point, nodes):
        min_dist = float('inf')
        nearest = None
        for node in nodes:
            dist = np.sqrt((point[0] - node[0])**2 + (point[1] - node[1])**2)
            if dist < min_dist:
                min_dist = dist
                nearest = node
        return nearest
    
    start_node = find_nearest_node(start, nodes)
    goal_node = find_nearest_node(goal, nodes)
    
    if start_node is None or goal_node is None:
        return None
    
    def heuristic(a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    open_set = []
    heapq.heappush(open_set, (0, start_node))
    came_from = {}
    g_score = defaultdict(lambda: float('inf'))
    g_score[start_node] = 0
    f_score = defaultdict(lambda: float('inf'))
    f_score[start_node] = heuristic(start_node, goal_node)
    
    while open_set:
        current = heapq.heappop(open_set)[1]
        
        if current == goal_node:
            # Reconstruct path
            path = [goal]
            node = current
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            path.reverse()
            print(f"✓ Path found! {len(path)} waypoints")
            return path
        
        for neighbor in graph[current]:
            tentative_g = g_score[current] + heuristic(current, neighbor)
            
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal_node)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    print("✗ No path found!")
    return None


cv2.namedWindow('Smart Box Navigation')
cv2.setMouseCallback('Smart Box Navigation', mouse_callback)

lines = []
obstacles = []
obstacle_map = None
graph = None
nodes = []
path = None

print("\n1. Draw a box around your drawing area")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    display = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Draw ROI box
    if roi_box_start and roi_box_end:
        cv2.rectangle(display, roi_box_start, roi_box_end, (255, 255, 0), 2)
        
        if roi_complete:
            # Detect lines and obstacles
            lines, obstacles, obstacle_map = detect_lines_and_contours(
                gray, (roi_box_start, roi_box_end)
            )
            
            # Draw obstacles (boxes to avoid)
            cv2.drawContours(display, obstacles, -1, (0, 0, 255), 2)
            for i, obs in enumerate(obstacles):
                M = cv2.moments(obs)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(display, f"Obs{i+1}", (cx-20, cy),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Draw lines (paths)
            for line in lines:
                lx1, ly1, lx2, ly2 = line
                cv2.line(display, (lx1, ly1), (lx2, ly2), (0, 255, 255), 1)
            
            # Build graph if we have start and goal
            if start_point and goal_point and obstacle_map is not None:
                graph, nodes = build_navigation_graph(
                    lines, obstacles, start_point, goal_point,
                    obstacle_map, gray.shape
                )
                
                # Draw nodes
                for node in nodes:
                    cv2.circle(display, node, 3, (255, 0, 255), -1)
                
                # Find path
                path = a_star_pathfinding(start_point, goal_point, graph, nodes)
                
                # Draw path
                if path:
                    for i in range(len(path) - 1):
                        cv2.line(display, path[i], path[i+1], (0, 255, 0), 3)
                    
                    cv2.putText(display, f"Path: {len(path)} waypoints", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.putText(display, f"Lines: {len(lines)} | Obstacles: {len(obstacles)}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Draw start and goal
    if start_point:
        cv2.circle(display, start_point, 10, (0, 255, 0), -1)
        cv2.putText(display, "START", (start_point[0] + 15, start_point[1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Show if start is in obstacle
        if obstacles and point_in_obstacle(start_point, obstacles):
            cv2.putText(display, "(inside box)", (start_point[0] + 15, start_point[1] + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    if goal_point:
        cv2.circle(display, goal_point, 10, (0, 0, 255), -1)
        cv2.putText(display, "GOAL", (goal_point[0] + 15, goal_point[1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Show if goal is in obstacle
        if obstacles and point_in_obstacle(goal_point, obstacles):
            cv2.putText(display, "(inside box)", (goal_point[0] + 15, goal_point[1] + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    # Instructions
    if not roi_complete:
        cv2.putText(display, "1. DRAW BOX (click & drag)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    elif not start_point:
        cv2.putText(display, "2. CLICK START", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    elif not goal_point:
        cv2.putText(display, "3. CLICK GOAL", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(display, "PATHFINDING! (click to reset)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Smart Box Navigation', display)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("Smart Box Navigation Complete!")
print("=" * 60)