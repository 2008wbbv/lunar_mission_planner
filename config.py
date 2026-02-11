"""
Configuration settings for Lunar Mission Planning System
"""

# Camera Settings
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
CAMERA_INDEX = 0  # Change if using different webcam

# Image Processing Settings
GAUSSIAN_BLUR_KERNEL = 5
CANNY_THRESHOLD_1 = 50
CANNY_THRESHOLD_2 = 150

# Line Detection (Hough Transform)
HOUGH_RHO = 1
HOUGH_THETA_DIVISIONS = 180
HOUGH_THRESHOLD = 100
MIN_LINE_LENGTH = 50
MAX_LINE_GAP = 10

# Obstacle Detection
ADAPTIVE_THRESHOLD_BLOCK_SIZE = 11
ADAPTIVE_THRESHOLD_C = 2
MIN_OBSTACLE_AREA = 100  # pixels
MORPH_KERNEL_SIZE = 5

# Path Planning
OBSTACLE_PENALTY = 50  # Additional cost for paths near obstacles
SAFETY_MARGIN = 20  # Pixels around obstacles to avoid
PATH_SMOOTHING_FACTOR = 0.5

# Communication
BASE_STATION_HOST = 'localhost'
BASE_STATION_PORT = 5555
TELEMETRY_UPDATE_INTERVAL = 0.1  # seconds

# Logging
LOG_LEVEL = 'INFO'
LOG_FILE = 'logs/mission.log'

# Display
SHOW_DEBUG_WINDOWS = True
WINDOW_SCALE = 0.8
