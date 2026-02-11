"""
Camera interface for capturing frames from webcam
"""

import cv2
import numpy as np
from logger import setup_logger
import config


class CameraController:
    """Handles camera initialization and frame capture"""
    
    def __init__(self, camera_index=config.CAMERA_INDEX):
        """
        Initialize camera controller
        
        Args:
            camera_index: Index of camera device (0 for default webcam)
        """
        self.logger = setup_logger('Camera')
        self.camera_index = camera_index
        self.cap = None
        self.is_streaming = False
        
    def initialize_camera(self):
        """
        Initialize the camera with specified settings
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open camera {self.camera_index}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
            
            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            self.logger.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps} FPS")
            return True
            
        except Exception as e:
            self.logger.error(f"Camera initialization failed: {e}")
            return False
    
    def capture_frame(self):
        """
        Capture a single frame from the camera
        
        Returns:
            numpy.ndarray: Captured frame, or None if failed
        """
        if self.cap is None or not self.cap.isOpened():
            self.logger.error("Camera not initialized")
            return None
        
        ret, frame = self.cap.read()
        
        if not ret:
            self.logger.error("Failed to capture frame")
            return None
        
        return frame
    
    def start_stream(self):
        """Start continuous frame streaming"""
        self.is_streaming = True
        self.logger.info("Camera streaming started")
    
    def stop_stream(self):
        """Stop frame streaming"""
        self.is_streaming = False
        self.logger.info("Camera streaming stopped")
    
    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.logger.info("Camera released")
    
    def __del__(self):
        """Cleanup on deletion"""
        self.release()


class FrameBuffer:
    """Thread-safe circular buffer for storing frames"""
    
    def __init__(self, max_size=30):
        """
        Initialize frame buffer
        
        Args:
            max_size: Maximum number of frames to store
        """
        self.max_size = max_size
        self.buffer = []
        self.logger = setup_logger('FrameBuffer')
    
    def add_frame(self, frame):
        """
        Add frame to buffer
        
        Args:
            frame: Image frame to add
        """
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(frame.copy())
    
    def get_latest(self):
        """
        Get the most recent frame
        
        Returns:
            Latest frame or None if buffer is empty
        """
        if not self.buffer:
            return None
        return self.buffer[-1]
    
    def get_frame(self, index):
        """
        Get frame at specific index
        
        Args:
            index: Frame index
            
        Returns:
            Frame at index or None
        """
        if 0 <= index < len(self.buffer):
            return self.buffer[index]
        return None
    
    def clear(self):
        """Clear all frames from buffer"""
        self.buffer.clear()
        self.logger.info("Frame buffer cleared")
