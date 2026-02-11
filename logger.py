"""
Logging utility for the Lunar Mission Planning System
"""

import logging
import os
from datetime import datetime
from config import LOG_LEVEL, LOG_FILE


def setup_logger(name='LunarMission'):
    """
    Set up logging configuration
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(levelname)s - %(name)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    
    # File handler
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def log_mission_event(event_type, details):
    """
    Log a mission event with timestamp
    
    Args:
        event_type: Type of event (e.g., 'PATH_COMPUTED', 'OBSTACLE_DETECTED')
        details: Event details dictionary
    """
    logger = logging.getLogger('LunarMission')
    timestamp = datetime.now().isoformat()
    logger.info(f"[{event_type}] @ {timestamp}: {details}")
