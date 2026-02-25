"""
Pipeline Logging Configuration
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Setup logger with console and optional file output
    
    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_pipeline_logger(output_dir: str, verbose: bool = False) -> logging.Logger:
    """
    Get configured pipeline logger
    
    Args:
        output_dir: Output directory for log file
        verbose: Enable verbose (DEBUG) logging
    
    Returns:
        Configured logger
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create logs directory
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_{timestamp}.log"
    
    return setup_logger("pipeline", str(log_file), level)
