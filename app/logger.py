"""Application-level logging utilities."""

from __future__ import annotations

import logging
import sys
from typing import Optional


def setup_logger(name: str = "maritime_rag", level: int = logging.DEBUG) -> logging.Logger:
    """Return a configured logger instance."""
    logger = logging.getLogger(name)
    
    # Force remove any existing handlers
    logger.handlers.clear()
    
    # Create handler that writes to stdout (Streamlit captures this)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)  # Handler level
    
    # Detailed formatter
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Set logger level
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # Don't propagate to root logger
    
    return logger


LOGGER: logging.Logger = setup_logger()

__all__ = ["LOGGER", "setup_logger"]
