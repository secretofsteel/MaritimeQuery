"""Application-level logging utilities."""

from __future__ import annotations

import logging
from typing import Optional


def setup_logger(name: str = "maritime_rag", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


LOGGER: logging.Logger = setup_logger()

__all__ = ["LOGGER", "setup_logger"]
