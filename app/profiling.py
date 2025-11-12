"""Profiling utilities for production code."""

import time
import functools
from typing import Callable, Any
from .logger import LOGGER


def time_function(func: Callable) -> Callable:
    """Decorator to time function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        LOGGER.info(f"⏱️  {func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper


def time_async_function(func: Callable) -> Callable:
    """Decorator to time async function execution."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        LOGGER.info(f"⏱️  {func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper