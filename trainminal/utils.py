"""Utility functions for trainminal."""

import time
from typing import Optional, Dict, Any
from datetime import timedelta


def format_time(seconds: float) -> str:
    """Format seconds into a human-readable time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_bytes(bytes: int) -> str:
    """Format bytes into a human-readable size string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} PB"


def format_percentage(value: float, total: float) -> str:
    """Format a percentage value."""
    if total == 0:
        return "0.00%"
    return f"{(value / total) * 100:.2f}%"


def calculate_eta(elapsed: float, progress: float) -> Optional[float]:
    """Calculate estimated time to completion."""
    if progress <= 0:
        return None
    if progress >= 1.0:
        return 0.0
    total_estimated = elapsed / progress
    remaining = total_estimated - elapsed
    return max(0.0, remaining)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    if denominator == 0:
        return default
    return numerator / denominator


def detect_framework() -> Optional[str]:
    """Detect which ML framework is being used."""
    try:
        import torch
        return "pytorch"
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        return "tensorflow"
    except ImportError:
        pass
    
    try:
        import jax
        return "jax"
    except ImportError:
        pass
    
    return None


def check_nan(value: Any) -> bool:
    """Check if a value is NaN."""
    try:
        import math
        if isinstance(value, float):
            return math.isnan(value)
        if hasattr(value, 'isnan'):
            return value.isnan()
        return False
    except (TypeError, AttributeError):
        return False

