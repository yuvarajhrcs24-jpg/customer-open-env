#!/usr/bin/env python3
"""Resilience and error handling utilities for the environment."""

from typing import Any, Callable, Dict, Optional, TypeVar


F = TypeVar("F", bound=Callable[..., Any])


def safe_call(func: F, *args: Any, default: Any = None, **kwargs: Any) -> Any:
    """Call a function with graceful error handling.
    
    Returns default if function raises any exception.
    Useful for avoiding cascading failures in critical paths.
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"[Warning] {func.__name__} failed: {e}")
        return default


def validate_dict_keys(data: Dict[str, Any], required_keys: set) -> bool:
    """Validate that dict has all required keys."""
    return all(key in data for key in required_keys)


def clamp(value: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
    """Clamp value to [min_val, max_val] range."""
    return max(min_val, min(max_val, value))


def safe_json_parse(text: str, default: Dict = None) -> Dict:
    """Parse JSON with fallback to empty dict."""
    import json
    try:
        return json.loads(text)
    except Exception:
        return default or {}
