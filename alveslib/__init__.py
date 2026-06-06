from importlib import import_module
from typing import Any

__all__ = ["get_logger", "ask", "stream", "ask_async", "stream_async", "Agent"]


def __getattr__(name: str) -> Any:
    if name == "get_logger":
        return import_module("alveslib.logger").get_logger
    if name in {"ask", "stream", "ask_async", "stream_async", "Agent"}:
        return getattr(import_module("alveslib.agent"), name)
    raise AttributeError(f"module 'alveslib' has no attribute {name!r}")
