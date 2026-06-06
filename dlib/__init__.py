from importlib import import_module
from typing import Any

__all__ = [
    "get_logger",
    "configure_tracing",
    "get_tracer",
    "start_span",
    "current_trace_context",
    "shutdown_tracing",
    "ask",
    "stream",
    "ask_async",
    "stream_async",
    "Agent",
]


def __getattr__(name: str) -> Any:
    if name == "get_logger":
        return import_module("dlib.logger").get_logger
    if name in {
        "configure_tracing",
        "get_tracer",
        "start_span",
        "current_trace_context",
        "shutdown_tracing",
    }:
        return getattr(import_module("dlib.tracing"), name)
    if name in {"ask", "stream", "ask_async", "stream_async", "Agent"}:
        return getattr(import_module("dlib.agent"), name)
    raise AttributeError(f"module 'dlib' has no attribute {name!r}")
