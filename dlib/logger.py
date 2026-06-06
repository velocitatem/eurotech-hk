import logging

try:
    from .tracing import configure_tracing, current_trace_context
except ImportError:
    def configure_tracing(*args, **kwargs):  # type: ignore[no-redef]
        return None

    def current_trace_context() -> dict[str, str]:  # type: ignore[no-redef]
        return {}


class TraceContextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base_message = super().format(record)
        trace_context = current_trace_context()
        if not trace_context:
            return base_message

        trace_id = trace_context["trace_id"]
        span_id = trace_context["span_id"]
        return f"{base_message} trace_id={trace_id} span_id={span_id}"


def get_logger(service_name: str, level: str = "INFO") -> logging.Logger:
    configure_tracing(service_name)

    logger = logging.getLogger(service_name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            TraceContextFormatter(
                "%(asctime)s %(name)s %(levelname)s %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.propagate = False

    return logger
