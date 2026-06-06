from __future__ import annotations

import os
from contextlib import contextmanager
from threading import Lock
from typing import Any, Iterator, Mapping

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Span

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    OTLP_HTTP_EXPORTER_AVAILABLE = True
except ImportError:
    OTLP_HTTP_EXPORTER_AVAILABLE = False


_PROVIDER_LOCK = Lock()
_TRACE_PROVIDER_CONFIGURED = False


def _normalize_otlp_endpoint(endpoint: str) -> str:
    normalized_endpoint = endpoint.rstrip("/")
    if normalized_endpoint.endswith("/v1/traces"):
        return normalized_endpoint
    if normalized_endpoint.startswith("http://") or normalized_endpoint.startswith(
        "https://"
    ):
        return f"{normalized_endpoint}/v1/traces"
    return endpoint


def _parse_otlp_headers(raw_headers: str | None) -> dict[str, str] | None:
    if not raw_headers:
        return None

    headers: dict[str, str] = {}
    for pair in raw_headers.split(","):
        if "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            headers[key] = value

    return headers or None


def configure_tracing(
    service_name: str,
    *,
    service_version: str = "0.1.0",
    environment: str | None = None,
    console_exporter: bool | None = None,
) -> trace.Tracer:
    global _TRACE_PROVIDER_CONFIGURED

    with _PROVIDER_LOCK:
        if _TRACE_PROVIDER_CONFIGURED:
            return trace.get_tracer(service_name)

        deployment_environment = environment or os.getenv(
            "OTEL_ENVIRONMENT", "development"
        )
        resource = Resource.create(
            {
                SERVICE_NAME: os.getenv("OTEL_SERVICE_NAME", service_name),
                SERVICE_VERSION: os.getenv("OTEL_SERVICE_VERSION", service_version),
                "deployment.environment": deployment_environment,
            }
        )

        provider = TracerProvider(resource=resource)

        otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT"
        )
        if otlp_endpoint:
            if not OTLP_HTTP_EXPORTER_AVAILABLE:
                raise RuntimeError(
                    "OTLP tracing requested but exporter dependency is missing. "
                    "Install opentelemetry-exporter-otlp-proto-http."
                )

            otlp_exporter = OTLPSpanExporter(
                endpoint=_normalize_otlp_endpoint(otlp_endpoint),
                headers=_parse_otlp_headers(os.getenv("OTEL_EXPORTER_OTLP_HEADERS")),
            )
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

        should_use_console_exporter = (
            console_exporter
            if console_exporter is not None
            else os.getenv("OTEL_CONSOLE_EXPORTER", "1") == "1"
        )
        if should_use_console_exporter or not otlp_endpoint:
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

        trace.set_tracer_provider(provider)
        _TRACE_PROVIDER_CONFIGURED = True

    return trace.get_tracer(service_name)


def get_tracer(name: str) -> trace.Tracer:
    return trace.get_tracer(name)


@contextmanager
def start_span(
    name: str,
    *,
    tracer_name: str = "dlib",
    attributes: Mapping[str, Any] | None = None,
) -> Iterator[Span]:
    tracer = get_tracer(tracer_name)
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                if value is None:
                    continue
                span.set_attribute(key, value)
        yield span


def current_trace_context() -> dict[str, str]:
    span = trace.get_current_span()
    context = span.get_span_context()
    if not context.is_valid:
        return {}

    return {
        "trace_id": f"{context.trace_id:032x}",
        "span_id": f"{context.span_id:016x}",
    }


def shutdown_tracing() -> None:
    provider = trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        provider.shutdown()
