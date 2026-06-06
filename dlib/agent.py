"""
Thin async + sync wrappers over the Anthropic SDK for quick scripting and agent
patterns. Use this when you want direct API access with streaming; for full
agentic loops with file tools use `claude-agent-sdk` (pip install claude-agent-sdk).

Usage:
    from dlib.agent import ask, stream, Agent

    # One-shot
    reply = ask("Summarize this data: ...")

    # Streaming to stdout
    stream("Write a FastAPI endpoint that ...")

    # Multi-turn agent
    agent = Agent(system="You are an expert Python dev.")
    reply = agent.chat("Generate a Celery task that processes CSV files")
    follow = agent.chat("Now add error handling and retries")
"""

import os
from typing import Iterator, AsyncIterator

from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from .tracing import get_tracer

try:
    import anthropic

    _client: anthropic.Anthropic | None = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )
    _async_client: anthropic.AsyncAnthropic | None = anthropic.AsyncAnthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )
except ImportError:
    _client = None
    _async_client = None


DEFAULT_MODEL = "claude-sonnet-4-5"
_TRACER = get_tracer(__name__)


def _set_llm_request_attributes(
    span: Span, prompt: str, system: str, model: str, history_length: int | None = None
) -> None:
    span.set_attribute("llm.vendor", "anthropic")
    span.set_attribute("llm.request.model", model)
    span.set_attribute("llm.request.prompt_length", len(prompt))
    span.set_attribute("llm.request.has_system_prompt", bool(system))
    if history_length is not None:
        span.set_attribute("llm.request.history_messages", history_length)


def _set_usage_attributes(span: Span, usage: object) -> None:
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    cache_read_tokens = getattr(usage, "cache_read_input_tokens", None)
    cache_creation_tokens = getattr(usage, "cache_creation_input_tokens", None)

    if input_tokens is not None:
        span.set_attribute("llm.usage.input_tokens", input_tokens)
    if output_tokens is not None:
        span.set_attribute("llm.usage.output_tokens", output_tokens)
    if cache_read_tokens is not None:
        span.set_attribute("llm.usage.cache_read_input_tokens", cache_read_tokens)
    if cache_creation_tokens is not None:
        span.set_attribute(
            "llm.usage.cache_creation_input_tokens", cache_creation_tokens
        )


def _require_client() -> "anthropic.Anthropic":
    if _client is None:
        raise ImportError("pip install anthropic")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    return _client


def ask(prompt: str, system: str = "", model: str = DEFAULT_MODEL) -> str:
    """One-shot blocking request; returns full text."""
    client = _require_client()
    with _TRACER.start_as_current_span("dlib.agent.ask", kind=SpanKind.CLIENT) as span:
        _set_llm_request_attributes(span, prompt, system, model)
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=8096,
                system=system or anthropic.NOT_GIVEN,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise

        reply = msg.content[0].text
        span.set_attribute("llm.response.length", len(reply))
        usage = getattr(msg, "usage", None)
        if usage is not None:
            _set_usage_attributes(span, usage)
        span.set_status(Status(StatusCode.OK))
        return reply


def stream(prompt: str, system: str = "", model: str = DEFAULT_MODEL) -> Iterator[str]:
    """Streaming generator; yields text deltas. Print as they arrive."""
    client = _require_client()
    with _TRACER.start_as_current_span(
        "dlib.agent.stream", kind=SpanKind.CLIENT
    ) as span:
        _set_llm_request_attributes(span, prompt, system, model)
        chunk_count = 0
        response_length = 0
        try:
            with client.messages.stream(
                model=model,
                max_tokens=8096,
                system=system or anthropic.NOT_GIVEN,
                messages=[{"role": "user", "content": prompt}],
            ) as s:
                for text in s.text_stream:
                    chunk_count += 1
                    response_length += len(text)
                    yield text
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise

        span.set_attribute("llm.stream.chunk_count", chunk_count)
        span.set_attribute("llm.response.length", response_length)
        span.set_status(Status(StatusCode.OK))


async def ask_async(prompt: str, system: str = "", model: str = DEFAULT_MODEL) -> str:
    """Async one-shot request."""
    if _async_client is None:
        raise ImportError("pip install anthropic")
    with _TRACER.start_as_current_span(
        "dlib.agent.ask_async", kind=SpanKind.CLIENT
    ) as span:
        _set_llm_request_attributes(span, prompt, system, model)
        try:
            msg = await _async_client.messages.create(
                model=model,
                max_tokens=8096,
                system=system or anthropic.NOT_GIVEN,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise

        reply = msg.content[0].text
        span.set_attribute("llm.response.length", len(reply))
        usage = getattr(msg, "usage", None)
        if usage is not None:
            _set_usage_attributes(span, usage)
        span.set_status(Status(StatusCode.OK))
        return reply


async def stream_async(
    prompt: str, system: str = "", model: str = DEFAULT_MODEL
) -> AsyncIterator[str]:
    """Async streaming generator."""
    if _async_client is None:
        raise ImportError("pip install anthropic")
    with _TRACER.start_as_current_span(
        "dlib.agent.stream_async", kind=SpanKind.CLIENT
    ) as span:
        _set_llm_request_attributes(span, prompt, system, model)
        chunk_count = 0
        response_length = 0
        try:
            async with _async_client.messages.stream(
                model=model,
                max_tokens=8096,
                system=system or anthropic.NOT_GIVEN,
                messages=[{"role": "user", "content": prompt}],
            ) as s:
                async for text in s.text_stream:
                    chunk_count += 1
                    response_length += len(text)
                    yield text
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise

        span.set_attribute("llm.stream.chunk_count", chunk_count)
        span.set_attribute("llm.response.length", response_length)
        span.set_status(Status(StatusCode.OK))


class Agent:
    """Stateful multi-turn conversation agent with optional system prompt."""

    def __init__(self, system: str = "", model: str = DEFAULT_MODEL):
        self.system = system
        self.model = model
        self.history: list[dict] = []

    def chat(self, prompt: str) -> str:
        client = _require_client()
        with _TRACER.start_as_current_span(
            "dlib.agent.chat", kind=SpanKind.CLIENT
        ) as span:
            _set_llm_request_attributes(
                span,
                prompt=prompt,
                system=self.system,
                model=self.model,
                history_length=len(self.history),
            )
            self.history.append({"role": "user", "content": prompt})
            try:
                msg = client.messages.create(
                    model=self.model,
                    max_tokens=8096,
                    system=self.system or anthropic.NOT_GIVEN,
                    messages=self.history,
                )
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

            reply = msg.content[0].text
            self.history.append({"role": "assistant", "content": reply})
            span.set_attribute("llm.response.length", len(reply))
            span.set_attribute("llm.response.history_messages", len(self.history))
            usage = getattr(msg, "usage", None)
            if usage is not None:
                _set_usage_attributes(span, usage)
            span.set_status(Status(StatusCode.OK))
            return reply

    async def chat_async(self, prompt: str) -> str:
        if _async_client is None:
            raise ImportError("pip install anthropic")
        with _TRACER.start_as_current_span(
            "dlib.agent.chat_async", kind=SpanKind.CLIENT
        ) as span:
            _set_llm_request_attributes(
                span,
                prompt=prompt,
                system=self.system,
                model=self.model,
                history_length=len(self.history),
            )
            self.history.append({"role": "user", "content": prompt})
            try:
                msg = await _async_client.messages.create(
                    model=self.model,
                    max_tokens=8096,
                    system=self.system or anthropic.NOT_GIVEN,
                    messages=self.history,
                )
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

            reply = msg.content[0].text
            self.history.append({"role": "assistant", "content": reply})
            span.set_attribute("llm.response.length", len(reply))
            span.set_attribute("llm.response.history_messages", len(self.history))
            usage = getattr(msg, "usage", None)
            if usage is not None:
                _set_usage_attributes(span, usage)
            span.set_status(Status(StatusCode.OK))
            return reply

    def reset(self) -> None:
        self.history.clear()
