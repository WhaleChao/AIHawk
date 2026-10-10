"""Base LLM provider interface."""

from __future__ import annotations

import asyncio
import json
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Literal, cast

import json_repair
from loguru import logger

from nanobot.events import (
    NO_EVENTS,
    EventSink,
    ResponseSource,
    ResponseSourceEvent,
    RetryStatusEvent,
    RetryWaitEvent,
)
from nanobot.utils.helpers import sanitize_surrogates_deep

STREAM_IDLE_TIMEOUT_S = 90.0
RETRY_AFTER_BUFFER = 1
CONTEXT_SAFETY_BUFFER = 1024

RetryEventCallback = Callable[[str], Awaitable[None]]
RetryStatusCallback = Callable[[RetryStatusEvent], Awaitable[None]]


@dataclass
class ToolCallRequest:
    """A tool call request from the LLM."""
    id: str
    name: str
    arguments: Any
    extra_content: dict[str, Any] | None = None
    provider_specific_fields: dict[str, Any] | None = None
    function_provider_specific_fields: dict[str, Any] | None = None

    def has_valid_name(self) -> bool:
        """Whether this call carries a usable (non-empty string) tool name.

        ToolCallRequest.name is typed ``str`` but not enforced at runtime: a
        model/gateway can emit a degenerate call with ``name=None`` or ``""``.
        Such a call cannot be executed and, if persisted and replayed, makes
        upstream APIs reject the whole request (e.g. with a
        ``tool_use.name: Input should be a valid string`` error),
        which permanently wedges the session.
        """
        runtime_name = cast(object, self.name)
        return isinstance(runtime_name, str) and bool(runtime_name)

    def to_openai_tool_call(self) -> dict[str, Any]:
        """Serialize to an OpenAI-style tool_call payload."""
        arguments = (
            self.arguments
            if isinstance(self.arguments, str)
            else json.dumps(self.arguments, ensure_ascii=False)
        )
        tool_call: dict[str, Any] = {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": arguments,
            },
        }
        if self.extra_content:
            tool_call["extra_content"] = self.extra_content
        if self.provider_specific_fields:
            tool_call["provider_specific_fields"] = self.provider_specific_fields
        if self.function_provider_specific_fields:
            tool_call["function"]["provider_specific_fields"] = self.function_provider_specific_fields
        return tool_call


def parse_tool_arguments(arguments: Any) -> Any:
    """Parse provider tool arguments without guessing executable parameters.

    Valid JSON object strings become dicts. Empty strings become no-arg calls.
    Malformed JSON and JSON array/scalar values are preserved so ToolRegistry
    can reject them before execution.
    """
    if arguments is None:
        return {}
    if not isinstance(arguments, str):
        return arguments

    stripped = arguments.strip()
    if not stripped:
        return {}

    try:
        parsed = json.loads(stripped)
    except Exception:
        return arguments
    return arguments if parsed is None else parsed


def tool_arguments_object_for_replay(arguments: Any) -> dict[str, Any]:
    """Return object-shaped arguments for provider history replay only.

    This compatibility path may repair malformed JSON because it only shapes
    existing conversation history for provider protocols. Do not use it for
    newly generated tool calls that are about to execute.
    """
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return cast(dict[str, Any], arguments)
    if not isinstance(arguments, str):
        return {}

    stripped = arguments.strip()
    if not stripped:
        return {}

    try:
        parsed = json.loads(stripped)
    except Exception:
        try:
            parsed = json_repair.loads(stripped)
        except Exception:
            return {}
    return cast(dict[str, Any], parsed) if isinstance(parsed, dict) else {}


def tool_arguments_json_for_replay(arguments: Any) -> str:
    """Return JSON object string arguments for provider history replay only."""
    return json.dumps(tool_arguments_object_for_replay(arguments), ensure_ascii=False)


@dataclass(frozen=True)
class ProviderCallContext:
    """What the retry chain needs to know about one model request.

    The ``chat_stream`` contract stays provider-agnostic; a provider that
    consumes this context does so through ``chat_stream_with_context``, and
    every other provider inherits the context-free delegation.
    """

    events: EventSink = field(default=NO_EVENTS, repr=False, compare=False)
    # None opts out (auxiliary calls); an empty name denotes an unnamed preset.
    response_preset: str | None = None
    response_is_fallback: bool = False


@dataclass(frozen=True, slots=True)
class LLMUsage:
    """Canonical token usage reported by, or estimated for, one or more LLM calls.

    ``input_tokens`` is the logical input total and therefore includes cache reads
    and writes.  ``None`` cache counts mean the wire protocol did not report that
    metric, while zero means it explicitly reported no cache activity.

    ``total_tokens`` preserves a provider-reported total when it exceeds the
    visible input plus output (for example, hidden reasoning or tool usage).  It
    must be at least ``input_tokens + output_tokens``.  The reported and estimated
    totals partition it exactly, including after multi-call aggregation.
    """

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    reported_tokens: int = 0
    estimated_tokens: int = 0
    generation_ms: int = 0
    measured_output_tokens: int = 0
    ttft_ms: int = 0
    timed_requests: int = 0
    context_tokens: int | None = None
    request_count: int = 0

    def __post_init__(self) -> None:
        token_fields = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "reported_tokens": self.reported_tokens,
            "estimated_tokens": self.estimated_tokens,
            "generation_ms": self.generation_ms,
            "measured_output_tokens": self.measured_output_tokens,
            "ttft_ms": self.ttft_ms,
            "timed_requests": self.timed_requests,
            "request_count": self.request_count,
        }
        for name, value in token_fields.items():
            runtime_value = cast(object, value)
            if (
                not isinstance(runtime_value, int)
                or isinstance(runtime_value, bool)
                or runtime_value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")
        for name, value in (
            ("cache_read_tokens", self.cache_read_tokens),
            ("cache_write_tokens", self.cache_write_tokens),
            ("context_tokens", self.context_tokens),
        ):
            runtime_value = cast(object, value)
            if runtime_value is not None and (
                not isinstance(runtime_value, int)
                or isinstance(runtime_value, bool)
                or runtime_value < 0
            ):
                raise ValueError(f"{name} must be None or a non-negative integer")

        visible_total = self.input_tokens + self.output_tokens
        if self.total_tokens < visible_total:
            raise ValueError("total_tokens must be at least input_tokens + output_tokens")
        if self.reported_tokens + self.estimated_tokens != self.total_tokens:
            raise ValueError("reported_tokens + estimated_tokens must equal total_tokens")
        cache_total = (self.cache_read_tokens or 0) + (self.cache_write_tokens or 0)
        if cache_total > self.input_tokens:
            raise ValueError("cache token counts cannot exceed logical input_tokens")

    @classmethod
    def reported(
        cls,
        *,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int | None = None,
        cache_read_tokens: int | None = None,
        cache_write_tokens: int | None = None,
    ) -> LLMUsage:
        """Build usage normalized from a provider response."""
        visible_total = input_tokens + output_tokens
        normalized_total = (
            visible_total if total_tokens is None else max(visible_total, total_tokens)
        )
        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=normalized_total,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            reported_tokens=normalized_total,
            context_tokens=input_tokens,
            request_count=1,
        )

    @classmethod
    def estimated(cls, *, input_tokens: int, output_tokens: int) -> LLMUsage:
        """Build usage estimated locally because the provider omitted it."""
        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            estimated_tokens=input_tokens + output_tokens,
            context_tokens=input_tokens,
            request_count=1,
        )

    @classmethod
    def empty_request(cls) -> LLMUsage:
        """Represent a completed model request with no measurable token usage."""
        return cls(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            request_count=1,
        )

    @property
    def source(self) -> Literal["reported", "estimated", "mixed"]:
        if self.estimated_tokens == 0:
            return "reported"
        if self.reported_tokens == 0:
            return "estimated"
        return "mixed"

    def with_timing(
        self,
        *,
        generation_ms: int | None,
        ttft_ms: int | None,
    ) -> LLMUsage:
        """Attach locally measured streaming telemetry to this usage value."""
        return LLMUsage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.total_tokens,
            cache_read_tokens=self.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens,
            reported_tokens=self.reported_tokens,
            estimated_tokens=self.estimated_tokens,
            generation_ms=max(0, generation_ms or 0),
            measured_output_tokens=self.output_tokens if generation_ms is not None else 0,
            ttft_ms=max(0, ttft_ms or 0),
            timed_requests=1 if ttft_ms is not None else 0,
            context_tokens=self.context_tokens,
            request_count=self.request_count,
        )

    def __add__(self, other: LLMUsage) -> LLMUsage:
        """Aggregate calls without turning partially reported cache data into a count."""

        def _sum_cache(left: int | None, right: int | None) -> int | None:
            return left + right if left is not None and right is not None else None

        return LLMUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cache_read_tokens=_sum_cache(self.cache_read_tokens, other.cache_read_tokens),
            cache_write_tokens=_sum_cache(self.cache_write_tokens, other.cache_write_tokens),
            reported_tokens=self.reported_tokens + other.reported_tokens,
            estimated_tokens=self.estimated_tokens + other.estimated_tokens,
            generation_ms=self.generation_ms + other.generation_ms,
            measured_output_tokens=(
                self.measured_output_tokens + other.measured_output_tokens
            ),
            ttft_ms=self.ttft_ms + other.ttft_ms,
            timed_requests=self.timed_requests + other.timed_requests,
            context_tokens=(
                other.context_tokens
                if other.context_tokens is not None
                else self.context_tokens
            ),
            request_count=self.request_count + other.request_count,
        )


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: LLMUsage | None = None
    # What the request cost in USD, as the gateway reported it (OpenRouter's usage.cost); None when it
    # did not. On the response, not on LLMUsage: a cost is one request's, there is nothing to merge.
    cost_usd: float | None = None
    # Locally measured streaming telemetry. ``generation_ms`` excludes time to
    # first token and provider retry gaps; ``ttft_ms`` measures the first
    # streamed reasoning/content delta from request start. They stay separate
    # from provider usage because providers do not report these consistently.
    generation_ms: int | None = None
    ttft_ms: int | None = None
    retry_after: float | None = None  # Provider supplied retry wait in seconds.
    reasoning_content: str | None = None  # Kimi, DeepSeek-R1, MiMo etc.
    thinking_blocks: list[dict[str, Any]] | None = None  # extended thinking blocks
    # Structured error metadata used by retry policy when finish_reason == "error".
    error_status_code: int | None = None
    error_kind: str | None = None  # e.g. "timeout", "connection"
    error_type: str | None = None  # Provider/type semantic, e.g. insufficient_quota.
    error_code: str | None = None  # Provider/code semantic, e.g. rate_limit_exceeded.
    error_retry_after_s: float | None = None
    error_should_retry: bool | None = None

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0

    @property
    def should_execute_tools(self) -> bool:
        """Tools execute only when has_tool_calls AND finish_reason is a tool-capable stop.
        Blocks gateway-injected calls under ``refusal`` / ``content_filter`` / ``error`` (#3220)."""
        if not self.has_tool_calls:
            return False
        return self.finish_reason in ("tool_calls", "function_call", "stop")


@dataclass(frozen=True)
class GenerationSettings:
    """Default generation settings.

    ``max_tokens`` None sends no limit of our own: the answer may be as long as the model gives. A runtime sets
    it to the model's published maximum (``ModelLimits``), which is sent so no provider default cuts it shorter.
    """

    temperature: float = 0.7
    max_tokens: int | None = None
    reasoning_effort: str | None = None


@dataclass(frozen=True)
class ModelLimits:
    """What one request to a model can hold, as its provider publishes it: the context window (prompt and answer
    together) and the longest answer. None where the provider publishes nothing (a router choosing the model
    per request); then no limit is assumed and none is sent."""

    context_tokens: int | None = None
    answer_tokens: int | None = None


_SYNTHETIC_USER_CONTENT = "(conversation continued)"

# The longest part of a provider's error body that is kept as text.
_ERROR_BODY_LIMIT = 500


class LLMProvider(ABC):
    """Base class for LLM providers."""

    _CHAT_RETRY_DELAYS = (1, 2, 4)
    _RETRY_HEARTBEAT_CHUNK = 30
    _TRANSIENT_ERROR_MARKERS = (
        "429",
        "rate limit",
        "500",
        "502",
        "503",
        "504",
        "overloaded",
        "timeout",
        "timed out",
        "connection",
        "server error",
        "server_error",
        "temporarily unavailable",
        "速率限制",
        "访问量过大",
    )
    _RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})
    _TRANSIENT_ERROR_KINDS = frozenset({"timeout", "connection"})
    _NON_RETRYABLE_429_ERROR_TOKENS = frozenset({
        "insufficient_quota",
        "quota_exceeded",
        "quota_exhausted",
        "billing_hard_limit_reached",
        "insufficient_balance",
        "insufficient_credits",
        "credit_balance_too_low",
        "billing_not_active",
        "payment_required",
    })
    _RETRYABLE_429_ERROR_TOKENS = frozenset({
        "rate_limit_exceeded",
        "rate_limit_error",
        "too_many_requests",
        "request_limit_exceeded",
        "requests_limit_exceeded",
        "overloaded_error",
    })
    _NON_RETRYABLE_429_TEXT_MARKERS = (
        "insufficient_quota",
        "insufficient quota",
        "quota exceeded",
        "quota exhausted",
        "billing hard limit",
        "billing_hard_limit_reached",
        "billing not active",
        "insufficient balance",
        "insufficient_balance",
        "insufficient credits",
        "insufficient_credits",
        "credit balance too low",
        "payment required",
        "out of credits",
        "out of quota",
        "exceeded your current quota",
    )
    _RETRYABLE_429_TEXT_MARKERS = (
        "rate limit",
        "rate_limit",
        "too many requests",
        "retry after",
        "try again in",
        "temporarily unavailable",
        "overloaded",
        "concurrency limit",
        "速率限制",
    )

    _SENTINEL = object()

    def __init__(
        self,
        api_base: str | None = None,
        *,
        provider_name: str,
        api_key: str | None = None,
    ):
        runtime_provider_name = cast(object, provider_name)
        if not isinstance(runtime_provider_name, str) or not runtime_provider_name.strip():
            raise ValueError("provider_name must be a non-empty configured identity")
        # The one place the credential lives: a client is built from it.
        self.api_key = api_key
        self.api_base = api_base
        self.provider_name = provider_name
        self.generation: GenerationSettings = GenerationSettings()

    async def model_limits(self, model: str) -> ModelLimits:
        """The limits `model` has at this provider; unknown for a provider that publishes none."""
        return ModelLimits()

    @staticmethod
    def _sanitize_empty_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Sanitize message content: fix empty blocks, strip internal _meta fields.

        Also strips unpaired UTF-16 surrogate code points from every string leaf
        as a defense-in-depth pass before the payload leaves the process. Lone
        surrogates (e.g. leaking from a Windows console, prompt_toolkit history,
        or a truncated JSON round-trip) otherwise cause ``UnicodeEncodeError:
        'utf-8' codec can't encode characters ... surrogates not allowed`` when
        the HTTP client serializes the request body.
        """
        result: list[dict[str, Any]] = []
        for raw_msg in messages:
            msg = {key: value for key, value in raw_msg.items() if key != "_meta"}
            content = msg.get("content")

            if isinstance(content, str) and not content:
                clean = dict(msg)
                clean["content"] = None if (msg.get("role") == "assistant" and msg.get("tool_calls")) else "(empty)"
                result.append(clean)
                continue

            if isinstance(content, list):
                new_items: list[Any] = []
                changed = False
                for raw_item in cast(list[object], content):
                    item = cast(dict[str, Any], raw_item) if isinstance(raw_item, dict) else None
                    if (
                        item is not None
                        and item.get("type") in ("text", "input_text", "output_text")
                        and not item.get("text")
                    ):
                        changed = True
                        continue
                    if item is not None and "_meta" in item:
                        new_items.append({k: v for k, v in item.items() if k != "_meta"})
                        changed = True
                    else:
                        new_items.append(raw_item)
                if changed:
                    clean = dict(msg)
                    if new_items:
                        clean["content"] = new_items
                    elif msg.get("role") == "assistant" and msg.get("tool_calls"):
                        clean["content"] = None
                    else:
                        clean["content"] = "(empty)"
                    result.append(clean)
                    continue

            if isinstance(content, dict):
                clean = dict(msg)
                clean["content"] = [content]
                result.append(clean)
                continue

            result.append(msg)
        # Defense-in-depth: scrub lone UTF-16 surrogates from every string leaf.
        # This is idempotent and no-op when messages are already clean.
        sanitized = sanitize_surrogates_deep(result)
        return cast(list[dict[str, Any]], sanitized) if isinstance(sanitized, list) else result

    @staticmethod
    def _tool_name(tool: dict[str, Any]) -> str:
        """Extract tool name from either a flat tool schema or an OpenAI function schema."""
        name = tool.get("name")
        if isinstance(name, str):
            return name
        fn = tool.get("function")
        fn_object = cast(dict[str, Any], fn) if isinstance(fn, dict) else None
        if fn_object is not None:
            fname = fn_object.get("name")
            if isinstance(fname, str):
                return fname
        return ""

    @classmethod
    def _tool_cache_marker_indices(cls, tools: list[dict[str, Any]]) -> list[int]:
        """Return cache marker indices: builtin/MCP boundary and tail index."""
        if not tools:
            return []

        tail_idx = len(tools) - 1
        last_builtin_idx: int | None = None
        for i in range(tail_idx, -1, -1):
            if not cls._tool_name(tools[i]).startswith("mcp_"):
                last_builtin_idx = i
                break

        ordered_unique: list[int] = []
        for idx in (last_builtin_idx, tail_idx):
            if idx is not None and idx not in ordered_unique:
                ordered_unique.append(idx)
        return ordered_unique

    @staticmethod
    def _sanitize_request_messages(
        messages: list[dict[str, Any]],
        allowed_keys: frozenset[str],
    ) -> list[dict[str, Any]]:
        """Keep only provider-safe message keys and normalize assistant content."""
        sanitized: list[dict[str, Any]] = []
        for msg in messages:
            clean = {k: v for k, v in msg.items() if k in allowed_keys}
            if clean.get("role") == "assistant" and "content" not in clean:
                clean["content"] = None
            sanitized.append(clean)
        return sanitized

    def failure_text(self, exc: BaseException) -> str:
        """The one place a provider failure becomes text (the model, the logs and the host all read it).

        The text is the error body when the exception carries one, else the exception's own
        message. A body the client already parsed (the OpenAI client's `body` is the JSON) is
        read for its message, `{"message": ...}` or `{"error": {"message": ...}}`: its Python
        repr would reach the person as `{'message': ..., 'code': 400}`.
        """
        body = (
            getattr(exc, "doc", None)
            or getattr(exc, "body", None)
            or getattr(getattr(exc, "response", None), "text", None)
        )
        if isinstance(body, dict):
            error = body.get("error", body)
            message = error.get("message") if isinstance(error, dict) else None
            body = message if isinstance(message, str) and message.strip() else json.dumps(body, ensure_ascii=False)
        body_text = body if isinstance(body, str) else str(body) if body is not None else ""
        body_text = body_text.strip()
        if body_text and getattr(exc, "status_code", None) == 401:
            # OpenRouter says "User not found." of a wrong or revoked key: the person has to read that it is the key.
            return f"Error: the API key was refused (401): {body_text[:_ERROR_BODY_LIMIT]}"
        if body_text:
            return f"Error: {body_text[:_ERROR_BODY_LIMIT]}"
        detail = str(exc).strip() or type(exc).__name__
        return f"Error calling LLM: {detail}"

    def _error_response_from_exception(self, exc: Exception) -> LLMResponse:
        """Convert an unexpected exception while retaining retry metadata."""
        error_names = tuple(cls.__name__.lower() for cls in type(exc).__mro__)
        error_kind: str | None = None
        error_should_retry: bool | None = None
        if any("timeout" in name for name in error_names):
            error_kind = "timeout"
            error_should_retry = True
        elif any(
            token in name
            for name in error_names
            for token in ("connect", "connection", "network", "protocol", "transport")
        ):
            error_kind = "connection"
            error_should_retry = True
        elif any(
            "ratelimit" in name or "throttl" in name
            for name in error_names
        ):
            error_kind = "rate_limit"
            error_should_retry = True
        elif any(
            "server" in name or "internal" in name
            for name in error_names
        ):
            error_kind = "server_error"
            error_should_retry = True
        elif any(
            token in name
            for name in error_names
            for token in ("auth", "credential", "permissiondenied", "unauthor")
        ):
            error_kind = "authentication"

        response = getattr(exc, "response", None)
        raw_status = getattr(exc, "status_code", None)
        if raw_status is None and response is not None:
            raw_status = getattr(response, "status_code", None)
        try:
            error_status_code = int(raw_status) if raw_status is not None else None
        except (TypeError, ValueError):
            error_status_code = None

        raw_error_type = getattr(exc, "error_type", None)
        raw_error_code = getattr(exc, "error_code", None)
        return LLMResponse(
            content=self.failure_text(exc),
            finish_reason="error",
            error_status_code=error_status_code,
            error_kind=error_kind,
            error_type=str(raw_error_type) if raw_error_type is not None else None,
            error_code=str(raw_error_code) if raw_error_code is not None else None,
            error_should_retry=error_should_retry,
        )

    @classmethod
    def _is_transient_error(cls, content: str | None) -> bool:
        err = (content or "").lower()
        return any(marker in err for marker in cls._TRANSIENT_ERROR_MARKERS)

    @classmethod
    def is_transient_response(cls, response: LLMResponse) -> bool:
        """Prefer structured error metadata, fallback to text markers for legacy providers."""
        if response.error_should_retry is not None:
            return bool(response.error_should_retry)

        if response.error_status_code is not None:
            status = int(response.error_status_code)
            if status == 429:
                return cls._is_retryable_429_response(response)
            if status in cls._RETRYABLE_STATUS_CODES or status >= 500:
                return True

        kind = (response.error_kind or "").strip().lower()
        if kind in cls._TRANSIENT_ERROR_KINDS:
            return True

        return cls._is_transient_error(response.content)

    @classmethod
    def is_arrearage_response(cls, response: LLMResponse) -> bool:
        """Detect API-key arrearage / quota / billing errors that won't clear on retry.

        These surface as HTTP 402 or as billing semantic tokens (e.g.
        ``insufficient_quota``, ``payment_required``); reuses the same token and
        text markers the 429 retry policy treats as non-retryable.
        """
        if response.error_status_code is not None and int(response.error_status_code) == 402:
            return True

        type_token = cls._normalize_error_token(response.error_type)
        code_token = cls._normalize_error_token(response.error_code)
        if any(
            token in cls._NON_RETRYABLE_429_ERROR_TOKENS
            for token in (type_token, code_token)
            if token is not None
        ):
            return True

        content = (response.content or "").lower()
        return any(marker in content for marker in cls._NON_RETRYABLE_429_TEXT_MARKERS)

    @staticmethod
    def _normalize_error_token(value: Any) -> str | None:
        if value is None:
            return None
        token = str(value).strip().lower()
        return token or None

    @classmethod
    def _extract_error_type_code(cls, payload: Any) -> tuple[str | None, str | None]:
        data: dict[str, Any] | None = None
        if isinstance(payload, dict):
            data = cast(dict[str, Any], payload)
        elif isinstance(payload, str):
            text = payload.strip()
            if text:
                try:
                    parsed = json.loads(text)
                except Exception:
                    parsed = None
                if isinstance(parsed, dict):
                    data = cast(dict[str, Any], parsed)
        if data is None:
            return None, None

        error_obj = data.get("error")
        type_value = data.get("type")
        code_value = data.get("code")
        error_object = cast(dict[str, Any], error_obj) if isinstance(error_obj, dict) else None
        if error_object is not None:
            type_value = error_object.get("type") or type_value
            code_value = error_object.get("code") or code_value

        return cls._normalize_error_token(type_value), cls._normalize_error_token(code_value)

    @classmethod
    def _is_retryable_429_response(cls, response: LLMResponse) -> bool:
        type_token = cls._normalize_error_token(response.error_type)
        code_token = cls._normalize_error_token(response.error_code)
        semantic_tokens = {
            token for token in (type_token, code_token)
            if token is not None
        }
        if any(token in cls._NON_RETRYABLE_429_ERROR_TOKENS for token in semantic_tokens):
            return False

        content = (response.content or "").lower()
        if any(marker in content for marker in cls._NON_RETRYABLE_429_TEXT_MARKERS):
            return False

        if any(token in cls._RETRYABLE_429_ERROR_TOKENS for token in semantic_tokens):
            return True
        if any(marker in content for marker in cls._RETRYABLE_429_TEXT_MARKERS):
            return True
        # Unknown 429 defaults to WAIT+retry.
        return True

    @staticmethod
    def _content_as_blocks(content: Any) -> list[dict[str, Any]]:
        """Convert message content to blocks so mixed user content can be merged."""
        if isinstance(content, list):
            return [
                dict(cast(dict[str, Any], item))
                if isinstance(item, dict)
                else {"type": "text", "text": str(item)}
                for item in cast(list[object], content)
            ]
        if content is None:
            return []
        return [{"type": "text", "text": str(content)}]

    @staticmethod
    def _enforce_role_alternation(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge consecutive same-role messages and drop trailing assistant messages.

        Some providers (OpenAI-compat, Azure, vLLM, Ollama, etc.) reject requests
        where the last message is 'assistant' (prefill not supported) or two
        consecutive non-system messages share the same role.
        """
        if not messages:
            return messages

        merged: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            if (
                merged
                and role != "system"
                and role not in ("tool",)
                and merged[-1].get("role") == role
                and role in ("user", "assistant")
            ):
                prev = merged[-1]
                if role == "assistant":
                    prev_has_tools = bool(prev.get("tool_calls"))
                    curr_has_tools = bool(msg.get("tool_calls"))
                    if curr_has_tools:
                        merged[-1] = dict(msg)
                        continue
                    if prev_has_tools:
                        continue
                prev_content = prev.get("content") or ""
                curr_content = msg.get("content") or ""
                if isinstance(prev_content, str) and isinstance(curr_content, str):
                    prev["content"] = (prev_content + "\n\n" + curr_content).strip()
                elif role == "user":
                    combined = dict(msg)
                    combined["content"] = [
                        *LLMProvider._content_as_blocks(prev_content),
                        *LLMProvider._content_as_blocks(curr_content),
                    ]
                    merged[-1] = combined
                else:
                    merged[-1] = dict(msg)
            else:
                merged.append(dict(msg))

        last_popped = None
        while merged and merged[-1].get("role") == "assistant":
            last_popped = merged.pop()

        # If removing trailing assistant messages left only system messages,
        # the request would be invalid for most providers (e.g. Zhipu/GLM
        # error 1214).  Recover by converting the last popped assistant
        # message to a user message so the LLM can still see the content.
        if (
            merged
            and last_popped is not None
            and not any(m.get("role") in ("user", "tool") for m in merged)
        ):
            recovered = dict(last_popped)
            recovered["role"] = "user"
            merged.append(recovered)

        # Safety net: ensure the first non-system message is not a bare
        # ``assistant`` message.  Providers like GLM reject system→assistant
        # with error 1214.  Insert a synthetic user message to keep the
        # sequence valid when replayed history starts at an assistant turn.
        for i, msg in enumerate(merged):
            if msg.get("role") != "system":
                if msg.get("role") == "assistant" and not msg.get("tool_calls"):
                    merged.insert(i, {"role": "user", "content": _SYNTHETIC_USER_CONTENT})
                break

        return merged

    @staticmethod
    def _strip_image_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        """Replace image_url blocks with text placeholder. Returns None if no images found."""
        found = False
        result: list[dict[str, Any]] = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                new_content: list[Any] = []
                for raw_block in cast(list[object], content):
                    block = cast(dict[str, Any], raw_block) if isinstance(raw_block, dict) else None
                    if block is not None and block.get("type") == "image_url":
                        placeholder = (
                            "[Image not delivered to model - "
                            "do not describe or reference it]"
                        )
                        new_content.append({"type": "text", "text": placeholder})
                        found = True
                    else:
                        new_content.append(raw_block)
                result.append({**msg, "content": new_content})
            else:
                result.append(msg)
        return result if found else None

    @staticmethod
    def _strip_image_content_inplace(messages: list[dict[str, Any]]) -> bool:
        """Replace image_url blocks with text placeholder *in-place*.

        Mutates the content lists of the original message dicts so that
        callers holding references to those dicts also see the stripped
        version.
        """
        found = False
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for i, raw_block in enumerate(cast(list[object], content)):
                    block = cast(dict[str, Any], raw_block) if isinstance(raw_block, dict) else None
                    if block is not None and block.get("type") == "image_url":
                        placeholder = (
                            "[Image not delivered to model - "
                            "do not describe or reference it]"
                        )
                        content[i] = {"type": "text", "text": placeholder}
                        found = True
        return found

    @abstractmethod
    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Send one chat completion request and return the whole answer.

        ``tool_choice`` is a strategy ("auto", "required") or a specific tool dict.
        A provider answers a failure with an ``LLMResponse`` whose ``finish_reason``
        is "error", built from ``_error_response_from_exception``.
        """

    async def chat_stream_with_context(
        self,
        *,
        provider_context: ProviderCallContext,
        **kwargs: Any,
    ) -> LLMResponse:
        """Streaming continuation hook with a context-free default."""
        _ = provider_context
        return await self.chat_stream(**kwargs)

    async def _safe_chat_stream(self, **kwargs: Any) -> LLMResponse:
        """Call chat_stream() and convert unexpected exceptions to error responses."""
        try:
            provider_context = kwargs.pop("provider_context", None)
            if isinstance(provider_context, ProviderCallContext):
                return await self.chat_stream_with_context(
                    provider_context=provider_context,
                    **kwargs,
                )
            return await self.chat_stream(**kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return self._error_response_from_exception(exc)

    async def chat_stream_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: object = _SENTINEL,
        temperature: object = _SENTINEL,
        reasoning_effort: object = _SENTINEL,
        tool_choice: str | dict[str, Any] | None = None,
        on_retry_wait: RetryEventCallback | None = None,
        provider_context: ProviderCallContext | None = None,
        on_retry_exhausted: RetryEventCallback | None = None,
        on_retry_status: RetryStatusCallback | None = None,
    ) -> LLMResponse:
        """Call chat_stream() with retry on transient provider failures."""
        if max_tokens is self._SENTINEL or max_tokens is None:
            max_tokens = self.generation.max_tokens
        if temperature is self._SENTINEL or temperature is None:
            temperature = self.generation.temperature
        if reasoning_effort is self._SENTINEL:
            reasoning_effort = self.generation.reasoning_effort

        kw: dict[str, Any] = dict(
            messages=messages, tools=tools, model=model,
            max_tokens=max_tokens, temperature=temperature,
            reasoning_effort=reasoning_effort, tool_choice=tool_choice,
        )
        if provider_context is not None:
            kw["provider_context"] = provider_context
        on_retry_wait, on_retry_exhausted, on_retry_status = await self._retry_notifications(
            provider_context, on_retry_wait, on_retry_exhausted, on_retry_status,
        )
        return await self._run_chat_with_retry(
            kw,
            messages,
            on_retry_wait=on_retry_wait,
            on_retry_exhausted=on_retry_exhausted,
            on_retry_status=on_retry_status,
        )

    @staticmethod
    async def _retry_notifications(
        context: ProviderCallContext | None,
        on_wait: RetryEventCallback | None,
        on_exhausted: RetryEventCallback | None,
        on_status: RetryStatusCallback | None,
    ) -> tuple[RetryEventCallback | None, RetryEventCallback | None, RetryStatusCallback | None]:
        """Adapt once at the retry-chain boundary, before candidate callbacks.

        Explicit callbacks retain precedence. In particular a fallback candidate
        exhaustion callback captures its result; it must not also notify the UI.
        """
        if on_wait is None and context is not None and context.events.publish is not None:
            async def publish(content: str) -> None:
                await context.events.emit(RetryWaitEvent(content))

            on_wait = publish
        if on_status is None and context is not None and context.events.accepts(RetryStatusEvent):
            on_status = context.events.emit
            # A turn may continue after an exhausted request; the new chain owns
            # its own retry state, including when its first attempt is terminal.
            await on_status(RetryStatusEvent("cleared", 1, None, "unknown"))
        return on_wait, on_exhausted or on_wait, on_status

    async def _run_chat_with_retry(
        self,
        kw: dict[str, Any],
        original_messages: list[dict[str, Any]],
        *,
        on_retry_wait: RetryEventCallback | None,
        on_retry_exhausted: RetryEventCallback | None,
        on_retry_status: RetryStatusCallback | None,
    ) -> LLMResponse:
        """Run one chat entry point through this provider's retry policy."""
        call = self._safe_chat_stream

        async def attributed_call(**kwargs: Any) -> LLMResponse:
            context = kwargs.get("provider_context")
            if (
                not isinstance(context, ProviderCallContext)
                or context.response_preset is None
                or not context.events.accepts(ResponseSourceEvent)
            ):
                return await call(**kwargs)
            source = (
                ResponseSource(
                    provider=self.provider_name,
                    model=kwargs.get("model") or self.get_default_model(),
                    preset=context.response_preset,
                    fallback=context.response_is_fallback,
                )
                if context.response_preset else None
            )
            await context.events.emit(ResponseSourceEvent(None))
            response = await call(**kwargs)
            await context.events.emit(ResponseSourceEvent(
                source if response.finish_reason != "error" and response.content else None,
                content=response.content if response.finish_reason != "error" else None,
            ))
            return response

        return await self._run_with_retry(
            attributed_call,
            kw,
            original_messages,
            on_retry_wait=on_retry_wait,
            on_retry_exhausted=on_retry_exhausted,
            on_retry_status=on_retry_status,
        )

    @classmethod
    def _extract_retry_after(cls, content: str | None) -> float | None:
        text = (content or "").lower()
        patterns = (
            r"retry after\s+(\d+(?:\.\d+)?)\s*(ms|milliseconds|s|sec|secs|seconds|m|min|minutes)?",
            r"try again in\s+(\d+(?:\.\d+)?)\s*(ms|milliseconds|s|sec|secs|seconds|m|min|minutes)",
            r"wait\s+(\d+(?:\.\d+)?)\s*(ms|milliseconds|s|sec|secs|seconds|m|min|minutes)\s*before retry",
            r"retry[_-]?after[\"'\s:=]+(\d+(?:\.\d+)?)",
        )
        for idx, pattern in enumerate(patterns):
            if idx == 1 and (compound := cls._extract_compound_try_again_in(text)) is not None:
                return compound
            match = re.search(pattern, text)
            if not match:
                continue
            value = float(match.group(1))
            unit = match.group(2) if idx < 3 else "s"
            return cls._to_retry_seconds(value, unit)
        return None

    @classmethod
    def _extract_compound_try_again_in(cls, text: str) -> float | None:
        """Sum Go-style durations such as OpenAI's ``try again in 1m30s``."""
        match = re.search(r"try again in\s+((?:\d+(?:\.\d+)?(?:ms|h|m|s))+)(?![a-z])", text)
        if not match:
            return None
        unit_seconds = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
        parts = re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", match.group(1))
        return max(0.1, sum(float(value) * unit_seconds[unit] for value, unit in parts))

    @classmethod
    def _to_retry_seconds(cls, value: float, unit: str | None = None) -> float:
        normalized_unit = (unit or "s").lower()
        if normalized_unit in {"ms", "milliseconds"}:
            return max(0.1, value / 1000.0)
        if normalized_unit in {"m", "min", "minutes"}:
            return max(0.1, value * 60.0)
        return max(0.1, value)

    @classmethod
    def _extract_retry_after_from_headers(cls, headers: Any) -> float | None:
        if not headers:
            return None

        def _header_value(name: str) -> Any:
            if hasattr(headers, "get"):
                value = headers.get(name) or headers.get(name.title())
                if value is not None:
                    return value
            if isinstance(headers, dict):
                for key, value in cast(dict[object, Any], headers).items():
                    if isinstance(key, str) and key.lower() == name.lower():
                        return value
            return None

        with suppress(TypeError, ValueError):
            retry_ms = _header_value("retry-after-ms")
            if retry_ms is not None:
                value = float(retry_ms) / 1000.0
                if value > 0:
                    return value

        retry_after = _header_value("retry-after")
        if retry_after is None:
            return None
        retry_after_text = str(retry_after).strip()
        if not retry_after_text:
            return None
        if re.fullmatch(r"\d+(?:\.\d+)?", retry_after_text):
            return cls._to_retry_seconds(float(retry_after_text), "s")
        try:
            retry_at = parsedate_to_datetime(retry_after_text)
        except Exception:
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        remaining = (retry_at - datetime.now(retry_at.tzinfo)).total_seconds()
        return max(0.1, remaining)

    @classmethod
    def _extract_retry_after_from_response(cls, response: LLMResponse) -> float | None:
        if response.error_retry_after_s is not None and response.error_retry_after_s > 0:
            return response.error_retry_after_s
        if response.retry_after is not None and response.retry_after > 0:
            return response.retry_after
        return cls._extract_retry_after(response.content)

    async def _sleep_with_heartbeat(
        self,
        delay: float,
        *,
        attempt: int,
        error_kind: str,
        max_attempts: int,
        on_retry_wait: RetryEventCallback | None = None,
        on_retry_status: RetryStatusCallback | None = None,
    ) -> None:
        next_retry_at = time.time() + max(0.0, delay)
        remaining = max(0.0, delay)
        while remaining > 0:
            if on_retry_wait:
                await on_retry_wait(
                    f"Model request failed, retry in {max(1, int(round(remaining)))}s "
                    f"(attempt {attempt})."
                )
            if on_retry_status:
                await on_retry_status(
                    RetryStatusEvent(
                        state="waiting",
                        attempt=attempt,
                        max_attempts=max_attempts,
                        error_kind=error_kind,
                        next_retry_at=next_retry_at,
                    )
                )
            chunk = min(remaining, self._RETRY_HEARTBEAT_CHUNK)
            await asyncio.sleep(chunk)
            remaining -= chunk

    @classmethod
    def public_error_kind(cls, response: LLMResponse) -> str:
        """Return a stable public category without exposing provider details."""
        if cls.is_arrearage_response(response):
            return "billing"
        kind = (response.error_kind or "").strip().lower()
        if kind in {"connection", "timeout"}:
            return kind
        if response.error_status_code == 429:
            return "rate_limit"
        if response.error_status_code is not None and response.error_status_code >= 500:
            return "server"
        return "unknown"

    async def _run_with_retry(
        self,
        call: Callable[..., Awaitable[LLMResponse]],
        kw: dict[str, Any],
        original_messages: list[dict[str, Any]],
        *,
        on_retry_wait: RetryEventCallback | None,
        on_retry_exhausted: RetryEventCallback | None,
        on_retry_status: RetryStatusCallback | None,
    ) -> LLMResponse:
        attempt = 0
        delays = list(self._CHAT_RETRY_DELAYS)
        last_response: LLMResponse | None = None

        async def _finish_retry_status(
            state: Literal["recovered", "cleared"],
            response: LLMResponse,
        ) -> None:
            if attempt > 1 and on_retry_status:
                await on_retry_status(
                    RetryStatusEvent(
                        state=state,
                        attempt=attempt,
                        max_attempts=len(delays) + 1,
                        error_kind=self.public_error_kind(response),
                    )
                )

        while True:
            attempt += 1
            response = await call(**kw)
            if response.finish_reason != "error":
                await _finish_retry_status("recovered", response)
                return response
            last_response = response
            if not self.is_transient_response(response):
                stripped = self._strip_image_content(kw["messages"])
                if stripped is not None:
                    logger.warning(
                        "Non-transient LLM error with image content, retrying without images"
                    )
                    retry_kw = dict(kw)
                    retry_kw["messages"] = stripped
                    result = await call(**retry_kw)
                    # Permanently strip images from the original messages so
                    # subsequent iterations do not repeat the error-retry cycle.
                    if result.finish_reason != "error":
                        self._strip_image_content_inplace(original_messages)
                    await _finish_retry_status(
                        "recovered" if result.finish_reason != "error" else "cleared",
                        result,
                    )
                    return result
                await _finish_retry_status("cleared", response)
                return response

            if attempt > len(delays):
                logger.warning(
                    "LLM request failed after {} attempts, giving up: {}",
                    attempt,
                    (response.content or "")[:120].lower(),
                )
                if on_retry_exhausted:
                    await on_retry_exhausted(
                        f"Model request failed after {attempt} attempts, giving up."
                    )
                if on_retry_status:
                    await on_retry_status(
                        RetryStatusEvent(
                            state="exhausted",
                            attempt=attempt,
                            max_attempts=len(delays) + 1,
                            error_kind=self.public_error_kind(response),
                        )
                    )
                break

            retry_after = self._extract_retry_after_from_response(response)
            base_delay = delays[min(attempt - 1, len(delays) - 1)]
            delay = retry_after + RETRY_AFTER_BUFFER if retry_after else base_delay

            logger.warning(
                "LLM transient error (attempt {}/{}), retrying in {}s: {}",
                attempt,
                len(delays),
                int(round(delay)),
                (response.content or "")[:120].lower(),
            )
            await self._sleep_with_heartbeat(
                delay,
                attempt=attempt,
                error_kind=self.public_error_kind(response),
                max_attempts=len(delays) + 1,
                on_retry_wait=on_retry_wait,
                on_retry_status=on_retry_status,
            )

        return last_response if last_response is not None else await call(**kw)  # pyright: ignore[reportUnnecessaryComparison]

    @abstractmethod
    def get_default_model(self) -> str:
        """Get the default model for this provider."""
        pass
