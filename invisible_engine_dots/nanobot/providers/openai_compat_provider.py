"""OpenAI-compatible provider: chat completions against OpenRouter."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import secrets
import string
import uuid
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterable
from ipaddress import ip_address
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlparse

from nanobot.providers.base import (
    STREAM_IDLE_TIMEOUT_S,
    LLMProvider,
    LLMResponse,
    LLMUsage,
    ModelLimits,
    ToolCallRequest,
    parse_tool_arguments,
    tool_arguments_json_for_replay,
)
from nanobot.providers.prompt_count import PromptCounts

if TYPE_CHECKING:
    from openai import AsyncOpenAI as AsyncOpenAIType

    from nanobot.providers.registry import ProviderSpec

# Module-level placeholder - set lazily by _ensure_client on first real
# use, or replaced by tests via ``patch(...)``.  Kept as a plain name so
# that ``unittest.mock.patch`` can find and replace it.
AsyncOpenAI: Any = None

_ALLOWED_MSG_KEYS = frozenset({
    "role", "content", "tool_calls", "tool_call_id", "name",
    "reasoning_content", "extra_content",
})
_ALNUM = string.ascii_letters + string.digits

_STANDARD_TC_KEYS = frozenset({"id", "type", "index", "function"})
_STANDARD_FN_KEYS = frozenset({"name", "arguments"})
# The Dot's attribution on OpenRouter. This is the only place these two values live,
# and they are sent only to an openrouter.ai base URL.
_DEFAULT_OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/feder-cr/invisible_dots",
    "X-Title": "invisible_dots",
}
_KIMI_K3_MODEL = "kimi-k3"
_KIMI_THINKING_MODELS: frozenset[str] = frozenset({
    "kimi-k2.5",
    "kimi-k2.6",
    "kimi-k2.7",
    "kimi-k2.7-code",
    "kimi-k2.7-code-highspeed",
    "k2.6-code-preview",
})
_KIMI_ALWAYS_THINKING_MODELS: frozenset[str] = frozenset({
    "kimi-k2.7-code",
    "kimi-k2.7-code-highspeed",
})
_TEXT_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
# Thinking-capable MiMo models per Xiaomi docs (see
# tests/providers/test_xiaomi_mimo_thinking.py). mimo-v2-flash is omitted
# because it does not support thinking.
_MIMO_THINKING_MODELS: frozenset[str] = frozenset({
    "mimo-v2.5-pro",
    "mimo-v2.5",
    "mimo-v2-pro",
    "mimo-v2-omni",
})
_OPENAI_COMPAT_REQUEST_TIMEOUT_S = 120.0

# Maps a model's thinking style → extra_body builder.
# Each builder takes a bool (thinking_enabled) and returns the dict to
# merge into extra_body, keeping the style→wire-format mapping in one place.
_THINKING_STYLE_MAP: dict[
    str,
    Callable[[bool], dict[str, Any]],
] = {
    "thinking_type": lambda on: {"thinking": {"type": "enabled" if on else "disabled"}},
    "enable_thinking": lambda on: {"enable_thinking": on},
    "reasoning_split": lambda on: {"reasoning_split": on},
}
_GATEWAY_REASONING_STYLE_MAP: dict[
    str,
    Callable[[str], dict[str, Any]],
] = {
    "reasoning_effort": lambda effort: {"reasoning": {"effort": effort}},
}
_QWEN_THINKING_MODELS: frozenset[str] = frozenset({
    "qwen3.7-max",
    "qwen3.7-plus",
    "qwen3.6-max-preview",
    "qwen3.6-plus",
    "qwen3.6-flash",
    "qwen3.5-plus",
    "qwen3.5-flash",
})

_MODEL_THINKING_STYLES: dict[str, str] = {
    **dict.fromkeys(_KIMI_THINKING_MODELS, "thinking_type"),
    **dict.fromkeys(_MIMO_THINKING_MODELS, "thinking_type"),
    **dict.fromkeys(_QWEN_THINKING_MODELS, "enable_thinking"),
}


def _model_slug(model_name: str) -> str:
    return model_name.lower().rsplit("/", 1)[-1]


def _requires_max_completion_tokens(model_name: str) -> bool:
    """Return True for models that require ``max_completion_tokens``."""
    slug = _model_slug(model_name)
    return slug == _KIMI_K3_MODEL or "gpt-5" in slug or any(
        slug == p or slug.startswith((p + "-", p + ".")) for p in ("o1", "o3", "o4")
    )


def _model_thinking_style(model_name: str) -> str:
    return _MODEL_THINKING_STYLES.get(_model_slug(model_name), "")


def _thinking_extra_body(style: str, thinking_enabled: bool) -> dict[str, Any] | None:
    builder = _THINKING_STYLE_MAP.get(style)
    return builder(thinking_enabled) if builder else None


def _gateway_reasoning_extra_body(style: str, effort: str | None) -> dict[str, Any] | None:
    if not effort:
        return None
    builder = _GATEWAY_REASONING_STYLE_MAP.get(style)
    return builder(effort) if builder else None


def _short_tool_id() -> str:
    """9-char alphanumeric ID compatible with all providers (incl. Mistral)."""
    return "".join(secrets.choice(_ALNUM) for _ in range(9))


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```") or not stripped.endswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 2:
        return stripped
    return "\n".join(lines[1:-1]).strip()


def _extract_text_tool_calls(content: str | None) -> tuple[str | None, list[ToolCallRequest]]:
    """Normalize common text-format tool call blocks into structured calls."""
    if not content or "<tool_call>" not in content:
        return content, []

    tool_calls: list[ToolCallRequest] = []
    spans: list[tuple[int, int]] = []
    for match in _TEXT_TOOL_CALL_RE.finditer(content):
        try:
            raw_payload: object = json.loads(
                _strip_json_fence(match.group(1))
            )
        except Exception:
            continue
        if not isinstance(raw_payload, dict):
            continue
        payload = cast(dict[str, Any], raw_payload)

        nested = cast(object, payload.get("tool_call"))
        if isinstance(nested, dict):
            payload = cast(dict[str, Any], nested)
        function = cast(object, payload.get("function"))
        if not isinstance(function, dict):
            function = payload
        function_data = cast(dict[str, Any], function)
        name = cast(object, function_data.get("name"))
        if not isinstance(name, str) or not name:
            continue

        arguments = function_data.get(
            "arguments",
            payload.get("arguments", {}),
        )
        tool_calls.append(ToolCallRequest(
            id=str(payload.get("id") or _short_tool_id()),
            name=name,
            arguments=parse_tool_arguments(arguments),
        ))
        spans.append(match.span())

    if not tool_calls:
        return content, []

    visible_parts: list[str] = []
    last = 0
    for start, end in spans:
        visible_parts.append(content[last:start])
        last = end
    visible_parts.append(content[last:])
    visible_content = "".join(visible_parts).strip() or None
    return visible_content, tool_calls


def _get(obj: object, key: str) -> Any:
    """Get a value from dict or object attribute, returning None if absent."""
    if isinstance(obj, dict):
        return cast(dict[str, Any], obj).get(key)
    return getattr(obj, key, None)


def _coerce_dict(value: object) -> dict[str, Any] | None:
    """Try to coerce *value* to a dict; return None if not possible or empty."""
    if value is None:
        return None
    if isinstance(value, dict):
        return cast(dict[str, Any], value) if value else None
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped: object = model_dump()
        if isinstance(dumped, dict) and dumped:
            return cast(dict[str, Any], dumped)
    return None


def _extract_tc_extras(tc: Any) -> tuple[
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    """Extract (extra_content, provider_specific_fields, fn_provider_specific_fields).

    Works for both SDK objects and dicts.  Captures Gemini ``extra_content``
    verbatim and any non-standard keys on the tool-call / function.
    """
    extra_content = _coerce_dict(_get(tc, "extra_content"))

    tc_dict = _coerce_dict(tc)
    prov = None
    fn_prov = None
    if tc_dict is not None:
        leftover = {k: v for k, v in tc_dict.items()
                    if k not in _STANDARD_TC_KEYS and k != "extra_content" and v is not None}
        if leftover:
            prov = leftover
        fn = _coerce_dict(tc_dict.get("function"))
        if fn is not None:
            fn_leftover = {k: v for k, v in fn.items()
                          if k not in _STANDARD_FN_KEYS and v is not None}
            if fn_leftover:
                fn_prov = fn_leftover
    else:
        prov = _coerce_dict(_get(tc, "provider_specific_fields"))
        fn_obj = _get(tc, "function")
        if fn_obj is not None:
            fn_prov = _coerce_dict(_get(fn_obj, "provider_specific_fields"))

    return extra_content, prov, fn_prov


def _uses_openrouter_attribution(api_base: str | None) -> bool:
    """Attribution headers go only to an openrouter.ai base URL, never to a stand-in."""
    host = (urlparse(api_base or "").hostname or "").rstrip(".").lower()
    return host == "openrouter.ai"


def _is_local_endpoint(api_base: str | None) -> bool:
    """Return True when the endpoint is a local or LAN server.

    Matches common private-network patterns in the base URL (localhost, 127.x,
    192.168.x, 10.x, 172.16-31.x, Docker ``host.docker.internal``).
    """
    if not api_base:
        return False
    raw = api_base.strip().lower()
    parsed = urlparse(raw if "://" in raw else f"//{raw}")
    try:
        host = parsed.hostname
    except ValueError:
        return False
    if host in {"localhost", "host.docker.internal"}:
        return True
    if not host:
        return False
    try:
        addr = ip_address(host)
    except ValueError:
        return False
    return addr.is_loopback or addr.is_private


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*, returning a new dict.

    Nested dicts are merged key-by-key; all other types in *override*
    replace the corresponding key in *base*.
    """
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(
                cast(dict[str, Any], merged[key]),
                cast(dict[str, Any], value),
            )
        else:
            merged[key] = value
    return merged


def _merge_chat_extra_body(
    kwargs: dict[str, Any],
    extra_body: dict[str, Any],
) -> dict[str, Any]:
    """Merge configured Chat Completions fields without clobbering tools."""
    regular_extra = {key: value for key, value in extra_body.items() if key != "tools"}
    merged = dict(kwargs)
    if regular_extra:
        existing = kwargs.get("extra_body", {})
        merged["extra_body"] = _deep_merge(existing, regular_extra)

    if "tools" in extra_body:
        current_tools = kwargs.get("tools")
        configured_tools = extra_body["tools"]
        if isinstance(current_tools, list) and isinstance(configured_tools, list):
            merged["tools"] = [*current_tools, *configured_tools]
        else:
            merged["tools"] = configured_tools

    return merged


class StreamIdleTimeout(TimeoutError):
    """The model's stream sent nothing for `STREAM_IDLE_TIMEOUT_S`; the text of the failure is `failure_text`'s."""


class OpenAICompatProvider(LLMProvider):
    """OpenAI-compatible provider speaking chat completions.

    Receives a resolved ``ProviderSpec`` from the caller; no internal
    registry lookups are needed.
    """

    def __init__(
        self,
        api_key: str,
        api_base: str | None = None,
        default_model: str = "gpt-4o",
        extra_headers: dict[str, str] | None = None,
        spec: ProviderSpec | None = None,
        extra_body: dict[str, Any] | None = None,
        extra_query: dict[str, str] | None = None,
        proxy: str | None = None,
        provider_name: str = "openai",
        request_timeout_s: float = _OPENAI_COMPAT_REQUEST_TIMEOUT_S,
    ):
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api_key must be a non-empty string: a provider never sends a request without its key")
        super().__init__(api_base, provider_name=provider_name, api_key=api_key)
        self.default_model = default_model
        self.extra_headers = extra_headers or {}
        self._spec = spec
        self._extra_body = dict(extra_body or {})
        self._extra_query = extra_query or {}
        self._proxy = proxy or None
        self._request_timeout_s = request_timeout_s

        effective_base = api_base or (spec.default_api_base if spec else None) or None
        self._effective_base = effective_base
        self._default_headers = {"x-session-affinity": uuid.uuid4().hex}
        if _uses_openrouter_attribution(effective_base):
            self._default_headers.update(_DEFAULT_OPENROUTER_HEADERS)
        if extra_headers:
            self._default_headers.update(extra_headers)
        self._is_local = _is_local_endpoint(effective_base)

        # Lazy-init: the OpenAI client and its httpx transport are expensive
        # to create (~700 ms on Windows). Defer until first use.
        self._client: AsyncOpenAIType | None = None
        self._client_lock = asyncio.Lock()
        # The limits of every model the endpoint lists, read once (GET /models) when a request first needs them.
        self._model_limits: dict[str, ModelLimits] | None = None
        self._model_limits_lock = asyncio.Lock()
        # What the endpoint counted of the prompts it took, which sizes the next requests (prompt_count.py).
        self._prompt_counts = PromptCounts()

    def _build_client(self) -> None:
        """Create the OpenAI client using the current module-level AsyncOpenAI."""
        import httpx

        timeout_s = self._request_timeout_s
        http_client: httpx.AsyncClient | None = None
        if self._proxy:
            http_client = httpx.AsyncClient(
                timeout=timeout_s,
                proxy=self._proxy,
                trust_env=False,
                follow_redirects=True,
            )
        elif self._is_local:
            # Local model servers (Ollama, llama.cpp, vLLM) often close idle
            # HTTP connections before the client-side keepalive expires. When
            # two LLM calls happen seconds apart (e.g. heartbeat _decide then
            # process_direct), the second call may grab a now-dead pooled
            # connection, causing a transient APIConnectionError on every first
            # attempt. Disabling keepalive for local endpoints avoids this by
            # opening a fresh connection for each request, which is cheap on a
            # LAN. Cloud providers benefit from keepalive, so we leave the
            # default pool settings for them.
            #
            # Also disable proxy for local endpoints: when the host has
            # HTTP_PROXY / HTTPS_PROXY / ALL_PROXY set, httpx would try to
            # route local traffic through the proxy, which typically cannot
            # reach localhost or LAN addresses.
            _local_limits = httpx.Limits(keepalive_expiry=0)
            http_client = httpx.AsyncClient(
                limits=_local_limits,
                timeout=timeout_s,
                transport=httpx.AsyncHTTPTransport(proxy=None, limits=_local_limits),
            )
        # else: http_client stays None → SDK creates DefaultAsyncHttpxClient
        # which already reads proxy env vars via trust_env=True, has proper
        # connection limits, and follows redirects.
        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self._effective_base,
            default_headers=self._default_headers,
            default_query=self._extra_query or None,
            max_retries=0,
            timeout=timeout_s,
            http_client=http_client,
        )

    async def _ensure_client(self) -> AsyncOpenAIType:
        """Return the shared OpenAI client, creating it on first call."""
        if self._client is not None:
            return self._client
        async with self._client_lock:
            if self._client is not None:
                return self._client
            global AsyncOpenAI
            if AsyncOpenAI is None:
                from openai import AsyncOpenAI as _AsyncOpenAI

                AsyncOpenAI = _AsyncOpenAI

            self._build_client()
            if self._client is None:
                raise RuntimeError("OpenAI client initialization did not produce a client")
            return self._client

    @classmethod
    def _apply_cache_control(
        cls,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        """Inject cache_control markers for prompt caching."""
        cache_marker = {"type": "ephemeral"}
        new_messages = list(messages)

        def _mark(msg: dict[str, Any]) -> dict[str, Any]:
            content = msg.get("content")
            if isinstance(content, str):
                return {**msg, "content": [
                    {"type": "text", "text": content, "cache_control": cache_marker},
                ]}
            if isinstance(content, list) and content:
                nc = list(cast(list[dict[str, Any]], content))
                nc[-1] = {**nc[-1], "cache_control": cache_marker}
                return {**msg, "content": nc}
            return msg

        if new_messages and new_messages[0].get("role") == "system":
            new_messages[0] = _mark(new_messages[0])
        if len(new_messages) >= 3:
            new_messages[-2] = _mark(new_messages[-2])

        new_tools = tools
        if tools:
            new_tools = list(tools)
            for idx in cls._tool_cache_marker_indices(new_tools):
                new_tools[idx] = {**new_tools[idx], "cache_control": cache_marker}
        return new_messages, new_tools

    @staticmethod
    def _normalize_tool_call_id(tool_call_id: Any) -> Any:
        """Normalize to a provider-safe 9-char alphanumeric form."""
        if not isinstance(tool_call_id, str):
            return tool_call_id
        if len(tool_call_id) == 9 and tool_call_id.isalnum():
            return tool_call_id
        return hashlib.sha1(tool_call_id.encode()).hexdigest()[:9]

    def _sanitize_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Strip non-standard keys and make tool_call IDs unique."""
        sanitized = LLMProvider._sanitize_request_messages(messages, _ALLOWED_MSG_KEYS)
        pending_tool_ids: dict[str, deque[str]] = {}

        def unique_tool_id(value: Any, used_ids: set[str], idx: int) -> str:
            if isinstance(value, str) and value:
                base: Any = value
            else:
                base = _short_tool_id()
            if not isinstance(base, str) or not base:
                base = _short_tool_id()
            if base not in used_ids:
                return base
            seed = value if isinstance(value, str) and value else base
            salt = 1
            while True:
                candidate = self._normalize_tool_call_id(f"{seed}:{idx}:{salt}")
                if isinstance(candidate, str) and candidate not in used_ids:
                    return candidate
                salt += 1

        def map_tool_result_id(value: Any) -> Any:
            if not isinstance(value, str):
                return value
            queue = pending_tool_ids.get(value)
            if queue:
                mapped = queue.popleft()
                if not queue:
                    pending_tool_ids.pop(value, None)
                return mapped
            return value

        for clean in sanitized:
            tool_calls_value = cast(object, clean.get("tool_calls"))
            if isinstance(tool_calls_value, list):
                normalized: list[Any] = []
                used_ids: set[str] = set()
                for idx, tc in enumerate(cast(list[object], tool_calls_value)):
                    if not isinstance(tc, dict):
                        normalized.append(tc)
                        continue
                    tc_clean = dict(cast(dict[str, Any], tc))
                    raw_id = tc_clean.get("id")
                    mapped_id = unique_tool_id(raw_id, used_ids, idx)
                    tc_clean["id"] = mapped_id
                    used_ids.add(mapped_id)
                    if isinstance(raw_id, str) and raw_id:
                        pending_tool_ids.setdefault(raw_id, deque()).append(mapped_id)
                    function = cast(object, tc_clean.get("function"))
                    if isinstance(function, dict):
                        function_clean = dict(cast(dict[str, Any], function))
                        if "arguments" in function_clean:
                            function_clean["arguments"] = tool_arguments_json_for_replay(
                                function_clean.get("arguments")
                            )
                        else:
                            function_clean["arguments"] = "{}"
                        tc_clean["function"] = function_clean
                    normalized.append(tc_clean)
                clean["tool_calls"] = normalized
            if "tool_call_id" in clean and clean["tool_call_id"]:
                clean["tool_call_id"] = map_tool_result_id(clean["tool_call_id"])
        return self._enforce_role_alternation(sanitized)

    # ------------------------------------------------------------------
    # Build kwargs
    # ------------------------------------------------------------------

    @staticmethod
    def _supports_temperature(
        model_name: str,
        reasoning_effort: str | None = None,
    ) -> bool:
        """Return True when the model accepts a temperature parameter.

        Temperature is omitted for fixed-temperature Kimi K3, GPT-5, and
        o-series models. GPT-6 requires explicit ``"none"`` effort; its
        default enables reasoning.
        """
        if _model_slug(model_name) == _KIMI_K3_MODEL:
            return False
        name = model_name.lower()
        if "gpt-6" in name:
            return bool(reasoning_effort and reasoning_effort.lower() == "none")
        return not any(token in name for token in ("gpt-5", "o1", "o3", "o4"))

    def _build_kwargs(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str | None,
        max_tokens: int | None,
        temperature: float,
        reasoning_effort: str | None,
        tool_choice: str | dict[str, Any] | None,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        model_name = model or self.default_model
        spec = self._spec

        if spec and spec.supports_prompt_caching and "claude" in model_name.lower():
            messages, tools = self._apply_cache_control(messages, tools)

        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": self._sanitize_messages(self._sanitize_empty_content(messages)),
        }

        if self._supports_temperature(model_name, reasoning_effort):
            kwargs["temperature"] = temperature

        # No limit is sent when none is known: the model's own maximum applies.
        if max_tokens is not None:
            if _requires_max_completion_tokens(model_name):
                kwargs["max_completion_tokens"] = max(1, max_tokens)
            else:
                kwargs["max_tokens"] = max(1, max_tokens)

        # Normalize reasoning_effort into a semantic form (OpenAI vocab)
        # used for internal decisions, and a wire form actually sent out.
        semantic_effort: str | None = None
        if isinstance(reasoning_effort, str):
            semantic_effort = reasoning_effort.lower()
            if semantic_effort == "minimum":
                semantic_effort = "minimal"

        wire_effort = reasoning_effort
        slug = _model_slug(model_name)
        if slug == _KIMI_K3_MODEL and semantic_effort is not None:
            # K3 always reasons and currently accepts only the top-level
            # reasoning_effort="max". Preserve disabled/default semantics by
            # omitting the field; normalize older enabled presets to "max" so
            # switching from a K2.x model does not send an unsupported value.
            if semantic_effort in ("none", "minimal"):
                wire_effort = None
            else:
                semantic_effort = "max"
                wire_effort = "max"

        if wire_effort and semantic_effort != "none":
            kwargs["reasoning_effort"] = wire_effort

        # Only send thinking controls when reasoning_effort is explicit so
        # omitting the config preserves each model's default.
        model_style = _model_thinking_style(model_name)
        if reasoning_effort is not None:
            thinking_enabled = semantic_effort not in ("none", "minimal")
            if model_style and (thinking_enabled or slug not in _KIMI_ALWAYS_THINKING_MODELS):
                extra = _thinking_extra_body(model_style, thinking_enabled)
                if extra:
                    kwargs.setdefault("extra_body", {}).update(extra)
                gateway_style = spec.gateway_reasoning_style if spec else ""
                if gateway_style:
                    extra = _gateway_reasoning_extra_body(gateway_style, semantic_effort)
                    if extra:
                        kwargs.setdefault("extra_body", {}).update(extra)

            # Moonshot rejects requests that carry both 'reasoning_effort'
            # and the native 'thinking' param.  We already expressed the
            # user's intent via the model-native shape, so drop the
            # redundant wire-level kwarg.  Only kimi models need this;
            # Xiaomi's API accepts both params.
            if slug in _KIMI_THINKING_MODELS:
                kwargs.pop("reasoning_effort", None)

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"

        # Backfill reasoning_content="" on assistants missing it: thinking
        # models reject history otherwise (#3554, #3584); "" reads as "no
        # thinking that turn".
        explicit_thinking = (
            reasoning_effort is not None
            and semantic_effort not in ("none", "minimal")
            and bool(model_style)
        )
        if explicit_thinking:
            for msg in kwargs["messages"]:
                if msg.get("role") == "assistant" and "reasoning_content" not in msg:
                    msg["reasoning_content"] = ""

        # Merge user-configured extra_body last so ordinary fields can override
        # defaults. Keep configured tools at the top level: the SDK
        # otherwise lets extra_body.tools replace nanobot's generated functions.
        if self._extra_body:
            kwargs = _merge_chat_extra_body(kwargs, self._extra_body)
        if extra_headers:
            kwargs["extra_headers"] = extra_headers

        return kwargs

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _maybe_mapping(value: object) -> dict[str, Any] | None:
        if isinstance(value, dict):
            return cast(dict[str, Any], value)
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            dumped: object = model_dump()
            if isinstance(dumped, dict):
                return cast(dict[str, Any], dumped)
        return None

    @classmethod
    def _extract_text_content(cls, value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts: list[str] = []
            for item in cast(list[object], value):
                item_map = cls._maybe_mapping(item)
                if item_map:
                    # Skip Mistral-style {"type":"thinking","thinking":[...]}
                    # blocks: their text belongs in reasoning_content.
                    if item_map.get("type") == "thinking":
                        continue
                    text = item_map.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                        continue
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
                    continue
                if isinstance(item, str):
                    parts.append(item)
            return "".join(parts) or None
        return str(value)

    @classmethod
    def _extract_thinking_content(cls, value: object) -> str | None:
        """Extract reasoning text from Mistral-style thinking blocks.

        Mistral returns content as a list mixing
        ``{"type":"thinking","thinking":[{"type":"text","text":...}]}`` and
        ``{"type":"text","text":...}``. The thinking text belongs in
        ``reasoning_content`` so the agent can surface it as a reasoning
        trace rather than as the assistant's reply.
        """
        if not isinstance(value, list):
            return None
        parts: list[str] = []
        for item in cast(list[object], value):
            item_map = cls._maybe_mapping(item)
            if not item_map:
                continue
            if item_map.get("type") != "thinking":
                continue
            inner = item_map.get("thinking")
            text = cls._extract_text_content(inner)
            if text:
                parts.append(text)
        return "".join(parts) or None

    @classmethod
    def _usage_object(cls, response: Any) -> Any:
        """The ``usage`` of a response or stream chunk (dict or SDK object), or None."""
        response_map = cls._maybe_mapping(response)
        if response_map is not None:
            return response_map.get("usage")
        if hasattr(response, "usage") and response.usage:
            return response.usage
        return None

    @classmethod
    def _extract_cost(cls, response: Any) -> float | None:
        """The money the request cost, in USD, as the gateway reported it; None when it did not.

        OpenRouter puts it in the usage of the final stream chunk: ``cost`` is the whole
        charge. For a BYOK request (``is_byok``) the upstream provider's bill is reported
        beside it in ``cost_details.upstream_inference_cost``; the two are added, which may
        count more than was charged and never less, the safe side for a cap.
        """
        usage_obj = cls._usage_object(response)
        cost = cls._get_nested_float(usage_obj, ("cost",))
        if cost is None:
            return None
        if cls._get_nested(usage_obj, ("is_byok",)) is True:
            cost += cls._get_nested_float(usage_obj, ("cost_details", "upstream_inference_cost")) or 0.0
        return cost

    @classmethod
    def _extract_usage(cls, response: Any) -> LLMUsage | None:
        """Extract token usage from an OpenAI-compatible response.

        Handles both dict-based (raw JSON) and object-based (SDK Pydantic)
        responses. Provider-specific cache fields are normalized once at
        this Chat Completions wire boundary.
        """
        usage_obj = cls._usage_object(response)
        usage_map = cls._maybe_mapping(usage_obj)
        if usage_map is not None:
            input_tokens = int(usage_map.get("prompt_tokens") or 0)
            output_tokens = int(usage_map.get("completion_tokens") or 0)
        elif usage_obj:
            input_tokens = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
            output_tokens = int(getattr(usage_obj, "completion_tokens", 0) or 0)
        else:
            return None

        wire_total = cls._get_nested_int(usage_obj, ("total_tokens",))

        cache_read: int | None = None
        # --- cached_tokens (normalised across Chat-compatible providers) ---
        # Try nested paths first (dict), fall back to attribute (SDK object).
        # Priority order ensures the most specific field wins.
        for path in (
            ("prompt_tokens_details", "cached_tokens"),  # OpenAI-style gateways
            ("cached_tokens",),                          # StepFun/Moonshot (top-level)
            ("prompt_cache_hit_tokens",),                # DeepSeek/SiliconFlow
        ):
            cached = cls._get_nested_int(usage_map, path)
            if cached is None and usage_obj:
                cached = cls._get_nested_int(usage_obj, path)
            if cached is not None:
                cache_read = cached
                break

        cache_write = cls._get_nested_int(
            usage_obj,
            ("prompt_tokens_details", "cache_write_tokens"),
        )

        return LLMUsage.reported(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=wire_total,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
        )

    @staticmethod
    def _get_nested(obj: object, path: tuple[str, ...]) -> object:
        """The value at `path` of a dict or an SDK object; None when a step is missing."""
        current: object = obj
        for segment in path:
            if current is None:
                return None
            if isinstance(current, dict):
                current = cast(dict[str, Any], current).get(segment)
            else:
                current = getattr(current, segment, None)
        return current

    @classmethod
    def _get_nested_float(cls, obj: object, path: tuple[str, ...]) -> float | None:
        """A reported amount of money: a finite number that is not negative (a bool, text or NaN is none)."""
        value = cls._get_nested(obj, path)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        amount = float(value)
        return amount if math.isfinite(amount) and amount >= 0 else None

    @classmethod
    def _get_nested_int(cls, obj: object, path: tuple[str, ...]) -> int | None:
        """Return a present usage count while preserving explicit zero.

        Supports both dict-key access and attribute access so it works
        uniformly with raw JSON dicts **and** SDK Pydantic models.
        """
        current = cls._get_nested(obj, path)
        if current is None or isinstance(current, bool):
            return None
        try:
            return int(cast(Any, current))
        except (TypeError, ValueError):
            return None

    @classmethod
    def _parse_chunks(cls, chunks: list[Any]) -> LLMResponse:
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tc_bufs: dict[int, dict[str, Any]] = {}
        finish_reason = "stop"
        usage: LLMUsage | None = None
        cost_usd: float | None = None

        def note_usage(chunk: Any) -> None:
            """Keep the last usage a chunk reported, and the cost with it."""
            nonlocal usage, cost_usd
            usage = cls._extract_usage(chunk) or usage
            cost = cls._extract_cost(chunk)
            if cost is not None:
                cost_usd = cost

        def _accum_tc(tc: Any, idx_hint: int) -> None:
            """Accumulate one streaming tool-call delta into *tc_bufs*."""
            tc_index: int = _get(tc, "index") if _get(tc, "index") is not None else idx_hint
            buf = tc_bufs.setdefault(tc_index, {
                "id": "", "name": "", "arguments": "",
                "extra_content": None, "prov": None, "fn_prov": None,
            })
            tc_id = _get(tc, "id")
            if tc_id:
                buf["id"] = str(tc_id)
            fn = _get(tc, "function")
            if fn is not None:
                fn_name = _get(fn, "name")
                if fn_name:
                    buf["name"] = str(fn_name)
                fn_args = _get(fn, "arguments")
                if fn_args:
                    buf["arguments"] += str(fn_args)
            ec, prov, fn_prov = _extract_tc_extras(tc)
            if ec:
                buf["extra_content"] = ec
            if prov:
                buf["prov"] = prov
            if fn_prov:
                buf["fn_prov"] = fn_prov

        def _accum_legacy_function_call(function_call: Any) -> None:
            """Accumulate legacy ``delta.function_call`` streaming chunks."""
            if not function_call:
                return
            buf = tc_bufs.setdefault(0, {
                "id": "", "name": "", "arguments": "",
                "extra_content": None, "prov": None, "fn_prov": None,
            })
            fn_name = _get(function_call, "name")
            if fn_name:
                buf["name"] = str(fn_name)
            fn_args = _get(function_call, "arguments")
            if fn_args:
                buf["arguments"] += str(fn_args)

        for chunk in chunks:
            if isinstance(chunk, str):
                content_parts.append(chunk)
                continue

            chunk_map = cls._maybe_mapping(chunk)
            if chunk_map is not None:
                choices = cast(
                    list[object],
                    chunk_map.get("choices") or [],
                )
                if not choices:
                    note_usage(chunk_map)
                    text = cls._extract_text_content(
                        chunk_map.get("content") or chunk_map.get("output_text")
                    )
                    if text:
                        content_parts.append(text)
                    continue
                choice = cls._maybe_mapping(choices[0]) or {}
                if choice.get("finish_reason"):
                    finish_reason = str(choice["finish_reason"])
                delta = cls._maybe_mapping(choice.get("delta")) or {}
                raw_delta_content = delta.get("content")
                text = cls._extract_text_content(raw_delta_content)
                if text:
                    content_parts.append(text)
                text = cls._extract_text_content(delta.get("reasoning_content"))
                if not text:
                    text = cls._extract_text_content(delta.get("reasoning"))
                if not text:
                    # Mistral streams thinking inside the content array as
                    # {"type":"thinking", thinking:[{"type":"text", ...}]}.
                    text = cls._extract_thinking_content(raw_delta_content)
                if text:
                    reasoning_parts.append(text)
                for idx, tc in enumerate(
                    cast(
                        Iterable[object],
                        delta.get("tool_calls") or [],
                    )
                ):
                    _accum_tc(tc, idx)
                _accum_legacy_function_call(delta.get("function_call"))
                note_usage(chunk_map)
                continue

            if not chunk.choices:
                note_usage(chunk)
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            delta = choice.delta
            if delta and delta.content:
                text = cls._extract_text_content(delta.content)
                if text:
                    content_parts.append(text)
                thinking_text = cls._extract_thinking_content(delta.content)
                if thinking_text:
                    reasoning_parts.append(thinking_text)
            if delta:
                reasoning = getattr(delta, "reasoning_content", None)
                if not reasoning:
                    reasoning = getattr(delta, "reasoning", None)
                if reasoning:
                    text = cls._extract_text_content(reasoning)
                    if text:
                        reasoning_parts.append(text)
            delta_tool_calls = (
                cast(Iterable[object], getattr(delta, "tool_calls", None) or [])
                if delta
                else ()
            )
            for tc in delta_tool_calls:
                _accum_tc(tc, getattr(tc, "index", 0))
            if delta:
                _accum_legacy_function_call(getattr(delta, "function_call", None))

        # Some providers (e.g. Zhipu/GLM) reuse the same tool_call id for
        # parallel tool calls in streaming mode. Deduplicate before building
        # the response so downstream tool messages don't collide.
        _seen_tc_ids: set[str] = set()
        for b in tc_bufs.values():
            if not b["id"] or b["id"] in _seen_tc_ids:
                b["id"] = _short_tool_id()
            _seen_tc_ids.add(b["id"])

        content = "".join(content_parts) or None
        tool_calls = [
            ToolCallRequest(
                id=b["id"] or _short_tool_id(),
                name=b["name"],
                arguments=parse_tool_arguments(b["arguments"]),
                extra_content=b.get("extra_content"),
                provider_specific_fields=b.get("prov"),
                function_provider_specific_fields=b.get("fn_prov"),
            )
            for b in tc_bufs.values()
        ]
        if not tool_calls:
            content, tool_calls = _extract_text_tool_calls(content)

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            cost_usd=cost_usd,
            reasoning_content="".join(reasoning_parts) or None,
        )

    @classmethod
    def _extract_error_metadata(cls, e: Exception) -> dict[str, Any]:
        response = getattr(e, "response", None)
        headers = getattr(response, "headers", None)
        payload = (
            getattr(e, "body", None)
            or getattr(e, "doc", None)
            or getattr(response, "text", None)
        )
        if payload is None and response is not None:
            response_json = getattr(response, "json", None)
            if callable(response_json):
                try:
                    payload = response_json()
                except Exception:
                    payload = None
        error_type, error_code = LLMProvider._extract_error_type_code(payload)

        status_code = getattr(e, "status_code", None)
        if status_code is None and response is not None:
            status_code = getattr(response, "status_code", None)

        should_retry: bool | None = None
        if headers is not None:
            raw = headers.get("x-should-retry")
            if isinstance(raw, str):
                lowered = raw.strip().lower()
                if lowered == "true":
                    should_retry = True
                elif lowered == "false":
                    should_retry = False

        error_kind: str | None = None
        error_name = e.__class__.__name__.lower()
        if "timeout" in error_name:
            error_kind = "timeout"
        elif "connection" in error_name:
            error_kind = "connection"

        return {
            "error_status_code": int(status_code) if status_code is not None else None,
            "error_kind": error_kind,
            "error_type": error_type,
            "error_code": error_code,
            "error_retry_after_s": cls._extract_retry_after_from_headers(headers),
            "error_should_retry": should_retry,
        }

    def _handle_error(self, e: Exception) -> LLMResponse:
        """A failure of a request as an error response; its text is `failure_text`, the one owner."""
        msg = self.failure_text(e)

        response = getattr(e, "response", None)
        retry_after = LLMProvider._extract_retry_after_from_headers(getattr(response, "headers", None))
        if retry_after is None:
            retry_after = LLMProvider._extract_retry_after(msg)
        return LLMResponse(
            content=msg,
            finish_reason="error",
            retry_after=retry_after,
            **OpenAICompatProvider._extract_error_metadata(e),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def model_limits(self, model: str) -> ModelLimits:
        """The limits the endpoint publishes for `model` (``GET /models``), read once per provider.

        OpenRouter gives each model the context window and the longest answer of the provider it routes to by
        default (``top_provider``); a model with none published there (a router) has unknown limits. A model the
        endpoint does not list is an error: a request for it would fail too.
        """
        async with self._model_limits_lock:
            if self._model_limits is None:
                client = await self._ensure_client()
                listed: dict[str, ModelLimits] = {}
                async for item in client.models.list():
                    top = (item.model_extra or {}).get("top_provider") or {}
                    context, answer = top.get("context_length"), top.get("max_completion_tokens")
                    listed[item.id] = ModelLimits(
                        context_tokens=context if isinstance(context, int) and context > 0 else None,
                        answer_tokens=answer if isinstance(answer, int) and answer > 0 else None,
                    )
                self._model_limits = listed
        found = self._model_limits.get(model)
        if found is None:
            raise LookupError(f"{self.provider_name} does not list the model {model}")
        return found

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
        client = await self._ensure_client()
        idle_timeout_s = STREAM_IDLE_TIMEOUT_S
        try:
            kwargs = self._build_kwargs(
                messages, tools, model, max_tokens, temperature,
                reasoning_effort, tool_choice,
            )
            kwargs["stream"] = True
            kwargs["timeout"] = idle_timeout_s
            kwargs["stream_options"] = {"include_usage": True}
            chat_stream = cast(
                Any,
                await client.chat.completions.create(**kwargs),
            )
            chunks: list[Any] = []
            completed = False
            stream_iter: AsyncIterator[Any] = chat_stream.__aiter__()
            while True:
                try:
                    chunk: Any = await asyncio.wait_for(
                        stream_iter.__anext__(),
                        timeout=idle_timeout_s,
                    )
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    raise StreamIdleTimeout(
                        f"stream stalled for more than {idle_timeout_s:g} seconds"
                    ) from None
                chunks.append(chunk)
                if chunk.choices:
                    completed |= bool(chunk.choices[0].finish_reason)
            if not completed:
                raise ConnectionError("Model stream ended before a finish reason was received")
            response = self._parse_chunks(chunks)
            usage = response.usage
            if usage is not None and usage.input_tokens > 0 and usage.estimated_tokens == 0:
                self._prompt_counts.observe(model or self.default_model, messages, tools, usage.input_tokens)
            return response
        except Exception as e:
            return self._handle_error(e)

    def estimate_prompt_tokens(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, model: str | None
    ) -> tuple[int, str]:
        """The tokens this endpoint will count for the prompt, from what it counted before (prompt_count.py)."""
        return self._prompt_counts.estimate(model or self.default_model, messages, tools)

    def get_default_model(self) -> str:
        return self.default_model
