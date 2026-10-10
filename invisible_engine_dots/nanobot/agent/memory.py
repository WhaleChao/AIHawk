"""Transcript summaries: the LLM checkpoint of older messages, with a mechanical fallback.

Context governance replaces an accepted prefix of the transcript with one
checkpoint when the next request would not fit. ``Consolidator`` writes that
checkpoint. It owns no files: the summary is returned to its caller, which
stores it with the session.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from loguru import logger

from nanobot.agent.context_governance import answer_limit, prompt_budget
from nanobot.providers.base import LLMResponse
from nanobot.session.history_visibility import is_hidden_history_message
from nanobot.session.summary import SUMMARY_CONTINUATION_TEXT
from nanobot.utils.helpers import (
    build_assistant_message,
    content_with_media_breadcrumbs,
    estimate_message_tokens,
    estimate_prompt_tokens_chain,
    strip_think,
    truncate_text_to_tokens,
)
from nanobot.utils.prompt_templates import render_template

if TYPE_CHECKING:
    from nanobot.utils.llm_runtime import LLMRuntime

# The person's latest messages are kept as they wrote them after the summary, newest first up to this many tokens
# (Codex's COMPACT_USER_MESSAGE_MAX_TOKENS, Apache-2.0: github.com/openai/codex, codex-rs/core/src/compact.rs).
RECENT_USER_MESSAGE_TOKENS = 20_000


def recent_user_message_tokens(prompt_budget: int) -> int:
    """How much of the person's latest messages a summary keeps as they wrote them: Codex's 20000 tokens, but no
    more than a quarter of the prompt budget of the requests the summary goes into (the share of recent context
    opencode keeps whole), so that a model with a small window still has room for the work after the summary.
    A budget of 0 (no window known) keeps the 20000."""
    if prompt_budget <= 0:
        return RECENT_USER_MESSAGE_TOKENS
    return min(RECENT_USER_MESSAGE_TOKENS, prompt_budget // 4)
_ARCHIVE_TOOL_RESULT = (
    "Session archival does not execute tools. Use only the supplied conversation and "
    "return the requested compact checkpoint now; do not call another tool."
)


def _normalize_summary(entry: str) -> str:
    """Return the model-safe text of a checkpoint: its thinking stripped."""
    return strip_think(entry.rstrip())


def _format_messages(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in messages:
        content = content_with_media_breadcrumbs(
            message.get("role"),
            message.get("content", ""),
            message.get("media"),
        )
        if not content:
            continue
        tools_used = message.get("tools_used")
        tools = (
            f" [tools: {', '.join(cast(list[str], tools_used))}]"
            if tools_used
            else ""
        )
        raw_timestamp = message.get("timestamp")
        timestamp = str(raw_timestamp) if raw_timestamp is not None else "?"
        role = str(message.get("role") or "unknown")
        lines.append(f"[{timestamp[:16]}] {role.upper()}{tools}: {content}")
    return "\n".join(lines)


def _build_raw_checkpoint(messages: list[dict[str, Any]]) -> str:
    """The mechanical checkpoint: the messages themselves, formatted, their thinking stripped."""
    return _normalize_summary(f"[RAW] {len(messages)} messages\n{_format_messages(messages)}")


def _combine_raw_checkpoint(
    raw: str,
    *,
    previous_summary: str | None,
    max_tokens: int | None,
) -> str:
    """Return a checkpoint that preserves prior and newly archived context, within `max_tokens` when one is given."""
    if not previous_summary and not max_tokens:
        return raw
    if not max_tokens:
        return f"[Previous archived context]\n{previous_summary}\n\n[Newly archived raw context]\n{raw}"
    token_limit = max_tokens
    if not previous_summary:
        return truncate_text_to_tokens(raw, token_limit)

    combined = (
        "[Previous archived context]\n"
        f"{previous_summary}\n\n"
        "[Newly archived raw context]\n"
        f"{raw}"
    )
    bounded = truncate_text_to_tokens(combined, token_limit)
    if bounded == combined:
        return combined

    # Keep evidence from both sides when their full concatenation cannot fit.
    section_limit = max(1, (token_limit - 32) // 2)
    return truncate_text_to_tokens(
        "[Previous archived context]\n"
        f"{truncate_text_to_tokens(previous_summary, section_limit)}\n\n"
        "[Newly archived raw context]\n"
        f"{truncate_text_to_tokens(raw, section_limit)}",
        token_limit,
    )


def _drop_oldest_message(messages: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """Remove from `messages` the oldest one after the instructions, with the results of its calls, and return
    what was removed; None when nothing but the instructions and the request for the summary (the last message)
    is left."""
    first = 0
    while first < len(messages) and messages[first].get("role") in {"system", "developer"}:
        first += 1
    if first >= len(messages) - 1:
        return None
    end = first + 1
    while end < len(messages) - 1 and messages[end].get("role") == "tool":
        end += 1
    removed = messages[first:end]
    del messages[first:end]
    return removed


def _with_recent_user_messages(summary: str, messages: list[dict[str, Any]], limit: int) -> str:
    """The summary, then the person's latest messages as they wrote them: newest first up to `limit` tokens, the
    one that crosses the limit cut to fit, put back in their order."""
    kept: list[str] = []
    used = 0
    for message in reversed(messages):
        if message.get("role") != "user" or is_hidden_history_message(message):
            continue
        text = content_with_media_breadcrumbs("user", message.get("content", ""), message.get("media"))
        if not isinstance(text, str) or not text.strip() or text == SUMMARY_CONTINUATION_TEXT:
            continue
        tokens = estimate_message_tokens({"role": "user", "content": text})
        if used + tokens > limit:
            if limit - used > 0:
                kept.append(truncate_text_to_tokens(text, limit - used))
            break
        kept.append(text)
        used += tokens
    if not kept:
        return summary
    latest = "\n\n---\n\n".join(reversed(kept))
    return f"{summary}\n\n## The person's latest messages, as they wrote them\n\n{latest}"


class Consolidator:
    """Summarize a transcript prefix into one replacement checkpoint."""

    async def summarize(
        self,
        source_messages: list[dict[str, Any]],
        *,
        runtime: LLMRuntime,
        session_key: str,
        history: list[dict[str, Any]],
        request_tools: list[dict[str, Any]],
        previous_summary: str | None = None,
        input_token_budget: int | None = None,
        fallback_max_tokens: int | None = None,
    ) -> str | None:
        """Generate a replacement checkpoint; fall back to a mechanical one."""
        if not source_messages:
            return None

        def raw_fallback() -> str:
            return _combine_raw_checkpoint(
                _build_raw_checkpoint(source_messages),
                previous_summary=previous_summary,
                max_tokens=fallback_max_tokens,
            )

        prompt = render_template(
            "agent/consolidator_archive.md",
            strip=True,
            archive_count=len(source_messages),
        )
        prompt_message = {"role": "user", "content": prompt}
        request_messages = [
            *[dict(message) for message in history],
            prompt_message,
        ]
        estimated = 0
        if input_token_budget:
            # Too long for the summary model: the oldest messages go, one at a time with the results of their
            # calls, until the rest fits (Codex drops its oldest item and tries again).
            estimated, source = estimate_prompt_tokens_chain(
                runtime.provider,
                runtime.model,
                request_messages,
                request_tools,
            )
            while estimated > input_token_budget:
                # Each message's own estimate says how many to drop; the whole is measured again after.
                over = estimated - input_token_budget
                while over > 0:
                    removed = _drop_oldest_message(request_messages)
                    if removed is None:
                        logger.warning(
                            "Summary input does not fit for {} with only the instructions left: {}/{} via {}; "
                            "using raw checkpoint",
                            session_key,
                            estimated,
                            input_token_budget,
                            source,
                        )
                        return raw_fallback()
                    over -= sum(estimate_message_tokens(message) for message in removed)
                estimated, source = estimate_prompt_tokens_chain(
                    runtime.provider,
                    runtime.model,
                    request_messages,
                    request_tools,
                )

        response: LLMResponse | None = None
        for attempt in range(2):
            try:
                response = await runtime.provider.chat_stream_with_retry(
                    model=runtime.model,
                    messages=request_messages,
                    tools=request_tools,
                    temperature=runtime.generation.temperature,
                    max_tokens=answer_limit(runtime.context_window_tokens, runtime.generation.max_tokens, estimated),
                    reasoning_effort=runtime.generation.reasoning_effort,
                )
            except Exception:
                phase = "provider call" if attempt == 0 else "tool-call recovery"
                logger.warning(
                    "Summary {} failed; using raw checkpoint",
                    phase,
                )
                return raw_fallback()
            if response.should_execute_tools is not True or attempt == 1:
                break

            logger.info(
                "Summary provider returned {} tool call(s); requesting checkpoint",
                len(response.tool_calls),
            )
            assistant_message = build_assistant_message(
                response.content,
                tool_calls=[call.to_openai_tool_call() for call in response.tool_calls],
                reasoning_content=response.reasoning_content,
                thinking_blocks=response.thinking_blocks,
            )
            tool_messages = [
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": _ARCHIVE_TOOL_RESULT,
                }
                for call in response.tool_calls
            ]
            request_messages = [
                *request_messages,
                assistant_message,
                *tool_messages,
            ]
        assert response is not None
        if response.finish_reason in {"error", "length"}:
            logger.warning(
                "Summary provider did not complete ({}); using raw checkpoint",
                response.finish_reason,
            )
            return raw_fallback()
        if response.has_tool_calls is True:
            logger.warning("Summary provider returned tool calls; using raw checkpoint")
            return raw_fallback()
        summary = response.content
        if not summary or not summary.strip():
            logger.warning("Summary provider returned no summary; using raw checkpoint")
            return raw_fallback()
        summary = _normalize_summary(summary)
        if not summary:
            logger.warning(
                "Summary provider output was not safe to replay; using raw checkpoint"
            )
            return raw_fallback()
        return summary

    async def summarize_transcript(
        self,
        accepted_messages: list[dict[str, Any]],
        previous_summary: str | None,
        *,
        runtime: LLMRuntime,
        session_key: str,
        tools: list[dict[str, Any]],
        recent_user_tokens: int = RECENT_USER_MESSAGE_TOKENS,
    ) -> str | None:
        """Summarize the exact transcript prefix already accepted by the model; the person's latest messages, up to
        `recent_user_tokens`, follow the summary as they wrote them (`recent_user_message_tokens`)."""
        source_messages = [
            dict(message)
            for message in accepted_messages
            if message.get("role") != "system"
        ]
        if not source_messages:
            return None

        # The summary model's own window, less the room for its answer: what its request may hold. A mechanical
        # checkpoint takes at most half of it, so the request it is replayed in leaves room for the work after it.
        input_token_budget = prompt_budget(runtime.context_window_tokens, runtime.generation.max_tokens)
        summary = await self.summarize(
            source_messages,
            runtime=runtime,
            session_key=session_key,
            history=accepted_messages,
            request_tools=tools,
            previous_summary=previous_summary,
            input_token_budget=input_token_budget,
            fallback_max_tokens=input_token_budget // 2 or None,
        )
        if summary is None:
            return None
        return _with_recent_user_messages(summary, source_messages, recent_user_tokens)
