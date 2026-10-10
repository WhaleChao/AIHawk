"""Shared execution loop for tool-using agents."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from loguru import logger

from nanobot.agent.context import TranscriptInput
from nanobot.agent.context_governance import (
    ContextCompactionState,
    ContextGovernanceConfig,
    ContextGovernor,
    HistoryConsolidator,
    ModelRequestState,
    TranscriptBuilder,
)
from nanobot.agent.hook import AgentHook, AgentHookContext, AgentRunHookContext
from nanobot.agent.tools.context import tool_log_content_allowed
from nanobot.agent.tools.execution import STATUS_PARKED, execute_tool_calls
from nanobot.agent.tools.gate_types import ToolGate
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.transcript_metadata import IS_ERROR, METADATA_KEY
from nanobot.events import NO_EVENTS, EventSink
from nanobot.providers.base import (
    LLMProvider,
    LLMResponse,
    LLMUsage,
    ProviderCallContext,
    ToolCallRequest,
)
from nanobot.session.history_visibility import is_hidden_history_message
from nanobot.session.summary import SessionSummaryCheckpoint
from nanobot.utils.helpers import (
    build_assistant_message,
    estimate_message_tokens,
    estimate_prompt_tokens_chain,
    extract_reasoning,
)
from nanobot.utils.llm_runtime import LLMRuntime
from nanobot.utils.prompt_templates import render_template
from nanobot.utils.runtime import (
    EMPTY_FINAL_RESPONSE_MESSAGE,
    build_finalization_retry_message,
    build_length_recovery_message,
    is_blank_text,
)

CheckpointCallback = Callable[[dict[str, Any]], Awaitable[None]]
InjectionCallback = Callable[[], Awaitable[Iterable[Any] | None]]

_DEFAULT_ERROR_MESSAGE = "Sorry, I encountered an error calling the AI model."
_ARREARAGE_ERROR_MESSAGE = (
    "The AI provider rejected the request because the API key is out of quota or the "
    "account is in arrears. Please top up / check the billing status of your API key and try again."
)
_PERSISTED_MODEL_ERROR_PLACEHOLDER = "[Assistant reply unavailable due to model error.]"
_MAX_EMPTY_RETRIES = 2
_MAX_LENGTH_RECOVERIES = 3


def _restore_outer_whitespace(content: str, original: str | None) -> str:
    """Restore boundary whitespace stripped while cleaning one recovered segment."""
    if not original:
        return content
    leading_size = len(original) - len(original.lstrip())
    trailing_size = len(original) - len(original.rstrip())
    leading = original[:leading_size]
    trailing = original[-trailing_size:] if trailing_size else ""
    return f"{leading}{content}{trailing}"


@dataclass(slots=True)
class AgentRunSpec:
    """Configuration for a single agent execution."""

    tools: ToolRegistry
    runtime: LLMRuntime
    max_iterations: int
    max_tool_result_chars: int
    # The policy every tool call crosses before it runs: there is no run without one.
    gate: ToolGate
    # The turn's inputs, and how they become the transcript the model reads.
    transcript_input: TranscriptInput
    transcript_builder: TranscriptBuilder
    consolidate_history: HistoryConsolidator
    hook: AgentHook | None = None
    concurrent_tools: bool = False
    workspace: Path | None = None
    session_key: str | None = None
    checkpoint_callback: CheckpointCallback | None = None
    injection_callback: InjectionCallback | None = None
    events: EventSink = NO_EVENTS
    # Given the messages of a model request, returns the ones to send. For what the model must see
    # and the transcript must not keep (a screenshot): the result is made for that request, never stored
    # and never part of the history the next request starts from.
    request_attachments: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None


@dataclass(slots=True)
class AgentRunResult:
    """Outcome of a shared agent execution."""

    final_content: str | None
    messages: list[dict[str, Any]]
    tools_used: list[str] = field(default_factory=list)
    usage: LLMUsage | None = None
    # One entry per runner-visible model round. Recovery dispatches needed to
    # produce that round's response are folded into the same usage value.
    round_usages: list[LLMUsage] = field(default_factory=list)
    stop_reason: str = "completed"
    error: str | None = None
    failure_error_kind: str | None = None
    tool_events: list[dict[str, str]] = field(default_factory=list)
    had_injections: bool = False
    summary_checkpoint: SessionSummaryCheckpoint | None = field(default=None, repr=False)


class AgentRunner:
    """Run a tool-capable LLM loop without product-layer concerns."""

    def __init__(self) -> None:
        self.context_governor = ContextGovernor()

    async def _commit(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        message: dict[str, Any],
        phase: str,
    ) -> None:
        """Add one message to the transcript, then hand it to the checkpoint callback.

        Every message the runner adds goes through here, one at a time, and the
        runner goes on only after the callback returned: whatever it durably records
        is never behind the transcript. The callback gets {"phase", "message"}; the
        phases are assistant_tool_calls, tool_result, final_response, injected_user,
        length_segment, length_notice, error_placeholder, empty_final_response and
        max_iterations_fallback. The message may carry the engine's metadata under
        METADATA_KEY: the callback sees it, the transcript the model reads never does.
        The callback gets a snapshot: the runner goes on marking its own copy.
        """
        if METADATA_KEY in message:
            messages.append({key: value for key, value in message.items() if key != METADATA_KEY})
        else:
            messages.append(message)
        callback = spec.checkpoint_callback
        if callback is not None:
            await callback({"phase": phase, "message": deepcopy(message)})

    async def _commit_notice(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        content: str | None,
        phase: str,
    ) -> None:
        """Commit a closing assistant text, unless the transcript already ends with it."""
        if not content:
            return
        last = messages[-1] if messages else None
        if (
            last is not None
            and last.get("role") == "assistant"
            and not last.get("tool_calls")
            and last.get("content") == content
        ):
            return
        await self._commit(spec, messages, build_assistant_message(content), phase)

    async def _try_drain_injections(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        assistant_message: dict[str, Any] | None,
        injection_cycles: int,
        *,
        phase: str = "after error",
        drain_callback: bool = True,
    ) -> tuple[bool, int]:
        """Append one pending-input snapshot and return whether execution continues."""
        injections = await self._drain_injections(spec) if drain_callback else []
        if not injections:
            return False, injection_cycles
        injection_cycles += 1
        if assistant_message is not None:
            await self._commit(spec, messages, assistant_message, "final_response")
        for injection in injections:
            await self._commit(spec, messages, injection, "injected_user")
        preview = "[content hidden]"
        if tool_log_content_allowed():
            preview = "\n\n".join(
                message["content"] for message in injections
                if isinstance(message.get("content"), str)
                and not is_hidden_history_message(message)
            )
            preview = preview[:80] + "..." if len(preview) > 80 else preview
        logger.info(
            "Injected {} follow-up message(s) {} (snapshot {}): {}",
            len(injections), phase, injection_cycles, preview,
        )
        return True, injection_cycles

    async def _drain_injections(self, spec: AgentRunSpec) -> list[dict[str, Any]]:
        """Drain one pending-input snapshot via the injection callback."""
        callback = spec.injection_callback
        if callback is None:
            return []
        try:
            items = await callback()
        except Exception:
            logger.opt(exception=tool_log_content_allowed()).error("injection_callback failed")
            return []
        if not items:
            return []
        injected_messages: list[dict[str, Any]] = []
        for item in items:
            if item is None:
                continue
            if isinstance(item, dict):
                message_item = cast(dict[str, Any], item)
                if message_item.get("role") == "user" and "content" in message_item:
                    if self._has_injection_content(message_item.get("content")):
                        injected_messages.append(message_item)
                continue
            content = getattr(item, "content") if hasattr(item, "content") else str(item)
            if self._has_injection_content(content):
                injected_messages.append({"role": "user", "content": content})
        return injected_messages

    @staticmethod
    def _has_injection_content(content: Any) -> bool:
        if content is None:
            return False
        if isinstance(content, str):
            return bool(content.strip())
        if isinstance(content, list):
            return bool(cast(list[Any], content))
        return True

    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        hook = spec.hook or AgentHook()
        messages, compaction = self._initial_transcript_and_compaction(spec)
        context = AgentRunHookContext(messages=deepcopy(messages))

        try:
            await hook.before_run(context)
            result = await self._run_core(spec, hook, messages, compaction)
        except asyncio.CancelledError as exc:
            context.messages = deepcopy(messages)
            context.stop_reason = "cancelled"
            context.error = None
            context.exception = exc
            raise
        except Exception as exc:
            context.messages = deepcopy(messages)
            context.stop_reason = "error"
            context.error = f"Error: {type(exc).__name__}: {exc}"
            context.exception = exc
            await hook.on_error(context)
            raise
        else:
            context.messages = deepcopy(result.messages)
            context.final_content = result.final_content
            context.tools_used = list(result.tools_used)
            context.usage = result.usage
            context.stop_reason = result.stop_reason
            context.error = result.error
            context.tool_events = deepcopy(result.tool_events)
            context.had_injections = result.had_injections
            context.exception = None
            if context.error is not None:
                await hook.on_error(context)
            await hook.after_run(context)
            return result
        finally:
            context.messages = deepcopy(messages)
            if context.exception is None:
                await hook.on_finally(context)
            else:
                try:
                    await hook.on_finally(context)
                except Exception:
                    logger.opt(exception=tool_log_content_allowed()).error(
                        "AgentHook.on_finally error after {}",
                        context.stop_reason or "run exception",
                    )

    @staticmethod
    def _initial_transcript_and_compaction(
        spec: AgentRunSpec,
    ) -> tuple[list[dict[str, Any]], ContextCompactionState]:
        """Build the initial transcript and its compaction state."""
        return ContextCompactionState.from_transcript(
            spec.transcript_input,
            spec.transcript_builder,
            spec.consolidate_history,
        )

    async def _run_core(
        self,
        spec: AgentRunSpec,
        hook: AgentHook,
        messages: list[dict[str, Any]],
        compaction: ContextCompactionState,
    ) -> AgentRunResult:
        final_content: str | None = None
        tools_used: list[str] = []
        usage: LLMUsage | None = None
        round_usages: list[LLMUsage] = []
        error: str | None = None
        failure_error_kind: str | None = None
        stop_reason = "completed"
        tool_events: list[dict[str, str]] = []
        empty_content_retries = 0
        # Segments from one uninterrupted length-recovery chain. Tool work or
        # injected user input starts a new logical answer and clears the chain.
        length_recovery_parts: list[str] = []
        pending_length_segment: str | None = None
        had_injections = False
        injection_cycles = 0
        governance_config = ContextGovernanceConfig(
            provider=spec.runtime.provider,
            model=spec.runtime.model,
            tools=spec.tools,
            workspace=spec.workspace,
            session_key=spec.session_key,
            max_tool_result_chars=spec.max_tool_result_chars,
            context_window_tokens=spec.runtime.context_window_tokens,
            max_tokens=spec.runtime.generation.max_tokens,
        )
        request_state = ModelRequestState(
            config=governance_config,
            compaction=compaction,
            events=spec.events,
        )

        async def end_length_segment(*, interrupted: bool) -> None:
            nonlocal pending_length_segment
            if pending_length_segment is None:
                return
            segment_content = pending_length_segment
            pending_length_segment = None
            if interrupted:
                length_recovery_parts.clear()
            else:
                await self._commit(
                    spec, messages, build_length_recovery_message(segment_content), "length_notice",
                )

        for iteration in range(spec.max_iterations):
            # A resumed iteration must not inherit a previous iteration's failure.
            stop_reason = "completed"
            error = None
            # The session inbox cuts a finite snapshot before every model call.
            # This includes follow-ups that arrived before the first request and
            # messages received while the previous request or tools were running.
            drained_before_request, injection_cycles = await self._try_drain_injections(
                spec,
                messages,
                None,
                injection_cycles,
                phase="before model call",
            )
            if drained_before_request:
                had_injections = True
            await end_length_segment(interrupted=drained_before_request)
            context = AgentHookContext(
                iteration=iteration,
                messages=messages,
                session_key=spec.session_key,
            )
            await hook.before_iteration(context)
            request_message_count = len(messages)
            request_messages = request_state.compaction.request_messages(messages)
            response, raw_usage = await self._request_model(
                spec,
                request_messages,
                request_state=request_state,
            )
            assert request_state.messages is not None
            messages_for_model = request_state.messages
            request_state.compaction.accept_request(
                messages_for_model,
                raw_boundary=request_message_count,
            )
            context.response = response
            context.tool_calls = list(response.tool_calls)

            original_content = response.content
            _, cleaned_content = extract_reasoning(
                response.reasoning_content,
                response.thinking_blocks,
                response.content,
            )
            response.content = cleaned_content
            round_usages.append(raw_usage)
            context.usage = raw_usage
            usage = self._merge_usage(usage, raw_usage)

            if response.should_execute_tools:
                context.tool_calls = list(response.tool_calls)

                assistant_message = build_assistant_message(
                    response.content or "",
                    tool_calls=[tc.to_openai_tool_call() for tc in response.tool_calls],
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                await self._commit(spec, messages, assistant_message, "assistant_tool_calls")

                await hook.before_execute_tools(context)

                async def commit_tool_result(
                    tool_call: ToolCallRequest,
                    result: Any,
                    event: dict[str, str],
                ) -> None:
                    tool_message: dict[str, Any] = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": self.context_governor.normalize_tool_result(
                            governance_config,
                            tool_call.id,
                            tool_call.name,
                            result,
                        ),
                    }
                    if event["status"] == "error":
                        tool_message[METADATA_KEY] = {IS_ERROR: True}
                    await self._commit(spec, messages, tool_message, "tool_result")

                results, new_events = await execute_tool_calls(
                    spec.tools,
                    response.tool_calls,
                    concurrent=spec.concurrent_tools,
                    hook=hook,
                    context=context,
                    model_messages=messages_for_model,
                    gate=spec.gate,
                    on_result=commit_tool_result,
                )
                tool_events.extend(new_events)
                tools_used.extend(
                    tool_call.name
                    for tool_call, event in zip(response.tool_calls, new_events)
                    if event.get("status") == "ok"
                )
                context.tool_results = list(results)
                context.tool_events = list(new_events)
                if any(event["status"] == STATUS_PARKED for event in new_events):
                    # The user's decision comes first: no further model request in this turn.
                    stop_reason = "parked"
                    context.stop_reason = stop_reason
                    await hook.after_iteration(context)
                    break
                empty_content_retries = 0
                length_recovery_parts.clear()
                await hook.after_iteration(context)
                continue

            if response.has_tool_calls:
                logger.warning(
                    "Ignoring tool calls under finish_reason='{}' for {}",
                    response.finish_reason,
                    spec.session_key or "default",
                )

            clean = hook.finalize_content(context, response.content)
            if (
                response.finish_reason
                not in {"error", "length", "refusal", "content_filter"}
                and is_blank_text(clean)
            ):
                empty_content_retries += 1
                if empty_content_retries < _MAX_EMPTY_RETRIES:
                    logger.warning(
                        "Empty response on turn {} for {} ({}/{}); retrying",
                        iteration,
                        spec.session_key or "default",
                        empty_content_retries,
                        _MAX_EMPTY_RETRIES,
                    )
                    await hook.after_iteration(context)
                    continue
                logger.warning(
                    "Empty response on turn {} for {} after {} retries; attempting finalization",
                    iteration,
                    spec.session_key or "default",
                    empty_content_retries,
                )
                response = await self._request_finalization_retry(
                    spec,
                    messages_for_model,
                    request_state=request_state,
                )
                retry_usage = self._record_request_usage(spec, request_state, response)
                round_usages.append(retry_usage)
                usage = self._merge_usage(usage, retry_usage)
                raw_usage = self._merge_usage(raw_usage, retry_usage)
                context.response = response
                context.usage = raw_usage
                context.tool_calls = list(response.tool_calls)
                original_content = response.content
                clean = hook.finalize_content(context, response.content)

            if response.finish_reason == "length":
                if len(length_recovery_parts) < _MAX_LENGTH_RECOVERIES:
                    length_recovery_parts.append(
                        _restore_outer_whitespace(clean or "", original_content)
                    )
                    logger.info(
                        "Output truncated on turn {} for {} ({}/{}); continuing",
                        iteration,
                        spec.session_key or "default",
                        len(length_recovery_parts),
                        _MAX_LENGTH_RECOVERIES,
                    )
                    await self._commit(
                        spec,
                        messages,
                        build_assistant_message(
                            clean,
                            reasoning_content=response.reasoning_content,
                            thinking_blocks=response.thinking_blocks,
                        ),
                        "length_segment",
                    )
                    # The next input snapshot decides whether to continue this
                    # answer or close its stream before answering a new question.
                    pending_length_segment = clean or ""
                    await hook.after_iteration(context)
                    continue

            assistant_message: dict[str, Any] | None = None
            if response.finish_reason != "error" and not is_blank_text(clean):
                assistant_message = build_assistant_message(
                    clean,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )

            # Inputs that arrived while the model answered are taken in, and the run goes on.
            can_make_followup_request = iteration + 1 < spec.max_iterations
            should_continue, injection_cycles = await self._try_drain_injections(
                spec, messages, assistant_message, injection_cycles,
                phase="after final response",
                drain_callback=can_make_followup_request,
            )
            if should_continue:
                had_injections = True
                length_recovery_parts.clear()
                await hook.after_iteration(context)
                continue

            if response.finish_reason == "error":
                if LLMProvider.is_arrearage_response(response):
                    # The provider's own words follow: how much the balance still allows, where to add credit.
                    said = (clean or "").strip().removeprefix("Error:").strip()
                    final_content = f"{_ARREARAGE_ERROR_MESSAGE} The provider said: {said}" if said else _ARREARAGE_ERROR_MESSAGE
                else:
                    final_content = clean or _DEFAULT_ERROR_MESSAGE
                stop_reason = "error"
                error = final_content
                await self._commit_model_error_placeholder(spec, messages)
                context.final_content = final_content
                context.error = error
                context.stop_reason = stop_reason
                await hook.after_iteration(context)
                should_continue, injection_cycles = await self._try_drain_injections(
                    spec, messages, None, injection_cycles,
                    phase="after LLM error",
                    drain_callback=can_make_followup_request,
                )
                if should_continue:
                    had_injections = True
                    length_recovery_parts.clear()
                    continue
                failure_error_kind = LLMProvider.public_error_kind(response)
                break
            if is_blank_text(clean):
                final_content = EMPTY_FINAL_RESPONSE_MESSAGE
                stop_reason = "empty_final_response"
                error = final_content
                await self._commit_notice(
                    spec, messages, final_content, "empty_final_response",
                )
                context.final_content = final_content
                context.error = error
                context.stop_reason = stop_reason
                await hook.after_iteration(context)
                should_continue, injection_cycles = await self._try_drain_injections(
                    spec, messages, None, injection_cycles,
                    phase="after empty response",
                    drain_callback=can_make_followup_request,
                )
                if should_continue:
                    had_injections = True
                    length_recovery_parts.clear()
                    continue
                break

            await self._commit(
                spec,
                messages,
                assistant_message
                or build_assistant_message(
                    clean,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                ),
                "final_response",
            )
            if length_recovery_parts:
                final_content = (
                    "".join(length_recovery_parts)
                    + _restore_outer_whitespace(clean or "", original_content)
                ).strip()
            else:
                final_content = clean
            context.final_content = final_content
            context.stop_reason = stop_reason
            await hook.after_iteration(context)
            break
        else:
            stop_reason = "max_iterations"
            await end_length_segment(interrupted=False)
            terminal_content = self._max_iterations_fallback(spec)
            if length_recovery_parts:
                terminal_tail = f"\n\n{terminal_content.lstrip()}"
                final_content = (
                    "".join(length_recovery_parts).rstrip() + terminal_tail
                ).strip()
            else:
                final_content = terminal_content
            await self._commit_notice(
                spec, messages, terminal_content, "max_iterations_fallback",
            )

        return AgentRunResult(
            final_content=final_content,
            messages=messages,
            tools_used=tools_used,
            usage=usage,
            round_usages=round_usages,
            stop_reason=stop_reason,
            error=error,
            failure_error_kind=failure_error_kind,
            tool_events=tool_events,
            had_injections=had_injections,
            summary_checkpoint=request_state.compaction.summary_checkpoint,
        )

    def _build_request_kwargs(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        answer_tokens: int | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "messages": spec.request_attachments(messages) if spec.request_attachments else messages,
            "tools": tools,
            "model": spec.runtime.model,
        }
        generation = spec.runtime.generation
        kwargs["temperature"] = generation.temperature
        # The model's longest answer within what its window leaves (ContextGovernor.answer_tokens).
        kwargs["max_tokens"] = answer_tokens
        kwargs["reasoning_effort"] = generation.reasoning_effort
        return kwargs

    @staticmethod
    def _provider_context(spec: AgentRunSpec) -> ProviderCallContext:
        """Where a request reports its retries, and the preset its answer is attributed to."""
        return ProviderCallContext(events=spec.events, response_preset=spec.runtime.model_preset or "")

    async def _request_model(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        *,
        request_state: ModelRequestState,
        malformed_retry: bool = False,
    ) -> tuple[LLMResponse, LLMUsage]:
        tool_definitions = spec.tools.get_definitions()
        messages = await self.context_governor.prepare_request(
            request_state,
            messages,
            tool_definitions=tool_definitions,
        )

        kwargs = self._build_request_kwargs(
            spec,
            messages,
            tools=tool_definitions,
            answer_tokens=request_state.answer_tokens,
        )
        response = await spec.runtime.provider.chat_stream_with_retry(
            **kwargs,
            provider_context=self._provider_context(spec),
        )
        round_usage = self._record_request_usage(spec, request_state, response)
        dropped, all_dropped, original_finish_reason = (
            self._drop_malformed_tool_calls(response)
        )
        if (
            all_dropped
            and original_finish_reason in ("tool_calls", "function_call")
            and not malformed_retry
        ):
            logger.warning(
                "Retrying LLM request after all {} malformed tool call(s) were dropped",
                dropped,
            )
            retry_messages = self._malformed_tool_call_retry_messages(
                messages, response.content,
            )
            retry_response, retry_usage = await self._request_model(
                spec, retry_messages,
                request_state=request_state,
                malformed_retry=True,
            )
            return retry_response, round_usage + retry_usage
        if (
            all_dropped
            and original_finish_reason in ("tool_calls", "function_call")
            and malformed_retry
        ):
            logger.warning(
                "Malformed tool calls persisted after retry; falling back to no-tools request",
            )
            fallback_messages = self._malformed_tool_call_retry_messages(
                messages, response.content,
            )
            fallback_response = await self._request_no_tools(
                spec,
                fallback_messages,
                request_state=request_state,
            )
            fallback_usage = self._record_request_usage(
                spec,
                request_state,
                fallback_response,
            )
            return fallback_response, round_usage + fallback_usage
        return response, round_usage

    @staticmethod
    def _drop_malformed_tool_calls(
        response: LLMResponse,
    ) -> tuple[int, bool, str | None]:
        """Strip tool calls whose name is missing/non-string from the response.

        Returns (dropped_count, all_dropped, original_finish_reason).

        A degenerate call (name=None or "") cannot be executed, and if it were
        persisted into the assistant message it would be replayed on every
        subsequent turn, causing upstream validation errors
        (``tool_use.name: Input should be a valid string``) that permanently
        wedge the session. Dropping it here keeps it out of execution, the
        assistant message, and the saved history in one place.
        """
        calls = getattr(response, "tool_calls", None)
        if not calls:
            return (0, False, getattr(response, "finish_reason", None))
        valid = [tc for tc in calls if tc.has_valid_name()]
        if len(valid) == len(calls):
            return (0, False, getattr(response, "finish_reason", None))
        dropped = len(calls) - len(valid)
        original_finish_reason = getattr(response, "finish_reason", None)
        logger.warning(
            "Dropped {} malformed tool call(s) with missing/non-string name "
            "from LLM response (finish_reason={!r})",
            dropped,
            original_finish_reason,
        )
        response.tool_calls = valid
        if not valid:
            response.finish_reason = "stop"
        return (dropped, not valid, original_finish_reason)

    @staticmethod
    def _malformed_tool_call_retry_messages(
        messages: list[dict[str, Any]],
        assistant_text: str | None,
    ) -> list[dict[str, Any]]:
        retry_messages = list(messages)
        note = (
            "The previous model response attempted to call tools, but every tool call "
            "was malformed: the tool_use blocks had missing or non-string tool names. "
            "Do not answer with a promise to use tools. Either call the required tools again "
            "using valid tool names from the provided tool list and JSON object inputs, or give "
            "a final answer only if no tool is required."
        )
        if assistant_text:
            note += (
                f"\n\nPrevious assistant text before the malformed calls:\n"
                f"{assistant_text}"
            )
        retry_messages.append({"role": "user", "content": note})
        return retry_messages

    async def _request_finalization_retry(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        *,
        request_state: ModelRequestState,
    ) -> LLMResponse:
        retry_messages = self._finalization_retry_messages(messages)
        return await self._request_no_tools(
            spec,
            retry_messages,
            request_state=request_state,
        )

    @staticmethod
    def _finalization_retry_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        retry_messages = list(messages)
        retry_messages.append(build_finalization_retry_message())
        return retry_messages

    async def _request_no_tools(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        *,
        request_state: ModelRequestState,
    ) -> LLMResponse:
        messages = await self.context_governor.prepare_request(
            request_state,
            messages,
            tool_definitions=None,
        )
        kwargs = self._build_request_kwargs(
            spec,
            messages,
            tools=None,
            answer_tokens=request_state.answer_tokens,
        )
        return await spec.runtime.provider.chat_stream_with_retry(
            **kwargs,
            provider_context=self._provider_context(spec),
        )

    @staticmethod
    def _max_iterations_fallback(spec: AgentRunSpec) -> str:
        return render_template(
            "agent/max_iterations_message.md",
            strip=True,
            max_iterations=spec.max_iterations,
        )

    def _usage_or_estimate(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        response: LLMResponse,
        *,
        tool_definitions: list[dict[str, Any]] | None,
    ) -> LLMUsage:
        usage = response.usage
        if response.finish_reason == "error":
            if usage is None or usage.total_tokens == 0:
                usage = LLMUsage.empty_request()
        elif usage is None or usage.total_tokens == 0:
            usage = self._estimate_response_usage(
                spec,
                messages,
                response,
                tool_definitions=tool_definitions,
            )
        return usage.with_timing(
            generation_ms=response.generation_ms,
            ttft_ms=response.ttft_ms,
        )

    def _record_request_usage(
        self,
        spec: AgentRunSpec,
        state: ModelRequestState,
        response: LLMResponse,
    ) -> LLMUsage:
        assert state.messages is not None
        state.usage = self._usage_or_estimate(
            spec,
            state.messages,
            response,
            tool_definitions=state.tool_definitions,
        )
        return state.usage

    def _estimate_response_usage(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        response: LLMResponse,
        *,
        tool_definitions: list[dict[str, Any]] | None,
    ) -> LLMUsage:
        prompt_tokens, _ = estimate_prompt_tokens_chain(
            spec.runtime.provider,
            spec.runtime.model,
            messages,
            tool_definitions,
        )
        assistant_message = build_assistant_message(
            response.content or "",
            tool_calls=[tc.to_openai_tool_call() for tc in response.tool_calls],
            reasoning_content=response.reasoning_content,
            thinking_blocks=response.thinking_blocks,
        )
        completion_tokens = estimate_message_tokens(assistant_message)
        return LLMUsage.estimated(
            input_tokens=max(0, prompt_tokens),
            output_tokens=max(0, completion_tokens),
        )

    @staticmethod
    def _merge_usage(
        left: LLMUsage | None,
        right: LLMUsage | None,
    ) -> LLMUsage | None:
        if left is None:
            return right
        if right is None:
            return left
        return left + right

    async def _commit_model_error_placeholder(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
    ) -> None:
        """Keep the transcript legal after a failed request: an assistant reply stands in for it."""
        if messages and messages[-1].get("role") == "assistant" and not messages[-1].get("tool_calls"):
            return
        await self._commit(
            spec,
            messages,
            build_assistant_message(_PERSISTED_MODEL_ERROR_PLACEHOLDER),
            "error_placeholder",
        )
