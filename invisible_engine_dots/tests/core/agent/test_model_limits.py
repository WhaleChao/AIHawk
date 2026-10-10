"""Every request uses its model's own limits: the whole context window, and the longest answer the model gives.

A request is compacted only at the window, less the room kept for the answer; compacting first clears old tool
results and writes a summary only when that is not enough.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.runner_helpers import make_run_spec
from nanobot.agent.context_governance import (
    ANSWER_ROOM_TOKENS,
    CLEAR_MINIMUM_TOKENS,
    CLEAR_PROTECT_TOKENS,
    CLEARED_TOOL_RESULT,
    ContextCompactionState,
    ContextGovernanceConfig,
    ContextGovernor,
    ModelRequestState,
    answer_limit,
    prompt_budget,
)
from nanobot.providers.base import CONTEXT_SAFETY_BUFFER, LLMProvider


@pytest.mark.parametrize(
    ("window", "longest_answer", "budget"),
    [
        # OpenRouter's numbers for three models (2026-10-08): the room for the answer is 20000 tokens, or the
        # model's longest answer when shorter, so a model with a huge longest answer still uses its window.
        (1_048_575, 943_717, 1_048_575 - ANSWER_ROOM_TOKENS - CONTEXT_SAFETY_BUFFER),
        (400_000, 128_000, 400_000 - ANSWER_ROOM_TOKENS - CONTEXT_SAFETY_BUFFER),
        (128_000, 4_000, 128_000 - 4_000 - CONTEXT_SAFETY_BUFFER),
        # A router that publishes no longest answer keeps the usual room.
        (2_000_000, None, 2_000_000 - ANSWER_ROOM_TOKENS - CONTEXT_SAFETY_BUFFER),
        # No window known: nothing is compacted.
        (None, 64_000, 0),
    ],
)
def test_a_prompt_may_take_the_window_less_the_room_for_the_answer(window, longest_answer, budget) -> None:
    assert prompt_budget(window, longest_answer) == budget


@pytest.mark.parametrize(
    ("window", "longest_answer", "prompt", "sent"),
    [
        (1_000_000, 64_000, 10_000, 64_000),  # all of the longest answer fits
        (100_000, 64_000, 50_000, 100_000 - 50_000 - CONTEXT_SAFETY_BUFFER),  # what the window leaves
        (1_048_575, 943_717, 1_000_000, 1_048_575 - 1_000_000 - CONTEXT_SAFETY_BUFFER),
        (None, 64_000, 10_000, 64_000),  # no window known: the longest answer
        (400_000, None, 10_000, None),  # no longest answer published: none is sent
    ],
)
def test_a_request_sends_the_longest_answer_its_window_leaves_room_for(window, longest_answer, prompt, sent) -> None:
    assert answer_limit(window, longest_answer, prompt) == sent


def _tokens_by_length(monkeypatch: pytest.MonkeyPatch) -> None:
    """A token is four characters of a message's content, for every estimate the governor makes."""

    def message_tokens(message: dict[str, Any]) -> int:
        return len(str(message.get("content") or "")) // 4

    def prompt_tokens(provider: Any, model: Any, messages: list[dict[str, Any]], tools: Any) -> tuple[int, str]:
        return sum(message_tokens(message) for message in messages), "test"

    monkeypatch.setattr("nanobot.agent.context_governance.estimate_message_tokens", message_tokens)
    monkeypatch.setattr("nanobot.agent.context_governance.estimate_prompt_tokens_chain", prompt_tokens)


def _tool_turns(count: int, tokens_each: int) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": "system"}, {"role": "user", "content": "do it"}]
    for index in range(count):
        call = {"id": f"c{index}", "type": "function", "function": {"name": "exec", "arguments": "{}"}}
        messages.append({"role": "assistant", "content": f"step {index}", "tool_calls": [call]})
        messages.append({"role": "tool", "tool_call_id": f"c{index}", "name": "exec", "content": "o" * (4 * tokens_each)})
    return messages


def test_old_tool_results_are_cleared_the_newest_and_every_call_stay(monkeypatch: pytest.MonkeyPatch) -> None:
    _tokens_by_length(monkeypatch)
    # Ten results of 10000 tokens: the newest four are the 40000 tokens kept, the six older free 60000.
    messages = _tool_turns(10, 10_000)

    cleared = ContextGovernor.clear_old_tool_results(messages)

    assert cleared is not None
    results = [m["content"] for m in cleared if m["role"] == "tool"]
    assert results[:6] == [CLEARED_TOOL_RESULT] * 6
    assert all(content == "o" * 40_000 for content in results[6:])
    # Calls, ids, reasoning and the person's message are untouched.
    assert [m for m in cleared if m["role"] != "tool"] == [m for m in messages if m["role"] != "tool"]
    assert [m["tool_call_id"] for m in cleared if m["role"] == "tool"] == [f"c{i}" for i in range(10)]
    assert CLEAR_PROTECT_TOKENS == 40_000 and CLEAR_MINIMUM_TOKENS == 20_000


def test_clearing_too_little_to_matter_clears_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    _tokens_by_length(monkeypatch)
    # Five results of 10000 tokens: only one is outside the newest 40000, and 10000 is under the 20000 minimum.
    assert ContextGovernor.clear_old_tool_results(_tool_turns(5, 10_000)) is None


def _request_state(messages: list[dict[str, Any]], window: int, longest_answer: int, consolidate: Any) -> ModelRequestState:
    provider = MagicMock(spec=LLMProvider)
    spec = make_run_spec(
        provider,
        initial_messages=messages,
        tools=MagicMock(),
        model="m",
        context_window_tokens=window,
        max_tokens=longest_answer,
        max_iterations=1,
        max_tool_result_chars=1_000_000,
        consolidate_history=consolidate,
    )
    _, compaction = ContextCompactionState.from_transcript(
        spec.transcript_input, spec.transcript_builder, spec.consolidate_history,
    )
    return ModelRequestState(
        config=ContextGovernanceConfig(
            provider=provider,
            model="m",
            tools=MagicMock(),
            workspace=None,
            session_key="s",
            max_tool_result_chars=1_000_000,
            context_window_tokens=window,
            max_tokens=longest_answer,
        ),
        compaction=compaction,
    )


async def test_a_request_over_the_window_that_fits_once_old_results_are_cleared_asks_for_no_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _tokens_by_length(monkeypatch)
    consolidate = AsyncMock(return_value="a summary")
    # 100000 tokens of results in a 100000 token window: over its budget; cleared, 40000 are left.
    messages = _tool_turns(10, 10_000)
    state = _request_state(messages, window=100_000, longest_answer=64_000, consolidate=consolidate)

    prepared = await ContextGovernor().prepare_request(state, messages, tool_definitions=[])

    consolidate.assert_not_awaited()
    assert [m["content"] for m in prepared if m["role"] == "tool"][:6] == [CLEARED_TOOL_RESULT] * 6
    prompt = sum(len(str(m.get("content") or "")) // 4 for m in prepared)
    assert state.answer_tokens == min(64_000, 100_000 - prompt - CONTEXT_SAFETY_BUFFER)


async def test_a_request_still_over_the_window_once_cleared_is_summarized(monkeypatch: pytest.MonkeyPatch) -> None:
    _tokens_by_length(monkeypatch)
    consolidate = AsyncMock(return_value="a summary")
    # The newest 40000 tokens of results alone are over a 40000 token window: clearing cannot be enough.
    messages = _tool_turns(10, 10_000)
    state = _request_state(messages, window=40_000, longest_answer=4_000, consolidate=consolidate)

    await ContextGovernor().prepare_request(state, messages, tool_definitions=[])

    consolidate.assert_awaited_once()


async def test_a_request_under_the_window_is_sent_as_it_is_with_the_longest_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _tokens_by_length(monkeypatch)
    consolidate = AsyncMock(return_value="a summary")
    messages = _tool_turns(2, 1_000)
    state = _request_state(messages, window=1_000_000, longest_answer=64_000, consolidate=consolidate)

    prepared = await ContextGovernor().prepare_request(state, messages, tool_definitions=[])

    consolidate.assert_not_awaited()
    assert CLEARED_TOOL_RESULT not in [m.get("content") for m in prepared]
    assert state.answer_tokens == 64_000
