"""Tests for core AgentRunner behavior: message passing, iteration limits,
empty-response handling, usage accumulation, and config passthrough."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.runner_helpers import ScriptedTools, make_run_spec
from nanobot.agent.context import TranscriptInput
from nanobot.agent.context_governance import ContextWindowExceededError
from nanobot.providers.base import (
    LLMProvider,
    LLMResponse,
    LLMUsage,
    ToolCallRequest,
)

_MAX_TOOL_RESULT_CHARS = 16_000


def _make_usage_spec(provider, tools):
    return make_run_spec(
        provider,
        initial_messages=[{"role": "user", "content": "hello"}],
        tools=tools,
        model="test-model",
        max_iterations=1,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
    )


def test_initial_transcript_is_built_from_structured_turn_input() -> None:
    from nanobot.agent.runner import AgentRunner

    provider = MagicMock(spec=LLMProvider)
    transcript_input = TranscriptInput(
        history=[{"role": "user", "content": "earlier"}],
        current_message="fresh",
    )
    expected = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "earlier"},
        {"role": "user", "content": "fresh"},
    ]
    transcript_builder = MagicMock(return_value=expected)
    spec = make_run_spec(
        provider,
        transcript_input=transcript_input,
        transcript_builder=transcript_builder,
        tools=MagicMock(),
        model="test-model",
        max_iterations=1,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
    )

    messages, compaction = AgentRunner._initial_transcript_and_compaction(spec)

    assert messages == expected
    assert compaction is not None
    assert compaction.raw_messages == expected
    transcript_builder.assert_called_once_with(transcript_input)


def test_usage_or_estimate_replaces_reported_zero_for_content(monkeypatch) -> None:
    from nanobot.agent.runner import AgentRunner

    provider = MagicMock(spec=LLMProvider)
    tools = MagicMock()
    tools.get_definitions.return_value = []
    monkeypatch.setattr(
        "nanobot.agent.runner.estimate_prompt_tokens_chain",
        lambda provider, model, messages, definitions: (12, "test"),
    )
    monkeypatch.setattr("nanobot.agent.runner.estimate_message_tokens", lambda message: 7)
    response = LLMResponse(
        content="answer",
        usage=LLMUsage.reported(input_tokens=0, output_tokens=0),
        generation_ms=25,
        ttft_ms=5,
    )

    usage = AgentRunner()._usage_or_estimate(
        _make_usage_spec(provider, tools),
        [{"role": "user", "content": "hello"}],
        response,
        tool_definitions=tools.get_definitions(),
    )

    assert usage == LLMUsage.estimated(input_tokens=12, output_tokens=7).with_timing(
        generation_ms=25,
        ttft_ms=5,
    )
    assert usage.source == "estimated"
    assert usage.total_tokens == 19


def test_usage_or_estimate_counts_tool_call_output_for_reported_zero(monkeypatch) -> None:
    from nanobot.agent.runner import AgentRunner

    provider = MagicMock(spec=LLMProvider)
    tools = MagicMock()
    tools.get_definitions.return_value = []
    captured_message: dict = {}
    monkeypatch.setattr(
        "nanobot.agent.runner.estimate_prompt_tokens_chain",
        lambda provider, model, messages, definitions: (13, "test"),
    )

    def estimate_output(message):
        captured_message.update(message)
        return 9

    monkeypatch.setattr("nanobot.agent.runner.estimate_message_tokens", estimate_output)
    response = LLMResponse(
        content=None,
        tool_calls=[
            ToolCallRequest(
                id="call_1",
                name="lookup",
                arguments={"query": "nanobot"},
            )
        ],
        finish_reason="tool_calls",
        usage=LLMUsage.reported(input_tokens=0, output_tokens=0),
    )

    usage = AgentRunner()._usage_or_estimate(
        _make_usage_spec(provider, tools),
        [{"role": "user", "content": "hello"}],
        response,
        tool_definitions=tools.get_definitions(),
    )

    assert usage == LLMUsage.estimated(input_tokens=13, output_tokens=9)
    assert usage.total_tokens == 22
    assert captured_message["tool_calls"][0]["function"]["name"] == "lookup"


@pytest.mark.parametrize(
    "provider_usage",
    [None, LLMUsage.reported(input_tokens=0, output_tokens=0)],
)
def test_usage_or_estimate_counts_error_without_estimating_tokens(
    monkeypatch,
    provider_usage: LLMUsage | None,
) -> None:
    from nanobot.agent.runner import AgentRunner

    provider = MagicMock(spec=LLMProvider)
    tools = MagicMock()
    estimate = MagicMock()
    runner = AgentRunner()
    monkeypatch.setattr(runner, "_estimate_response_usage", estimate)
    response = LLMResponse(
        content="upstream failed",
        finish_reason="error",
        usage=provider_usage,
    )

    usage = runner._usage_or_estimate(
        _make_usage_spec(provider, tools),
        [{"role": "user", "content": "hello"}],
        response,
        tool_definitions=tools.get_definitions(),
    )

    assert usage is not None
    assert usage.total_tokens == 0
    assert usage.request_count == 1
    assert usage.context_tokens is None
    aggregate = LLMUsage.reported(input_tokens=12, output_tokens=3) + usage
    assert aggregate.context_tokens == 12
    assert aggregate.request_count == 2
    estimate.assert_not_called()


def test_usage_or_estimate_trusts_positive_reported_total(monkeypatch) -> None:
    from nanobot.agent.runner import AgentRunner

    provider = MagicMock(spec=LLMProvider)
    tools = MagicMock()
    estimate = MagicMock()
    runner = AgentRunner()
    monkeypatch.setattr(runner, "_estimate_response_usage", estimate)
    response = LLMResponse(
        content="answer",
        usage=LLMUsage.reported(
            input_tokens=15,
            output_tokens=18,
            total_tokens=175,
        ),
        generation_ms=30,
        ttft_ms=6,
    )

    usage = runner._usage_or_estimate(
        _make_usage_spec(provider, tools),
        [{"role": "user", "content": "hello"}],
        response,
        tool_definitions=tools.get_definitions(),
    )

    assert usage is not None
    assert usage.source == "reported"
    assert usage.input_tokens == 15
    assert usage.output_tokens == 18
    assert usage.total_tokens == 175
    assert usage.reported_tokens == 175
    assert usage.generation_ms == 30
    assert usage.ttft_ms == 6
    estimate.assert_not_called()


@pytest.mark.asyncio
async def test_runner_preserves_reasoning_fields_and_tool_results():
    from nanobot.agent.runner import AgentRunner

    provider = MagicMock(spec=LLMProvider)
    captured_second_call: list[dict] = []
    call_count = {"n": 0}

    async def chat_stream_with_retry(*, messages, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return LLMResponse(
                content="thinking",
                tool_calls=[ToolCallRequest(id="call_1", name="list_dir", arguments={"path": "."})],
                reasoning_content="hidden reasoning",
                thinking_blocks=[{"type": "thinking", "thinking": "step"}],
                usage=LLMUsage.reported(input_tokens=5, output_tokens=3),
            )
        captured_second_call[:] = messages
        return LLMResponse(content="done", tool_calls=[], usage=None)

    provider.chat_stream_with_retry = chat_stream_with_retry
    tools = ScriptedTools(AsyncMock(return_value="tool result"), definitions=[])

    runner = AgentRunner()
    result = await runner.run(make_run_spec(provider,
        initial_messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "do task"},
        ],
        tools=tools,
        model="test-model",
        max_iterations=3,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
    ))

    assert result.final_content == "done"
    assert result.tools_used == ["list_dir"]
    assert result.tool_events == [
        {"name": "list_dir", "status": "ok", "detail": "tool result"}
    ]

    assistant_messages = [
        msg for msg in captured_second_call
        if msg.get("role") == "assistant" and msg.get("tool_calls")
    ]
    assert len(assistant_messages) == 1
    assert assistant_messages[0]["reasoning_content"] == "hidden reasoning"
    assert assistant_messages[0]["thinking_blocks"] == [{"type": "thinking", "thinking": "step"}]
    assert any(
        msg.get("role") == "tool" and msg.get("content") == "tool result"
        for msg in captured_second_call
    )


@pytest.mark.asyncio
async def test_runner_preserves_tool_result_before_rejecting_unfit_followup():
    from nanobot.agent.runner import AgentRunner

    provider = MagicMock(spec=LLMProvider)
    calls = 0
    checkpoints: list[dict] = []

    async def chat_stream_with_retry(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="read_file",
                        arguments={"path": "large.txt"},
                    ),
                ],
            )
        return LLMResponse(content="done")

    provider.chat_stream_with_retry = chat_stream_with_retry
    tools = ScriptedTools(AsyncMock(return_value="x" * 5_000), definitions=[])

    async def checkpoint(payload: dict) -> None:
        checkpoints.append(payload)

    with pytest.raises(ContextWindowExceededError):
        await AgentRunner().run(make_run_spec(
            provider,
            initial_messages=[
                {"role": "system", "content": "system"},
                {"role": "user", "content": "read the file"},
            ],
            tools=tools,
            model="gpt-5.6",
            context_window_tokens=2_224,
            max_tokens=1_000,
            max_iterations=3,
            max_tool_result_chars=10_000,
            checkpoint_callback=checkpoint,
        ))

    assert calls == 1
    tool_result = next(
        checkpoint["message"]
        for checkpoint in checkpoints
        if checkpoint["phase"] == "tool_result"
    )
    assert tool_result == {
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "read_file",
        "content": "x" * 5_000,
    }


@pytest.mark.asyncio
async def test_injected_final_response_is_committed_before_the_injected_message():
    from nanobot.agent.runner import AgentRunner

    provider = MagicMock(spec=LLMProvider)
    provider.chat_stream_with_retry = AsyncMock(side_effect=[
        LLMResponse(content="first answer"),
        LLMResponse(content="second answer"),
    ])
    tools = MagicMock()
    tools.get_definitions.return_value = []
    checkpoints: list[dict] = []
    # One snapshot per drain: before the first request, after the first answer (a
    # follow-up arrived while it was written), before the second request, after the second answer.
    injections = [[], [{"role": "user", "content": "follow up"}], [], []]

    async def checkpoint(payload: dict) -> None:
        checkpoints.append(payload)

    async def inject() -> list[dict]:
        return injections.pop(0)

    await AgentRunner().run(make_run_spec(
        provider,
        initial_messages=[{"role": "user", "content": "start"}],
        tools=tools,
        model="gpt-5.6",
        max_iterations=3,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        checkpoint_callback=checkpoint,
        injection_callback=inject,
    ))

    assert [(c["phase"], c["message"]["role"], c["message"]["content"]) for c in checkpoints] == [
        ("final_response", "assistant", "first answer"),
        ("injected_user", "user", "follow up"),
        ("final_response", "assistant", "second answer"),
    ]


@pytest.mark.asyncio
async def test_runner_returns_max_iterations_fallback():
    from nanobot.agent.runner import AgentRunner

    provider = MagicMock(spec=LLMProvider)
    provider.chat_stream_with_retry = AsyncMock(return_value=LLMResponse(
        content="still working",
        tool_calls=[ToolCallRequest(id="call_1", name="list_dir", arguments={"path": "."})],
    ))
    tools = ScriptedTools(AsyncMock(return_value="tool result"), definitions=[])

    runner = AgentRunner()
    result = await runner.run(make_run_spec(provider,
        initial_messages=[{"role": "user", "content": "start"}],
        tools=tools,
        model="test-model",
        max_iterations=2,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
    ))

    assert result.stop_reason == "max_iterations"
    assert result.final_content == (
        "I reached the maximum number of tool call iterations (2) "
        "without completing the task. You can try breaking the task into smaller steps."
    )
    assert result.messages[-1]["role"] == "assistant"
    assert result.messages[-1]["content"] == result.final_content
    # The run ends on its fallback message: no further request is made for a last answer.
    assert provider.chat_stream_with_retry.await_count == 2
    assert tools.execute.await_count == 2


@pytest.mark.asyncio
async def test_runner_replaces_empty_tool_result_with_marker():
    from nanobot.agent.runner import AgentRunner

    provider = MagicMock(spec=LLMProvider)
    captured_second_call: list[dict] = []
    call_count = {"n": 0}

    async def chat_stream_with_retry(*, messages, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return LLMResponse(
                content="working",
                tool_calls=[ToolCallRequest(id="call_1", name="noop", arguments={})],
                usage=None,
            )
        captured_second_call[:] = messages
        return LLMResponse(content="done", tool_calls=[], usage=None)

    provider.chat_stream_with_retry = chat_stream_with_retry
    tools = ScriptedTools(AsyncMock(return_value=""), definitions=[])

    runner = AgentRunner()
    result = await runner.run(make_run_spec(provider,
        initial_messages=[{"role": "user", "content": "do task"}],
        tools=tools,
        model="test-model",
        max_iterations=2,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
    ))

    assert result.final_content == "done"
    tool_message = next(msg for msg in captured_second_call if msg.get("role") == "tool")
    assert tool_message["content"] == "(noop completed with no output)"


@pytest.mark.asyncio
async def test_runner_retries_empty_final_response_with_summary_prompt():
    """Empty responses get 2 silent retries before finalization kicks in."""
    from nanobot.agent.runner import AgentRunner

    provider = MagicMock(spec=LLMProvider)
    calls: list[dict] = []

    async def chat_stream_with_retry(*, messages, tools=None, **kwargs):
        calls.append({"messages": messages, "tools": tools})
        if len(calls) <= 2:
            return LLMResponse(
                content=None,
                tool_calls=[],
                usage=LLMUsage.reported(input_tokens=5, output_tokens=1),
            )
        return LLMResponse(
            content="final answer",
            tool_calls=[],
            usage=LLMUsage.reported(input_tokens=3, output_tokens=7),
        )

    provider.chat_stream_with_retry = chat_stream_with_retry
    tools = MagicMock()
    tools.get_definitions.return_value = []

    runner = AgentRunner()
    result = await runner.run(make_run_spec(provider,
        initial_messages=[{"role": "user", "content": "do task"}],
        tools=tools,
        model="test-model",
        max_iterations=3,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
    ))

    assert result.final_content == "final answer"
    # 2 silent retries (iterations 0,1) + finalization on iteration 1
    assert len(calls) == 3
    assert calls[0]["tools"] is not None
    assert calls[1]["tools"] is not None
    assert calls[2]["tools"] is None
    assert result.usage is not None
    assert result.usage.input_tokens == 13
    assert result.usage.output_tokens == 9
    assert [(item.input_tokens, item.output_tokens) for item in result.round_usages] == [
        (5, 1),
        (5, 1),
        (3, 7),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason", ["refusal", "content_filter"])
async def test_runner_does_not_retry_blank_policy_terminal(
    finish_reason: str,
) -> None:
    from nanobot.agent.runner import AgentRunner
    from nanobot.utils.runtime import EMPTY_FINAL_RESPONSE_MESSAGE

    provider = MagicMock(spec=LLMProvider)
    provider.chat_stream_with_retry = AsyncMock(return_value=LLMResponse(
        content=None,
        finish_reason=finish_reason,
    ))
    tools = MagicMock()
    tools.get_definitions.return_value = []

    result = await AgentRunner().run(make_run_spec(
        provider,
        initial_messages=[{"role": "user", "content": "do task"}],
        tools=tools,
        model="test-model",
        max_iterations=3,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
    ))

    assert provider.chat_stream_with_retry.await_count == 1
    assert result.final_content == EMPTY_FINAL_RESPONSE_MESSAGE
    assert result.stop_reason == "empty_final_response"


@pytest.mark.asyncio
async def test_runner_uses_specific_message_after_empty_finalization_retry():
    """After silent retries + finalization all return empty, stop_reason is empty_final_response."""
    from nanobot.agent.runner import AgentRunner
    from nanobot.utils.runtime import EMPTY_FINAL_RESPONSE_MESSAGE

    provider = MagicMock(spec=LLMProvider)

    async def chat_stream_with_retry(*, messages, **kwargs):
        return LLMResponse(content=None, tool_calls=[], usage=None)

    provider.chat_stream_with_retry = chat_stream_with_retry
    tools = MagicMock()
    tools.get_definitions.return_value = []

    runner = AgentRunner()
    result = await runner.run(make_run_spec(provider,
        initial_messages=[{"role": "user", "content": "do task"}],
        tools=tools,
        model="test-model",
        max_iterations=3,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
    ))

    assert result.final_content == EMPTY_FINAL_RESPONSE_MESSAGE
    assert result.stop_reason == "empty_final_response"


@pytest.mark.asyncio
async def test_empty_finalization_retry_runs_no_tool_it_returns():
    from nanobot.agent.runner import AgentRunner

    provider = MagicMock(spec=LLMProvider)
    provider.chat_stream_with_retry = AsyncMock(side_effect=[
        LLMResponse(content=None, tool_calls=[], usage=None),
        LLMResponse(content=None, tool_calls=[], usage=None),
        LLMResponse(
            content="finalized without tools",
            tool_calls=[ToolCallRequest(id="call_1", name="exec", arguments={})],
            finish_reason="stop",
            usage=None,
        ),
    ])
    tools = ScriptedTools(AsyncMock(return_value="must not run"), definitions=[])

    runner = AgentRunner()
    result = await runner.run(make_run_spec(
        provider,
        initial_messages=[{"role": "user", "content": "do task"}],
        tools=tools,
        model="test-model",
        max_iterations=3,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
    ))

    tools.execute.assert_not_awaited()
    assert result.final_content == "finalized without tools"


@pytest.mark.asyncio
async def test_runner_length_recovery_returns_all_segments():
    """Recovered output segments are returned together instead of only the tail."""
    from nanobot.agent.runner import AgentRunner

    provider = MagicMock(spec=LLMProvider)
    provider.chat_stream_with_retry = AsyncMock(side_effect=[
        LLMResponse(content="first ", finish_reason="length"),
        LLMResponse(content="second ", finish_reason="length"),
        LLMResponse(content="third", finish_reason="stop"),
    ])
    tools = MagicMock()
    tools.get_definitions.return_value = []

    runner = AgentRunner()
    result = await runner.run(make_run_spec(provider,
        initial_messages=[{"role": "user", "content": "give a long answer"}],
        tools=tools,
        model="test-model",
        max_iterations=5,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
    ))

    assert result.final_content == "first second third"
    assert [
        message["content"]
        for message in result.messages
        if message.get("role") == "assistant"
    ] == ["first", "second", "third"]
    assert provider.chat_stream_with_retry.await_count == 3


@pytest.mark.asyncio
async def test_runner_length_recovery_preserves_prefix_at_max_iterations():
    """Budget exhaustion must not replace output already produced by recovery."""
    from nanobot.agent.runner import AgentRunner

    provider = MagicMock(spec=LLMProvider)
    provider.chat_stream_with_retry = AsyncMock(
        return_value=LLMResponse(content="partial answer", finish_reason="length")
    )
    tools = MagicMock()
    tools.get_definitions.return_value = []

    runner = AgentRunner()
    result = await runner.run(make_run_spec(
        provider,
        initial_messages=[{"role": "user", "content": "give a long answer"}],
        tools=tools,
        model="test-model",
        max_iterations=1,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
    ))

    limit_reached = (
        "I reached the maximum number of tool call iterations (1) "
        "without completing the task. You can try breaking the task into smaller steps."
    )
    assert result.stop_reason == "max_iterations"
    assert result.final_content == f"partial answer\n\n{limit_reached}"
    assert [
        message["content"]
        for message in result.messages
        if message.get("role") == "assistant"
    ] == ["partial answer", limit_reached]


@pytest.mark.asyncio
async def test_runner_length_recovery_does_not_leak_across_tool_calls():
    """A recovered prefix belongs only to its contiguous response chain."""
    from nanobot.agent.runner import AgentRunner

    provider = MagicMock(spec=LLMProvider)
    provider.chat_stream_with_retry = AsyncMock(side_effect=[
        LLMResponse(content="working", finish_reason="length"),
        LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id="call_1", name="read_file", arguments={"path": "x"})],
            finish_reason="tool_calls",
        ),
        LLMResponse(content="final answer", finish_reason="stop"),
    ])
    tools = ScriptedTools(AsyncMock(return_value="file content"), definitions=[])

    runner = AgentRunner()
    result = await runner.run(make_run_spec(provider,
        initial_messages=[{"role": "user", "content": "inspect a file"}],
        tools=tools,
        model="test-model",
        max_iterations=5,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
    ))

    assert result.final_content == "final answer"
    assert result.tools_used == ["read_file"]


@pytest.mark.asyncio
async def test_runner_empty_response_does_not_break_tool_chain():
    """An empty intermediate response must not kill an ongoing tool chain.

    Sequence: tool_call -> empty -> tool_call -> final text.
    The runner should recover via silent retry and complete normally.
    """
    from nanobot.agent.runner import AgentRunner

    provider = MagicMock(spec=LLMProvider)
    call_count = 0

    async def chat_stream_with_retry(*, messages, tools=None, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(id="tc1", name="read_file", arguments={"path": "a.txt"})],
                usage=LLMUsage.reported(input_tokens=10, output_tokens=5),
            )
        if call_count == 2:
            return LLMResponse(content=None, tool_calls=[], usage=LLMUsage.reported(input_tokens=10, output_tokens=1))
        if call_count == 3:
            return LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(id="tc2", name="read_file", arguments={"path": "b.txt"})],
                usage=LLMUsage.reported(input_tokens=10, output_tokens=5),
            )
        return LLMResponse(
            content="Here are the results.",
            tool_calls=[],
            usage=LLMUsage.reported(input_tokens=10, output_tokens=10),
        )

    provider.chat_stream_with_retry = chat_stream_with_retry

    async def fake_tool(name, args, **kw):
        return "file content"

    tool_registry = ScriptedTools(AsyncMock(side_effect=fake_tool), definitions=[{"type": "function", "function": {"name": "read_file"}}])

    runner = AgentRunner()
    result = await runner.run(make_run_spec(provider,
        initial_messages=[{"role": "user", "content": "read both files"}],
        tools=tool_registry,
        model="test-model",
        max_iterations=10,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
    ))

    assert result.final_content == "Here are the results."
    assert result.stop_reason == "completed"
    assert call_count == 4
    assert "read_file" in result.tools_used


@pytest.mark.asyncio
async def test_runner_accumulates_usage_and_preserves_cache_reads():
    """Runner accumulates usage across iterations, including cache reads."""
    from nanobot.agent.runner import AgentRunner

    provider = MagicMock(spec=LLMProvider)
    call_count = {"n": 0}

    async def chat_stream_with_retry(*, messages, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return LLMResponse(
                content="thinking",
                tool_calls=[ToolCallRequest(id="call_1", name="read_file", arguments={"path": "x"})],
                usage=LLMUsage.reported(input_tokens=100, output_tokens=10, cache_read_tokens=80),
            )
        return LLMResponse(
            content="done",
            tool_calls=[],
            usage=LLMUsage.reported(input_tokens=200, output_tokens=20, cache_read_tokens=150),
        )

    provider.chat_stream_with_retry = chat_stream_with_retry
    tools = ScriptedTools(AsyncMock(return_value="file content"), definitions=[])

    runner = AgentRunner()
    result = await runner.run(make_run_spec(provider,
        initial_messages=[{"role": "user", "content": "do task"}],
        tools=tools,
        model="test-model",
        max_iterations=3,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
    ))

    # Usage should be accumulated across iterations
    assert result.usage is not None
    assert result.usage.input_tokens == 300  # 100 + 200
    assert result.usage.output_tokens == 30  # 10 + 20
    assert result.usage.cache_read_tokens == 230  # 80 + 150
    assert result.usage.context_tokens == 200
    assert result.usage.request_count == 2
    assert [
        (item.input_tokens, item.cache_read_tokens)
        for item in result.round_usages
    ] == [
        (100, 80),
        (200, 150),
    ]


@pytest.mark.asyncio
async def test_runner_carries_retry_notifications_in_provider_context():
    """The runner carries a generic scope, not an event-specific callback."""
    from nanobot.agent.runner import AgentRunner
    from nanobot.events import EventSink, RetryWaitEvent

    captured: dict = {}

    async def chat_stream_with_retry(**kwargs):
        captured.update(kwargs)
        return LLMResponse(content="done", tool_calls=[], usage=None)

    provider = MagicMock(spec=LLMProvider)
    provider.chat_stream_with_retry = chat_stream_with_retry
    tools = MagicMock()
    tools.get_definitions.return_value = []

    retry_wait_cb = AsyncMock()

    runner = AgentRunner()
    await runner.run(make_run_spec(provider,
        initial_messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "hi"},
        ],
        tools=tools,
        model="test-model",
        max_iterations=1,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        events=EventSink(retry_wait_cb),
    ))

    assert "on_retry_wait" not in captured
    event = RetryWaitEvent("waiting")
    await captured["provider_context"].events.emit(event)
    retry_wait_cb.assert_awaited_once_with(event)


# ---------------------------------------------------------------------------
# Config passthrough tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runner_passes_temperature_to_provider():
    """temperature from AgentRunSpec should reach provider.chat_stream_with_retry."""
    from nanobot.agent.runner import AgentRunner

    captured: dict = {}

    async def chat_stream_with_retry(**kwargs):
        captured.update(kwargs)
        return LLMResponse(content="done", tool_calls=[], usage=None)

    provider = MagicMock(spec=LLMProvider)
    provider.chat_stream_with_retry = chat_stream_with_retry
    tools = MagicMock()
    tools.get_definitions.return_value = []

    runner = AgentRunner()
    await runner.run(make_run_spec(provider,
        initial_messages=[{"role": "user", "content": "hi"}],
        tools=tools,
        model="test-model",
        max_iterations=1,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        temperature=0.7,
    ))

    assert captured["temperature"] == 0.7


@pytest.mark.asyncio
async def test_runner_passes_max_tokens_to_provider():
    """max_tokens from AgentRunSpec should reach provider.chat_stream_with_retry."""
    from nanobot.agent.runner import AgentRunner

    captured: dict = {}

    async def chat_stream_with_retry(**kwargs):
        captured.update(kwargs)
        return LLMResponse(content="done", tool_calls=[], usage=None)

    provider = MagicMock(spec=LLMProvider)
    provider.chat_stream_with_retry = chat_stream_with_retry
    tools = MagicMock()
    tools.get_definitions.return_value = []

    runner = AgentRunner()
    await runner.run(make_run_spec(provider,
        initial_messages=[{"role": "user", "content": "hi"}],
        tools=tools,
        model="test-model",
        max_iterations=1,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        max_tokens=8192,
    ))

    assert captured["max_tokens"] == 8192


@pytest.mark.asyncio
async def test_runner_passes_reasoning_effort_to_provider():
    """reasoning_effort from AgentRunSpec should reach provider.chat_stream_with_retry."""
    from nanobot.agent.runner import AgentRunner

    captured: dict = {}

    async def chat_stream_with_retry(**kwargs):
        captured.update(kwargs)
        return LLMResponse(content="done", tool_calls=[], usage=None)

    provider = MagicMock(spec=LLMProvider)
    provider.chat_stream_with_retry = chat_stream_with_retry
    tools = MagicMock()
    tools.get_definitions.return_value = []

    runner = AgentRunner()
    await runner.run(make_run_spec(provider,
        initial_messages=[{"role": "user", "content": "hi"}],
        tools=tools,
        model="test-model",
        max_iterations=1,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        reasoning_effort="high",
    ))

    assert captured["reasoning_effort"] == "high"
