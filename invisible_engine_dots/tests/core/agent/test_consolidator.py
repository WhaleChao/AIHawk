"""Tests for the transcript summary (Consolidator) and its mechanical fallback."""

from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.memory import (
    _ARCHIVE_TOOL_RESULT,
    RECENT_USER_MESSAGE_TOKENS,
    Consolidator,
    _build_raw_checkpoint,
    _format_messages,
)
from nanobot.providers.base import (
    GenerationSettings,
    LLMResponse,
    ToolCallRequest,
)
from nanobot.utils.llm_runtime import LLMRuntime
from nanobot.utils.prompt_templates import render_template

_ARCHIVE_PROMPT = render_template("agent/consolidator_archive.md", strip=True)


@pytest.fixture
def mock_provider():
    p = MagicMock()
    p.chat_stream_with_retry = AsyncMock()
    p.generation = GenerationSettings(max_tokens=100)
    return p


@pytest.fixture
def runtime(mock_provider):
    return LLMRuntime.capture(
        mock_provider,
        "test-model",
        context_window_tokens=1000,
    )


@pytest.fixture
def consolidator():
    return Consolidator()


async def _archive(
    consolidator,
    messages,
    runtime,
    *,
    session_key="test:session",
    previous_summary=None,
):
    return await consolidator.summarize(
        messages,
        runtime=runtime,
        session_key=session_key,
        history=[
            {"role": "system", "content": "system prompt"},
            *messages,
        ],
        request_tools=[],
        previous_summary=previous_summary,
    )


class TestTurnTranscriptSummary:
    async def test_uses_exact_accepted_prefix(
        self,
        consolidator,
        mock_provider,
        runtime,
    ):
        summary = "replacement checkpoint"
        runtime = replace(runtime, context_window_tokens=4096)
        accepted = [
            {"role": "system", "content": "stable system"},
            {"role": "user", "content": "accepted history"},
        ]
        tools = [{"type": "function", "function": {"name": "inspect"}}]
        mock_provider.chat_stream_with_retry.return_value = LLMResponse(
            content=summary,
        )

        result = await consolidator.summarize_transcript(
            accepted,
            "previous checkpoint",
            runtime=runtime,
            session_key="test:turn",
            tools=tools,
        )

        # The person's latest message follows the summary as they wrote it.
        assert result == summary + "\n\n## The person's latest messages, as they wrote them\n\naccepted history"
        call = mock_provider.chat_stream_with_retry.await_args.kwargs
        assert call["messages"][:-1] == accepted
        assert call["messages"][-1]["role"] == "user"
        assert "CONTEXT CHECKPOINT COMPACTION" in call["messages"][-1]["content"]
        assert call["tools"] == tools

    async def test_failure_returns_raw_checkpoint(
        self,
        consolidator,
        mock_provider,
        runtime,
    ):
        accepted = [
            {"role": "system", "content": "stable system"},
            {"role": "user", "content": "accepted history"},
        ]
        mock_provider.chat_stream_with_retry.return_value = LLMResponse(content="")

        result = await consolidator.summarize_transcript(
            accepted,
            None,
            runtime=runtime,
            session_key="test:ephemeral-turn",
            tools=[],
        )

        assert result is not None
        assert "[RAW]" in result
        assert "accepted history" in result

    async def test_a_tool_call_is_answered_and_the_checkpoint_asked_for_again(
        self,
        consolidator,
        mock_provider,
        runtime,
    ):
        accepted = [
            {"role": "system", "content": "stable system"},
            {"role": "user", "content": "accepted history"},
        ]
        tools = [{"type": "function", "function": {"name": "inspect"}}]
        mock_provider.chat_stream_with_retry.side_effect = [
            LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(id="call-1", name="inspect", arguments={})],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="replacement checkpoint", finish_reason="stop"),
        ]

        result = await consolidator.summarize_transcript(
            accepted,
            "previous checkpoint",
            runtime=runtime,
            session_key="test:turn",
            tools=tools,
        )

        assert result == "replacement checkpoint\n\n## The person's latest messages, as they wrote them\n\naccepted history"
        first_call, recovery_call = mock_provider.chat_stream_with_retry.await_args_list
        assert first_call.kwargs["tools"] == recovery_call.kwargs["tools"] == tools
        assert first_call.kwargs["messages"] == [*accepted, {"role": "user", "content": _ARCHIVE_PROMPT}]
        assistant, tool_result = recovery_call.kwargs["messages"][-2:]
        assert assistant["role"] == "assistant"
        assert [call["id"] for call in assistant["tool_calls"]] == ["call-1"]
        assert tool_result == {
            "role": "tool",
            "tool_call_id": "call-1",
            "name": "inspect",
            "content": _ARCHIVE_TOOL_RESULT,
        }


class TestConsolidatorSummarize:
    def test_format_messages_keeps_media_only_user_turn(self):
        path = "/workspace/clip.mp4"

        formatted = _format_messages([
            {
                "role": "user",
                "content": "",
                "media": [path],
                "timestamp": "2026-07-27",
            }
        ])

        assert formatted == f"[2026-07-27] USER: [image: {path}]"

    async def test_archive_uses_captured_generation(
        self, consolidator, mock_provider, runtime
    ):
        admitted = replace(
            runtime,
            context_window_tokens=100_000,
            generation=GenerationSettings(
                temperature=0.25,
                max_tokens=321,
                reasoning_effort="medium",
            ),
        )
        mock_provider.generation = GenerationSettings(
            temperature=0.9,
            max_tokens=999,
            reasoning_effort="high",
        )
        mock_provider.chat_stream_with_retry.return_value = MagicMock(
            content="Summary.",
            finish_reason="stop",
        )

        await _archive(consolidator, [{"role": "user", "content": "hello"}], admitted)

        call = mock_provider.chat_stream_with_retry.call_args.kwargs
        assert call["model"] == admitted.model
        assert call["temperature"] == 0.25
        assert call["max_tokens"] == 321
        assert call["reasoning_effort"] == "medium"

    async def test_summarize_returns_the_model_summary(
        self, consolidator, mock_provider, runtime
    ):
        mock_provider.chat_stream_with_retry.return_value = MagicMock(
            content="User fixed a bug in the auth module."
        )
        messages = [
            {"role": "user", "content": "fix the auth bug"},
            {"role": "assistant", "content": "Done, fixed the race condition."},
        ]
        result = await _archive(consolidator, messages, runtime)
        assert result == "User fixed a bug in the auth module."

    async def test_summarize_raw_dumps_on_llm_failure(
        self, consolidator, mock_provider, runtime
    ):
        """On LLM failure the messages themselves become the checkpoint."""
        mock_provider.chat_stream_with_retry.side_effect = Exception("API error")
        messages = [{"role": "user", "content": "hello"}]
        result = await _archive(consolidator, messages, runtime)
        assert result is not None
        assert "[RAW]" in result
        assert "hello" in result

    async def test_raw_fallback_represents_previous_checkpoint_and_new_chunk(
        self,
        consolidator,
        mock_provider,
        runtime,
    ):
        mock_provider.chat_stream_with_retry.side_effect = RuntimeError("API error")
        messages = [{"role": "user", "content": "NEW_MARKER " + "new " * 200}]

        # The caller bounds the mechanical checkpoint (half of the summary model's prompt budget in a turn).
        result = await consolidator.summarize(
            messages,
            runtime=runtime,
            session_key="test:session",
            history=[{"role": "system", "content": "system prompt"}, *messages],
            request_tools=[],
            previous_summary="OLD_MARKER " + "old " * 200,
            fallback_max_tokens=256,
        )

        assert result is not None
        assert "[Previous archived context]" in result
        assert "OLD_MARKER" in result
        assert "[Newly archived raw context]" in result
        assert "NEW_MARKER" in result
        assert "... (truncated)" in result

    async def test_summarize_skips_empty_messages(self, consolidator, runtime):
        result = await _archive(consolidator, [], runtime)
        assert result is None


class TestConsolidatorPromptContract:
    def test_archive_prompt_requests_a_handoff_that_replaces_the_previous_checkpoint(self):
        prompt = _ARCHIVE_PROMPT

        # Codex's handoff framing, OpenHands' headings and the Dot's own, and the merge rule.
        assert "CONTEXT CHECKPOINT COMPACTION" in prompt
        assert "handoff summary for another model that will resume this work" in prompt
        assert "[Archived Context Summary]" in prompt
        assert "the conversation wins" in prompt
        for heading in (
            "USER_CONTEXT:", "TASK_TRACKING:", "COMPLETED:", "PENDING:", "CURRENT_STATE:", "APPROVALS:", "FILES:",
            "CODE_STATE:", "TESTS:", "CHANGES:", "DEPS:", "VERSION_CONTROL_STATUS:",
        ):
            assert heading in prompt
        assert "Do not call a tool." in prompt
        assert "they are kept after your summary as they wrote them" in prompt
        assert "history.jsonl" not in prompt


class TestConsolidatorArchiveErrorHandling:
    """summarize() must fall back when the LLM does not complete its overview.

    Error responses include overloaded / quota failures from #3244; length
    responses contain a partial overview that is likewise unsafe to replay.
    """

    @pytest.mark.parametrize("finish_reason", ["error", "length"])
    async def test_archive_falls_back_on_incomplete_finish_reason(
        self,
        consolidator,
        mock_provider,
        runtime,
        finish_reason: str,
    ):
        """Incomplete LLM output should trigger the raw checkpoint, not partial text."""
        invalid_output = f"INVALID_{finish_reason.upper()}_OUTPUT"
        mock_provider.chat_stream_with_retry.return_value = MagicMock(
            content=invalid_output,
            finish_reason=finish_reason,
        )
        messages = [
            {"role": "user", "content": "fix the auth bug"},
            {"role": "assistant", "content": "Done, fixed the race condition."},
        ]
        result = await _archive(consolidator, messages, runtime)
        assert result is not None
        assert "[RAW]" in result
        assert invalid_output not in result

    async def test_archive_preserves_summary_on_success(
        self, consolidator, mock_provider, runtime
    ):
        """Normal LLM response should still produce a proper summary."""
        mock_provider.chat_stream_with_retry.return_value = MagicMock(
            content="User fixed a bug in the auth module.",
            finish_reason="stop",
        )
        messages = [
            {"role": "user", "content": "fix the auth bug"},
            {"role": "assistant", "content": "Done."},
        ]
        result = await _archive(consolidator, messages, runtime)
        assert result == "User fixed a bug in the auth module."
        assert "[RAW]" not in result

    async def test_summarize_propagates_template_failure_without_fallback(
        self, consolidator, mock_provider, runtime, monkeypatch
    ):
        runtime = replace(runtime, context_window_tokens=128_000)
        monkeypatch.setattr(
            "nanobot.agent.memory.render_template",
            MagicMock(side_effect=RuntimeError("template failed")),
        )

        with pytest.raises(RuntimeError, match="template failed"):
            await consolidator.summarize_transcript(
                [{"role": "user", "content": "important"}],
                None,
                runtime=runtime,
                session_key="test:template",
                tools=[],
            )

        mock_provider.chat_stream_with_retry.assert_not_awaited()


class TestRawCheckpoint:
    """The mechanical checkpoint is the messages, formatted, sanitized and bounded."""

    def test_raw_checkpoint_strips_thinking_before_truncating(self):
        # A thinking block longer than the cap must not leave its head in the checkpoint.
        content = "<think>PRIVATE" + "x" * 40_000 + "</think>VISIBLE_TAIL"

        checkpoint = _build_raw_checkpoint([{"role": "assistant", "content": content}])

        assert "PRIVATE" not in checkpoint
        assert "VISIBLE_TAIL" in checkpoint

    def test_raw_checkpoint_keeps_large_content_whole(self):
        # No fixed size: the turn bounds the checkpoint by its model's own window (`fallback_max_tokens`).
        checkpoint = _build_raw_checkpoint([{"role": "user", "content": "x" * 50_000}])

        assert "x" * 50_000 in checkpoint
        assert checkpoint.startswith("[RAW]")

    def test_raw_checkpoint_preserves_small_content(self):
        checkpoint = _build_raw_checkpoint([{"role": "user", "content": "hello"}])

        assert checkpoint == f"[RAW] 1 messages\n{_format_messages([{'role': 'user', 'content': 'hello'}])}"
        assert "hello" in checkpoint

    def test_raw_checkpoint_is_sanitized(self):
        checkpoint = _build_raw_checkpoint([
            {"role": "user", "content": "<think>PRIVATE_REASONING</think>visible result"}
        ])

        assert "PRIVATE_REASONING" not in checkpoint
        assert "visible result" in checkpoint


class TestSummaryBounds:
    async def test_summary_is_sanitized(self, consolidator, mock_provider, runtime):
        mock_provider.chat_stream_with_retry.return_value = MagicMock(
            content="<think>PRIVATE_REASONING</think>safe summary",
            finish_reason="stop",
            has_tool_calls=False,
        )

        summary = await _archive(
            consolidator,
            [{"role": "user", "content": "hi"}],
            runtime,
        )

        assert summary == "safe summary"

    async def test_a_long_summary_comes_back_whole(self, consolidator, mock_provider, runtime):
        """No size of our own: the model's longest answer bounds its summary."""
        mock_provider.chat_stream_with_retry.return_value = MagicMock(content="S" * 200_000, finish_reason="stop")

        summary = await _archive(consolidator, [{"role": "user", "content": "hi"}], runtime)

        assert summary == "S" * 200_000


class TestCompactionOfALongThread:
    async def test_a_thread_too_long_for_the_summary_model_loses_its_oldest_messages_with_their_results(
        self, consolidator, mock_provider, runtime, monkeypatch
    ):
        def ten_tokens_a_message(provider, model, messages, tools):
            return 10 * len(messages), "test"

        monkeypatch.setattr("nanobot.agent.memory.estimate_prompt_tokens_chain", ten_tokens_a_message)
        monkeypatch.setattr("nanobot.agent.memory.estimate_message_tokens", lambda message: 10)
        mock_provider.chat_stream_with_retry.return_value = LLMResponse(content="the summary")
        call = {"id": "c1", "type": "function", "function": {"name": "exec", "arguments": "{}"}}
        thread = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "c1", "name": "exec", "content": "out"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "done"},
        ]
        history = [{"role": "system", "content": "system prompt"}, *thread]

        # Room for the instructions, three messages and the request for the summary: 50 tokens.
        summary = await consolidator.summarize(
            thread,
            runtime=runtime,
            session_key="test:long",
            history=history,
            request_tools=[],
            input_token_budget=50,
        )

        assert summary == "the summary"
        sent = mock_provider.chat_stream_with_retry.await_args.kwargs["messages"]
        # "first" went, then its call went with its result: nothing is split, the newest stay.
        assert [m.get("content") for m in sent[:-1]] == ["system prompt", "second", "done"]
        assert sent[-1]["content"] == _ARCHIVE_PROMPT

    def test_the_latest_messages_kept_are_the_newest_up_to_the_limit_the_one_across_it_cut(self):
        from nanobot.agent.memory import _with_recent_user_messages

        big = "word " * RECENT_USER_MESSAGE_TOKENS
        messages = [
            {"role": "user", "content": "oldest"},
            {"role": "user", "content": big},
            {"role": "assistant", "content": "not the person's"},
            {"role": "user", "content": "newest"},
        ]

        text = _with_recent_user_messages("S", messages, RECENT_USER_MESSAGE_TOKENS)

        head, latest = text.split("## The person's latest messages, as they wrote them\n\n")
        assert head == "S\n\n"
        parts = latest.split("\n\n---\n\n")
        assert parts[-1] == "newest"
        assert "oldest" not in latest and "not the person's" not in latest
        assert parts[0].startswith("word word") and parts[0] != big

    @pytest.mark.parametrize(
        ("budget", "kept"),
        [
            (1_000_000, RECENT_USER_MESSAGE_TOKENS),  # a large window: Codex's 20000
            (40_000, 10_000),  # a small one: a quarter of the budget, room left for the work after it
            (2_880, 720),
            (0, RECENT_USER_MESSAGE_TOKENS),  # no window known
        ],
    )
    def test_the_latest_messages_kept_are_at_most_a_quarter_of_the_budget_of_the_requests_they_go_into(
        self, budget, kept
    ):
        from nanobot.agent.memory import recent_user_message_tokens

        assert recent_user_message_tokens(budget) == kept
