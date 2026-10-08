"""A prompt's tokens as the provider will count them: its own count of what it took, and a scaled estimate of the rest."""

from __future__ import annotations

import math
from typing import Any

from test_openai_compat_stream import FakeStream, as_sdk_object, chunk, provider_streaming

from nanobot.agent.context_governance import answer_limit
from nanobot.providers.prompt_count import UNCOUNTED_RATIO, PromptCounts
from nanobot.utils.helpers import estimate_prompt_tokens, estimate_prompt_tokens_chain

MODEL = "z-ai/glm-5.3-flash"
SYSTEM = {"role": "system", "content": "You are the Dot. " * 200}
TOOLS = [{"type": "function", "function": {"name": "grep", "parameters": {"type": "object"}}}]


def user(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def scaled(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, ratio: float = UNCOUNTED_RATIO) -> int:
    return math.ceil(estimate_prompt_tokens(messages, tools) * ratio)


class TestPromptCounts:
    def test_a_prompt_nothing_was_counted_of_is_its_estimate_scaled_up(self) -> None:
        messages = [SYSTEM, user("hello")]

        assert PromptCounts().estimate(MODEL, messages, TOOLS) == (scaled(messages, TOOLS), "scaled estimate")

    def test_a_prompt_that_extends_a_counted_one_is_its_count_plus_what_was_added_scaled(self) -> None:
        counts = PromptCounts()
        first = [SYSTEM, user("hello")]
        counts.observe(MODEL, first, TOOLS, 3_000)
        added = [{"role": "assistant", "content": "hi"}, user("a file\n" * 5_000)]

        tokens, source = counts.estimate(MODEL, [*first, *added], TOOLS)

        assert (tokens, source) == (3_000 + scaled(added), "provider count plus scaled estimate")

    def test_other_tools_or_another_model_are_another_prompt(self) -> None:
        counts = PromptCounts()
        first = [SYSTEM, user("hello")]
        counts.observe(MODEL, first, TOOLS, 3_000)
        longer = [*first, user("more")]

        assert counts.estimate(MODEL, longer, None)[1] == "scaled estimate"
        assert counts.estimate("anthropic/claude-sonnet-4.5", longer, TOOLS)[1] == "scaled estimate"

    def test_the_longest_counted_prefix_is_used_when_the_chat_and_a_task_alternate(self) -> None:
        counts = PromptCounts()
        chat = [SYSTEM, user("chat")]
        task = [SYSTEM, user("task")]
        counts.observe(MODEL, chat, TOOLS, 1_000)
        counts.observe(MODEL, [*chat, user("more chat")], TOOLS, 1_500)
        counts.observe(MODEL, task, TOOLS, 2_000)
        added = user("still the chat")

        tokens, _ = counts.estimate(MODEL, [*chat, user("more chat"), added], TOOLS)

        assert tokens == 1_500 + scaled([added])

    def test_a_model_that_counts_more_than_the_ratio_scales_by_its_own(self) -> None:
        counts = PromptCounts()
        first = [SYSTEM, user("x " * 3_000)]
        estimated = estimate_prompt_tokens(first, None)
        counts.observe(MODEL, first, None, 2 * estimated)

        assert counts.estimate(MODEL, [user("other")], None)[0] == scaled([user("other")], ratio=2.0)

    def test_a_small_prompt_teaches_no_ratio(self) -> None:
        # Around each message the provider adds tokens the estimate does not see: on a few tokens they would look
        # like a tokenizer that counts far more.
        counts = PromptCounts()
        counts.observe(MODEL, [user("hi")], None, 40)

        assert counts.estimate(MODEL, [user("other")], None)[0] == scaled([user("other")])


class TestTheProvider:
    async def test_what_the_endpoint_counted_sizes_the_next_request(self) -> None:
        first = [SYSTEM, user("hello")]
        stream = FakeStream([
            as_sdk_object(chunk({"role": "assistant", "content": "hi"}, finish="stop")),
            as_sdk_object(chunk(usage={"prompt_tokens": 4_321, "completion_tokens": 1, "total_tokens": 4_322})),
        ])
        provider, _ = provider_streaming(stream)
        await provider.chat_stream(first, tools=TOOLS, model=MODEL)
        added = [{"role": "assistant", "content": "hi"}, user("next")]

        tokens, source = estimate_prompt_tokens_chain(provider, MODEL, [*first, *added], TOOLS)

        assert (tokens, source) == (4_321 + scaled(added), "provider count plus scaled estimate")

    def test_an_answer_limit_from_it_fits_the_window_of_a_model_that_counts_a_third_more(self) -> None:
        # Claude counts Python a third above tiktoken (prompt_count.py): a limit from the plain estimate overflowed.
        provider, _ = provider_streaming(FakeStream([]))
        code = [SYSTEM, user("def handler(request):\n    return request.json()['items'][0]\n" * 4_000)]
        counted = math.ceil(estimate_prompt_tokens(code, None) * 1.334)
        window = 1_000_000

        tokens, _ = estimate_prompt_tokens_chain(provider, MODEL, code, None)
        limit = answer_limit(window, window, tokens)

        assert limit is not None and counted + limit <= window
