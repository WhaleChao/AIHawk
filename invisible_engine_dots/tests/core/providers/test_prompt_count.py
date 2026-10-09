"""A prompt's tokens as the endpoint will count them: the estimate scaled by the largest ratio measured or shown."""

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
    def test_a_prompt_is_its_estimate_scaled_by_the_largest_ratio_measured(self) -> None:
        messages = [SYSTEM, user("hello")]

        assert PromptCounts().estimate(MODEL, messages, TOOLS) == (scaled(messages, TOOLS), "estimate scaled by 1.35")

    def test_a_model_that_counts_less_keeps_the_measured_ratio(self) -> None:
        # OpenRouter refuses by a count of its own, above the model's: a model's low count does not lower it.
        counts = PromptCounts()
        first = [SYSTEM, user("x " * 3_000)]
        counts.observe(MODEL, first, None, estimate_prompt_tokens(first, None))

        assert counts.estimate(MODEL, [user("other")], None)[0] == scaled([user("other")])

    def test_a_model_that_counts_more_than_the_ratio_scales_by_its_own(self) -> None:
        counts = PromptCounts()
        first = [SYSTEM, user("x " * 3_000)]
        counts.observe(MODEL, first, None, 2 * estimate_prompt_tokens(first, None))

        assert counts.estimate(MODEL, [user("other")], None)[0] == scaled([user("other")], ratio=2.0)
        assert counts.estimate("openai/gpt-4o-mini", [user("other")], None)[0] == scaled([user("other")])

    def test_a_small_prompt_teaches_no_ratio(self) -> None:
        # Around each message the endpoint adds tokens the estimate does not see: on a few tokens they would look
        # like a tokenizer that counts far more.
        counts = PromptCounts()
        counts.observe(MODEL, [user("hi")], None, 40)

        assert counts.estimate(MODEL, [user("other")], None)[0] == scaled([user("other")])


class TestTheProvider:
    async def test_what_the_endpoint_counted_teaches_the_provider_the_models_ratio(self) -> None:
        first = [SYSTEM, user("x " * 3_000)]
        counted = 2 * estimate_prompt_tokens(first, TOOLS)
        stream = FakeStream([
            as_sdk_object(chunk({"role": "assistant", "content": "hi"}, finish="stop")),
            as_sdk_object(chunk(usage={"prompt_tokens": counted, "completion_tokens": 1, "total_tokens": counted + 1})),
        ])
        provider, _ = provider_streaming(stream)
        await provider.chat_stream(first, tools=TOOLS, model=MODEL)
        later = [*first, {"role": "assistant", "content": "hi"}, user("next")]

        tokens, source = estimate_prompt_tokens_chain(provider, MODEL, later, TOOLS)

        assert tokens == scaled(later, TOOLS, ratio=2.0)
        assert source == "estimate scaled by 2.00"

    def test_an_answer_limit_from_it_fits_the_window_of_a_model_that_counts_a_third_more(self) -> None:
        # Claude counts Python a third above tiktoken (prompt_count.py): a limit from the plain estimate overflowed.
        provider, _ = provider_streaming(FakeStream([]))
        code = [SYSTEM, user("def handler(request):\n    return request.json()['items'][0]\n" * 4_000)]
        counted = math.ceil(estimate_prompt_tokens(code, None) * 1.334)
        window = 1_000_000

        tokens, _ = estimate_prompt_tokens_chain(provider, MODEL, code, None)
        limit = answer_limit(window, window, tokens)

        assert limit is not None and counted + limit <= window

    def test_the_glm_refusal_measured_on_a_dot_does_not_happen_again(self) -> None:
        # 2026-10-08, a Dot's memory pass: the engine took the prompt for 103346 tokens, OpenRouter counted 105796
        # and refused the model's longest answer (943717) on a window of 1048576.
        window, longest, openrouter_counted, plain_estimate = 1_048_576, 943_717, 105_796, 103_346
        limit = answer_limit(window, longest, math.ceil(plain_estimate * UNCOUNTED_RATIO))

        assert limit is not None and openrouter_counted + limit <= window
