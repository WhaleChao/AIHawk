"""How many tokens a prompt takes for a model, as the endpoint will count it.

The engine sizes every request against the model's window (when to compact, how long an answer may be) before
sending it, and only the endpoint knows the model's tokenizer. tiktoken's count differs from it by model and
by kind of text; against OpenRouter's own counts (2026-10-08, ratio counted / estimated):

| model | English chat | Python | TypeScript | JSON | Italian | Chinese |
|---|---|---|---|---|---|---|
| anthropic/claude-sonnet-4.5 | 1.115 | 1.334 | 1.242 | 1.119 | 1.040 | 1.097 |
| google/gemini-2.5-flash | 1.039 | 1.275 | 1.172 | 1.051 | 0.880 | 0.548 |
| deepseek/deepseek-chat-v3.1 | 1.008 | 1.120 | 1.066 | 1.004 | 0.920 | 0.516 |
| qwen/qwen3-235b-a22b-2507 | 1.020 | 1.001 | 1.011 | 1.028 | 0.960 | 0.549 |
| openai/gpt-4o-mini | 0.990 | 1.005 | 1.007 | 0.989 | 0.920 | 0.774 |

So a margin of fixed size cannot hold: a third more than the estimate is a request of code for Claude. Nor can
the model's count of the last request settle it: OpenRouter refuses a request that does not fit the window by
a count of its own, made before it routes the request and reported only in the refusal, and on
z-ai/glm-5.3-flash that count ran 2.4% above the model's count of the same prompt plus an estimate of what was
added. So the estimate is scaled whole, by UNCOUNTED_RATIO (the largest ratio measured above, with room), or
by the largest ratio the model's own counts have shown, when that is more.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from nanobot.utils.helpers import estimate_prompt_tokens

# How many of the endpoint's tokens one estimated token may be.
UNCOUNTED_RATIO = 1.35
# The smallest estimate a ratio is learned from: below it, the tokens the endpoint adds around each message
# (which the estimate does not see) would make the ratio of the text itself look larger than it is.
RATIO_MIN_ESTIMATE = 1000


@dataclass
class PromptCounts:
    """The largest ratio of counted to estimated tokens each model has shown."""

    _ratio: dict[str, float] = field(default_factory=dict)

    def observe(self, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, tokens: int) -> None:
        """The endpoint counted `tokens` for a request of `model` with `messages` and `tools`."""
        estimated = estimate_prompt_tokens(messages, tools)
        if estimated >= RATIO_MIN_ESTIMATE:
            self._ratio[model] = max(self._ratio.get(model, 0.0), tokens / estimated)

    def estimate(self, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> tuple[int, str]:
        """The tokens the endpoint will count for this request: never fewer, as far as the counts above go."""
        ratio = max(UNCOUNTED_RATIO, self._ratio.get(model, 0.0))
        return math.ceil(estimate_prompt_tokens(messages, tools) * ratio), f"estimate scaled by {ratio:.2f}"
