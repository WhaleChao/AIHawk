"""How many tokens a prompt takes for a model, as the provider will count it.

The engine sizes every request against the model's window (when to compact, how long an answer may be) before
sending it, and only the provider knows the model's tokenizer. tiktoken's count differs from it by model and
by kind of text; against OpenRouter's own counts (2026-10-08, ratio counted / estimated):

| model | English chat | Python | TypeScript | JSON | Italian | Chinese |
|---|---|---|---|---|---|---|
| anthropic/claude-sonnet-4.5 | 1.115 | 1.334 | 1.242 | 1.119 | 1.040 | 1.097 |
| google/gemini-2.5-flash | 1.039 | 1.275 | 1.172 | 1.051 | 0.880 | 0.548 |
| deepseek/deepseek-chat-v3.1 | 1.008 | 1.120 | 1.066 | 1.004 | 0.920 | 0.516 |
| qwen/qwen3-235b-a22b-2507 | 1.020 | 1.001 | 1.011 | 1.028 | 0.960 | 0.549 |
| openai/gpt-4o-mini | 0.990 | 1.005 | 1.007 | 0.989 | 0.920 | 0.774 |

So a margin of fixed size cannot hold: a third more than the estimate is a request of code for Claude. What
the provider counted is used instead: every response reports the tokens of the prompt it took, so a request
that extends one of the last of its model is that count plus an estimate of what was added, and only the
estimated part is scaled up, by UNCOUNTED_RATIO (the largest ratio measured above, with room) or by the
largest ratio this model has shown, when that is more.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from nanobot.utils.helpers import estimate_prompt_tokens

# How many of the provider's tokens one estimated token may be, for text whose count is not known yet.
UNCOUNTED_RATIO = 1.35
# The smallest estimate a ratio is learned from: below it, the provider's tokens around each message (which the
# estimate does not see) would make the ratio of the text itself look larger than it is.
RATIO_MIN_ESTIMATE = 1000
# The counted requests kept per model: the chat and the tasks running beside it each extend their own.
COUNTED_KEPT = 8


@dataclass
class _Counted:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    tokens: int


@dataclass
class PromptCounts:
    """What the provider counted, per model: the last requests, and the largest ratio to the estimate seen."""

    _counted: dict[str, list[_Counted]] = field(default_factory=dict)
    _ratio: dict[str, float] = field(default_factory=dict)

    def observe(self, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, tokens: int) -> None:
        """The provider counted `tokens` for a request of `model` with `messages` and `tools`."""
        estimated = estimate_prompt_tokens(messages, tools)
        if estimated >= RATIO_MIN_ESTIMATE:
            self._ratio[model] = max(self._ratio.get(model, 0.0), tokens / estimated)
        kept = self._counted.setdefault(model, [])
        kept.append(_Counted(deepcopy(messages), deepcopy(tools), tokens))
        del kept[:-COUNTED_KEPT]

    def estimate(self, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> tuple[int, str]:
        """The tokens the provider will count for this request: never fewer, as far as the counts above go."""
        ratio = max(UNCOUNTED_RATIO, self._ratio.get(model, 0.0))
        extended = [
            counted
            for counted in self._counted.get(model, [])
            if counted.tools == tools and messages[: len(counted.messages)] == counted.messages
        ]
        if extended:
            base = max(extended, key=lambda counted: len(counted.messages))
            added = messages[len(base.messages) :]
            uncounted = estimate_prompt_tokens(added, None) if added else 0
            return base.tokens + math.ceil(uncounted * ratio), "provider count plus scaled estimate"
        return math.ceil(estimate_prompt_tokens(messages, tools) * ratio), "scaled estimate"
