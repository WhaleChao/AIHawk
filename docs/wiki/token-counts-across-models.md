---
title: "Token counts are not portable between models"
description: "tiktoken against the counts of five models on OpenRouter, by kind of text: from 0.52 to 1.33 times. Why a fixed safety margin got a request refused, and how a Dot sizes its requests now."
parent: "Studies"
nav_order: 3
---

# Token counts are not portable between models

Before an agent sends a request it has to know how big the prompt is: to
decide when to compact a long thread, and how long an answer it may ask for
(the prompt and the answer together must fit the model's window). Only the
provider knows its model's tokenizer, so engines estimate, usually with
OpenAI's tiktoken, and leave a small margin. A Dot did this with a margin of
1,024 tokens, and a long task on a one-million-token window was refused.

## The refusal

A memory task on z-ai/glm-5.3-flash (window 1,048,576 tokens, longest answer
943,717) asked OpenRouter for the longest answer that fitted after a prompt
the engine had estimated at 103,346 tokens. OpenRouter answered:

> This endpoint's maximum context length is 1048576 tokens. However, you
> requested about 1049513 tokens (100506 of text input, 5290 of tool input,
> 943717 in the output).

Its count of the prompt was 105,796: the estimate was 2,450 tokens (2.4%)
short, more than twice the margin, and the request overflowed the window by
937 tokens.

## The measurement

Six kinds of text were sent to five models through OpenRouter, asking for a
one-word answer, and the prompt tokens each model reported were compared with
tiktoken's count of the same request (ratio counted / estimated):

| Model | English chat | Python | TypeScript | JSON | Italian | Chinese |
|---|---|---|---|---|---|---|
| anthropic/claude-sonnet-4.5 | 1.115 | **1.334** | 1.242 | 1.119 | 1.040 | 1.097 |
| google/gemini-2.5-flash | 1.039 | 1.275 | 1.172 | 1.051 | 0.880 | 0.548 |
| deepseek/deepseek-chat-v3.1 | 1.008 | 1.120 | 1.066 | 1.004 | 0.920 | 0.516 |
| qwen/qwen3-235b-a22b-2507 | 1.020 | 1.001 | 1.011 | 1.028 | 0.960 | 0.549 |
| openai/gpt-4o-mini | 0.990 | 1.005 | 1.007 | 0.989 | 0.920 | 0.774 |

(October 2026; texts of 8,000 to 28,000 tokens.) The spread runs from about
half to a third more, and it depends on the model **and** on the text: Claude
counts Python a third above tiktoken, the Chinese text is counted at half by
four of the five. A margin of fixed size cannot cover a relative error: on a
600,000-token prompt of code, a third is 200,000 tokens.

## Why the provider's own count is not enough either

The obvious fix is to use what the provider reports: every response says how
many prompt tokens it took, so the next request is that count plus an estimate
of what was added. It was built, and the refusal came back. OpenRouter checks
a request against the window with **a count of its own**, made before it routes
the request and reported only in the refusal; on glm-5.3-flash it ran above the
model's count plus a scaled estimate of the rest. The judge is a number the
engine never sees on a request that succeeds.

## What a Dot does now

The whole estimate is scaled: a prompt is taken for its tiktoken count times
1.35 (the largest ratio measured above, with room), or times the largest ratio
the model's own counts have shown when that is more (learned from every
response of more than 1,000 tokens). Being wrong upwards costs little: an
answer limit of 900,000 tokens instead of 943,717, and a thread compacted a
little earlier. Being wrong downwards fails the request. Each request now logs
its size, how it was measured and the answer limit it sent, so the next
surprise shows its numbers.

A related fix: the system prompt carried the time to the minute, so it changed
on every turn, and neither the provider's prompt cache nor any count of the
previous request could carry over. It carries the day now.

Code: `invisible_engine_dots/nanobot/providers/prompt_count.py`.
