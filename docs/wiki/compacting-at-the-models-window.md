---
title: "Compacting a thread at the model's own window"
description: "Why a Dot sets no token limit of its own, when it compacts a long thread, and why it clears old tool results before it summarizes anything: the rules taken from opencode, Codex, OpenHands and a SWE-bench study."
parent: "Studies"
nav_order: 4
---

# Compacting a thread at the model's own window

An agent working for hours fills its context. Something has to give, and
every choice costs something: a cap on the answer cuts work in half, a summary
loses detail, a truncation loses more. This page is how a Dot decides, and the
failures that decided it.

## What went wrong with fixed limits

Until October 2026 a Dot sent `max_tokens: 4096` on every request and kept each
thread under a configurable 32,000 tokens. On long benchmark tasks (writing a
Scheme interpreter, a key-value server: see
[Benchmarking an agent on real VMs](benchmarking-agents-on-real-vms.md)) this
failed in a specific way: the model wrote a whole program in one `write_file`
call, the answer was cut at 4,096 tokens in the middle of the call, the engine
asked it to "continue from its exact endpoint", and with nothing delivered to
continue from, the model made the same call again until the task failed. A
model with a one-million-token window was being summarized at 32,000.

So the limits went, all of them: a Dot has no token setting. Every request uses
its model's own limits, read once from OpenRouter's list of models (the window
and the longest answer of the provider OpenRouter routes to by default), and
asks for the longest answer the window leaves room for. Without `max_tokens`
each provider would apply a default of its own, often far lower. A response
cut before it delivered anything is now told that nothing ran and to work in
smaller steps, instead of to continue.

## When to compact

A thread is compacted only when the next request would not fit: at the window,
less the room kept for the answer (20,000 tokens, or the longest answer when
that is shorter) and a safety margin. That is opencode's rule. How the prompt is
measured turned out to matter as much as the rule: see
[Token counts are not portable between models](token-counts-across-models.md).

## How: clear first, summarize second

1. **Old tool results are cleared.** Walking back from the newest message, the
   first 40,000 tokens of tool results stay; older ones become a short note that
   the tool can be called again, if that frees at least 20,000 tokens. The calls,
   the reasoning and every message stay. No model call is needed. On SWE-bench
   this costs about half as much as summarizing for the same solve rate
   (JetBrains, "The Complexity Trap", arXiv 2508.21433); the thresholds are
   opencode's.
2. **Only if that is not enough, a summary.** It is written as a handoff for the
   model that resumes the work (Codex's framing), under the headings of
   OpenHands' summarizing prompt, plus what only a Dot has: approvals given and
   files written. After it, the person's latest messages are kept as they wrote
   them (Codex keeps 20,000 tokens; a Dot keeps at most a quarter of the budget,
   because a model with a small window still needs room after them). The summary
   may be written by a cheaper model (`models.summary`), within that model's own
   window; a thread too long even for it loses its oldest messages first, never
   a call without its result.

OpenRouter's own `context-compression` plugin was looked at and not used: it is
a middle-out truncation, which drops the middle of the thread without a trace.

## What it changed

On the five long tasks (two to three hours each), the key-value server and the
Scheme interpreter went from failing most runs to passing 2 out of 2; all five
passed in the final round, in runs of 18 to 75 minutes.

One side effect is worth knowing: OpenRouter holds the cost of `max_tokens` up
front, so asking for the longest answer makes a nearly empty balance show
sooner. When it refuses for credit, the Dot now shows OpenRouter's own message
(how much the balance still allows, where to add credit) after its own.

## Sources

- opencode: [code](https://github.com/sst/opencode) (compaction thresholds)
- Codex: [code](https://github.com/openai/codex) (handoff summary, recent user messages)
- OpenHands: [code](https://github.com/All-Hands-AI/OpenHands) (summarizing condenser)
- JetBrains, "The Complexity Trap": [arXiv 2508.21433](https://arxiv.org/abs/2508.21433)
