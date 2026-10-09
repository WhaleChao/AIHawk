---
title: "Agent memory systems, measured on the same questions"
description: "Mastra Observational Memory, LangMem, Hindsight, an agentic dream pass and a single request, run on the same LongMemEval questions on real agents, with accuracy and cost per pass. Plus what changes with years of history."
parent: "Studies"
nav_order: 2
---

# Agent memory systems, measured on the same questions

A Dot keeps a profile of the person, `MEMORY.md`, in every prompt (see
[How a Dot remembers](how-a-dot-remembers.md)). Something has to write it from
the conversations. Many open-source projects do this; most publish their own
scores on their own setups, and the setups differ. So the candidates were run
here on the same questions, the same agent and the same model.

## The setup

- **Questions:** the 30 LongMemEval questions whose answer depends on what the
  person likes ("can you suggest a hotel for my trip?"), each with about 48
  dated chats, 115,000 tokens, as its history. These are the ones a profile is
  for; the other kinds were checked separately on 116 mixed questions.
- **Agent:** a new Dot per question, with the history as its conversation
  files. Each method writes `MEMORY.md` from that history; then the question
  goes to the Dot as a task. The Dot can also grep the files, so every method
  is measured on top of the same baseline.
- **Model:** z-ai/glm-5.3-flash for everything, through OpenRouter; GPT-4o
  grades with LongMemEval's own prompt.
- **Runs:** two per method where the budget allowed. Two identical runs with no
  profile scored 19 and 20 out of 30, and others moved by up to three
  questions, so a single run of 30 cannot tell methods apart; a difference of
  two questions out of 60 is noise.
- **Faithfulness:** each project's own prompts, copied verbatim (18 of 18
  checked against the sources), its own call pattern, and its own benchmark
  settings where it publishes them. Every adaptation is marked in the adapters,
  which are kept in commit `de3d9a1b` of the repository.

## Results

| Method | Correct | % | Cost of a pass over 115K tokens | Calls | Needs |
|---|---|---|---|---|---|
| No profile (the Dot greps the files) | 39/60 | 65% | 0 | 0 | nothing |
| **One request (ours)** | **50/60** | **83%** | **~$0.015** | **1** | nothing |
| Hindsight (MIT) | 25/30 | 83% | ~$0.25 | ~255 | Postgres + pgvector, ~1.9 GB RAM |
| Mastra Observational Memory (Apache-2.0) | 48/60 | 80% | ~$0.05 | ~6 | (TypeScript) |
| LangMem profile manager (MIT) | 48/60 | 80% | ~$0.02 | 1 | LangChain |
| Agentic pass with file tools (nanobot's Dream, MIT) | 44/59 | 75% | ~$0.08 | ~20 steps | nothing |
| Memobase (Apache-2.0) | stopped | | ~$0.25-0.40 | ~200 | Postgres + Redis |
| Mem0 (Apache-2.0) | not run | | ~$0.40 | ~250 | a vector store |
| Letta sleep-time agents (Apache-2.0) | not run | | ~$1.10-1.60 | ~120, growing | a server + Postgres |

The costs of the last three are estimated from their own call patterns,
measured against a stand-in model: running them through would have cost more
than the rest of the study together, for designs the measured ones already
covered (Letta's is an agentic pass like Dream's, which was measured).

## What the numbers say

**The profile matters more than the method.** Every method that writes one
moves the score from 65% to about 80%. Between the one request, Hindsight,
Mastra and LangMem the differences are two questions out of 60 or less:
inside the noise.

**At equal quality, cost decides, and it varies seventeenfold.** The single
request reads the conversations once and answers with the whole new profile.
Mastra reads them in 30,000-token chunks and rewrites its notes when they grow.
Hindsight extracts facts from every ~1,800 characters, one call each. The
agentic pass opens files one at a time with tools, and every step sends the
whole growing context again: 115,000 tokens of history become about 500,000
tokens paid. Letta's sleep-time agent does the same while keeping its own
transcript, which is why its estimate is the highest.

**Simpler is also more robust.** The single request has no tools to call and
no state of its own; the agentic pass failed in 13 of 30 runs before a token
sizing bug was fixed (see
[Token counts are not portable between models](token-counts-across-models.md)),
and Memobase's run ended when the credit did.

**On the 116 mixed questions the profile helped where expected and hurt
nowhere:** updates from 78% to 94% (the profile keeps the newest version of a
fact, with its date), preferences from 57% to 71%, the other kinds within one
question.

## What was chosen

The single request, as a background pass: once the Dot has been quiet for
five minutes, the conversation files that changed go to the summary model with
the current `MEMORY.md`, and the answer is the new `MEMORY.md`. From Mastra it
takes two ideas rather than code: reading in pieces when the new conversations
do not fit one request, and saying that a changed fact replaces the old one.
From LangMem, waiting for a quiet spell rather than running after every
message.

## With years of history (reasoning, not measured)

A Dot used every day for a year holds on the order of ten million tokens of
conversation. What changes:

- **The files and grep hold.** Searching 40 MB of text is instant, and nothing
  is lost, because the files are the words as said. What degrades is
  precision: a common word matches hundreds of times. The first remedy costs no
  model call: a ranked keyword index (BM25) inside the Dot's computer.
- **One profile stops being enough.** A year of a working life does not fit in
  25,000 characters. The pass then needs two levels: `MEMORY.md` as the core,
  and notes by subject (clients, health, home) that it writes when the core is
  full, each named by a line in the core. That is the part of Mastra's design
  (its Reflector) worth taking later.
- **The cost of the pass stays flat**, because it reads only what is new:
  around a cent a day, a few dollars a year. A monthly rebuild of the profile
  from the files, about $2 at that size, would stop small errors from
  compounding over hundreds of rewrites.
- **Hindsight gains, and so does its bill.** Retrieving facts for each question
  from millions of tokens is its strength, and the gap to a bounded profile
  would grow for specific details. Its cost grows with the history (about $22 a
  year of extraction at this rate, plus a database per Dot), and its facts are
  extractions, lossy where the files are verbatim. For specific details the
  files already do that job.
- **The agentic designs and per-message extraction get worse**: the first
  reread more every pass, the second pay on every message.
- **Contradictions are the real risk** ("I live in Milan", then months later "I
  moved to Rome"). A rewrite with dated facts handles them; systems that only
  accumulate facts keep both.

## Sources

- LongMemEval: [paper](https://arxiv.org/abs/2410.10813), [code](https://github.com/xiaowu0162/LongMemEval)
- Mastra Observational Memory: [research](https://mastra.ai/research/observational-memory), [code](https://github.com/mastra-ai/mastra)
- LangMem: [code](https://github.com/langchain-ai/langmem)
- Hindsight: [paper](https://arxiv.org/abs/2512.12818), [code](https://github.com/vectorize-io/hindsight), [benchmark harness](https://github.com/vectorize-io/agent-memory-benchmark)
- Memobase: [code](https://github.com/memodb-io/memobase)
- Mem0: [paper](https://arxiv.org/abs/2504.19413), [code](https://github.com/mem0ai/mem0)
- Letta: [code](https://github.com/letta-ai/letta), [sleep-time compute](https://arxiv.org/abs/2504.13171)
- nanobot (Dream): [code](https://github.com/HKUDS/nanobot)
- Is Grep All You Need?: [arXiv 2605.15184](https://arxiv.org/html/2605.15184)
