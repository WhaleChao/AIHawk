---
title: "How a Dot remembers"
description: "Past conversations kept as files the agent greps, a profile of the person in every prompt, and a background pass that keeps it current. Measured on LongMemEval: 7.8% to 90.4%."
parent: "Studies"
nav_order: 1
---

# How a Dot remembers

A Dot talks to one person for weeks: in its chat, and in the tasks it is
given. Three things make it remember what it was told, and each one was
added because a measurement asked for it.

| Layer | What it holds | Who writes it | In the prompt? |
|---|---|---|---|
| Conversations | every word said, by day and by task | the engine, after every turn | no: the Dot searches it with grep |
| `MEMORY.md` | what the Dot should always know about the person | the Dot, and a pass in the background | always |
| Notes and skills | details by subject, and how to do things | the Dot | when it opens them |

## The measure: LongMemEval

[LongMemEval](https://github.com/xiaowu0162/LongMemEval) (ICLR 2025, MIT)
gives each question a history of about 48 dated chats, around 115,000
tokens, and then asks something only that history answers: a detail the
person mentioned, one that changed later, how many days passed between two
events, what they would like, or something they never said (the right answer
is then that it is not known). The answer is graded by GPT-4o with the
benchmark's own prompts.

On Dots it runs as a person would meet it: a new Dot per question, the
history written as its past conversations, the question sent as a task, the
grading done outside the Dot so the OpenRouter key never enters it. 116
questions, drawn with a fixed seed in the benchmark's proportions; model
z-ai/glm-5.3-flash. The harness is `tests/bench/longmemeval.py`.

| | Correct | % |
|---|---|---|
| No history at all (the control) | 9/116 | 7.8% |
| Conversations as files | 96/116 | 82.8% |
| + the fixes that run found | 102/116 | 87.9% |
| + `MEMORY.md`, written outside the Dot as the pass would | 105/116 | 90.5% |
| + the memory pass in the product | 104/115 | **90.4%** |

The control only gets the four questions whose answer is "you never told
me", and five it can guess. The last row is the product as shipped: each Dot
made its own pass five quiet minutes after it started, then answered (one
question of the 116 did not run).

## Conversations as files

The transcripts live in the engine's database, which the model's commands
cannot read, and a long chat reaches the model only as a summary of its older
part. So after every turn the engine also writes what was said to the Dot's
own computer:

```
/home/dot/conversations/chat/2023-05-20.md         one file a day of the chat
/home/dot/conversations/tasks/2023-05-21-t1.md     one file a task
```

Each message sits under a heading with its time, and each call the Dot made
is one line (not what it returned). The prompt says to search there with grep
when something said before matters. This one change took the Dot from 7.8% to
82.8%. Plain files and grep were chosen over embeddings or a vector store
because that is what wins in the published measures: in "Is Grep All You
Need?" grep beat vector search in every harness and model tried, and on
LongMemEval-V2 the best result came from an agent searching files (74.9%)
rather than from extracted facts with retrieval (58.6%).

## What the first run found

Twenty answers were wrong, and reading each one showed where:

- **4 of the 5 preference misses searched nothing.** Asked "can you suggest
  accessories for my phone?", the Dot gave generic advice: nothing in the
  question says "remember". The prompt now says to look in past conversations
  before advice or a recommendation.
- **One task was refused by OpenRouter** for asking an answer 2,779 tokens
  too long for the window: the token estimate was off. See
  [Token counts are not portable between models](token-counts-across-models.md).
- The system prompt carried the time to the minute, so it changed on every
  turn and the provider's cache never held. It now carries the day.

Together: 82.8% to 87.9%.

## MEMORY.md, and the pass that keeps it

The remaining preference misses had a common cause grep cannot fix: the fact
that matters shares no word with the question. "I've been sneezing, could it
be my living room?" needs "my cat Luna sheds", and a search for "sneez|dust"
does not find the cat. What fixes that is a short profile of the person in
every prompt, so the Dot does not have to guess what to look for.

`MEMORY.md` is that profile: who the person is, the people and animals in
their life, what they have and use, what they like and dislike, their
routines and plans, each fact with the day it was said, and a line for every
other note. The Dot sees it whole in every prompt, up to 25,000 characters
(cut there, with a word to shorten it, as Claude Code does with its own).

A model often does not think to write it while it works, so the engine keeps
it current itself: once the Dot has been quiet for five minutes, it takes the
conversation files that changed since the last time and has the summary model
rewrite `MEMORY.md` from them, in one request with no tools. A changed fact
replaces the old one with its date; nothing a web page or a command said is
copied in as an instruction. If the conversations do not fit one request they
go in several, oldest first. A pass that is cut, empty, unpriced, or races
with the Dot editing the file writes nothing, and the next one takes the same
conversations again. Its cost goes to the person's usage like any other turn.

On the 30 preference questions it raised the right answers from 65% to 83%,
and on the 116 mixed ones updates went from 78% to 94% (the profile keeps the
newest version of a fact), with nothing else moving beyond the noise of a run.
How this was chosen among five methods is in
[Agent memory systems, measured on the same questions](agent-memory-systems-compared.md).

## What it costs

With glm-5.3-flash a pass over a whole 115,000-token history cost $0.025 on
average in the product run ($0.015 in the study, where the request was sent
from outside the Dot). In use a pass reads only what changed since the last
one, so a normal day of chat should come to a cent or two. Answering a question that needs the
history cost the Dot $0.003 to $0.005 on average.
