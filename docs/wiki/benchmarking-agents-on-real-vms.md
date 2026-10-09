---
title: "Benchmarking an agent on real VMs"
description: "How Dots are tested as a person meets them: Harbor tasks in a new VM each, long builds of two to three hours, eight Dots on one host, and the defects these runs found."
parent: "Studies"
nav_order: 5
---

# Benchmarking an agent on real VMs

A Dot is an agent with its own computer: a QEMU virtual machine on the host,
reached through an API. Unit tests and a stand-in model check the engine's
logic; they cannot tell whether a Dot gets real work done. These runs do, on
real VMs and a real model, and every one of them found something.

## The harness

[Harbor](https://github.com/harbor-framework/harbor) (Apache-2.0), the harness
of Terminal-Bench, runs a task in an environment and grades what is left. Here
the environment is a Dot: a new one per trial, created through the product's
API, with the task's Dockerfile replayed inside it as the Dot's own user (there
is no root: `apt-get install` becomes the Dot's `sudo dot-install`), and the
task's paths moved under its home. The agent is the Dot itself, given the
instruction as a task. A task counts only once its reference solution scores 1
and doing nothing scores 0 **in a Dot**: a task that needs root fails its
oracle, which is how such a task is found. Code: `tests/bench/`.

## Short tasks

25 tasks written for this, in shell, files and documents, coding, data
analysis, the computer's settings, questions with one answer, two long
pipelines, and three about what a Dot is for (its own automation must run
twice; clean up leftovers and nothing else; say a request is impossible). The
data is generated with a fixed seed and the expected answers are computed by
the authoring script, never by an agent.

- Oracle 25/25, doing nothing 0/25.
- A Dot on z-ai/glm-5.3-flash: **73 of 75 trials** (each task three times).
  Both misses were the same task, a Bash script to port to Python.

## Long tasks: two to three hours

Five tasks where the Dot builds or researches something whole, each graded by
checking the result itself, never by asking a model:

| Task | What the Dot does | What the grader checks |
|---|---|---|
| chess-perft | a chess move generator, every rule included | published perft counts of five positions to depth 4 and 5 (up to 4.9 million), and four positions it has not seen |
| lisp-interpreter | an interpreter for a Scheme subset | 30 programs it has not seen: closures, tail calls a million deep, recursion 5000 deep, errors |
| kv-store | a key-value server over HTTP | the API, TTLs, 1000 writes from 20 clients at once, every acknowledged write after `kill -9` |
| language-history | research on 18 programming languages, on the web | years and creators, influences, a cited 1500-word report from five sites |
| fraud-investigation | an investigation of 292,467 generated card transactions | exactly the planted cases of four kinds of fraud, a script that finds them again |

These found the token limits described in
[Compacting a thread at the model's own window](compacting-at-the-models-window.md):
with a 4,096-token answer cap, a Dot writing a whole program in one call looped
until it failed. After the fix all five passed, the server and the interpreter
two out of two, in runs of 18 to 75 minutes.

## Memory: LongMemEval

The same harness runs [LongMemEval](https://github.com/xiaowu0162/LongMemEval)
on Dots: a history of 48 chats becomes a Dot's past conversations, the question
is a task, GPT-4o grades outside the Dot. The results and what they changed are
in [How a Dot remembers](how-a-dot-remembers.md) and
[Agent memory systems, measured on the same questions](agent-memory-systems-compared.md).
This run found two defects of its own: a token estimate OpenRouter refuses
(see [Token counts are not portable between models](token-counts-across-models.md)),
and in the harness a call to a Dot that could wait forever, which once held
four questions for seven hours; every call now has a bound.

## Many Dots on one host

`tests/e2e/scale.ts` starts 1, 2, 4 and 8 Dots at once on one host; each must
come up, answer a message and do a task, within budgets that are ratios to one
Dot. All four steps passed. Measured: the time to READY at the 95th percentile
went from 21 s with one Dot to 45 s with eight; the API answered in 6 to 10 ms
at the 95th percentile throughout; each Dot's QEMU took about 700 MB.

## What the other tests found

- **Approvals**: every tool under allow, deny, ask-then-approve and
  ask-then-reject, in the chat and in a task (258 cases), plus a race of an
  approve and a reject sent at the same moment, 50 times on PostgreSQL: one
  wins, the other is refused, the Dot hears one decision. It found that a task
  cancelled while a call waited for approval left the approval pending forever,
  so the Dot never went back to READY and never slept.
- **Upgrades**: the databases a Dot and a host of a released version left
  behind open under the new build and keep working.

Every run here costs real money on a real model; the numbers above were bought
for a few dollars each, and the harness keeps each Dot's events, engine log and
memory with its result so a wrong answer can be read after the Dot is gone.
