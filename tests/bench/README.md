# Benchmarks on real Dots

Agent benchmarks run on real Dots: each task gets a new Dot, its own computer
is set up the way the task says, the Dot is given the task through the API,
and the task's grader checks what is left on that computer. The runs use
[Harbor](https://github.com/harbor-framework/harbor) (Apache-2.0), the harness
of Terminal-Bench, with the Dot as its environment and its agent, so our own
tasks and Harbor's benchmarks run the same way.

TEST HARNESS ONLY: nothing here is part of the product.

## How a Dot runs a Harbor task

| Harbor | Here |
|---|---|
| environment | `dots_harbor/environment.py`: a new Dot per trial, created through the API with the CPU and memory the task asks for, every permission allowed (nobody is there to answer an approval), deleted after the trial |
| the task's `environment/Dockerfile` | replayed in the Dot's computer as its user `dot` (`dots_harbor/dockerfile.py`): `apt-get install` becomes `sudo dot-install`, `pip install` goes to dot's user site, `COPY` uploads, `WORKDIR` and `ENV` are followed |
| `/app`, `/tests`, `/solution`, `/logs` | moved under `/home/dot/bench` (`dots_harbor/paths.py`), in commands, uploaded text files and the instruction: the model's user has no root, and the host reaches only dot's home |
| agent | `dots_harbor/agent.py`: the instruction is sent to the Dot as a task; the Dot works with its own tools until the task ends; its spend is the trial's cost |
| `oracle` and `nop` | Harbor's own: the task's reference solution, or nothing |

Every call to the product goes through `bridge.ts`, which uses the end-to-end
run's outside driver (`tests/e2e/driver.ts`): the HTTP API and dot-agentd on a
running computer.

A task that needs root in its computer cannot run in a Dot. Its oracle fails,
which is how such a task is found: **a task counts only once its oracle scores
1 and `nop` scores 0 in a Dot.**

## The tasks

`tasks/` holds our own suite, 25 tasks written by `author.py` (run it with
the directory to write to; the data is generated with a fixed seed and the
expected answers are computed there, never by the agent). Each grader is
`tests/grade.py`, run by `tests/test.sh`.

| Category | Tasks |
|---|---|
| shell | `top-ips`, `large-files`, `fix-script`, `user-service` |
| files and documents | `csv-to-json`, `organize-downloads`, `extract-emails`, `invoice-report` |
| coding | `fix-pagination`, `wordfreq-cli`, `bash-to-python`, `git-history` |
| data analysis | `sales-by-region`, `temperature-stats`, `join-orders` |
| the computer's settings | `git-identity`, `ssh-config` |
| questions with one answer | `meeting-speaker`, `recipes-question`, `shipment-weights` |
| long tasks (45 minutes allowed) | `static-site`, `log-pipeline` |
| what a Dot is for | `automation-tick` (its own automation must run twice), `careful-cleanup` (remove the leftovers and nothing else), `impossible-request` (it must say it cannot) |

The oracle of `automation-tick` cannot use the Dot's automations, which belong
to the Dot: it writes the two lines the grader checks itself.

## The long tasks

`tasks-long/` holds five tasks of two to three hours (`author_long.py`, with
the reference solutions in `references/`), where the Dot researches or builds
something whole. Each is graded by checking the result itself, never by
asking a model:

| Task | What the Dot does | What the grader checks |
|---|---|---|
| `chess-perft` | a chess move generator, every rule included | the published perft counts of five positions at depth 4 and 5 (up to 4.9 million) and four positions it has not seen, each within 15 minutes |
| `lisp-interpreter` | an interpreter for a Scheme subset | 30 programs it has not seen: closures, tail calls a million deep, recursion 5000 deep, errors, output format |
| `kv-store` | a key-value database server over HTTP | the API, TTLs, 1000 writes from 20 clients at once, and every acknowledged write after a `kill -9` and a restart |
| `language-history` | research on 18 programming languages, on the web | the years and creators (16 of 18 right), the best-known influences between them, a cited report of 1500 words from five websites |
| `fraud-investigation` | an investigation of 292,467 generated card transactions | exactly the planted cases of four kinds of fraud, a script that finds them again, and a report with their numbers |

They run with more room than the product's defaults:

```bash
docker exec -w /work/dots -e E2E_OPENROUTER_KEY_FILE=/run/secrets/openrouter \
  -e BENCH_MAX_STEPS=1000 -e BENCH_MAX_COST_USD=10 idots-bench \
  bash tests/bench/run.sh dot -p tests/bench/tasks-long -n 5
```

## Running

In the bench container: the end-to-end run's Linux host plus Python and
Harbor (`Dockerfile`), on the same volumes, so the images the end-to-end run
built are used. Build `idots-linux-host` first (`tests/e2e/README.md`), prepare
`/work/dots` with `prepare.sh` as for the end-to-end run, then:

```bash
docker build -t idots-bench tests/bench
docker run -d --init --name idots-bench \
  --device /dev/kvm --group-add "$(stat -c %g /dev/kvm)" \
  -v "$PWD:/src:ro" -v ~/openrouter.key:/run/secrets/openrouter:ro \
  -v idots-e2e-home:/data -v idots-e2e-work:/work \
  -e INVISIBLE_DOTS_HOME=/data/home idots-bench sleep infinity

# The tasks are sound: the oracle scores 1 on each, doing nothing scores 0.
docker exec -w /work/dots -e E2E_OPENROUTER_KEY_FILE=/run/secrets/openrouter idots-bench \
  bash tests/bench/run.sh oracle -p tests/bench/tasks -n 4
docker exec -w /work/dots -e E2E_OPENROUTER_KEY_FILE=/run/secrets/openrouter idots-bench \
  bash tests/bench/run.sh nop -p tests/bench/tasks -n 4

# The Dot does them; -k 3 runs each three times (pass rate per task).
docker exec -w /work/dots -e E2E_OPENROUTER_KEY_FILE=/run/secrets/openrouter \
  -e BENCH_MODEL=z-ai/glm-5.3-flash idots-bench \
  bash tests/bench/run.sh dot -p tests/bench/tasks -n 4 -k 3
```

`run.sh` starts the server, stores the key, runs `harbor run` with the Dot
environment and stops the server. The arguments after the agent are
`harbor run`'s: `-p` a task or a folder of tasks, `-d terminal-bench@2.0` a
dataset of Harbor's registry, `-n` the trials at once, `-k` the attempts per
task, `-i`/`-x` to include or exclude tasks. Results go to `BENCH_JOBS_DIR`
(default `/work/bench-jobs`): `harbor view` shows them.

| Variable | Default | |
|---|---|---|
| `BENCH_MODEL` | `z-ai/glm-5.3-flash` | the Dot's model (OpenRouter id) |
| `BENCH_MAX_STEPS` | 150 | the Dot's `limits.max_steps_per_task` |
| `BENCH_MAX_COST_USD` | 3 | the Dot's `limits.max_cost_per_task_usd` |

## LongMemEval: does a Dot remember?

`longmemeval.py` runs [LongMemEval](https://github.com/xiaowu0162/LongMemEval)
(MIT; ICLR 2025): each question has a history of about 48 dated chats (115K
tokens) and asks what only that history answers. Each question gets a new Dot:
the history becomes its conversation files, as the engine writes them; the Dot
makes its own memory pass over them; the question is a task; LongMemEval's own
prompts grade the answer with GPT-4o, outside the Dot. Questions: a fixed
subset of 116 in the benchmark's proportions, or every one of a kind (`--kind`).

```bash
docker exec -w /work/dots -e E2E_OPENROUTER_KEY_FILE=/run/secrets/openrouter idots-bench \
  bash tests/bench/run.sh longmemeval run --data /data-lme/longmemeval_s_cleaned.json --out /work/bench-jobs/lme -n 6
docker exec -w /work/dots -e E2E_OPENROUTER_KEY_FILE=/run/secrets/openrouter idots-bench \
  python3 tests/bench/longmemeval.py judge --data /data-lme/longmemeval_s_cleaned.json --out /work/bench-jobs/lme
```

Measured with z-ai/glm-5.3-flash (October 2026), 116 questions: 7.8% with no
history (the control, `--history none`), 82.8% with the conversation files, 88%
after the token and prompt fixes that run found, 90.5% with MEMORY.md written
by the memory pass (updates 78 to 94%, preferences 57 to 71%).

How MEMORY.md is written was chosen on the 30 preference questions, the same
for every method:

| Method | Correct | Cost of a pass over 115K tokens |
|---|---|---|
| no MEMORY.md (the Dot searches the files) | 39/60, 65% | 0 |
| **one request (the memory pass)** | **50/60, 83%** | **~$0.015, 1 call** |
| Hindsight (MIT), recall for the question | 25/30, 83% | ~$0.25, ~255 calls, Postgres |
| Mastra Observational Memory (Apache-2.0) | 48/60, 80% | ~$0.05, ~6 calls |
| LangMem profile manager (MIT) | 48/60, 80% | ~$0.02, 1 call |
| an agent pass with file tools (upstream nanobot's Dream) | 44/59, 75% | ~$0.08 |

Mem0, Letta and Memobase were not run to the end: by their own call patterns a
pass would cost $0.25 to $1.60. The adapters of that study (their prompts
ported; Mem0, Hindsight and Letta in their own venvs) are in commit `de3d9a1b`.

## External benchmarks

Harbor's registry datasets run the same way, for instance Terminal-Bench 2.0
(89 shell, coding and system tasks):

```bash
docker exec -w /work/dots -e E2E_OPENROUTER_KEY_FILE=/run/secrets/openrouter idots-bench \
  bash tests/bench/run.sh oracle -d terminal-bench@2.0 -n 4
```

Run the oracle first: the tasks whose oracle fails in a Dot (they need root, a
program dot-install cannot give, or a path outside the mapped roots) are left
out of the Dot's run with `-x`.
