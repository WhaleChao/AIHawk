"""LongMemEval on real Dots: does a Dot remember what it was told in past conversations?

LongMemEval (github.com/xiaowu0162/LongMemEval, MIT; ICLR 2025) gives each question a history of about 48
dated chat sessions (115K tokens), then asks something only that history answers: a detail the person
mentioned, one that changed later, how long ago something happened, or one it never said (the answer is
then that it is not known). Here each question gets a new Dot. Its history reaches the Dot the way the
product keeps past conversations (`--history`), the question is sent as a task, the task's answer is the
hypothesis, and LongMemEval's own judge prompts grade it with the model its paper uses (GPT-4o), outside
the Dot. TEST HARNESS ONLY.

    python tests/bench/longmemeval.py run   --data <longmemeval_s_cleaned.json> --out <dir> [--history none] [-n 4]
    python tests/bench/longmemeval.py judge --data <longmemeval_s_cleaned.json> --out <dir>

`run` answers the questions not yet in <dir>/hypotheses.jsonl (a stopped run goes on where it stopped);
`judge` grades the answers not yet in <dir>/judged.jsonl and writes <dir>/report.md. The questions are a
fixed subset of 116, proportional to the benchmark's six kinds of question (`--questions 500` for all);
`--kind <question_type>` takes every question of one kind instead.

--history conversations (the default): the sessions become the Dot's chat files, by day, in
/home/dot/conversations, written by the engine's own code in the Dot (nanobot/dots/conversations.py), as if
the person had had those conversations with it.

--history none: nothing of the history reaches the Dot. It is the control, what a Dot that keeps nothing
of its past conversations scores: about the share of questions whose right answer is "not known".
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dots_harbor import bridge  # noqa: E402
from dots_harbor.environment import dot_yaml  # noqa: E402

SUBSET_SEED = 20261008
JUDGE_MODEL = "openai/gpt-4o-2024-08-06"  # evaluate_qa.py's "gpt-4o", through OpenRouter
LONGMEMEVAL = Path(os.environ.get("LONGMEMEVAL_DIR", "/opt/longmemeval"))
TASK_TIMEOUT_MS = 60 * 60 * 1000


def subset(questions: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    """`size` questions with the benchmark's share of each kind (largest remainder), drawn with a fixed seed."""
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for question in sorted(questions, key=lambda q: q["question_id"]):
        by_type[question["question_type"]].append(question)
    exact = {kind: size * len(items) / len(questions) for kind, items in by_type.items()}
    counts = {kind: int(share) for kind, share in exact.items()}
    for kind in sorted(exact, key=lambda k: (counts[k] - exact[k], k))[: size - sum(counts.values())]:
        counts[kind] += 1
    rng = random.Random(SUBSET_SEED)
    chosen = [q for kind in sorted(by_type) for q in rng.sample(by_type[kind], counts[kind])]
    return sorted(chosen, key=lambda q: q["question_id"])


def task_text(question: dict[str, Any]) -> str:
    # The history is dated in 2023: the Dot is told the question's date, as LongMemEval's own prompts tell it.
    return f"(Today is {question['question_date']}.)\n\n{question['question']}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(row, ensure_ascii=False) + "\n")


HISTORY_JSON = "/home/dot/bench-history.json"
# Run by the engine's own Python in the Dot: the history's sessions become the chat's files exactly as the
# engine writes its chat (nanobot/dots/conversations.py), each message at its session's time.
WRITE_CHAT_FILES = f"""
import json, pathlib
from nanobot.dots import conversations
from nanobot.dots.transcript_outbox import INBOUND_ID
from nanobot.agent.transcript_metadata import METADATA_KEY
history = json.loads(pathlib.Path({HISTORY_JSON!r}).read_text(encoding="utf-8"))
messages = []
for date, session in zip(history["dates"], history["sessions"]):
    stamp = date[:10].replace("/", "-") + "T" + date[-5:] + ":00"
    for turn in session:
        message = {{"timestamp": stamp, "role": turn["role"], "content": turn["content"]}}
        if turn["role"] == "user":
            message[METADATA_KEY] = {{INBOUND_ID: "longmemeval"}}
        messages.append(message)
for path, text in conversations.chat_files(messages, 0, target=lambda name, arguments: None).items():
    file = pathlib.Path(conversations.CONVERSATIONS_DIR, path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(text, encoding="utf-8")
pathlib.Path({HISTORY_JSON!r}).unlink()
"""


async def give_history(dot_id: str, question: dict[str, Any], history: str) -> None:
    """Put the question's history where the product keeps past conversations (nothing for "none")."""
    if history == "none":
        return
    sessions = {"dates": question["haystack_dates"], "sessions": question["haystack_sessions"]}
    await bridge.call("put", dot_id, HISTORY_JSON, stdin=json.dumps(sessions).encode())
    result = await bridge.call(
        "exec", dot_id, "120000", stdin=f"/opt/invisible-dots-engine/bin/python -I -B - <<'PY'\n{WRITE_CHAT_FILES}\nPY\n".encode()
    )
    if result["exit_code"] != 0:
        raise RuntimeError(f"writing the history failed: {result['stderr'][-1500:]}")


# Run by the engine's own Python in the Dot: the prompt the Dot's memory pass works from (templates/agent/dream.md),
# over every conversation file.
DREAM_PROMPT = """
import pathlib
from nanobot.dots.conversations import CONVERSATIONS_DIR
from nanobot.dots.skills import DOT_SKILLS_DIR
from nanobot.dots.turns import MEMORY_DIR
from nanobot.utils.prompt_templates import render_template
files = sorted(str(path) for path in pathlib.Path(CONVERSATIONS_DIR).rglob("*.md"))
print(render_template("agent/dream.md", conversations=files, memory_dir=MEMORY_DIR, dot_skills_dir=DOT_SKILLS_DIR))
"""


async def dream(dot_id: str) -> dict[str, Any]:
    """The memory pass over the history, as a task before the question."""
    rendered = await bridge.call(
        "exec", dot_id, "120000", stdin=f"/opt/invisible-dots-engine/bin/python -I -B - <<'PY'\n{DREAM_PROMPT}\nPY\n".encode()
    )
    if rendered["exit_code"] != 0:
        raise RuntimeError(f"rendering the memory pass failed: {rendered['stderr'][-1500:]}")
    return await bridge.call("task", dot_id, str(TASK_TIMEOUT_MS), stdin=rendered["stdout"].encode())


# Run by the engine's own Python in the Dot: the one request that writes MEMORY.md from the conversations
# (templates/agent/memory_update.md).
PROFILE_PROMPT = """
import pathlib
from nanobot.dots.conversations import CONVERSATIONS_DIR
from nanobot.dots.turns import MEMORY_DIR, MEMORY_INDEX
from nanobot.utils.prompt_templates import render_template
memory = pathlib.Path(MEMORY_DIR, MEMORY_INDEX)
files = sorted(pathlib.Path(CONVERSATIONS_DIR).rglob("*.md"))
print(render_template(
    "agent/memory_update.md",
    memory_md=memory.read_text(encoding="utf-8") if memory.exists() else "",
    conversations=[{"path": str(f), "text": f.read_text(encoding="utf-8")} for f in files],
))
"""


# Run by the engine's own Python in the Dot: the conversation files, oldest first, as JSON.
CONVERSATIONS_JSON = """
import json, pathlib
from nanobot.dots.conversations import CONVERSATIONS_DIR
files = sorted(pathlib.Path(CONVERSATIONS_DIR).rglob("*.md"))
print(json.dumps([{"path": str(f), "text": f.read_text(encoding="utf-8")} for f in files]))
"""


async def _engine_python(dot_id: str, script: str, what: str) -> str:
    result = await bridge.call(
        "exec", dot_id, "120000", stdin=f"/opt/invisible-dots-engine/bin/python -I -B - <<'PY'\n{script}\nPY\n".encode()
    )
    if result["exit_code"] != 0:
        raise RuntimeError(f"{what} failed: {result['stderr'][-1500:]}")
    return result["stdout"]


# Memory systems that run as their own programs, each in its venv in the bench container (memory_<name>.py), calling
# the model through the metering proxy (openrouter_meter.py), which records what each question cost.
EXTERNAL = ("mem0", "hindsight", "letta")
METER_PORT = 8790
METER_LOG = Path(os.environ.get("BENCH_METER_LOG", "/work/bench-jobs/meter.jsonl"))
LETTA_URL = os.environ.get("BENCH_LETTA_URL", "http://127.0.0.1:8283")


async def external_memory(dot_id: str, variant: str, question: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """MEMORY.md as Mem0, Hindsight or Letta make it from the question's history, and the requests it cost."""
    conversations = json.loads(await _engine_python(dot_id, CONVERSATIONS_JSON, "reading the conversations"))
    request = {
        "conversations": conversations,
        "sessions": [
            {"date": date, "turns": [{"role": t["role"], "content": t["content"]} for t in session]}
            for date, session in zip(question["haystack_dates"], question["haystack_sessions"])
        ],
        "question": question["question"],
        "question_date": question["question_date"],
        "base_url": f"http://127.0.0.1:{METER_PORT}/q/{question['question_id']}/api/v1",
        "api_key": Path(os.environ["E2E_OPENROUTER_KEY_FILE"]).read_text(encoding="utf-8").strip(),
        "model": os.environ.get("BENCH_MODEL", "z-ai/glm-5.3-flash"),
        "embedding_model": "openai/text-embedding-3-small",
        "letta_url": LETTA_URL,
        "question_id": question["question_id"],
        # Mem0 per session rather than per pair (its own runner's choice), at about a fifth of the calls.
        "add_per": "session",
        # What Hindsight recalls, kept within the 25,000 characters a Dot carries of MEMORY.md (its benchmark asks
        # for 32768 + 16384 tokens): its own ranking chooses, not a cut.
        "recall_max_tokens": 4000,
        "chunk_max_tokens": 2000,
    }
    process = await asyncio.create_subprocess_exec(
        f"/opt/mem-{variant}/bin/python",
        str(Path(__file__).resolve().parent / f"memory_{variant}.py"),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await process.communicate(json.dumps(request).encode())
    if process.returncode != 0:
        raise RuntimeError(f"{variant} failed: {err.decode(errors='replace')[-1500:]}")
    usages = [line for line in read_jsonl(METER_LOG) if line["question"] == question["question_id"]]
    return json.loads(out)["memory_md"], usages


async def profile(dot_id: str, variant: str, question: dict[str, Any]) -> dict[str, Any]:
    """MEMORY.md written from the conversations, outside the Dot with the Dot's model, then put in the Dot: by our one
    request (templates/agent/memory_update.md, "ours") or by an open-source method ported in memory_variants.py."""
    from openai import AsyncOpenAI

    key = Path(os.environ["E2E_OPENROUTER_KEY_FILE"]).read_text(encoding="utf-8").strip()
    client = AsyncOpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")
    finishes: list[str | None] = []

    async def complete(messages: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        completion = await client.chat.completions.create(
            model=os.environ.get("BENCH_MODEL", "z-ai/glm-5.3-flash"),
            messages=messages,
            extra_body={"usage": {"include": True}},
        )
        finishes.append(completion.choices[0].finish_reason)
        return (completion.choices[0].message.content or "").strip(), (completion.usage.model_dump() if completion.usage else {})

    if variant in EXTERNAL:
        text, usages = await external_memory(dot_id, variant, question)
    elif variant == "ours":
        prompt = await _engine_python(dot_id, PROFILE_PROMPT, "rendering the memory update")
        text, usage = await complete([{"role": "user", "content": prompt}])
        usages = [usage]
    else:
        import memory_variants

        conversations = json.loads(await _engine_python(dot_id, CONVERSATIONS_JSON, "reading the conversations"))
        text, usages = await memory_variants.write_memory(variant, conversations, "", complete)
    await bridge.call("put", dot_id, "/home/dot/memory/MEMORY.md", stdin=(text + "\n").encode())
    return {
        "profile_variant": variant,
        "profile_calls": len(usages),
        "profile_spent_usd": sum(u.get("cost") or 0 for u in usages),
        "profile_prompt_tokens": sum(u.get("prompt_tokens") or 0 for u in usages),
        "profile_answer_tokens": sum(u.get("completion_tokens") or 0 for u in usages),
        "profile_finish": sorted({str(f) for f in finishes}),
    }


async def answer(question: dict[str, Any], history: str, out: Path, with_dream: bool, profile_variant: str | None) -> dict[str, Any]:
    name = f"bench-lme-{hashlib.sha256(question['question_id'].encode()).hexdigest()[:10]}"
    started = time.monotonic()
    created = await bridge.call("create", name, stdin=dot_yaml(name, cpus=2, memory_gb=4).encode())
    dot_id = created["id"]
    try:
        await give_history(dot_id, question, history)
        dreamt = await dream(dot_id) if with_dream else {}
        profiled = await profile(dot_id, profile_variant, question) if profile_variant else {}
        if with_dream:
            (out / "dream-events").mkdir(exist_ok=True)
            (out / "dream-events" / f"{question['question_id']}.json").write_text(
                json.dumps(await bridge.call("events", dot_id, dreamt["id"]), indent=1), encoding="utf-8"
            )
        task = await bridge.call("task", dot_id, str(TASK_TIMEOUT_MS), stdin=task_text(question).encode())
        events = await bridge.call("events", dot_id, task["id"])
        (out / "events").mkdir(exist_ok=True)
        (out / "events" / f"{question['question_id']}.json").write_text(json.dumps(events, indent=1), encoding="utf-8")
        # The engine's own log goes with the Dot: kept for every question, since a wrong answer is known only
        # once judged.
        journal = await bridge.call(
            "exec", dot_id, "60000", stdin=b"journalctl -u invisible-dots-agent --no-pager -o cat | tail -n 3000"
        )
        (out / "engine").mkdir(exist_ok=True)
        (out / "engine" / f"{question['question_id']}.log").write_text(journal["stdout"] or journal["stderr"], encoding="utf-8")
        if with_dream or profile_variant:
            # What the memory pass left the question to work from.
            memory = await bridge.call("exec", dot_id, "60000", stdin=b"cat /home/dot/memory/MEMORY.md")
            (out / "memory").mkdir(exist_ok=True)
            (out / "memory" / f"{question['question_id']}.md").write_text(memory["stdout"], encoding="utf-8")
    finally:
        await bridge.call("delete", dot_id)
    return {
        "question_id": question["question_id"],
        "hypothesis": task.get("summary") or "",
        "status": task.get("status"),
        "error": task.get("error"),
        "spent_usd": task.get("spent_usd"),
        "seconds": round(time.monotonic() - started),
        **(
            {"dream_status": dreamt.get("status"), "dream_error": dreamt.get("error"), "dream_spent_usd": dreamt.get("spent_usd")}
            if with_dream
            else {}
        ),
        **profiled,
    }


async def run(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    every = json.loads(Path(args.data).read_text(encoding="utf-8"))
    if args.kind:
        questions = sorted((q for q in every if q["question_type"] == args.kind), key=lambda q: q["question_id"])
    else:
        questions = subset(every, args.questions)
    if args.limit:
        questions = questions[: args.limit]
    done = {row["question_id"] for row in read_jsonl(out / "hypotheses.jsonl")}
    todo = [q for q in questions if q["question_id"] not in done]
    print(f"{len(questions)} questions, {len(done)} answered, {len(todo)} to go, {args.n} at once", flush=True)
    gate = asyncio.Semaphore(args.n)

    async def one(question: dict[str, Any]) -> None:
        async with gate:
            try:
                row = await answer(question, args.history, out, args.dream, args.profile)
            except Exception as error:  # a question the product could not run is reported, not graded
                print(f"{question['question_id']}: {error}", file=sys.stderr, flush=True)
                return
            append_jsonl(out / "hypotheses.jsonl", row)
            print(f"{row['question_id']} {row['status']} {row['seconds']}s ${row['spent_usd']}", flush=True)

    meter = None
    if args.profile in EXTERNAL:
        meter = await asyncio.create_subprocess_exec(
            sys.executable, str(Path(__file__).resolve().parent / "openrouter_meter.py"),
            "--port", str(METER_PORT), "--log", str(METER_LOG),
        )
    try:
        await asyncio.gather(*(one(q) for q in todo))
    finally:
        if meter is not None:
            meter.terminate()
            await meter.wait()


def judge_prompt():
    """LongMemEval's own grading prompts (src/evaluation/evaluate_qa.py), one per kind of question."""
    spec = importlib.util.spec_from_file_location("evaluate_qa", LONGMEMEVAL / "src" / "evaluation" / "evaluate_qa.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_anscheck_prompt


def judge(args: argparse.Namespace) -> None:
    from openai import OpenAI

    out = Path(args.out)
    references = {q["question_id"]: q for q in json.loads(Path(args.data).read_text(encoding="utf-8"))}
    key = Path(os.environ["E2E_OPENROUTER_KEY_FILE"]).read_text(encoding="utf-8").strip()
    client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")
    prompt_for = judge_prompt()
    judged = {row["question_id"] for row in read_jsonl(out / "judged.jsonl")}
    for row in read_jsonl(out / "hypotheses.jsonl"):
        if row["question_id"] in judged:
            continue
        reference = references[row["question_id"]]
        prompt = prompt_for(
            reference["question_type"],
            reference["question"],
            reference["answer"],
            row["hypothesis"],
            abstention="_abs" in row["question_id"],
        )
        completion = client.chat.completions.create(
            model=JUDGE_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0, max_tokens=10
        )
        verdict = completion.choices[0].message.content.strip()
        append_jsonl(out / "judged.jsonl", {**row, "judge": verdict, "correct": "yes" in verdict.lower()})
    report(out, references)


def report(out: Path, references: dict[str, dict[str, Any]]) -> None:
    rows = read_jsonl(out / "judged.jsonl")
    kinds: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        kinds[references[row["question_id"]]["question_type"]].append(row["correct"])
        if row["question_id"].endswith("_abs"):
            kinds["(abstention)"].append(row["correct"])
    spent = [row["spent_usd"] for row in rows if isinstance(row.get("spent_usd"), (int, float))]
    seconds = sorted(row["seconds"] for row in rows)
    statuses = Counter(row["status"] for row in rows)
    lines = [
        f"# LongMemEval: {out.name}",
        "",
        f"**{sum(r['correct'] for r in rows)}/{len(rows)} = {100 * sum(r['correct'] for r in rows) / max(1, len(rows)):.1f}%**",
        "",
        "| Kind | Correct | Accuracy |",
        "|---|---|---|",
        *(
            f"| {kind} | {sum(v)}/{len(v)} | {100 * sum(v) / len(v):.1f}% |"
            for kind, v in sorted(kinds.items())
        ),
        "",
        f"Tasks: {dict(statuses)}. Spent ${sum(spent):.2f} (${sum(spent) / max(1, len(spent)):.3f} a question). "
        f"Time per question: median {seconds[len(seconds) // 2] if seconds else 0}s, longest {seconds[-1] if seconds else 0}s.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "judge"):
        sub = commands.add_parser(command)
        sub.add_argument("--data", required=True)
        sub.add_argument("--out", required=True)
    commands.choices["run"].add_argument("--history", default="conversations", choices=["conversations", "none"])
    commands.choices["run"].add_argument(
        "--dream", action="store_true", help="the memory pass (templates/agent/dream.md) runs as a task before the question"
    )
    commands.choices["run"].add_argument(
        "--profile",
        choices=["ours", "mastra", "langmem", "memobase", *EXTERNAL],
        help="MEMORY.md is written before the question: by our one request (templates/agent/memory_update.md) "
        "or by an open-source method (memory_variants.py; memory_<name>.py for Mem0, Hindsight and Letta, through "
        "openrouter_meter.py)",
    )
    commands.choices["run"].add_argument("--questions", type=int, default=116)
    commands.choices["run"].add_argument("--kind", help="every question of this question_type instead of the subset")
    commands.choices["run"].add_argument("--limit", type=int, help="only the first N of the questions (a pilot)")
    commands.choices["run"].add_argument("-n", type=int, default=4)
    args = parser.parse_args()
    if args.command == "run":
        asyncio.run(run(args))
    else:
        judge(args)


if __name__ == "__main__":
    main()
