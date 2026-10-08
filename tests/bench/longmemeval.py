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
fixed subset of 116, proportional to the benchmark's six kinds of question (`--questions 500` for all).

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


async def answer(question: dict[str, Any], history: str, out: Path) -> dict[str, Any]:
    name = f"bench-lme-{hashlib.sha256(question['question_id'].encode()).hexdigest()[:10]}"
    started = time.monotonic()
    created = await bridge.call("create", name, stdin=dot_yaml(name, cpus=2, memory_gb=4).encode())
    dot_id = created["id"]
    try:
        await give_history(dot_id, question, history)
        task = await bridge.call("task", dot_id, str(TASK_TIMEOUT_MS), stdin=task_text(question).encode())
        events = await bridge.call("events", dot_id, task["id"])
        (out / "events").mkdir(exist_ok=True)
        (out / "events" / f"{question['question_id']}.json").write_text(json.dumps(events, indent=1), encoding="utf-8")
    finally:
        await bridge.call("delete", dot_id)
    return {
        "question_id": question["question_id"],
        "hypothesis": task.get("summary") or "",
        "status": task.get("status"),
        "error": task.get("error"),
        "spent_usd": task.get("spent_usd"),
        "seconds": round(time.monotonic() - started),
    }


async def run(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    questions = subset(json.loads(Path(args.data).read_text(encoding="utf-8")), args.questions)
    done = {row["question_id"] for row in read_jsonl(out / "hypotheses.jsonl")}
    todo = [q for q in questions if q["question_id"] not in done]
    print(f"{len(questions)} questions, {len(done)} answered, {len(todo)} to go, {args.n} at once", flush=True)
    gate = asyncio.Semaphore(args.n)

    async def one(question: dict[str, Any]) -> None:
        async with gate:
            try:
                row = await answer(question, args.history, out)
            except Exception as error:  # a question the product could not run is reported, not graded
                print(f"{question['question_id']}: {error}", file=sys.stderr, flush=True)
                return
            append_jsonl(out / "hypotheses.jsonl", row)
            print(f"{row['question_id']} {row['status']} {row['seconds']}s ${row['spent_usd']}", flush=True)

    await asyncio.gather(*(one(q) for q in todo))


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
    commands.choices["run"].add_argument("--questions", type=int, default=116)
    commands.choices["run"].add_argument("-n", type=int, default=4)
    args = parser.parse_args()
    if args.command == "run":
        asyncio.run(run(args))
    else:
        judge(args)


if __name__ == "__main__":
    main()
