"""TEST HARNESS ONLY: Mem0 open source (https://github.com/mem0ai/mem0, Apache-2.0, `mem0ai` 2.2.1, the venv
/opt/mem-mem0 from requirements-mem0.txt) writing MEMORY.md for one LongMemEval question, so the bench can weigh it
against our own memory methods. Nothing here ships.

    /opt/mem-mem0/bin/python tests/bench/memory_mem0.py < input.json > output.json

input:  {"sessions": [{"date": "2023/05/20 (Sat) 02:21", "turns": [{"role", "content"}]}] oldest first, "question",
         "question_date", "base_url" (OpenAI-compatible, ending in /api/v1), "api_key", "model", "embedding_model",
         optional "top_k" (50) and "add_per" ("pair" or "session")}
output: {"memory_md", "notes"}; on failure the reason goes to stderr and the exit status is 1.

Used as Mem0 is, with its own prompts and its v3 pipeline (one additive extraction call per add(), hybrid search:
semantic, BM25 by fastembed, entity boost by spaCy), in a fresh local Qdrant and history db per question, telemetry
off. Configured: the LLM and the embedder (OpenAI-compatible, through `base_url`); everything else is Mem0's default.
The history is fed and searched the way Mem0's own LongMemEval runner does it (github.com/mem0ai/memory-benchmarks
@ 4b61c5d, benchmarks/longmemeval/run.py): one add() per user/assistant pair, oldest session first, pairs with an
empty message skipped; search() with the question as it is; the top k shown oldest first, grouped under their date
(prompts.get_answer_generation_prompt). That runner uses k = 200 and also reports k = 50; 50 is the default here
because 200 memories pass the 25,000 characters of MEMORY.md the Dot is given.

ADAPTED: the session's date. Mem0's platform takes it as add(timestamp=...), which the open-source add() refuses, and
the open-source pipeline gives its extraction prompt today's date as "Observation Date", the only date it resolves
"last week" against. So each add() gets, as the platform's timestamp would set them, the session's date as the
memory's created_at (through metadata, which Mem0 stores as given) and as the prompt's Observation Date (the
`timestamp` argument of Mem0's own generate_additive_extraction_prompt, which the open-source add() leaves unset).
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
from datetime import datetime
from importlib.metadata import version

TOP_K = 50


def session_time(date: str) -> datetime:
    """LongMemEval's "2023/05/20 (Sat) 02:21"."""
    day, _, clock = date.split(" ")
    return datetime.strptime(f"{day} {clock}", "%Y/%m/%d %H:%M")


def chunks(turns: list[dict], add_per: str) -> list[list[dict]]:
    messages = [{"role": t["role"], "content": t["content"]} for t in turns]
    if add_per == "session":
        return [messages] if messages else []
    pairs = [messages[i : i + 2] for i in range(0, len(messages), 2)]
    return [p for p in pairs if all(m["content"].strip() for m in p)]


def render(results: list[dict], top_k: int) -> str:
    """As Mem0's LongMemEval answerer is shown them: oldest first, under a line per date."""
    lines = [f"Memories Mem0 retrieved from past conversations for the current question ({len(results)} of at most {top_k}, oldest first):"]
    last = None
    for result in sorted(results, key=lambda r: r.get("created_at") or ""):
        created = result.get("created_at")
        if created:
            day = datetime.fromisoformat(created).strftime("%A, %B %d, %Y")
            if day != last:
                lines.append(f"\n--- {day} ---")
                last = day
        lines.append(f"- {result['memory']}")
    return "\n".join(lines)


def run(job: dict, store: str) -> dict:
    os.environ["MEM0_TELEMETRY"] = "False"
    os.environ["MEM0_DIR"] = store
    os.environ.pop("OPENROUTER_API_KEY", None)  # Mem0's OpenAI LLM would go to OpenRouter directly with it
    os.environ.setdefault("FASTEMBED_CACHE_PATH", os.path.join(sys.prefix, "fastembed_cache"))
    import mem0.memory.main as mem0_main
    from mem0 import Memory

    observation = {"date": None}
    extraction_prompt = mem0_main.generate_additive_extraction_prompt
    mem0_main.generate_additive_extraction_prompt = lambda *a, **k: extraction_prompt(*a, **{"timestamp": observation["date"], **k})

    llm_calls: list[str | None] = []
    memory = Memory.from_config(
        {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": job["model"],
                    "api_key": job["api_key"],
                    "openai_base_url": job["base_url"],
                    "response_callback": lambda llm, response, params: llm_calls.append(response.choices[0].finish_reason),
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {"model": job["embedding_model"], "api_key": job["api_key"], "openai_base_url": job["base_url"]},
            },
            "vector_store": {"provider": "qdrant", "config": {"path": os.path.join(store, "qdrant"), "on_disk": True}},
            "history_db_path": os.path.join(store, "history.db"),
        }
    )
    embed_calls = [0]
    embeddings = memory.embedding_model.client.embeddings
    create = embeddings.create

    def counted(*a, **k):
        embed_calls[0] += 1
        return create(*a, **k)

    embeddings.create = counted

    user = "longmemeval"
    add_per = job.get("add_per", "pair")
    adds = extracted = 0
    for session in job["sessions"]:
        when = session_time(session["date"])
        observation["date"] = when.date().isoformat()
        for messages in chunks(session["turns"], add_per):
            result = memory.add(messages, user_id=user, metadata={"created_at": when.isoformat() + "+00:00"})
            adds += 1
            extracted += len(result.get("results", []))
    add_llm, add_embed = len(llm_calls), embed_calls[0]

    top_k = int(job.get("top_k", TOP_K))
    found = memory.search(job["question"], filters={"user_id": user}, top_k=top_k)["results"]
    memory_md = render(found, top_k)
    truncated = sum(1 for f in llm_calls if f == "length")
    notes = (
        f"mem0ai {version('mem0ai')}; {len(job['sessions'])} sessions, {adds} add() calls (per {add_per}), {extracted} memories extracted; "
        f"adding: {add_llm} chat completions ({truncated} cut at max_tokens), {add_embed} embedding requests; "
        f"search: {len(llm_calls) - add_llm} chat completions, {embed_calls[0] - add_embed} embedding requests, "
        f"{len(found)} memories of top_k {top_k}, {len(memory_md)} characters"
    )
    return {"memory_md": memory_md, "notes": notes}


def main() -> None:
    job = json.load(sys.stdin)
    with tempfile.TemporaryDirectory(prefix="mem0-") as store:
        try:
            with contextlib.redirect_stdout(sys.stderr):  # stdout carries only the answer
                out = run(job, store)
        except Exception as error:
            print(f"memory_mem0: {type(error).__name__}: {error}", file=sys.stderr)
            sys.exit(1)
    json.dump(out, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
