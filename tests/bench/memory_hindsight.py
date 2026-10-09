"""TEST HARNESS ONLY: Hindsight writes MEMORY.md for one LongMemEval question, so the bench can weigh it against our
own memory methods. Nothing here ships.

Hindsight, MIT, https://github.com/vectorize-io/hindsight (hindsight-api / hindsight-client 0.10.3, pinned in
requirements-hindsight.txt), run the way its own LongMemEval runs run it. Hindsight's benchmarks live in AMB,
https://github.com/vectorize-io/agent-memory-benchmark (MIT; hindsight-system-evals/README.md: "AMB is now the only
copy"), main @ 8ea4c837cb00903453515929df4f80bcc793530c, provider `hindsight-http`:

  src/memory_bench/dataset/longmemeval.py  load_documents: one document a session, content = json.dumps(the
      session's turns), context "Session <id> - you are the assistant in this conversation - happened on
      <YYYY-mm-dd HH:MM:SS> UTC.", timestamp = the session's date (UTC ISO); load_queries: query_timestamp = the
      question's date (UTC ISO). One bank per question (isolation_unit = "question").
  src/memory_bench/memory/hindsight.py     _bank_kwargs (enable_observations=False), async_ingest (retain_batch
      of 20 items, retain_async=True, then _await_bank_ingest: wait until no operation is pending or processing,
      never going on past a failed one), _doc_to_items, _recall_kwargs (budget "high", max_tokens 32768,
      include_chunks with max_chunk_tokens 16384, include_entities False, query cut at 1900 characters,
      query_timestamp), _deduplicate_results and _format_result(s).
  src/memory_bench/modes/rag.py            the context the answer model is given: "## Memory <n>\\n<result>" joined
      by blank lines. That text is MEMORY.md.

ADAPTED: one Hindsight server per question, started here as its README's bare-metal way (`hindsight-api`, embedded
Postgres "pg0" under a name of its own, Hindsight's default local embeddings BAAI/bge-small-en-v1.5 and reranker
cross-encoder/ms-marco-MiniLM-L-6-v2), and dropped afterwards; AMB runs one long-lived server. Only the LLM is
configured: Hindsight's "openrouter" provider (OpenAI-compatible; it adds OpenRouter's require_parameters when it
asks for JSON) with its base URL set to the bench's metering proxy, which forwards to OpenRouter. Session ids are
"<question id>_<index>" because the input carries no session ids. An operation that still fails after Hindsight's
own retries (3 a task, each LLM call retried with backoff) ends the run; AMB re-queues it up to 5 more times, 180
seconds apart, to ride out outages of a shared server. AMB answers with an LLM over the raw recall JSON
(dataset.build_rag_prompt); here the Dot answers with the rendered text in MEMORY.md.

"observations": true in the input turns on Hindsight's own default instead of AMB's setting: observation
consolidation after each retain, waited for like the retains (its operations are in the same queue).

    /opt/mem-hindsight/bin/python memory_hindsight.py < input.json > output.json
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from hindsight_client import Hindsight
from pg0 import Pg0

HINDSIGHT_API = str(Path(sys.executable).parent / "hindsight-api")
BATCH_SIZE = 20  # AMB async_ingest
STARTUP_TIMEOUT_S = 900  # the first start downloads the embedding and reranker models
INGEST_TIMEOUT_S = 8 * 3600  # AMB_INGEST_MAX_WAIT_S


def parse_date(text: str) -> datetime:
    """LongMemEval's "2023/05/20 (Sat) 02:21" as UTC, as AMB's LongMemEvalDataset._parse_date reads it."""
    return datetime.strptime(text.split("(")[0].strip() + " " + text.split(")")[-1].strip(), "%Y/%m/%d %H:%M").replace(
        tzinfo=timezone.utc
    )


def items_for(sessions: list[dict], question_id: str) -> list[dict]:
    """AMB LongMemEvalDataset.load_documents + _HindsightBase._doc_to_items, one item a session."""
    items = []
    for index, session in enumerate(sessions):
        doc_id = f"{question_id}_{index}"
        date = parse_date(session["date"])
        turns = [{"role": t["role"], "content": t["content"]} for t in session["turns"]]
        items.append(
            {
                "content": json.dumps(turns).replace("\x00", ""),
                "document_id": doc_id,
                "metadata": {"doc_id": doc_id},
                "timestamp": date.isoformat(),
                "context": f"Session {doc_id} - you are the assistant in this conversation - happened on "
                f"{date.strftime('%Y-%m-%d %H:%M:%S')} UTC.",
            }
        )
    return items


def format_result(result: dict, chunks: dict, seen_chunks: set) -> str:
    """AMB _format_result."""
    lines = [f"**[{result['type']}]** {result['text']}" if result.get("type") else result["text"]]
    meta = []
    start, end = result.get("occurred_start"), result.get("occurred_end")
    if start and end and start != end:
        meta.append(f"occurred: {start} – {end}")
    elif start:
        meta.append(f"occurred: {start}")
    if result.get("mentioned_at"):
        meta.append(f"mentioned: {result['mentioned_at']}")
    if result.get("chunk_id"):
        meta.append(f"chunk: {result['chunk_id']}")
    if meta:
        lines.append("_" + " · ".join(meta) + "_")
    chunk_id = result.get("chunk_id")
    if chunk_id and chunk_id in chunks and chunk_id not in seen_chunks:
        lines.append(f"> {chunks[chunk_id]['text']}")
        seen_chunks.add(chunk_id)
    return "\n".join(lines)


def render(recalled: dict) -> str:
    """AMB _deduplicate_results + _format_results, joined as modes/rag.py joins them for the answer model."""
    seen_ids: set = set()
    results = [r for r in recalled.get("results") or [] if not (r["id"] in seen_ids or seen_ids.add(r["id"]))]
    chunks = {k: v or {} for k, v in (recalled.get("chunks") or {}).items()}
    seen_chunks: set = set()
    return "\n\n".join(f"## Memory {i + 1}\n{format_result(r, chunks, seen_chunks)}" for i, r in enumerate(results))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def operations(http: httpx.Client, bank_id: str, status: str | None = None, limit: int = 1, offset: int = 0) -> dict:
    params = {"limit": limit, "offset": offset, **({"status": status} if status else {})}
    response = http.get(f"/v1/default/banks/{bank_id}/operations", params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def await_bank(http: httpx.Client, bank_id: str) -> dict:
    """AMB _await_bank_ingest: until nothing is pending or processing, refusing to go on past a failed operation."""
    start = time.monotonic()
    while time.monotonic() - start < INGEST_TIMEOUT_S:
        total = operations(http, bank_id)["total"]
        completed = operations(http, bank_id, "completed")["total"]
        failed = operations(http, bank_id, "failed")
        if failed["total"]:
            op = failed["operations"][0]
            raise RuntimeError(
                f"{failed['total']} operation(s) failed after Hindsight's own retries, e.g. {op['task_type']}: "
                f"{op.get('error_message')}"
            )
        if total - completed == 0:
            kinds: dict[str, int] = {}
            offset = 0
            while offset < total:
                page = operations(http, bank_id, "completed", limit=100, offset=offset)
                for op in page["operations"]:
                    kind = op.get("task_type") or op.get("operation_type") or "?"
                    kinds[kind] = kinds.get(kind, 0) + 1
                offset += len(page["operations"]) or total
            return {"seconds": round(time.monotonic() - start), "operations": kinds}
        time.sleep(5)
    raise RuntimeError(f"bank {bank_id}: operations still running after {INGEST_TIMEOUT_S}s")


def main() -> None:
    # A bench that stops this process still gets the server stopped and its database dropped (the finally below).
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    request = json.load(sys.stdin)
    question_id = request.get("question_id") or "q"
    bank_id = f"longmemeval-{question_id}"
    observations = bool(request.get("observations"))
    port = free_port()
    instance = f"lme-{uuid.uuid4().hex[:12]}"
    workdir = Path(tempfile.mkdtemp(prefix="hindsight-"))
    log_path = workdir / "server.log"
    env = {
        **os.environ,
        "HINDSIGHT_API_DATABASE_URL": f"pg0://{instance}",
        "HINDSIGHT_API_LLM_PROVIDER": "openrouter",
        "HINDSIGHT_API_LLM_BASE_URL": request["base_url"],
        "HINDSIGHT_API_LLM_API_KEY": request["api_key"],
        "HINDSIGHT_API_LLM_MODEL": request["model"],
    }
    started = time.monotonic()
    with log_path.open("wb") as log:
        server = subprocess.Popen(
            [HINDSIGHT_API, "--host", "127.0.0.1", "--port", str(port)],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=workdir,
            start_new_session=True,
        )
    try:
        base = f"http://127.0.0.1:{port}"
        with httpx.Client(base_url=base) as http:
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"hindsight-api exited with {server.returncode} while starting")
                if time.monotonic() - started > STARTUP_TIMEOUT_S:
                    raise RuntimeError(f"hindsight-api not ready after {STARTUP_TIMEOUT_S}s")
                try:
                    if http.get("/health", timeout=5).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(1)
            startup_s = round(time.monotonic() - started)

            client = Hindsight(base_url=base, timeout=600)
            client.create_bank(bank_id=bank_id, name=f"Benchmark Bank ({bank_id})", enable_observations=observations)
            items = items_for(request["sessions"], question_id)
            for i in range(0, len(items), BATCH_SIZE):
                client.retain_batch(bank_id=bank_id, items=items[i : i + BATCH_SIZE], retain_async=True)
            waited = await_bank(http, bank_id)

            # AMB's 32768 and 16384 unless the bench asks for less (ADAPTED: a Dot keeps 25,000 characters of MEMORY.md).
            recall_tokens = int(request.get("recall_max_tokens") or 32768)
            chunk_tokens = int(request.get("chunk_max_tokens") or 16384)
            recall = {
                "query": request["question"][:1900],
                "budget": "high",
                "max_tokens": recall_tokens,
                "include": {"entities": None, "chunks": {"max_tokens": chunk_tokens}},
                "query_timestamp": parse_date(request["question_date"]).isoformat(),
            }
            response = http.post(f"/v1/default/banks/{bank_id}/memories/recall", json=recall, timeout=600)
            response.raise_for_status()
            recalled = response.json()
            stats = http.get(f"/v1/default/banks/{bank_id}/stats", timeout=60)
            client.close()
        memory_md = render(recalled)
        kinds: dict[str, int] = {}
        for r in recalled.get("results") or []:
            kinds[r.get("type") or "?"] = kinds.get(r.get("type") or "?", 0) + 1
        notes = (
            f"Hindsight 0.10.3, bank {bank_id}, observations {'on' if observations else 'off (AMB)'}: "
            f"{len(items)} sessions ({sum(len(i['content']) for i in items)} characters) retained; "
            f"server ready in {startup_s}s; operations done in {waited['seconds']}s {waited['operations']}; "
            f"bank stats {stats.json() if stats.status_code == 200 else stats.status_code}; "
            f"recall (budget high, max_tokens {recall_tokens}, chunks {chunk_tokens}, query_timestamp) returned {kinds} and "
            f"{len(recalled.get('chunks') or {})} chunks, {len(memory_md)} characters"
        )
        json.dump({"memory_md": memory_md, "notes": notes}, sys.stdout, ensure_ascii=False)
    except Exception as error:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:] if log_path.exists() else ""
        print(f"memory_hindsight: {type(error).__name__}: {error}\n--- hindsight-api log (end) ---\n{tail}", file=sys.stderr)
        sys.exit(1)
    finally:
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait()
        try:
            Pg0(name=instance).drop()
        except Exception as error:
            print(f"memory_hindsight: dropping pg0 instance {instance}: {error}", file=sys.stderr)
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
