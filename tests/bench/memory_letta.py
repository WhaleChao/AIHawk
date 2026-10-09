"""TEST HARNESS ONLY: MEMORY.md written by Letta's sleep-time agent, for the LongMemEval bench. Nothing here ships.

Letta, Apache-2.0, https://github.com/letta-ai/letta, release 0.16.8 (tag 0.16.8, commit 1131535716e8, the last
release of the server; the PyPI name `letta` now ships Letta Code). Letta keeps memory blocks ("human", "persona") in
the agent's prompt; with enable_sleeptime=True a background sleep-time agent (agent_type sleeptime_agent, its own
prompt prompts/system_prompts/sleeptime_v2.py and tools memory_insert / memory_replace / memory_rethink /
memory_finish_edits) rewrites them from the conversation, every `sleeptime_agent_frequency` turns (5, as
server.create_sleeptime_agent_async sets it); the main agent then has no memory tools of its own.

Per question this creates one agent with enable_sleeptime=True and Letta's default blocks (schemas/block.py
DEFAULT_BLOCKS: an empty "human" and "persona"), records the whole history into it oldest first, waits for every
sleep-time run, renders the blocks the main agent sees as MEMORY.md, and deletes the agents and their blocks.
Letta's prompts, tools, tool rules, frequency and compaction are left as they are; only the model is configured.

    /opt/mem-letta/bin/python memory_letta.py < input.json > output.json      (server: tests/bench/letta-server.sh)

ADAPTED:
- The history is recorded, not replayed: each user message(s) + assistant reply of a session is one turn sent to
  POST /v1/agents/{id}/messages/capture, Letta's own endpoint for persisting a turn that was answered elsewhere; it
  stores the dataset's assistant reply as the agent's and runs the sleep-time group exactly as after a real turn.
  The main agent therefore makes no model call. Consecutive assistant messages are one reply.
- Dates: capture stamps messages with the current time and the sleep-time transcript (groups/helpers.py
  stringify_message) carries no time, so each user message is prefixed with its session's date, "[2023/05/20 (Sat)
  02:21] ...".
- Each sleep-time run is waited for before the next turn is recorded (a person's next message comes after it ends),
  and the last turn is recorded with the frequency set to 1, so the turns after the last multiple of 5 are processed
  too (each run gets every message since the previous run).
- Model: the LLMConfig Letta's OpenRouterProvider.list_llm_models_async builds (model_endpoint_type "openrouter",
  context_window from the endpoint's /models, 128000 if absent; max_tokens 16384), with model_endpoint the
  per-question base_url, so the sleep-time agent (which copies the main agent's config) also goes through it. No
  handle is set: compaction then summarizes with this same config instead of resolving a server-wide provider.
  The key is not per agent in Letta (env or a BYOK provider); the metering proxy adds it, so `api_key` is unused.
- Embeddings (required for a sleep-time agent to be created; none of its tools embed): `embedding_model` at the same
  base_url, 1536 dimensions.
- MEMORY.md: every block of the main agent with text, as "## <label>", Letta's description of the block, then its
  value (Letta shows the same three in the prompt, as XML).
"""

from __future__ import annotations

import json
import sys
import time
import uuid

import httpx
from letta_client import Letta

RUN_TIMEOUT_S = 3600
FINAL_FREQUENCY = 1


def turns(sessions: list[dict]) -> list[tuple[list[str], str]]:
    """(user messages, assistant reply) per turn, oldest first; a turn never spans two sessions."""
    out: list[tuple[list[str], str]] = []
    for session in sessions:
        users: list[str] = []
        replies: list[str] = []
        for turn in session["turns"]:
            if turn["role"] == "user":
                if replies:
                    out.append((users, "\n\n".join(replies)))
                    users, replies = [], []
                users.append(f"[{session['date']}] {turn['content']}")
            else:
                replies.append(turn["content"])
        if users or replies:
            out.append((users, "\n\n".join(replies)))
    return out


def context_window(http: httpx.Client, base_url: str, model: str) -> int:
    response = http.get(base_url.rstrip("/") + "/models", params={"supported_parameters": "tools"})
    response.raise_for_status()
    for entry in response.json().get("data", []):
        if entry.get("id") == model and entry.get("context_length"):
            return int(entry["context_length"])
    return 128000


def wait_for_run(client: Letta, run_id: str) -> dict:
    deadline = time.monotonic() + RUN_TIMEOUT_S
    while True:
        run = client.runs.retrieve(run_id)
        if run.status in ("completed", "failed", "cancelled"):
            return run.model_dump()
        if time.monotonic() > deadline:
            raise RuntimeError(f"sleep-time run {run_id} still {run.status} after {RUN_TIMEOUT_S}s")
        time.sleep(1)


def render(blocks: list) -> str:
    parts = []
    for block in sorted(blocks, key=lambda b: b.label):
        if (block.value or "").strip():
            parts += [f"## {block.label}", (block.description or "").strip(), block.value.strip()]
    return "\n\n".join(part for part in parts if part)


def main() -> None:
    job = json.load(sys.stdin)
    base_url, model = job["base_url"], job["model"]
    client = Letta(base_url=job["letta_url"], timeout=600)
    http = httpx.Client(timeout=600)
    llm_config = {
        "model": model,
        "model_endpoint_type": "openrouter",
        "model_endpoint": base_url,
        "provider_name": "openrouter",
        "provider_category": "base",
        "context_window": context_window(http, base_url, model),
        "max_tokens": 16384,
    }
    embedding_config = {
        "embedding_endpoint_type": "openai",
        "embedding_endpoint": base_url,
        "embedding_model": job["embedding_model"],
        "embedding_dim": 1536,
    }
    agent = client.agents.create(
        name=f"longmemeval-{uuid.uuid4().hex[:12]}",
        agent_type="letta_v1_agent",
        enable_sleeptime=True,
        memory_blocks=[{"label": "human", "value": ""}, {"label": "persona", "value": ""}],
        llm_config=llm_config,
        embedding_config=embedding_config,
    )
    block_ids: set[str] = set()
    try:
        group = agent.multi_agent_group
        if group is None or not group.agent_ids:
            raise RuntimeError("Letta created no sleep-time agent")
        sleeptime_id = group.agent_ids[0]
        block_ids |= {b.id for b in client.agents.blocks.list(agent.id)}
        block_ids |= {b.id for b in client.agents.blocks.list(sleeptime_id)}

        history = turns(job["sessions"])
        runs: list[dict] = []
        for i, (users, reply) in enumerate(history):
            if i == len(history) - 1:
                http.patch(
                    f"{job['letta_url']}/v1/groups/{group.id}",
                    json={"manager_config": {"manager_type": "sleeptime", "sleeptime_agent_frequency": FINAL_FREQUENCY}},
                ).raise_for_status()
            response = http.post(
                f"{job['letta_url']}/v1/agents/{agent.id}/messages/capture",
                json={
                    "provider": "openrouter",
                    "model": model,
                    "request_messages": [{"role": "user", "content": text} for text in users],
                    "response_dict": {"content": reply},
                },
            )
            response.raise_for_status()
            for run_id in response.json().get("run_ids") or []:
                runs.append(wait_for_run(client, run_id))

        group_state = http.get(f"{job['letta_url']}/v1/groups/{group.id}").json()
        newest = http.get(f"{job['letta_url']}/v1/agents/{agent.id}/messages", params={"limit": 1, "order": "desc"}).json()
        if not newest or group_state.get("last_processed_message_id") != newest[0]["id"]:
            raise RuntimeError("the sleep-time agent was not handed the last message")
        failed = [r for r in runs if r["status"] != "completed"]
        if runs and len(failed) == len(runs):
            raise RuntimeError(f"every sleep-time run failed: {failed[0].get('metadata')}")

        blocks = list(client.agents.blocks.list(agent.id))
        steps = sum(((r.get("metadata") or {}).get("result") or {}).get("usage", {}).get("step_count", 0) for r in runs)
        notes = {
            "turns_recorded": len(history),
            "user_messages": sum(len(u) for u, _ in history),
            "assistant_replies": sum(1 for _, r in history if r),
            "sleeptime_runs": len(runs),
            "sleeptime_runs_failed": len(failed),
            "sleeptime_steps": steps,
            "block_chars": {b.label: len(b.value or "") for b in blocks},
            "context_window": llm_config["context_window"],
        }
        json.dump({"memory_md": render(blocks), "notes": json.dumps(notes)}, sys.stdout, ensure_ascii=False)
    finally:
        client.agents.delete(agent.id)  # also deletes its sleep-time agent and their group
        for block_id in block_ids:
            try:
                client.blocks.delete(block_id)
            except Exception:  # already gone with the agent
                pass


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"memory_letta: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
