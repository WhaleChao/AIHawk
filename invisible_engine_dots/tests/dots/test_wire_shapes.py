"""What the engine answers to `GET /tools` and `GET /skills`, and the data of every outbound event it writes, pinned in
a file the host checks.

The host describes these answers with schemas in `packages/shared/src/protocol.ts` (`toolInfoSchema`). The engine is Python and cannot import them, so this test writes what the engine really answers into
`wire_shapes.json`, and `apps/scheduler/test/engine-shapes.test.ts` parses that file with the schemas and runs the
host's fake guest on the same permissions. A key renamed, added or dropped here fails this test until the file is
written again (`UPDATE_WIRE_SHAPES=1 pytest tests/dots/test_wire_shapes.py`), and then fails the host's test until its
schema says the same.

The events are not written by hand here: a real engine runs a chat, tasks (one that reports progress, one that
fails, one cut in the middle of a call), calls that ask for approval, a terminal, a file, and the browser identities,
and what its own writers put in the outbox is what is pinned (one event of each type and each set of keys, with the
values that vary from run to run, such as ids and durations, replaced by stable ones). The host's side of the same
contract, the events and the config it sends, is written by `apps/scheduler/test/host-shapes.test.ts` into
`host_wire_shapes.json`, which `test_host_wire_shapes.py` parses with this engine's own parsers.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fakes.dot_config import ALLOW_ALL, runtime_config_body
from fakes.engine_harness import EngineHarness, task_cancelled, task_created, user_message
from fakes.scripted_provider import ScriptEntry, call, calls, says

from nanobot.cron.types import MAX_RUN_AT_MS
from nanobot.dots import store as dots_store
from nanobot.dots.permissions import TOOL_PERMISSIONS, offered_tools, tool_table
from nanobot.dots.skills import builtin_skills
from nanobot.dots.protocol import OUTBOUND_EVENT_TYPES
from nanobot.providers.base import LLMResponse

FIXTURE = Path(__file__).with_name("wire_shapes.json")


class _Registry:
    """Stands in for the tool registry: a row's description is the only thing `tool_table` asks it for."""

    def get(self, name: str) -> Any:
        return SimpleNamespace(description=f"description of {name}")


_OFFERED_CASES = [
    {"permissions": {"computer.exec": "allow", "files.read": "ask", "files.write": "deny", "automations": "deny"}},
    {"permissions": {}},
    {"permissions": {"computer.exec": "ask", "files.read": "ask", "files.write": "ask", "automations": "ask"}},
    {"permissions": {"browser.identity.list": "allow", "browser.identity.create": "allow", "browser.identity.delete": "ask"}},
]


MakeEngine = Callable[..., EngineHarness]

_ISO_MS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


async def _engine_events(make_engine: MakeEngine) -> list[dict[str, Any]]:
    """Run a real engine through everything it writes events about, and return its outbox, oldest first."""
    script: list[ScriptEntry] = [
        # A chat turn that opens a terminal and writes a file, then answers.
        calls(
            call("c1", "exec", command="echo hi", tty=True),
            call("c2", "write_file", path="/home/dot/memory/trips/lisbon.md", content="TAP 41 EUR"),
            cost=0.125,
        ),
        says("Noted.", cost=0.25),
        # A task that says what it is about to do, calls a tool and finishes.
        calls(call("c3", "list_dir", path="/home/dot/workspace"), text="Comparing the fares", cost=0.5),
        says("The fares are compared.", cost=0.25),
        # A task whose model fails.
        LLMResponse(content="provider down", finish_reason="error"),
        # A task cut in the middle of a call.
        calls(call("c4", "exec", command="sleep 30")),
        # A chat turn and a task whose call asks for approval.
        calls(call("c5", "exec", command="ls -la")),
        calls(call("c6", "exec", command="ls -la")),
        # The memory pass over the conversations above.
        says("## The person\n- looks for the cheapest fares (2026-10-04)", cost=0.0625),
    ]
    h = make_engine(script)
    h.engine.start()
    h.configure(runtime_config_body(permissions=ALLOW_ALL))

    h.engine.accept(user_message("m1", "find the cheapest fare"))
    await h.idle()
    h.engine.accept(task_created("a", "compare the fares", 5))
    await h.idle()
    h.engine.accept(task_created("b", "this one fails", 0))
    await h.idle()
    h.engine.accept(task_created("c", "this one is cut", 0))
    await h.wait_until(lambda: h.count("dots_tool_intents") == 1)
    h.engine.accept(task_cancelled("c"))
    await h.idle()

    h.configure(runtime_config_body(permissions={**ALLOW_ALL, "computer.exec": "ask"}))
    h.engine.accept(user_message("m2", "list the build"))
    await h.idle()
    h.engine.accept(task_created("d", "list the build too", 0))
    await h.idle()

    identity = await h.browser.create("Shopping Account")
    await h.browser.launch(identity.id)
    await h.browser.close(identity.id)
    await h.browser.delete(identity.id)

    h.engine.automations_next_run(1_790_000_000_000)
    h.engine.automations_next_run(None)
    # The pass the engine starts once the Dot has been quiet, run now.
    assert (await h.engine._memory.run()).kind == "updated"
    return h.events()


def _stable(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One event of each type and each set of keys and kinds of value, as the engine wrote it, with what varies from run to run made stable.

    The ids and the times are checked for their shape and dropped (the host's schema takes any, and the fixture is
    about the data); an approval or an identity gets a name in the order it first appears; the time a call took is a
    stand-in unless it is the zero of a call that was cut.
    """
    names: dict[tuple[str, str], str] = {}

    def named(prefix: str, value: str) -> str:
        return names.setdefault((prefix, value), f"{prefix}-{sum(1 for kind, _ in names if kind == prefix) + 1}")

    seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    pinned: list[dict[str, Any]] = []
    for event in events:
        assert _UUID.match(event["id"]) and _ISO_MS.match(event["ts"]), event
        data = dict(event["data"])
        for key, prefix in (("approval_id", "approval"), ("identity_id", "identity")):
            if key in data:
                data[key] = named(prefix, data[key])
        if event["type"] == "tool.called" and not data.get("interrupted"):
            data["duration_ms"] = 10
        # A key and the kind of value it holds: a time or none (`automation.next_run`) are two shapes.
        shape = (event["type"], tuple(sorted((key, type(value).__name__) for key, value in data.items())))
        if shape not in seen:
            seen.add(shape)
            pinned.append({"type": event["type"], "data": data})
    return pinned


def _shapes(outbound_events: list[dict[str, Any]]) -> dict[str, Any]:
    cases = []
    for case in _OFFERED_CASES:
        offered = offered_tools(case["permissions"])
        cases.append({**case, "offered": offered, "tools": tool_table(_Registry(), offered)})
    return {
        "limits": {"max_run_at_ms": MAX_RUN_AT_MS},
        "outbound_events": outbound_events,
        "tool_offering": cases,
        # What `GET /skills` answers for a Dot that wrote none of its own (Engine.skills), the path made stable.
        "skills": [
            {"name": s.name, "description": s.description, "source": s.source, "path": f"/engine/skills/{s.name}/SKILL.md", "content": s.content}
            for s in builtin_skills()
        ],
    }


@pytest.fixture
async def outbound_events(make_engine: MakeEngine) -> list[dict[str, Any]]:
    return _stable(await _engine_events(make_engine))


def test_the_answers_of_the_engine_are_the_ones_the_host_checks(outbound_events: list[dict[str, Any]]) -> None:
    shapes = _shapes(outbound_events)
    text = json.dumps(shapes, indent=2, sort_keys=True) + "\n"
    if os.environ.get("UPDATE_WIRE_SHAPES") == "1":
        FIXTURE.write_bytes(text.encode("utf-8"))
    assert FIXTURE.exists(), "wire_shapes.json is missing: write it with UPDATE_WIRE_SHAPES=1"
    assert json.loads(FIXTURE.read_text(encoding="utf-8")) == shapes


def test_the_cases_reach_every_kind_of_tool(outbound_events: list[dict[str, Any]]) -> None:
    shapes = _shapes(outbound_events)
    for case in shapes["tool_offering"]:
        assert [row["name"] for row in case["tools"]] == list(TOOL_PERMISSIONS)


def test_the_events_pinned_are_one_of_every_type_the_engine_writes_and_every_set_of_keys_it_writes_them_with(
    outbound_events: list[dict[str, Any]],
) -> None:
    assert {event["type"] for event in outbound_events} == set(OUTBOUND_EVENT_TYPES)
    keys = {(event["type"], tuple(sorted(event["data"]))) for event in outbound_events}
    # The optional keys of the events the host's schemas describe: each is written, and each is also left out.
    assert ("tool.called", ("decision", "duration_ms", "ok", "permission", "target", "tool", "tty")) in keys
    assert ("tool.called", ("decision", "duration_ms", "interrupted", "ok", "permission", "target", "task_id", "tool")) in keys
    assert ("tool.called", ("decision", "duration_ms", "ok", "permission", "target", "task_id", "tool")) in keys
    assert ("approval.requested", ("approval_id", "arguments", "permission", "reason", "task_id", "tool")) in keys
    assert ("approval.requested", ("approval_id", "arguments", "permission", "reason", "tool")) in keys
    assert ("message.assistant", ("in_reply_to", "spent_usd", "text")) in keys
    assert ("task.progress", ("spent_usd", "task_id", "text")) in keys
    assert ("task.failed", ("error", "spent_usd", "task_id")) in keys
    assert ("task.completed", ("spent_usd", "summary", "task_id")) in keys
    assert ("automation.next_run", ("next_run_at_ms",)) in keys
    assert {e["data"]["next_run_at_ms"] for e in outbound_events if e["type"] == "automation.next_run"} == {1_790_000_000_000, None}
