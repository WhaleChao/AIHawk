"""An engine on a real store, the real tools on a local computer and a scripted model."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fakes.browser_manager import make_browser_manager
from fakes.local_computer import LocalComputer
from fakes.scripted_provider import ScriptedProvider, ScriptEntry
from fakes.turn_harness import KEY, FixedProviders

from nanobot.agent.tools.exec_session import ExecSessionManager
from nanobot.cron.service import CronService
from nanobot.dots import store as s
from nanobot.dots.engine import Engine
from nanobot.dots.permissions import ToolDeps, build_registry
from nanobot.dots.protocol import InboundEvent, parse_inbound_event
from nanobot.dots.secrets import KeyHolder
from nanobot.dots.store import DotStore

TS = "2026-10-04T10:00:00.000Z"


def inbound(event_id: str, event_type: str, data: dict[str, Any]) -> Any:
    """A parsed inbound event, as the server hands it to the engine."""
    event: InboundEvent = parse_inbound_event({"id": event_id, "type": event_type, "ts": TS, "data": data})
    return event


def user_message(event_id: str, text: str = "hello") -> Any:
    return inbound(event_id, "user.message", {"text": text})


def task_created(task_id: str, description: str = "do it", priority: int = 0) -> Any:
    return inbound(f"e-{task_id}", "task.created", {"task_id": task_id, "description": description, "priority": priority})


def task_cancelled(task_id: str) -> Any:
    return inbound(f"c-{task_id}", "system.event", {"name": "task.cancelled", "data": {"task_id": task_id}})


def decision(event_id: str, approval_id: str, verdict: str, note: str | None = None) -> Any:
    data: dict[str, Any] = {"approval_id": approval_id, "decision": verdict}
    if note is not None:
        data["note"] = note
    return inbound(event_id, "approval.received", data)


class EngineHarness:
    def __init__(
        self,
        tmp_path: Path,
        store: DotStore,
        script: list[ScriptEntry],
        *,
        key: bool = True,
        stop_grace_s: float = 0.05,
        browser: dict[str, Any] | None = None,
        memory_quiet_s: float | None = None,
    ) -> None:
        self.tmp_path = tmp_path
        self.store = store
        self.provider = ScriptedProvider(script)
        self.providers = FixedProviders(self.provider)
        self.keys = KeyHolder()
        if key:
            self.keys.set(KEY)
        workspace = tmp_path / "home" / "dot" / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        self.computer = LocalComputer(tmp_path, workspace)
        self.stop_grace_s = stop_grace_s
        # None keeps the engine's own quiet spell before a memory pass, longer than any test waits.
        self.memory_quiet_s = memory_quiet_s
        self.browser_options = browser or {}
        self.engines: list[Engine] = []
        self.cron = CronService(tmp_path / "cron" / "jobs.json")
        self.engine = self.new_engine()

    def new_engine(self) -> Engine:
        """An engine on the same store, as a restarted process has: nothing in memory."""
        # A restarted process has no browser open: the manager is new, the rows are the store's.
        self.browser = make_browser_manager(self.tmp_path, self.store, self.computer, **self.browser_options)
        registry = build_registry(
            ToolDeps(
                computer=self.computer,
                exec_session_manager=ExecSessionManager(),
                cron_service=self.cron,
                browser=self.browser,
            )
        )
        engine = Engine(
            store=self.store,
            computer=self.computer,
            base_registry=registry,
            browser=self.browser,
            providers=self.providers,
            key_holder=self.keys,
            workspace=self.computer.workspace,
            stop_grace_s=self.stop_grace_s,
            **({} if self.memory_quiet_s is None else {"memory_quiet_s": self.memory_quiet_s}),
        )
        self.engines.append(engine)
        return engine

    def restart(self) -> Engine:
        self.engine = self.new_engine()
        return self.engine

    def configure(self, body: dict[str, Any]) -> None:
        self.engine.set_config(body)

    def give_key(self) -> None:
        self.keys.set(KEY)
        self.engine.key_received()

    async def idle(self) -> None:
        """Return when no turn is in flight, counting the turns the end of one starts."""
        turns = self.engine._turns
        while turns:
            # A finished turn is retired, and the work loop kicked, by a callback that runs before this returns.
            await asyncio.wait([turn.task for turn in turns.values()])

    async def wait_until(self, predicate: Callable[[], bool], timeout_s: float = 10.0) -> None:
        """Return when `predicate` holds, looking again after every commit that adds an outbox row."""
        wake = asyncio.Event()
        remove = self.store.on_append(wake.set)
        try:
            async with asyncio.timeout(timeout_s):
                while True:
                    wake.clear()
                    if predicate():
                        return
                    await wake.wait()
        finally:
            remove()

    # --- what the store holds -------------------------------------------------

    def events(self) -> list[dict[str, Any]]:
        return self.store.read(lambda conn: s.read_outbox_after(conn, 0, 10_000))

    def types(self) -> list[str]:
        return [event["type"] for event in self.events()]

    def events_of(self, event_type: str) -> list[dict[str, Any]]:
        return [event["data"] for event in self.events() if event["type"] == event_type]

    def states(self) -> list[str]:
        return [data["state"] for data in self.events_of("agent.state")]

    def task_events(self) -> list[tuple[str, str]]:
        return [
            (event["type"], event["data"]["task_id"])
            for event in self.events()
            if event["type"].startswith("task.")
        ]

    def messages(self, session_key: str = s.CHAT_SESSION_KEY) -> list[dict[str, Any]]:
        return self.store.read(lambda conn: s.read_messages(conn, session_key))

    def task(self, task_id: str) -> s.TaskRow | None:
        return self.store.read(lambda conn: s.get_task(conn, task_id))

    def approval(self, approval_id: str) -> s.Approval:
        found = self.store.read(lambda conn: s.get_approval(conn, approval_id))
        assert found is not None
        return found

    def pending_approvals(self) -> list[s.Approval]:
        return self.store.read(lambda conn: s.list_approvals(conn, "pending"))

    def inbound_state(self, inbound_id: str) -> str:
        row = self.store.read(
            lambda conn: conn.execute("SELECT state FROM dots_inbound WHERE id = ?", (inbound_id,)).fetchone()
        )
        return str(row[0])

    def count(self, table: str) -> int:
        return int(self.store.read(lambda conn: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]))

    def asked(self) -> int:
        return len(self.provider.requests)
