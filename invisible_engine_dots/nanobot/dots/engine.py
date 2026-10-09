"""The Dot's engine: inbound events in, turns of the model out, outbound events into the outbox.

Everything the host sees is a row of `dots_outbox`, written by `DotStore` in the
transaction of the state change it describes. This module decides what work
runs, and what an outcome of a turn means; the turn itself is `TurnRunner`'s.

- `user.message` and `automation.fired` rows wait in `dots_inbound` until the
  chat turn takes them as its opening messages (or, while one runs, injects them
  into it). The transcript marks them in the transaction that stores their
  text, and the transaction that appends the answer applies them
  (`transcript_outbox.py`).
- `task.created` queues a task; tasks run one at a time, highest priority
  first, each in its own session. The answer that ends a task completes it in
  the transcript's transaction; a failed or empty run fails it here.
- `system.event task.cancelled` ends a queued or running task and cancels its turn.
- `approval.received` records the decision on the approval it names, in the
  transaction that accepts it. The work loop then tells the session that made
  the call, in a turn of its own: an approved call is to be made again with
  exactly its arguments (the gate lets that one through, once, `gate.py`), a
  rejected one did not run. Nothing waits in memory.
- An answer a chat owes (inputs in the transcript, none answered, no approval
  holding the session) is given by a turn with no opening message: one rule
  for a restart, a prepare-sleep and any turn that ended early.

- Once the Dot has been quiet for a while after a turn (and after a start), a memory pass brings MEMORY.md up
  to date from the conversations that changed (`memory_update.py`). It runs beside nothing: a turn that starts
  does not wait for it, and a pass waits for the next quiet spell when a turn runs.

One asyncio loop runs everything; the one SQLite connection is used on it only.
At most one chat turn and one task turn run at a time, and they may overlap.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from loguru import logger

from nanobot.agent.memory import Consolidator
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.cron.types import CronJob
from nanobot.dots import store as dots_store
from nanobot.dots.browser import CLOSE_TIMEOUT_S, BrowserManager
from nanobot.dots.computer import Computer
from nanobot.dots.gate import DotsGate, close_open_calls
from nanobot.dots.memory_update import QUIET_S, MemoryUpdater
from nanobot.dots.permissions import tool_table, tool_target
from nanobot.dots.projection import EngineSettings, project
from nanobot.dots.protocol import (
    TASK_CANCELLED_EVENT,
    ApprovalReceivedEvent,
    DotRuntimeConfig,
    DotsConfigError,
    SystemEvent,
    TaskCreatedEvent,
    UserMessageEvent,
    parse_runtime_config,
)
from nanobot.dots.provider import OpenRouterProviders
from nanobot.dots.secrets import KeyHolder
from nanobot.dots.skills import all_skills
from nanobot.dots.store import (
    AUTOMATION_FIRED,
    CHAT_SESSION_KEY,
    Approval,
    DotStore,
    InboundRow,
    ToolIntent,
)
from nanobot.dots.transcript_outbox import APPROVAL_ID, APPROVAL_LINE, INBOUND_ID
from nanobot.dots.turns import OpeningMessage, TurnOutcome, TurnRunner, TurnUnit

# How many times a task may be started before an interruption fails it (architecture 8.7).
TASK_MAX_ATTEMPTS = 3
# How long a tool in flight may finish when the Dot prepares to sleep (architecture 8.7), in seconds.
STOP_GRACE_S = 20.0
# After the grace, how long the cancelled turns get to end before the sleep goes on without them.
_CANCEL_WAIT_S = 5.0
# How long the open browsers get to close (Firefox flushes its profile) once the turns have ended. On a
# stop (SIGTERM) it is what is left of systemd's TimeoutStopSec=30 after the grace (architecture 8.7). On
# a prepare-sleep nothing but the host's 60 s waits, so a browser gets the whole of its own close timeout:
# 20 s of grace, 5 s of cancel wait and this stay inside them.
_CLOSE_BROWSERS_ON_STOP_S = 4.0
_CLOSE_BROWSERS_ON_SLEEP_S = CLOSE_TIMEOUT_S

RESUME_TASK_NOTE = (
    "[The previous attempt at this task was interrupted by a restart. Check what it already did "
    "before you continue, and do not repeat an action that may have taken effect.]"
)

InboundEvent = UserMessageEvent | TaskCreatedEvent | ApprovalReceivedEvent | SystemEvent
Slot = Literal["chat", "task"]

# The inbound rows a chat turn takes in.
_CHAT_INPUT_TYPES = ("user.message", AUTOMATION_FIRED)


class EngineStopped(RuntimeError):
    """The engine is shutting down and takes no more events."""


@dataclass(frozen=True)
class StateAnswer:
    """What `GET /state` says."""

    state: str
    current_task_id: str | None
    # The oldest approval that waits for the host's decision.
    pending_approval: str | None


@dataclass(frozen=True)
class _Turn:
    """A turn in flight."""

    slot: Slot
    unit: TurnUnit
    task: asyncio.Task[bool]
    # Whether starting it counted an attempt of its task (a resume does, a continuation does not).
    counted_attempt: bool


def _with_note(approval: Approval) -> str:
    return f' The user\'s note: "{approval.note}".' if approval.note else ""


def approval_line(approval: Approval, approved: bool) -> str:
    """What the Dot's conversation files say of a decision: the call and what it acted on (a target never shows a
    secret), never the arguments the continuation hands the model."""
    target = tool_target(approval.tool, approval.arguments)
    call = f"the {approval.tool} call" + (f" ({target})" if target else "")
    return f"The person {'approved' if approved else 'rejected'} {call}.{_with_note(approval)}"


def approval_granted_continuation(approval: Approval) -> str:
    # The arguments as JSON with no spaces: the model reads them back and makes the call with exactly these.
    arguments = json.dumps(approval.arguments, separators=(",", ":"), ensure_ascii=False)
    return (
        f"[The user approved your {approval.tool} call ({approval.approval_id}).{_with_note(approval)} "
        f"It has not run yet. Call {approval.tool} again now with exactly these arguments, "
        f"and it will run once: {arguments}]"
    )


def approval_rejected_continuation(approval: Approval) -> str:
    return (
        f"[The user rejected your {approval.tool} call ({approval.approval_id}).{_with_note(approval)} "
        "It did not run. Do not try it again unless the user asks you to.]"
    )


def opening_for_row(row: InboundRow) -> OpeningMessage:
    """The user message an accepted input becomes in the chat transcript."""
    if row.type == AUTOMATION_FIRED:
        text = f'[Automation "{row.data.get("name", "")}" fired] {row.data.get("message", "")}'
    else:
        text = row.data.get("text", "")
    return OpeningMessage(text if isinstance(text, str) else "", {INBOUND_ID: row.id})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _pending_chat_inputs(conn: sqlite3.Connection) -> list[InboundRow]:
    return [row for row in dots_store.list_inbound(conn, "accepted") if row.type in _CHAT_INPUT_TYPES]


class Engine:
    """Runs the Dot: accepts the host's events, decides which turn runs, ends what a turn leaves."""

    def __init__(
        self,
        *,
        store: DotStore,
        computer: Computer,
        base_registry: ToolRegistry,
        browser: BrowserManager,
        providers: OpenRouterProviders,
        key_holder: KeyHolder,
        workspace: str,
        openrouter_base_url: str | None = None,
        stop_grace_s: float = STOP_GRACE_S,
        memory_quiet_s: float = QUIET_S,
    ) -> None:
        self._store = store
        self._computer = computer
        self._browser = browser
        self._registry = base_registry
        self._key_holder = key_holder
        self._workspace = workspace
        self._openrouter_base_url = openrouter_base_url
        self._stop_grace_s = stop_grace_s
        self._config: DotRuntimeConfig | None = None
        self._settings: EngineSettings | None = None
        self._started = False
        self._stopped = False
        self._suspended = False
        # The suspend in flight: a second one waits for it instead of running beside it.
        self._suspension: asyncio.Task[None] | None = None
        # Calls running now, per session, for every turn whose run has started.
        self._tools_running: dict[str, int] = {}
        self._turns: dict[Slot, _Turn] = {}
        self._pumping = False
        self._repump = False
        # The memory pass: whether one is owed (a turn ended, or the process started, since the last), since when
        # the Dot has been quiet, the timer that looks again once it has been long enough, and the pass running.
        self._memory_quiet_s = memory_quiet_s
        self._memory_due = True
        self._quiet_since = 0.0
        self._memory_timer: asyncio.TimerHandle | None = None
        self._memory_pass: asyncio.Task[None] | None = None
        self._memory = MemoryUpdater(
            store=store,
            computer=computer,
            providers=providers,
            key_holder=key_holder,
            settings_getter=lambda: self._settings,
        )
        self._runner = TurnRunner(
            store=store,
            computer=computer,
            base_registry=base_registry,
            providers=providers,
            key_holder=key_holder,
            settings_getter=lambda: self._settings,
            gate=DotsGate(lambda: self._config, store),
            consolidator=Consolidator(),
            host=self,
            injection_source=self._chat_inputs,
        )

    # --- what the rest of the process reads ---------------------------------

    @property
    def started(self) -> bool:
        return self._started

    @property
    def state(self) -> str:
        """The agent state the outbox last announced."""
        return str(self._store.read(lambda conn: dots_store.read_kv(conn, dots_store.KV_AGENT_STATE, "IDLE")))

    @property
    def config(self) -> DotRuntimeConfig | None:
        return self._config

    @property
    def browser(self) -> BrowserManager:
        """The Dot's browser identities: the engine closes their browsers when it stops work."""
        return self._browser

    def state_answer(self) -> StateAnswer:
        def answer(conn: sqlite3.Connection) -> StateAnswer:
            running = dots_store.get_running_task(conn)
            pending = dots_store.list_approvals(conn, "pending")
            return StateAnswer(
                state=str(dots_store.read_kv(conn, dots_store.KV_AGENT_STATE, "IDLE")),
                current_task_id=running.task_id if running else None,
                pending_approval=pending[0].approval_id if pending else None,
            )

        return self._store.read(answer)

    def tool_table(self) -> list[dict[str, Any]]:
        """The Dot's tools and whether the model is offered each now (none before a config arrived)."""
        offered = self._settings.offered_tools if self._settings else ()
        return tool_table(self._registry, offered)

    async def skills(self) -> list[dict[str, Any]]:
        """The Dot's skills as `GET /skills` shows them: its own and the built-in ones, each with its whole file."""
        return [
            {"name": s.name, "description": s.description, "source": s.source, "path": s.path, "content": s.content}
            for s in await all_skills(self._computer)
        ]

    def read_outbox_after(self, after: int, limit: int) -> list[dict[str, Any]]:
        return self._store.read(lambda conn: dots_store.read_outbox_after(conn, after, limit))

    def on_append(self, listener: Callable[[], None]) -> Callable[[], None]:
        """Call `listener` after every commit that adds outbox rows; returns its remover."""
        return self._store.on_append(listener)

    # --- start and stop -----------------------------------------------------

    def start(self) -> None:
        """Announce this start (`agent.started`, the host pushes the key again on it), close what
        the last process left open, and pick up work."""
        def recover(conn: sqlite3.Connection) -> int:
            dots_store.append_outbox(conn, "agent.started", {})
            # A call with an intent and no result was cut by the stop; its outcome is unknown.
            intents = dots_store.list_tool_intents(conn)
            sessions = {CHAT_SESSION_KEY, *(intent.session_key for intent in intents)}
            running = dots_store.get_running_task(conn)
            if running is not None:
                sessions.add(running.session_key)
            for session_key in sorted(sessions):
                close_open_calls(conn, session_key)
            dots_store.take_all_tool_intents(conn)
            # An approved call that was running was closed above as interrupted with "ask"; its approval is over.
            for approval in dots_store.list_approvals(conn, "running"):
                dots_store.advance_approval(conn, approval.approval_id, "running", "done")
            # A turn telling a session a decision was cut: tell it again.
            for approval in dots_store.list_approvals(conn, "granted"):
                dots_store.advance_approval(conn, approval.approval_id, "granted", "approved")
            for approval in dots_store.list_approvals(conn, "told"):
                dots_store.advance_approval(conn, approval.approval_id, "told", "rejected")
            # The outbox may end on a busy state from before the stop.
            dots_store.record_agent_state(conn, "IDLE", force=True)
            return len(intents)

        stored = self._store.read(lambda conn: dots_store.read_kv(conn, dots_store.KV_RUNTIME_CONFIG))
        interrupted = self._store.write(recover)
        if stored is not None:
            try:
                self._apply_config(parse_runtime_config(stored))
            except DotsConfigError as error:
                logger.error("the stored Dot config is not valid, waiting for the host to push one: {}", error)
        self._started = True
        self._quiet_since = asyncio.get_running_loop().time()
        logger.info("Dot engine started configured={} interrupted_calls={}", self._config is not None, interrupted)
        self.kick()

    async def stop(self) -> None:
        """Stop taking work and end the turns in flight; used on shutdown. It returns when the
        suspend does, within the stop grace and the cancel wait, whatever a turn does."""
        self._stopped = True
        await self.suspend()

    # --- host requests ------------------------------------------------------

    def set_config(self, body: object) -> DotRuntimeConfig:
        """`PUT /config`: validate, store, project. The same config again changes nothing."""
        config = parse_runtime_config(body)
        if config == self._config:
            return config
        # What the host sent, as it sent it: a field it left out is not stored as null, which the parse refuses.
        stored = config.model_dump(mode="json", exclude_unset=True)
        self._store.write(lambda conn: dots_store.write_kv(conn, dots_store.KV_RUNTIME_CONFIG, stored))
        self._apply_config(config)
        logger.info("Dot config updated name={} model={}", config.name, config.model.id)
        self.kick()
        return config

    def _apply_config(self, config: DotRuntimeConfig) -> None:
        self._settings = project(
            config, workspace=self._workspace, openrouter_base_url=self._openrouter_base_url
        )
        self._config = config

    def key_received(self) -> None:
        """The OpenRouter key arrived (`POST /secrets`).

        The host pushes it on every READY transition, so it also ends a prepare-sleep that no
        shutdown followed.
        """
        if self._suspended and not self._stopped:
            logger.info("resuming work: the host pushed the key, so no shutdown is coming")
        self._suspended = False
        self.kick()

    def accept(self, event: InboundEvent) -> bool:
        """`POST /events`: record the event and its effect in one transaction.

        False for an event id accepted before, which is still a success.
        """
        if self._stopped:
            raise EngineStopped("the agent is shutting down")
        record = event.model_dump(mode="json")
        cancel_session: list[str] = []

        def apply(conn: sqlite3.Connection) -> bool:
            applied = event.type != "user.message"
            if not dots_store.record_inbound(conn, record, "applied" if applied else "accepted"):
                return False
            if isinstance(event, TaskCreatedEvent):
                if not dots_store.enqueue_task(
                    conn,
                    task_id=event.data.task_id,
                    description=event.data.description,
                    priority=event.data.priority,
                ):
                    logger.warning("task already known, not queued again task_id={}", event.data.task_id)
            elif isinstance(event, ApprovalReceivedEvent):
                to = "approved" if event.data.decision == "approve" else "rejected"
                if not dots_store.advance_approval(
                    conn, event.data.approval_id, "pending", to, note=event.data.note
                ):
                    logger.warning(
                        "decision for an approval that is unknown or already decided; ignored approval_id={}",
                        event.data.approval_id,
                    )
            elif isinstance(event, SystemEvent) and event.data.name == TASK_CANCELLED_EVENT:
                raw = event.data.data.get("task_id")
                task_id = raw if isinstance(raw, str) else ""
                task = dots_store.get_task(conn, task_id) if task_id else None
                if task is None or not dots_store.finish_task(conn, task_id, "cancelled"):
                    logger.warning("cancel for a task that is unknown or already finished; ignored task_id={}", task_id)
                else:
                    if (
                        dots_store.end_waiting_approvals(conn, task.session_key)
                        and dots_store.read_kv(conn, dots_store.KV_AGENT_STATE) == "WAITING_APPROVAL"
                        and not dots_store.list_approvals(conn, "pending")
                    ):
                        # Nothing runs while the agent waits: with nothing left to wait for, it is idle.
                        dots_store.record_agent_state(conn, "IDLE")
                    if task.status == "running":
                        cancel_session.append(task.session_key)
            return True

        if not self._store.write(apply):
            logger.debug("inbound event already accepted id={} type={}", event.id, event.type)
            return False
        logger.info("inbound event accepted id={} type={}", event.id, event.type)
        for session_key in cancel_session:
            self._cancel_session(session_key)
        # New work after a prepare-sleep means the host changed its mind.
        self._suspended = False
        self.kick()
        return True

    async def automation_fired(self, job: CronJob) -> None:
        """A cron job fired: record it as an input of the chat, once per firing."""
        if self._stopped:
            return
        run_at = job.state.next_run_at_ms or dots_store.clock_ms()
        event = {
            "id": f"cron:{job.id}:{run_at}",
            "type": AUTOMATION_FIRED,
            "ts": _now_iso(),
            "data": {"name": job.name, "message": job.payload.message},
        }
        if self._store.write(lambda conn: dots_store.record_inbound(conn, event, "accepted")):
            logger.info("automation fired id={} name={}", event["id"], job.name)
            self.kick()

    def automations_next_run(self, next_run_at_ms: int | None) -> None:
        """The cron service armed its timer: tell the host when the earliest job is due, once per change.

        The host wakes a computer that is off for it, and does not put one to sleep that is about to need
        it (architecture section 9.5).
        """
        if self._stopped:
            return
        if self._store.write(lambda conn: dots_store.record_next_run(conn, next_run_at_ms)):
            logger.info("next automation run reported at_ms={}", next_run_at_ms)

    async def suspend(self) -> None:
        """`POST /prepare-sleep`: start no new work; cancel a turn with no tool running at once; give
        a turn with a tool running up to the stop grace to record its result and end; cancel what is
        left; flush the database. A running task stays running and resumes at the next start.

        One suspend runs at a time, and a caller that arrives meanwhile waits for it: two would
        cancel the same turns and give the same attempt back twice. The suspend runs to its end
        even when the caller goes away (the host hanging up must not cut a sleep short).
        """
        self._suspended = True
        if self._suspension is None:
            suspension = asyncio.get_running_loop().create_task(self._suspend_now(), name="suspend")
            suspension.add_done_callback(self._suspension_done)
            self._suspension = suspension
        await asyncio.shield(self._suspension)

    def _suspension_done(self, done: asyncio.Task[None]) -> None:
        self._suspension = None
        if not done.cancelled() and done.exception() is not None:
            logger.error("suspending failed error={!r}", done.exception())

    async def _suspend_now(self) -> None:
        loop = asyncio.get_running_loop()
        # A memory pass is only bookkeeping: it stops at once, and the next quiet spell takes it up again.
        if self._memory_timer is not None:
            self._memory_timer.cancel()
            self._memory_timer = None
        if self._memory_pass is not None:
            self._memory_pass.cancel()
            await asyncio.wait([self._memory_pass], timeout=_CANCEL_WAIT_S)
        deadline = loop.time() + self._stop_grace_s
        cut_task = next(
            (turn.unit.task_id for turn in self._turns.values() if turn.counted_attempt and turn.unit.task_id),
            None,
        )
        cancelled: set[str] = set()
        while self._turns:
            now = loop.time()
            for turn in list(self._turns.values()):
                session_key = turn.unit.session_key
                if session_key not in cancelled and (
                    self._tools_running.get(session_key, 0) == 0 or now >= deadline
                ):
                    cancelled.add(session_key)
                    turn.task.cancel()
            if now >= deadline + _CANCEL_WAIT_S:
                logger.warning("a turn did not end after it was cancelled; going on without it")
                break
            pending = [turn.task for turn in self._turns.values()]
            await asyncio.wait(pending, timeout=0.05)
        if cut_task is not None:
            # A sleep is not a failed attempt (architecture 8.7).
            self._store.write(lambda conn: dots_store.uncount_task_attempt(conn, cut_task))
        # No turn is left to call a browser: close them while the engine still lives, so Firefox flushes.
        # The closes are tasks of the manager, so the end of the wait cuts none of them short.
        budget = _CLOSE_BROWSERS_ON_STOP_S if self._stopped else _CLOSE_BROWSERS_ON_SLEEP_S
        try:
            await asyncio.wait_for(self._browser.close_all(), budget)
        except asyncio.TimeoutError:
            logger.error("the open browsers did not close within {} s; the process ends them", budget)
            if self._stopped:
                # The store closes before these closes end: record them now, as what the exit makes them.
                self._browser.closed_by_exit()
        self._store.checkpoint()

    # --- the host of a turn (TurnHost) --------------------------------------

    def is_suspending(self) -> bool:
        return self._suspended or self._stopped

    def run_started(self, unit: TurnUnit) -> None:
        self._tools_running[unit.session_key] = 0
        self._store.write(lambda conn: dots_store.record_agent_state(conn, "THINKING"))

    def tool_started(self, intent: ToolIntent) -> None:
        self._tools_running[intent.session_key] = self._tools_running.get(intent.session_key, 0) + 1

        def record(conn: sqlite3.Connection) -> None:
            dots_store.record_tool_intent(conn, intent)
            dots_store.record_agent_state(conn, "EXECUTING")

        self._store.write(record)

    def tool_ended(self, session_key: str) -> None:
        running = self._tools_running.get(session_key, 0)
        if running > 0:
            self._tools_running[session_key] = running - 1
        if sum(self._tools_running.values()) == 0:
            self._store.write(lambda conn: dots_store.record_agent_state(conn, "THINKING"))

    async def _chat_inputs(self) -> list[OpeningMessage]:
        """The inputs accepted since the chat turn started, for it to take in."""
        return [opening_for_row(row) for row in self._store.read(_pending_chat_inputs)]

    # --- work ---------------------------------------------------------------

    def _can_work(self) -> bool:
        return (
            self._started
            and not self._stopped
            and not self._suspended
            and self._config is not None
            and self._key_holder.configured
        )

    def kick(self) -> None:
        """Look for work that can start now. Safe to call any time."""
        if self._pumping:
            self._repump = True
            return
        self._pumping = True
        try:
            while True:
                self._repump = False
                try:
                    self._pump()
                except Exception:
                    logger.exception("Dot work loop failed")
                if not self._repump:
                    return
        finally:
            self._pumping = False

    def _pump(self) -> None:
        if not self._can_work():
            return
        self._pump_approvals()
        self._pump_chat()
        self._pump_task()
        self._pump_memory()

    def _pump_memory(self) -> None:
        """Start the memory pass that is owed once no turn has run for `memory_quiet_s`."""
        if self._memory_pass is not None or not self._memory_due or self._turns:
            return
        loop = asyncio.get_running_loop()
        wait = self._quiet_since + self._memory_quiet_s - loop.time()
        if wait > 0:
            if self._memory_timer is None:
                self._memory_timer = loop.call_later(wait, self._memory_timer_fired)
            return
        self._memory_due = False
        memory_pass = loop.create_task(self._run_memory_pass(), name="memory pass")
        self._memory_pass = memory_pass
        memory_pass.add_done_callback(self._memory_pass_done)

    def _memory_timer_fired(self) -> None:
        self._memory_timer = None
        self.kick()

    async def _run_memory_pass(self) -> None:
        try:
            outcome = await self._memory.run()
        except asyncio.CancelledError:
            # Cut by a sleep or a stop: it is owed again, and what it took in so far is kept.
            self._memory_due = True
            raise
        if outcome.kind == "failed":
            # Owed again only once a turn ends: a pass that fails at once is not tried over and over.
            logger.warning("memory pass failed: {}", outcome.reason)

    def _memory_pass_done(self, done: asyncio.Task[None]) -> None:
        self._memory_pass = None
        if not done.cancelled() and done.exception() is not None:
            logger.opt(exception=done.exception()).error("memory pass failed")
        self.kick()

    def _pump_approvals(self) -> None:
        """Decided approvals: tell the session that made the call."""
        decided = self._store.read(
            lambda conn: [
                *dots_store.list_approvals(conn, "approved"),
                *dots_store.list_approvals(conn, "rejected"),
            ]
        )
        for approval in decided:
            self._act_on_approval(approval)

    def _act_on_approval(self, approval: Approval) -> None:
        """Tell the session of a decided call what was decided, in a turn of its own.

        The approval moves to granted (or told) before the turn starts, so the call the turn makes
        again finds it, and to done when the turn ends; a turn cut by a stop is told again at the
        next start.
        """
        approval_id = approval.approval_id
        source = approval.status
        approved = source == "approved"
        telling = "granted" if approved else "told"
        text = approval_granted_continuation(approval) if approved else approval_rejected_continuation(approval)
        opening = (OpeningMessage(text, {APPROVAL_ID: approval_id, APPROVAL_LINE: approval_line(approval, approved)}),)
        if approval.session_key == CHAT_SESSION_KEY:
            if "chat" in self._turns:
                # Its end kicks again.
                return
            if not self._store.write(lambda conn: dots_store.advance_approval(conn, approval_id, source, telling)):
                return
            self._launch("chat", TurnUnit(CHAT_SESSION_KEY, None, opening, approval_id))
            return
        task = self._store.read(lambda conn: dots_store.get_task_by_session(conn, approval.session_key))
        if task is None or task.status != "running":
            # The task ended (cancelled) while it waited: nobody is left to tell.
            self._store.write(lambda conn: dots_store.advance_approval(conn, approval_id, source, "done"))
            return
        if "task" in self._turns:
            # A task turn is in flight: its end kicks again.
            return
        if not self._store.write(lambda conn: dots_store.advance_approval(conn, approval_id, source, telling)):
            return
        self._launch("task", TurnUnit(task.session_key, task.task_id, opening, approval_id))

    def _pump_chat(self) -> None:
        if "chat" in self._turns:
            return
        rows = self._store.read(_pending_chat_inputs)
        if rows:
            self._launch(
                "chat",
                TurnUnit(CHAT_SESSION_KEY, None, tuple(opening_for_row(row) for row in rows)),
            )
            return
        # An answer owed: the transcript holds inputs nobody answered, and no approval holds the chat.
        owed = self._store.read(
            lambda conn: bool(dots_store.list_inbound(conn, "in_transcript"))
            and dots_store.open_approval_for_session(conn, CHAT_SESSION_KEY) is None
        )
        if owed:
            self._launch("chat", TurnUnit(CHAT_SESSION_KEY, None))

    def _pump_task(self) -> None:
        if "task" in self._turns:
            return

        def next_task(conn: sqlite3.Connection) -> tuple[dots_store.TaskRow, bool] | Literal["failed"] | None:
            running = dots_store.get_running_task(conn)
            if running is not None and dots_store.open_approval_for_session(conn, running.session_key):
                # Parked on an approval: the decision, not the queue, moves it on.
                return None
            if running is not None:
                # Left running by the last process: resume it, or give up on it.
                if running.attempts >= TASK_MAX_ATTEMPTS:
                    error = f"stopped: the task was interrupted {running.attempts} times"
                    dots_store.finish_task(conn, running.task_id, "failed", error=error)
                    dots_store.append_outbox_spent(
                        conn, "task.failed", {"task_id": running.task_id, "error": error}, running.session_key
                    )
                    return "failed"
                dots_store.start_task(conn, running.task_id)
                return running, True
            queued = dots_store.next_queued_task(conn)
            if queued is None:
                return None
            dots_store.start_task(conn, queued.task_id)
            dots_store.append_outbox(conn, "task.started", {"task_id": queued.task_id})
            return queued, False

        next_up = self._store.write(next_task)
        if next_up is None:
            return
        if next_up == "failed":
            self._repump = True
            return
        task, resumed = next_up
        description = f"{task.description}\n\n{RESUME_TASK_NOTE}" if resumed else task.description
        self._launch(
            "task",
            TurnUnit(task.session_key, task.task_id, (OpeningMessage(description),)),
            counted_attempt=True,
        )

    # --- turns --------------------------------------------------------------

    def _launch(self, slot: Slot, unit: TurnUnit, *, counted_attempt: bool = False) -> None:
        logger.info(
            "turn starts slot={} session={} task={} approval={}",
            slot, unit.session_key, unit.task_id, unit.approval_id,
        )
        task = asyncio.get_running_loop().create_task(self._run_turn(unit), name=f"turn:{unit.session_key}")
        turn = _Turn(slot, unit, task, counted_attempt)
        self._turns[slot] = turn
        task.add_done_callback(lambda done: self._turn_done(turn, done))

    async def _run_turn(self, unit: TurnUnit) -> bool:
        """Run a turn and record its end; False when the end could not be recorded."""
        outcome: TurnOutcome
        try:
            outcome = await self._runner.run(unit)
        except asyncio.CancelledError:
            # The engine cancels a turn only to suspend, stop or cancel its task: it is abandoned.
            outcome = TurnOutcome("abandoned")
        try:
            self._turn_ended(unit, outcome)
        except Exception:
            logger.exception("ending a turn failed session={}", unit.session_key)
            return False
        return True

    def _turn_done(self, turn: _Turn, done: asyncio.Task[bool]) -> None:
        if self._turns.get(turn.slot) is turn:
            del self._turns[turn.slot]
        # The turn wrote its conversation: a memory pass is owed once the Dot has been quiet again.
        self._memory_due = True
        self._quiet_since = asyncio.get_running_loop().time()
        if not done.cancelled() and done.exception() is not None:
            logger.error("turn task failed session={} error={!r}", turn.unit.session_key, done.exception())
        elif not done.cancelled() and not done.result() and turn.slot == "chat":
            # Its end was not recorded, so the inputs it took may still be accepted: starting the same
            # turn again from here would repeat the failure at once. The next event kicks. (A task turn
            # is bounded by its attempts, and fails with a reason when they run out.)
            return
        self.kick()

    def _cancel_session(self, session_key: str) -> None:
        for turn in self._turns.values():
            if turn.unit.session_key == session_key:
                turn.task.cancel()

    def _turn_ended(self, unit: TurnUnit, outcome: TurnOutcome) -> None:
        """What the end of a turn means, in one transaction: the approval it told, the answer or the
        failure it leaves, the agent state."""
        session_key = unit.session_key
        was_running = self._tools_running.pop(session_key, None) is not None
        last_run = not self._tools_running
        stopping = self.is_suspending()
        reason = outcome.reason or "the run failed"

        def end(conn: sqlite3.Connection) -> None:
            is_chat = session_key == CHAT_SESSION_KEY
            task = None if is_chat else dots_store.get_task_by_session(conn, session_key)
            task_running = task is not None and task.status == "running"
            if unit.approval_id:
                if stopping and outcome.kind in ("abandoned", "failed"):
                    # Cut by a sleep or a stop before it was told: tell it again once work resumes.
                    dots_store.advance_approval(conn, unit.approval_id, "granted", "approved")
                    dots_store.advance_approval(conn, unit.approval_id, "told", "rejected")
                else:
                    # The turn that told the session ends the approval; a call it made again ended it already.
                    dots_store.advance_approval(conn, unit.approval_id, "granted", "done")
                    dots_store.advance_approval(conn, unit.approval_id, "told", "done")
            if outcome.kind == "failed" and not stopping:
                close_open_calls(conn, session_key)
                if is_chat:
                    self._answer_failure(conn, unit, reason)
                elif task is not None and task_running:
                    # A run that parked a call for approval ended on purpose: the task waits.
                    if dots_store.open_approval_for_session(conn, session_key) is None:
                        dots_store.finish_task(conn, task.task_id, "failed", error=reason)
                        dots_store.append_outbox_spent(
                            conn, "task.failed", {"task_id": task.task_id, "error": reason}, session_key
                        )
            elif outcome.kind == "abandoned" and task is not None and not task_running:
                # The task was cancelled under the turn: its open calls end here.
                close_open_calls(conn, session_key)
            if was_running and last_run:
                dots_store.record_agent_state(conn, "DONE")
                waiting = bool(dots_store.list_approvals(conn, "pending"))
                dots_store.record_agent_state(conn, "WAITING_APPROVAL" if waiting else "IDLE")

        self._store.write(end)

    @staticmethod
    def _answer_failure(conn: sqlite3.Connection, unit: TurnUnit, reason: str) -> None:
        """A chat turn that ended without an answer still owes the person one.

        That includes the inputs the turn itself took in when it failed before its opening reached
        the transcript: left accepted, the next turn would take them in again and fail the same way.
        """
        for opening in unit.opening:
            inbound_id = opening.metadata.get(INBOUND_ID)
            if isinstance(inbound_id, str):
                dots_store.mark_inbound_in_transcript(conn, inbound_id)
        if not dots_store.list_inbound(conn, "in_transcript"):
            return
        answered = dots_store.apply_answered_inputs(conn)
        reply = {"in_reply_to": answered[-1]} if answered else {}
        dots_store.append_outbox_spent(
            conn, "message.assistant", {"text": f"I could not answer: {reason}", **reply}, unit.session_key
        )
