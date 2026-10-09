"""The Dot's durable state: one SQLite file holding its own tables and the transcripts.

Keeping the Dot's tables in the same file as the transcripts (`sessions`,
`messages`) is what lets an outbox row commit in the same transaction as the
transcript row it describes (architecture section 8.7): SQLite makes a
transaction atomic per file only.

- dots_outbox: outbound events, numbered by `seq` (AUTOINCREMENT, so a seq that
  was committed is never handed out again);
- dots_inbound: every inbound event id the engine accepted (idempotency), with
  how far it was applied;
- dots_tasks: the local task queue (one at a time, priority then arrival);
- dots_tool_intents: a tool call that started and has no result yet (an intent
  left by a crash is a call the agent stopped during), with the line that names
  what it acts on (`target`) and whether it started a terminal session (`tty`),
  which its result reports in `tool.called`. A file made before the notes were
  the Dot's alone has a `memory_keys_json` column too, which its default fills;
- dots_spend: what the model requests of a session have cost, in USD, and whether one of them reported
  no cost (see `nanobot.dots.spend`);
- dots_browser_identities: the browser identities of the Dot (id, name, proxy, created, last used,
  archived). Whether one is open is never stored: it is derived from the live browser sessions of
  this process, so a file never says "open" about a process that is gone. Its profile is a
  directory of the Computer (`browsers/<id>`), not a row;
- dots_kv: the runtime config the host pushed and the last agent state;
- dots_approvals: a tool call the policy answered "ask", with its full
  arguments, from the request to the call that ran it or the rejection the
  model was told about. Nothing of an approval lives only in memory;
- dots_tool_decisions: the gate's decision for a call of a turn, until the
  call's result reaches the transcript;
- sessions, messages: the transcripts.

A provider's tool_call id names a call only inside its own response: models behind
OpenRouter reuse "call_0" in every response and every session. Every row about a
call (intent, decision, approval) is therefore keyed by the session AND the id,
and an approval is a row per ask, never one per id.

Every row function takes the connection and runs inside the caller's
transaction; none opens one. `DotStore.write` is the one place a write
transaction begins and ends.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypeVar, get_args

from loguru import logger

from nanobot.dots import transcript_outbox
from nanobot.dots.protocol import AGENT_STATES, INBOUND_EVENT_TYPES, OUTBOUND_EVENT_TYPES
from nanobot.session.manager import Session

T = TypeVar("T")

# How long open() waits for the file's lock before it says another engine owns it.
OPEN_TIMEOUT_S = 2.0

# The Dot's single persistent conversation, and the key of a task's own one.
CHAT_SESSION_KEY = "chat"
TASK_SESSION_PREFIX = "task:"


def task_session_key(task_id: str) -> str:
    return f"{TASK_SESSION_PREFIX}{task_id}"


# Runs once, on a file with no tables (see `DotStore.open`).
_SCHEMA: tuple[str, ...] = (
    """
    CREATE TABLE dots_outbox (
      seq INTEGER PRIMARY KEY AUTOINCREMENT,
      id TEXT NOT NULL UNIQUE,
      type TEXT NOT NULL,
      ts TEXT NOT NULL,
      data_json TEXT NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE dots_inbound (
      id TEXT PRIMARY KEY,
      type TEXT NOT NULL,
      ts TEXT NOT NULL,
      data_json TEXT NOT NULL,
      state TEXT NOT NULL,
      accepted_order INTEGER NOT NULL,
      accepted_at INTEGER NOT NULL,
      applied_at INTEGER
    ) STRICT
    """,
    "CREATE INDEX dots_inbound_pending ON dots_inbound (state, accepted_order)",
    """
    CREATE TABLE dots_tasks (
      task_id TEXT PRIMARY KEY,
      description TEXT NOT NULL,
      priority INTEGER NOT NULL,
      status TEXT NOT NULL,
      created_order INTEGER NOT NULL,
      session_key TEXT NOT NULL UNIQUE,
      attempts INTEGER NOT NULL DEFAULT 0,
      summary TEXT,
      error TEXT,
      updated_at INTEGER NOT NULL
    ) STRICT
    """,
    "CREATE INDEX dots_tasks_queue ON dots_tasks (status, priority, created_order)",
    """
    CREATE TABLE dots_tool_intents (
      session_key TEXT NOT NULL,
      tool_call_id TEXT NOT NULL,
      tool TEXT NOT NULL,
      task_id TEXT,
      started_at INTEGER NOT NULL,
      target TEXT,
      tty INTEGER NOT NULL DEFAULT 0 CHECK (tty IN (0, 1)),
      PRIMARY KEY (session_key, tool_call_id)
    ) STRICT
    """,
    """
    CREATE TABLE dots_spend (
      session_key TEXT PRIMARY KEY,
      usd REAL NOT NULL,
      unpriced INTEGER NOT NULL DEFAULT 0 CHECK (unpriced IN (0, 1))
    ) STRICT
    """,
    """
    CREATE TABLE dots_browser_identities (
      id TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      proxy TEXT,
      created_at INTEGER NOT NULL,
      last_used_at INTEGER,
      archived INTEGER NOT NULL DEFAULT 0 CHECK (archived IN (0, 1))
    ) STRICT
    """,
    "CREATE INDEX dots_browser_identities_order ON dots_browser_identities (created_at, id)",
    """
    CREATE TABLE dots_kv (
      key TEXT PRIMARY KEY,
      value_json TEXT NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE dots_approvals (
      approval_id TEXT PRIMARY KEY,
      session_key TEXT NOT NULL,
      task_id TEXT,
      tool_call_id TEXT NOT NULL,
      tool TEXT NOT NULL,
      permission TEXT NOT NULL,
      arguments_json TEXT NOT NULL,
      status TEXT NOT NULL,
      note TEXT,
      run_tool_call_id TEXT,
      created_at INTEGER NOT NULL,
      resolved_at INTEGER
    ) STRICT
    """,
    "CREATE INDEX dots_approvals_status ON dots_approvals (status, created_at)",
    "CREATE INDEX dots_approvals_call ON dots_approvals (session_key, tool_call_id)",
    """
    CREATE TABLE dots_tool_decisions (
      session_key TEXT NOT NULL,
      tool_call_id TEXT NOT NULL,
      decision TEXT NOT NULL,
      PRIMARY KEY (session_key, tool_call_id)
    ) STRICT
    """,
    """
    CREATE TABLE sessions (
      key TEXT PRIMARY KEY,
      metadata_json TEXT NOT NULL,
      updated_at INTEGER NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE messages (
      session_key TEXT NOT NULL,
      idx INTEGER NOT NULL,
      message_json TEXT NOT NULL,
      PRIMARY KEY (session_key, idx)
    ) STRICT
    """,
)


class StoreOwnedError(RuntimeError):
    """The database file is held by another engine."""


class DotStore:
    """One connection to the Dot's database, owned by this process.

    The connection is used on one thread only (sqlite3's own check enforces
    it). `locking_mode=EXCLUSIVE` makes the first writer the only user of the
    file, so a second engine on it fails at open instead of interleaving.
    """

    def __init__(self, conn: sqlite3.Connection, path: Path) -> None:
        self._conn: sqlite3.Connection | None = conn
        self.path = path
        self._listeners: list[Callable[[], None]] = []

    @classmethod
    def open(cls, path: str | Path, *, open_timeout_s: float = OPEN_TIMEOUT_S) -> DotStore:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, isolation_level=None, timeout=open_timeout_s)
        conn.row_factory = sqlite3.Row
        try:
            # EXCLUSIVE first: entering WAL under it keeps the wal-index in
            # memory, and the first access below then takes the file for good.
            conn.execute("PRAGMA locking_mode=EXCLUSIVE")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("PRAGMA foreign_keys=ON")
            # The schema is the first write transaction, so the write lock is
            # held from here on and a second process cannot get past it.
            conn.execute("BEGIN IMMEDIATE")
            try:
                # Decided under the lock: another engine may have come and gone since.
                if _table_count(conn) == 0:
                    _create_schema(conn)
                conn.execute("COMMIT")
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
        except sqlite3.OperationalError as error:
            conn.close()
            if "locked" in str(error) or "busy" in str(error):
                raise StoreOwnedError(f"another engine owns {path}") from None
            raise
        except BaseException:
            conn.close()
            raise
        return cls(conn, path)

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError(f"the store {self.path} is closed")
        return self._conn

    def on_append(self, listener: Callable[[], None]) -> Callable[[], None]:
        """Call `listener` after every commit that added outbox rows; returns its remover."""
        self._listeners.append(listener)

        def remove() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return remove

    def write(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        """Run `fn` in one IMMEDIATE transaction: commit when it returns, roll back when it raises."""
        conn = self._connection()
        conn.execute("BEGIN IMMEDIATE")
        try:
            before = _outbox_high_water(conn)
            result = fn(conn)
            grew = _outbox_high_water(conn) > before
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        if grew:
            for listener in list(self._listeners):
                try:
                    listener()
                except Exception:
                    logger.exception("an outbox listener failed")
        return result

    def read(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        return fn(self._connection())

    def checkpoint(self) -> None:
        self._connection().execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


def _table_count(conn: sqlite3.Connection) -> int:
    return conn.execute(
        "SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchone()[0]


def _create_schema(conn: sqlite3.Connection) -> None:
    """Make the tables on an empty file."""
    for statement in _SCHEMA:
        conn.execute(statement)


def _outbox_high_water(conn: sqlite3.Connection) -> int:
    """The highest seq ever handed out (AUTOINCREMENT keeps it even if rows are deleted)."""
    row = conn.execute("SELECT seq FROM sqlite_sequence WHERE name = 'dots_outbox'").fetchone()
    return int(row[0]) if row else 0


def clock_ms() -> int:
    """The wall clock in milliseconds: the one clock every timestamp of a row comes from."""
    return time.time_ns() // 1_000_000


def _dumps(value: Any) -> str:
    # ASCII only: a lone surrogate a model emitted survives as an escape, where
    # raw text would fail to encode into the database.
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# The outbox
# ---------------------------------------------------------------------------


def _iso_ms(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def iso_from_ms(milliseconds: int) -> str:
    """A time the store keeps in milliseconds since the epoch, as the API shows it: ISO 8601 in UTC."""
    return _iso_ms(datetime.fromtimestamp(milliseconds / 1000, tz=timezone.utc))


def append_outbox(
    conn: sqlite3.Connection,
    event_type: str,
    data: Mapping[str, Any],
    now: datetime | None = None,
) -> dict[str, Any]:
    """Append one outbound event; returns it with its seq."""
    if event_type not in OUTBOUND_EVENT_TYPES:
        raise ValueError(f"not an outbound event type: {event_type}")
    event_id = str(uuid.uuid4())
    ts = _iso_ms(now or datetime.now(timezone.utc))
    payload = dict(data)
    cursor = conn.execute(
        "INSERT INTO dots_outbox (id, type, ts, data_json) VALUES (?, ?, ?, ?)",
        (event_id, event_type, ts, _dumps(payload)),
    )
    return {"seq": int(cursor.lastrowid), "id": event_id, "type": event_type, "ts": ts, "data": payload}


def read_outbox_after(conn: sqlite3.Connection, after: int, limit: int) -> list[dict[str, Any]]:
    """Outbound events after `after`, oldest first, at most `limit`."""
    rows = conn.execute(
        "SELECT seq, id, type, ts, data_json FROM dots_outbox WHERE seq > ? ORDER BY seq LIMIT ?",
        (after, limit),
    ).fetchall()
    return [
        {
            "seq": int(row["seq"]),
            "id": row["id"],
            "type": row["type"],
            "ts": row["ts"],
            "data": json.loads(row["data_json"]),
        }
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Inbound events
# ---------------------------------------------------------------------------

# How far an inbound event is applied:
# - accepted: recorded; a user.message or automation.fired is not in the transcript yet;
# - in_transcript: the chat transcript holds its text, so a restart must not send it again;
# - applied: done (once the chat turn that holds it was answered).
InboundState = Literal["accepted", "in_transcript", "applied"]

# The inbound types the table holds: the host's, and the automation a cron firing records.
AUTOMATION_FIRED = "automation.fired"
STORED_INBOUND_TYPES = (*INBOUND_EVENT_TYPES, AUTOMATION_FIRED)


@dataclass(frozen=True)
class InboundRow:
    id: str
    type: str
    ts: str
    data: dict[str, Any]
    state: InboundState


def record_inbound(
    conn: sqlite3.Connection,
    event: Mapping[str, Any],
    state: InboundState,
    now_ms: int | None = None,
) -> bool:
    """Record an inbound event; False when its id was accepted before (nothing changes)."""
    if event["type"] not in STORED_INBOUND_TYPES:
        raise ValueError(f"not an inbound event type: {event['type']}")
    now = clock_ms() if now_ms is None else now_ms
    cursor = conn.execute(
        """
        INSERT INTO dots_inbound (id, type, ts, data_json, state, accepted_order, accepted_at, applied_at)
        VALUES (?, ?, ?, ?, ?, (SELECT COALESCE(MAX(accepted_order), 0) + 1 FROM dots_inbound), ?, ?)
        ON CONFLICT (id) DO NOTHING
        """,
        (
            event["id"],
            event["type"],
            event["ts"],
            _dumps(event["data"]),
            state,
            now,
            now if state == "applied" else None,
        ),
    )
    return cursor.rowcount == 1


def list_inbound(conn: sqlite3.Connection, state: InboundState) -> list[InboundRow]:
    rows = conn.execute(
        "SELECT id, type, ts, data_json, state FROM dots_inbound WHERE state = ? ORDER BY accepted_order",
        (state,),
    ).fetchall()
    return [
        InboundRow(row["id"], row["type"], row["ts"], json.loads(row["data_json"]), row["state"])
        for row in rows
    ]


def mark_inbound_in_transcript(conn: sqlite3.Connection, inbound_id: str) -> bool:
    """Move a user.message or automation.fired from accepted to in_transcript; False when it was not accepted."""
    cursor = conn.execute(
        """
        UPDATE dots_inbound SET state = 'in_transcript'
        WHERE id = ? AND type IN ('user.message', 'automation.fired') AND state = 'accepted'
        """,
        (inbound_id,),
    )
    return cursor.rowcount == 1


def apply_answered_inputs(conn: sqlite3.Connection, now_ms: int | None = None) -> list[str]:
    """Mark every input the transcript holds as answered, in the transaction that commits the answer.

    Returns the ids of the user.message rows only, oldest first: an automation
    firing is answered too, but nobody replied to it.
    """
    now = clock_ms() if now_ms is None else now_ms
    ids = [
        row["id"]
        for row in conn.execute(
            """
            SELECT id FROM dots_inbound
            WHERE type = 'user.message' AND state = 'in_transcript' ORDER BY accepted_order
            """
        )
    ]
    conn.execute(
        """
        UPDATE dots_inbound SET state = 'applied', applied_at = ?
        WHERE type IN ('user.message', 'automation.fired') AND state = 'in_transcript'
        """,
        (now,),
    )
    return ids


# ---------------------------------------------------------------------------
# The task queue
# ---------------------------------------------------------------------------

TaskStatus = Literal["queued", "running", "completed", "failed", "cancelled"]


@dataclass(frozen=True)
class TaskRow:
    task_id: str
    description: str
    priority: int
    status: TaskStatus
    session_key: str
    attempts: int


_TASK_COLUMNS = "task_id, description, priority, status, session_key, attempts"


def _task(row: sqlite3.Row | None) -> TaskRow | None:
    if row is None:
        return None
    return TaskRow(
        row["task_id"],
        row["description"],
        int(row["priority"]),
        row["status"],
        row["session_key"],
        int(row["attempts"]),
    )


def enqueue_task(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    description: str,
    priority: int,
    now_ms: int | None = None,
) -> bool:
    """Queue a task; False when its id is known already."""
    cursor = conn.execute(
        """
        INSERT INTO dots_tasks (task_id, description, priority, status, created_order, session_key, attempts, updated_at)
        VALUES (?, ?, ?, 'queued', (SELECT COALESCE(MAX(created_order), 0) + 1 FROM dots_tasks), ?, 0, ?)
        ON CONFLICT (task_id) DO NOTHING
        """,
        (task_id, description, priority, task_session_key(task_id), clock_ms() if now_ms is None else now_ms),
    )
    return cursor.rowcount == 1


def get_task(conn: sqlite3.Connection, task_id: str) -> TaskRow | None:
    return _task(conn.execute(f"SELECT {_TASK_COLUMNS} FROM dots_tasks WHERE task_id = ?", (task_id,)).fetchone())


def get_task_by_session(conn: sqlite3.Connection, session_key: str) -> TaskRow | None:
    return _task(
        conn.execute(f"SELECT {_TASK_COLUMNS} FROM dots_tasks WHERE session_key = ?", (session_key,)).fetchone()
    )


def list_tasks(conn: sqlite3.Connection) -> list[TaskRow]:
    """Every task, in the order they arrived."""
    rows = conn.execute(f"SELECT {_TASK_COLUMNS} FROM dots_tasks ORDER BY created_order").fetchall()
    return [task for row in rows if (task := _task(row))]


def get_running_task(conn: sqlite3.Connection) -> TaskRow | None:
    """The running task, if any (at most one runs)."""
    return _task(
        conn.execute(
            f"SELECT {_TASK_COLUMNS} FROM dots_tasks WHERE status = 'running' ORDER BY created_order LIMIT 1"
        ).fetchone()
    )


def next_queued_task(conn: sqlite3.Connection) -> TaskRow | None:
    """The next task to run: highest priority first, then the oldest."""
    return _task(
        conn.execute(
            f"""
            SELECT {_TASK_COLUMNS} FROM dots_tasks WHERE status = 'queued'
            ORDER BY priority DESC, created_order ASC LIMIT 1
            """
        ).fetchone()
    )


def start_task(conn: sqlite3.Connection, task_id: str, now_ms: int | None = None) -> None:
    """Start a task: running, one more attempt."""
    conn.execute(
        "UPDATE dots_tasks SET status = 'running', attempts = attempts + 1, updated_at = ? WHERE task_id = ?",
        (clock_ms() if now_ms is None else now_ms, task_id),
    )


def uncount_task_attempt(conn: sqlite3.Connection, task_id: str) -> None:
    """Give back the attempt of a run abandoned on purpose (a sleep), so it is not counted."""
    conn.execute(
        "UPDATE dots_tasks SET attempts = attempts - 1 WHERE task_id = ? AND status = 'running' AND attempts > 0",
        (task_id,),
    )


def finish_task(
    conn: sqlite3.Connection,
    task_id: str,
    status: Literal["completed", "failed", "cancelled"],
    *,
    summary: str | None = None,
    error: str | None = None,
    now_ms: int | None = None,
) -> bool:
    """End a task that is queued or running; False when it had ended already.

    A second ending (a late run result after a cancel) changes nothing.
    """
    if status == "completed" and summary is None:
        raise ValueError("a completed task needs a summary")
    if status == "failed" and error is None:
        raise ValueError("a failed task needs an error")
    cursor = conn.execute(
        """
        UPDATE dots_tasks SET status = ?, summary = ?, error = ?, updated_at = ?
        WHERE task_id = ? AND status IN ('queued', 'running')
        """,
        (
            status,
            summary if status == "completed" else None,
            error if status == "failed" else None,
            clock_ms() if now_ms is None else now_ms,
            task_id,
        ),
    )
    return cursor.rowcount == 1


# ---------------------------------------------------------------------------
# Key-value rows and the agent state
# ---------------------------------------------------------------------------

KV_RUNTIME_CONFIG = "runtime_config"
KV_AGENT_STATE = "agent_state"
KV_NEXT_RUN = "automation_next_run"


def read_kv(conn: sqlite3.Connection, key: str, default: Any = None) -> Any:
    row = conn.execute("SELECT value_json FROM dots_kv WHERE key = ?", (key,)).fetchone()
    return json.loads(row["value_json"]) if row else default


def write_kv(conn: sqlite3.Connection, key: str, value: Any) -> None:
    conn.execute(
        """
        INSERT INTO dots_kv (key, value_json) VALUES (?, ?)
        ON CONFLICT (key) DO UPDATE SET value_json = excluded.value_json
        """,
        (key, _dumps(value)),
    )


def record_agent_state(conn: sqlite3.Connection, state: str, force: bool = False) -> bool:
    """Record a new agent state and its event, unless it is the state already recorded.

    `force` records it anyway (a start, where the outbox may end on a busy
    state the last process left).
    """
    if state not in AGENT_STATES:
        raise ValueError(f"not an agent state: {state}")
    if not force and read_kv(conn, KV_AGENT_STATE) == state:
        return False
    write_kv(conn, KV_AGENT_STATE, state)
    append_outbox(conn, "agent.state", {"state": state})
    return True


def record_next_run(conn: sqlite3.Connection, next_run_at_ms: int | None) -> bool:
    """Record when the earliest automation is next due (None: none is) and its event, unless the host was told so.

    What the host was last told is kept with the event in the same transaction, so a restart does not say it again
    and a change is never missed. A computer that never had an automation has told the host nothing, and the host
    takes that as none being due.
    """
    if read_kv(conn, KV_NEXT_RUN) == next_run_at_ms:
        return False
    write_kv(conn, KV_NEXT_RUN, next_run_at_ms)
    append_outbox(conn, "automation.next_run", {"next_run_at_ms": next_run_at_ms})
    return True


# ---------------------------------------------------------------------------
# Spend
# ---------------------------------------------------------------------------


def add_spend(conn: sqlite3.Connection, session_key: str, usd: float) -> float:
    """Add what a model request cost to the session's spend; returns the new total."""
    conn.execute(
        """
        INSERT INTO dots_spend (session_key, usd) VALUES (?, ?)
        ON CONFLICT (session_key) DO UPDATE SET usd = usd + excluded.usd
        """,
        (session_key, usd),
    )
    return get_spend(conn, session_key)


def note_unpriced(conn: sqlite3.Connection, session_key: str) -> None:
    """Note that a model request of the session came back with no cost, so its spend is no longer the whole."""
    conn.execute(
        """
        INSERT INTO dots_spend (session_key, usd, unpriced) VALUES (?, 0, 1)
        ON CONFLICT (session_key) DO UPDATE SET unpriced = 1
        """,
        (session_key,),
    )


def get_spend(conn: sqlite3.Connection, session_key: str) -> float:
    row = conn.execute("SELECT usd FROM dots_spend WHERE session_key = ?", (session_key,)).fetchone()
    return float(row["usd"]) if row else 0.0


def has_unpriced(conn: sqlite3.Connection, session_key: str) -> bool:
    """Whether a request of the session reported no cost (see `note_unpriced`)."""
    row = conn.execute("SELECT unpriced FROM dots_spend WHERE session_key = ?", (session_key,)).fetchone()
    return bool(row["unpriced"]) if row else False


def reset_spend(conn: sqlite3.Connection, session_key: str) -> None:
    conn.execute("DELETE FROM dots_spend WHERE session_key = ?", (session_key,))


# The events that report the spend of the session they belong to (architecture section 5.4).
SPEND_EVENT_TYPES = ("message.assistant", "task.progress", "task.completed", "task.failed", "memory.updated")
# The events that end a unit of spend whose ledger then starts again: the chat's answer, and a memory pass.
SPEND_RESET_EVENT_TYPES = ("message.assistant", "memory.updated")
# USD are reported to the hundred-millionth: the cost OpenRouter reports has at most that many decimals,
# and the sum of several of them must not show the noise of a float addition.
SPENT_USD_DECIMALS = 8


def append_outbox_spent(
    conn: sqlite3.Connection,
    event_type: str,
    data: Mapping[str, Any],
    session_key: str,
) -> dict[str, Any]:
    """Append one outbound event with `spent_usd`, what `session_key` has spent so far, read in this transaction.

    The one place the spend of an event is told: a task's events carry the task's spend, the chat's
    `message.assistant` what the chat spent since its last answer. The answer takes that spend with it
    (the chat's row starts again), so each dollar of the chat is reported by exactly one answer, however
    many turns it took to give it: a call parked for approval, a restart, a sleep. A `memory.updated` does the
    same for the memory passes (memory_update.py), a pass that failed included.
    """
    if event_type not in SPEND_EVENT_TYPES:
        raise ValueError(f"not an event that reports spend: {event_type}")
    spent = round(get_spend(conn, session_key), SPENT_USD_DECIMALS)
    if event_type in SPEND_RESET_EVENT_TYPES:
        reset_spend(conn, session_key)
    return append_outbox(conn, event_type, {**data, "spent_usd": spent})


# ---------------------------------------------------------------------------
# Tool intents
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolIntent:
    tool_call_id: str
    tool: str
    session_key: str
    task_id: str | None
    started_at: int
    # The line `tool.called` shows of the call (permissions.tool_target); None when there is none.
    target: str | None = None
    # The call started a terminal session (permissions.tool_starts_terminal): `tool.called` says so.
    tty: bool = False


_INTENT_COLUMNS = "session_key, tool_call_id, tool, task_id, started_at, target, tty"


def _intent(row: sqlite3.Row) -> ToolIntent:
    return ToolIntent(
        row["tool_call_id"],
        row["tool"],
        row["session_key"],
        row["task_id"],
        int(row["started_at"]),
        row["target"],
        bool(row["tty"]),
    )


def record_tool_intent(conn: sqlite3.Connection, intent: ToolIntent) -> None:
    """Record that a tool call started. A second start of the same call of the same session keeps the first."""
    conn.execute(
        f"""
        INSERT INTO dots_tool_intents ({_INTENT_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (session_key, tool_call_id) DO NOTHING
        """,
        (
            intent.session_key,
            intent.tool_call_id,
            intent.tool,
            intent.task_id,
            intent.started_at,
            intent.target,
            int(intent.tty),
        ),
    )


def peek_tool_intent(conn: sqlite3.Connection, session_key: str, tool_call_id: str) -> ToolIntent | None:
    """The intent recorded for a call of a session, left in place."""
    row = conn.execute(
        f"SELECT {_INTENT_COLUMNS} FROM dots_tool_intents WHERE session_key = ? AND tool_call_id = ?",
        (session_key, tool_call_id),
    ).fetchone()
    return _intent(row) if row else None


def take_tool_intent(conn: sqlite3.Connection, session_key: str, tool_call_id: str) -> ToolIntent | None:
    """Remove and return the intent of a call of a session, if one was recorded."""
    intent = peek_tool_intent(conn, session_key, tool_call_id)
    if intent is not None:
        conn.execute(
            "DELETE FROM dots_tool_intents WHERE session_key = ? AND tool_call_id = ?", (session_key, tool_call_id)
        )
    return intent


def list_tool_intents(conn: sqlite3.Connection) -> list[ToolIntent]:
    """Every intent, oldest first, left in place: start recovery reads them before it closes the calls they belong to."""
    rows = conn.execute(f"SELECT {_INTENT_COLUMNS} FROM dots_tool_intents ORDER BY started_at").fetchall()
    return [_intent(row) for row in rows]


def take_all_tool_intents(conn: sqlite3.Connection) -> list[ToolIntent]:
    """Remove and return every intent: called once at startup, when no call can be running."""
    intents = list_tool_intents(conn)
    conn.execute("DELETE FROM dots_tool_intents")
    return intents


# ---------------------------------------------------------------------------
# Browser identities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BrowserIdentityRow:
    id: str
    name: str
    # The proxy as the person gave it, password included: the browser needs it at every launch, as it is. Nothing of
    # it is shown to a model, a person, an event or a log: they are told only that there is one.
    proxy: str | None
    created_at: int
    last_used_at: int | None
    archived: bool


_IDENTITY_COLUMNS = "id, name, proxy, created_at, last_used_at, archived"


def _identity(row: sqlite3.Row) -> BrowserIdentityRow:
    return BrowserIdentityRow(
        row["id"],
        row["name"],
        row["proxy"],
        int(row["created_at"]),
        None if row["last_used_at"] is None else int(row["last_used_at"]),
        bool(row["archived"]),
    )


def insert_identity(
    conn: sqlite3.Connection,
    *,
    identity_id: str,
    name: str,
    proxy: str | None = None,
    now_ms: int | None = None,
) -> bool:
    """Record a new identity, never used and not archived; False when its id is known already.

    The rules a name, a proxy and an id meet are `nanobot.dots.identity_rules`'s: the caller checks
    them, this stores what it is given.
    """
    cursor = conn.execute(
        f"""
        INSERT INTO dots_browser_identities ({_IDENTITY_COLUMNS}) VALUES (?, ?, ?, ?, NULL, 0)
        ON CONFLICT (id) DO NOTHING
        """,
        (identity_id, name, proxy, clock_ms() if now_ms is None else now_ms),
    )
    return cursor.rowcount == 1


def get_identity(conn: sqlite3.Connection, identity_id: str) -> BrowserIdentityRow | None:
    row = conn.execute(
        f"SELECT {_IDENTITY_COLUMNS} FROM dots_browser_identities WHERE id = ?", (identity_id,)
    ).fetchone()
    return _identity(row) if row else None


def list_identities(conn: sqlite3.Connection) -> list[BrowserIdentityRow]:
    """Every identity, oldest first."""
    rows = conn.execute(f"SELECT {_IDENTITY_COLUMNS} FROM dots_browser_identities ORDER BY created_at, id").fetchall()
    return [_identity(row) for row in rows]


def count_identities(conn: sqlite3.Connection) -> int:
    """How many identities exist (archived ones too: they still hold a profile)."""
    return int(conn.execute("SELECT count(*) FROM dots_browser_identities").fetchone()[0])


def touch_identity(conn: sqlite3.Connection, identity_id: str, now_ms: int | None = None) -> bool:
    """Record that an identity was launched now; False when there is no such identity."""
    cursor = conn.execute(
        "UPDATE dots_browser_identities SET last_used_at = ? WHERE id = ?",
        (clock_ms() if now_ms is None else now_ms, identity_id),
    )
    return cursor.rowcount == 1


def set_identity_archived(conn: sqlite3.Connection, identity_id: str, archived: bool) -> bool:
    """Archive or restore an identity; False when there is no such identity."""
    cursor = conn.execute("UPDATE dots_browser_identities SET archived = ? WHERE id = ?", (int(archived), identity_id))
    return cursor.rowcount == 1


def delete_identity(conn: sqlite3.Connection, identity_id: str) -> bool:
    """Remove an identity's row; False when there is no such identity. Its profile directory is the caller's."""
    return conn.execute("DELETE FROM dots_browser_identities WHERE id = ?", (identity_id,)).rowcount == 1


# ---------------------------------------------------------------------------
# Approvals
# ---------------------------------------------------------------------------

# Where an approval stands. Every step commits before the next one starts:
# - pending: the host was asked; the turn that made the call has ended;
# - approved / rejected: the decision is recorded, the session not told yet;
# - granted / told: the session is being told, in a turn of its own (an approved
#   call is to be made again, with exactly its arguments; a rejected one did not
#   run). A task's turn that ends without the call ends the approval; one cut by
#   a stop is told again at the next start;
# - running: the session made the approved call again and the gate let it
#   through (`run_tool_call_id`); it runs once;
# - done: the call's result reached the transcript, or the session was told and
#   nothing more is owed.
ApprovalStatus = Literal["pending", "approved", "rejected", "granted", "told", "running", "done"]

# The statuses in which an approval still holds its session: a parked task waits for it.
_OPEN_APPROVAL_STATUSES = "'pending', 'approved', 'rejected', 'granted', 'told', 'running'"


@dataclass(frozen=True)
class Approval:
    approval_id: str
    session_key: str
    task_id: str | None
    tool_call_id: str
    tool: str
    permission: str
    arguments: dict[str, Any]
    status: ApprovalStatus
    note: str | None
    # The call that ran it, once the session made the approved call again.
    run_tool_call_id: str | None
    created_at: int
    resolved_at: int | None


_APPROVAL_COLUMNS = (
    "approval_id, session_key, task_id, tool_call_id, tool, permission, arguments_json, "
    "status, note, run_tool_call_id, created_at, resolved_at"
)


def _approval(row: sqlite3.Row) -> Approval:
    return Approval(
        approval_id=row["approval_id"],
        session_key=row["session_key"],
        task_id=row["task_id"],
        tool_call_id=row["tool_call_id"],
        tool=row["tool"],
        permission=row["permission"],
        arguments=json.loads(row["arguments_json"]),
        status=row["status"],
        note=row["note"],
        run_tool_call_id=row["run_tool_call_id"],
        created_at=int(row["created_at"]),
        resolved_at=None if row["resolved_at"] is None else int(row["resolved_at"]),
    )


def canonical_arguments(value: object) -> str:
    """JSON with object keys sorted at every level: two argument objects are the same call when this is equal.

    The one owner of what "the same arguments" means: an approval holds its arguments in this form, and
    the gate compares a call with the approved one by it. It is applied to arguments that went through
    their tool's schema (cast and validated), where an integer field cannot arrive as 1.0, so 1 and 1.0
    never have to be told apart (tests/dots/test_permissions.py walks every tool of the table).
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def request_approval(
    conn: sqlite3.Connection,
    *,
    session_key: str,
    task_id: str | None,
    tool_call_id: str,
    tool: str,
    permission: str,
    arguments: Mapping[str, Any],
    now_ms: int | None = None,
) -> tuple[Approval, bool]:
    """The approval of a tool call, created once per call that waits.

    The same call asked again while its approval is still pending (same session, id, tool and
    arguments: a retried turn) returns that approval, with `created` False. A call that only shares
    its id with an earlier one (a provider that numbers calls "call_0" in every response) is a new
    call and gets its own approval: once an approval is decided, its id belongs to the past.
    """
    arguments_json = canonical_arguments(dict(arguments))
    existing = conn.execute(
        f"""
        SELECT {_APPROVAL_COLUMNS} FROM dots_approvals
        WHERE session_key = ? AND tool_call_id = ? AND tool = ? AND arguments_json = ? AND status = 'pending'
        """,
        (session_key, tool_call_id, tool, arguments_json),
    ).fetchone()
    if existing is not None:
        return _approval(existing), False
    approval_id = f"appr_{uuid.uuid4()}"
    conn.execute(
        f"""
        INSERT INTO dots_approvals ({_APPROVAL_COLUMNS})
        VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', NULL, NULL, ?, NULL)
        """,
        (
            approval_id,
            session_key,
            task_id,
            tool_call_id,
            tool,
            permission,
            arguments_json,
            clock_ms() if now_ms is None else now_ms,
        ),
    )
    created = get_approval(conn, approval_id)
    assert created is not None
    return created, True


def get_approval(conn: sqlite3.Connection, approval_id: str) -> Approval | None:
    row = conn.execute(
        f"SELECT {_APPROVAL_COLUMNS} FROM dots_approvals WHERE approval_id = ?", (approval_id,)
    ).fetchone()
    return _approval(row) if row else None


def get_approval_by_tool_call(conn: sqlite3.Connection, session_key: str, tool_call_id: str) -> Approval | None:
    """The approval the newest call with this id in `session_key` asked for."""
    row = conn.execute(
        f"""
        SELECT {_APPROVAL_COLUMNS} FROM dots_approvals WHERE session_key = ? AND tool_call_id = ?
        ORDER BY created_at DESC, rowid DESC LIMIT 1
        """,
        (session_key, tool_call_id),
    ).fetchone()
    return _approval(row) if row else None


def list_approvals(conn: sqlite3.Connection, status: ApprovalStatus) -> list[Approval]:
    """Approvals in one status, oldest first: by time, then in the order they were inserted (an id is random, so
    within one millisecond it would not say which came first)."""
    rows = conn.execute(
        f"SELECT {_APPROVAL_COLUMNS} FROM dots_approvals WHERE status = ? ORDER BY created_at, rowid",
        (status,),
    ).fetchall()
    return [_approval(row) for row in rows]


def open_approval_for_session(conn: sqlite3.Connection, session_key: str) -> Approval | None:
    """The oldest approval that still holds a session."""
    row = conn.execute(
        f"""
        SELECT {_APPROVAL_COLUMNS} FROM dots_approvals
        WHERE session_key = ? AND status IN ({_OPEN_APPROVAL_STATUSES})
        ORDER BY created_at, rowid LIMIT 1
        """,
        (session_key,),
    ).fetchone()
    return _approval(row) if row else None


def advance_approval(
    conn: sqlite3.Connection,
    approval_id: str,
    from_status: ApprovalStatus,
    to_status: ApprovalStatus,
    *,
    note: str | None = None,
    run_tool_call_id: str | None = None,
    now_ms: int | None = None,
) -> bool:
    """Move an approval from one status to the next.

    Returns False, changing nothing, when it is not in `from_status` (a decision
    for an approval already decided, or a step a concurrent path already took).
    """
    now = clock_ms() if now_ms is None else now_ms
    cursor = conn.execute(
        """
        UPDATE dots_approvals
        SET status = ?, note = COALESCE(?, note), run_tool_call_id = COALESCE(?, run_tool_call_id),
            resolved_at = CASE WHEN ? IN ('approved', 'rejected') THEN ? ELSE resolved_at END
        WHERE approval_id = ? AND status = ?
        """,
        (to_status, note, run_tool_call_id, to_status, now, approval_id, from_status),
    )
    return cursor.rowcount == 1


def finish_approval_run(conn: sqlite3.Connection, session_key: str, tool_call_id: str) -> bool:
    """The approved call `tool_call_id` of `session_key` ran and its result is in the transcript: the approval is done."""
    cursor = conn.execute(
        "UPDATE dots_approvals SET status = 'done' "
        "WHERE session_key = ? AND run_tool_call_id = ? AND status = 'running'",
        (session_key, tool_call_id),
    )
    return cursor.rowcount == 1


def return_approval_run(conn: sqlite3.Connection, session_key: str, tool_call_id: str) -> bool:
    """The gate let the approved call `tool_call_id` of `session_key` through and it never started: approved again, so the session is told again."""
    cursor = conn.execute(
        "UPDATE dots_approvals SET status = 'approved' "
        "WHERE session_key = ? AND run_tool_call_id = ? AND status = 'running'",
        (session_key, tool_call_id),
    )
    return cursor.rowcount == 1


def end_waiting_approvals(conn: sqlite3.Connection, session_key: str) -> int:
    """The calls of `session_key` still waiting for a decision end undecided: its task was cancelled, and the
    host expires their approvals, so no decision will come."""
    cursor = conn.execute(
        "UPDATE dots_approvals SET status = 'done' WHERE session_key = ? AND status = 'pending'",
        (session_key,),
    )
    return cursor.rowcount


def end_approval_telling(conn: sqlite3.Connection, session_key: str) -> bool:
    """The turn that told `session_key` of a decision gave its final answer: the approval is done."""
    cursor = conn.execute(
        "UPDATE dots_approvals SET status = 'done' WHERE session_key = ? AND status IN ('granted', 'told')",
        (session_key,),
    )
    return cursor.rowcount > 0


# ---------------------------------------------------------------------------
# Gate decisions
# ---------------------------------------------------------------------------

# What the gate decided for a call, kept until the call's result reaches the
# transcript, so `tool.called` reports the real decision:
# - deny: blocked, not run;
# - ask: an approved call that ran;
# - park: waiting for approval, no `tool.called` since it did not run (the call
#   that runs it later reports itself);
# - skipped: not run because an earlier call of the same response parked, no
#   `tool.called` either.
# A call with no row was allowed.
ToolDecision = Literal["deny", "ask", "park", "skipped"]


def record_tool_decision(conn: sqlite3.Connection, session_key: str, tool_call_id: str, decision: ToolDecision) -> None:
    if decision not in get_args(ToolDecision):
        raise ValueError(f"not a tool decision: {decision}")
    conn.execute(
        """
        INSERT INTO dots_tool_decisions (session_key, tool_call_id, decision) VALUES (?, ?, ?)
        ON CONFLICT (session_key, tool_call_id) DO UPDATE SET decision = excluded.decision
        """,
        (session_key, tool_call_id, decision),
    )


def peek_tool_decision(conn: sqlite3.Connection, session_key: str, tool_call_id: str) -> ToolDecision | None:
    """The decision recorded for a call of a session, left in place; None means it was allowed."""
    row = conn.execute(
        "SELECT decision FROM dots_tool_decisions WHERE session_key = ? AND tool_call_id = ?",
        (session_key, tool_call_id),
    ).fetchone()
    return row["decision"] if row else None


def take_tool_decision(conn: sqlite3.Connection, session_key: str, tool_call_id: str) -> ToolDecision | None:
    """Remove and return the decision recorded for a call of a session; None means it was allowed."""
    decision = peek_tool_decision(conn, session_key, tool_call_id)
    if decision is not None:
        conn.execute(
            "DELETE FROM dots_tool_decisions WHERE session_key = ? AND tool_call_id = ?", (session_key, tool_call_id)
        )
    return decision


# ---------------------------------------------------------------------------
# Transcripts
# ---------------------------------------------------------------------------


def append_messages(
    conn: sqlite3.Connection,
    session_key: str,
    messages: Sequence[Mapping[str, Any]],
    *,
    final_index: int | None,
) -> None:
    """Append messages to a session's transcript, each with the outbox rows it earns.

    `final_index` names the message of this call that is the final answer of
    the turn (an index into `messages`), or None. The transcript row and the
    outbox row it describes commit together or not at all, because both are
    written in the caller's transaction.
    """
    if final_index is not None and not 0 <= final_index < len(messages):
        raise ValueError(f"final_index {final_index} is outside the {len(messages)} messages")
    next_idx = int(
        conn.execute(
            "SELECT COALESCE(MAX(idx), -1) + 1 FROM messages WHERE session_key = ?", (session_key,)
        ).fetchone()[0]
    )
    for offset, message in enumerate(messages):
        conn.execute(
            "INSERT INTO messages (session_key, idx, message_json) VALUES (?, ?, ?)",
            (session_key, next_idx + offset, _dumps(dict(message))),
        )
        transcript_outbox.record_transcript_append(conn, session_key, message, final=(offset == final_index))
    conn.execute(
        """
        INSERT INTO sessions (key, metadata_json, updated_at) VALUES (?, '{}', ?)
        ON CONFLICT (key) DO UPDATE SET updated_at = excluded.updated_at
        """,
        (session_key, clock_ms()),
    )


def read_messages(conn: sqlite3.Connection, session_key: str) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT message_json FROM messages WHERE session_key = ? ORDER BY idx", (session_key,))
    return [json.loads(row["message_json"]) for row in rows]


# Where a session's last_consolidated offset rides in the metadata row (Session keeps it as a field).
_LAST_CONSOLIDATED = "last_consolidated"


def read_session_metadata(conn: sqlite3.Connection, session_key: str) -> dict[str, Any]:
    row = conn.execute("SELECT metadata_json FROM sessions WHERE key = ?", (session_key,)).fetchone()
    return json.loads(row["metadata_json"]) if row else {}


def write_session_metadata(
    conn: sqlite3.Connection, session_key: str, metadata: Mapping[str, Any], now_ms: int | None = None
) -> None:
    conn.execute(
        """
        INSERT INTO sessions (key, metadata_json, updated_at) VALUES (?, ?, ?)
        ON CONFLICT (key) DO UPDATE SET metadata_json = excluded.metadata_json, updated_at = excluded.updated_at
        """,
        (session_key, _dumps(dict(metadata)), clock_ms() if now_ms is None else now_ms),
    )


def load_session(conn: sqlite3.Connection, session_key: str) -> Session:
    """The session as the replay reads it: every message and the metadata row."""
    metadata = read_session_metadata(conn, session_key)
    last_consolidated = metadata.pop(_LAST_CONSOLIDATED, 0)
    return Session(
        key=session_key,
        messages=read_messages(conn, session_key),
        metadata=metadata,
        last_consolidated=last_consolidated,
    )


def commit_summary_checkpoint(conn: sqlite3.Connection, session_key: str, summary: str, boundary: int) -> None:
    """Replace the replay of the messages before `boundary` with `summary`.

    The checkpoint is a hidden marker message inserted at `boundary` (the rows from
    there on move up by one) and the summary and offset in the metadata row:
    `Session.commit_summary_checkpoint` decides what that is, this stores it.
    """
    session = load_session(conn, session_key)
    if not session.last_consolidated <= boundary <= len(session.messages):
        raise ValueError(
            f"summary boundary {boundary} is outside [{session.last_consolidated}, {len(session.messages)}]"
        )
    session.commit_summary_checkpoint(summary, insert_at=boundary)
    # Two steps through negative indices: a shift in place would collide with the primary key.
    conn.execute(
        "UPDATE messages SET idx = -idx - 2 WHERE session_key = ? AND idx >= ?", (session_key, boundary)
    )
    conn.execute("UPDATE messages SET idx = -idx - 1 WHERE session_key = ? AND idx < 0", (session_key,))
    conn.execute(
        "INSERT INTO messages (session_key, idx, message_json) VALUES (?, ?, ?)",
        (session_key, boundary, _dumps(session.messages[boundary])),
    )
    write_session_metadata(
        conn, session_key, {**session.metadata, _LAST_CONSOLIDATED: session.last_consolidated}
    )


def open_tool_calls(conn: sqlite3.Connection, session_key: str) -> list[dict[str, Any]]:
    """The tool calls of the newest assistant message that have no tool result after it."""
    answered: set[str] = set()
    rows = conn.execute("SELECT message_json FROM messages WHERE session_key = ? ORDER BY idx DESC", (session_key,))
    for row in rows:
        message = json.loads(row["message_json"])
        if message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if isinstance(call_id, str):
                answered.add(call_id)
        elif message.get("role") == "assistant":
            calls = message.get("tool_calls") or []
            return [call for call in calls if isinstance(call.get("id"), str) and call["id"] not in answered]
    return []
