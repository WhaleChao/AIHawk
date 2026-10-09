"""Outbox rows written in the transaction that appends a transcript message.

The event and the state change it describes commit together or not at all
(architecture section 8.7). `DotStore.append_messages` calls
`record_transcript_append` once a message row is inserted, in the caller's
transaction: everything it needs is read from the same database, nothing from
this process's memory.

- a user message of the chat that carries the id of an inbound event: that
  event's text is in the transcript now, so a restart must not send it again;
- the final assistant message of the chat: `message.assistant`, answering every
  input the transcript holds unanswered (`in_reply_to` names the newest
  user.message);
- the final assistant message of a running task: the task completes
  (`task.completed`, the text is the summary);
- either final answer, when the turn was telling the session of an approval's
  decision: that approval is done, in the same transaction;
- an assistant message of a running task that is not the final answer, has tool
  calls and text beside them: `task.progress` with that text (the model saying
  what it is about to do), never for the chat and never for the final answer;
- a tool result in the chat or in a task: `tool.called`, with the duration
  measured from the call's intent row, the decision the gate recorded for
  it (gate.py) and the target the intent holds (permissions.tool_target: what
  the call acted on, redacted; a call that never started, such as a denied
  one, has no intent and so no target), and `tty: true` when the intent says the call started a
  terminal session (permissions.tool_starts_terminal); a call that did not run (parked,
  skipped, closed as not run) gets none.

`message.assistant`, `task.progress` and `task.completed` carry `spent_usd`, what the session has spent so
far (`store.append_outbox_spent`), read in the same transaction.

Metadata of a message travels inside the message dict under one key, `_dots`.
The runner keeps it out of the transcript the model reads (`AgentRunner._commit`), and
the replay of a stored session copies only the keys of a model message
(`Session.get_history`), so it is never sent.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from typing import Any

from nanobot.agent.transcript_metadata import IS_ERROR, METADATA_KEY
from nanobot.dots import store
from nanobot.dots.permissions import tool_permission

# user message: the id of the inbound event whose text it is.
INBOUND_ID = "dots_inbound_id"
# user message: the id of the approval whose decision it tells the session.
APPROVAL_ID = "dots_approval_id"
# user message: what the Dot's conversation files say of that decision (the call and what it acted on, never its
# arguments, which may hold a secret the events mask).
APPROVAL_LINE = "dots_approval_line"
# tool result the engine wrote itself to close a call that has no result of its own.
CLOSED = "dots_closed"
CLOSED_INTERRUPTED = "interrupted"
CLOSED_NOT_RUN = "not_run"

# The longest text of a `task.progress` event, in characters, the ellipsis included.
PROGRESS_TEXT_MAX = 2000

# What the result of a call closed by the engine says to the model (gate.close_open_calls).
CLOSED_INTERRUPTED_TEXT = (
    "This call was interrupted before its result was recorded. It may have taken effect, "
    "and it may still be running. Check the current state before calling it again."
)
CLOSED_APPROVAL_USED_TEXT = " The approval was used."
CLOSED_DENIED_TEXT = "The Dot's policy denied this call."
CLOSED_NOT_RUN_TEXT = "Not executed: the unit ended before this call ran."


def message_metadata(message: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = message.get(METADATA_KEY)
    return metadata if isinstance(metadata, Mapping) else {}


def message_text(message: Mapping[str, Any]) -> str:
    """The text of a message: its string content, or its text parts joined, stripped."""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    return "".join(
        part["text"]
        for part in content
        if isinstance(part, Mapping) and part.get("type") == "text" and isinstance(part.get("text"), str)
    ).strip()


def tool_result_is_error(message: Mapping[str, Any]) -> bool:
    """Whether a tool result reports a failure: flagged in the metadata, or text starting with "Error".

    The one definition: the runner flags its results with it and this module
    reports with it.
    """
    return message_metadata(message).get(IS_ERROR) is True or message_text(message).startswith("Error")


def record_transcript_append(
    conn: sqlite3.Connection,
    session_key: str,
    message: Mapping[str, Any],
    *,
    final: bool,
    now_ms: int | None = None,
) -> None:
    """Record what a newly appended transcript message means for the Dot. Raises only on database errors."""
    is_chat = session_key == store.CHAT_SESSION_KEY
    task = None if is_chat else store.get_task_by_session(conn, session_key)
    if not is_chat and task is None:
        return
    role = message.get("role")

    if role == "user":
        inbound_id = message_metadata(message).get(INBOUND_ID)
        if is_chat and isinstance(inbound_id, str) and inbound_id:
            store.mark_inbound_in_transcript(conn, inbound_id)
        return

    if role == "tool":
        _record_tool_result(conn, session_key, message, task, store.clock_ms() if now_ms is None else now_ms)
        return

    if role != "assistant":
        return
    if not final:
        if task is not None and task.status == "running" and message.get("tool_calls"):
            _record_progress(conn, session_key, task.task_id, message_text(message))
        return
    text = message_text(message)
    # The turn that told the session of a decision ends with this answer: the answer and the end of
    # the approval commit together, or a restart would tell the session again.
    store.end_approval_telling(conn, session_key)
    if is_chat:
        answered = store.apply_answered_inputs(conn)
        reply = {"in_reply_to": answered[-1]} if answered else {}
        store.append_outbox_spent(conn, "message.assistant", {"text": text, **reply}, session_key)
        return
    if task is not None and task.status == "running":
        if store.finish_task(conn, task.task_id, "completed", summary=text):
            store.append_outbox_spent(conn, "task.completed", {"task_id": task.task_id, "summary": text}, session_key)


def _record_progress(conn: sqlite3.Connection, session_key: str, task_id: str, text: str) -> None:
    if not text:
        return
    if len(text) > PROGRESS_TEXT_MAX:
        text = text[: PROGRESS_TEXT_MAX - 1] + "…"
    store.append_outbox_spent(conn, "task.progress", {"task_id": task_id, "text": text}, session_key)


def _record_tool_result(
    conn: sqlite3.Connection, session_key: str, message: Mapping[str, Any], task: store.TaskRow | None, now_ms: int
) -> None:
    tool_call_id = message.get("tool_call_id")
    call_id = tool_call_id if isinstance(tool_call_id, str) else ""
    name = message.get("name")
    tool = name if isinstance(name, str) and name else "unknown"
    intent = store.take_tool_intent(conn, session_key, call_id) if call_id else None
    decision = store.take_tool_decision(conn, session_key, call_id) if call_id else None
    closed = message_metadata(message).get(CLOSED)
    # A call that did not run reports nothing: one waiting for approval (the call
    # that runs it once approved reports itself, with "ask"), one skipped because
    # an earlier call of its response parked, one closed before it ever started.
    if decision in ("park", "skipped") or closed == CLOSED_NOT_RUN:
        return
    if decision == "ask":
        store.finish_approval_run(conn, session_key, call_id)
    interrupted = closed == CLOSED_INTERRUPTED
    event: dict[str, Any] = {
        **({"task_id": task.task_id} if task is not None else {}),
        "tool": tool,
        "permission": tool_permission(tool),
        "decision": decision or "allow",
        "ok": decision != "deny" and not interrupted and not tool_result_is_error(message),
        "duration_ms": 0 if interrupted or intent is None else max(0, now_ms - intent.started_at),
    }
    if intent is not None and intent.target:
        event["target"] = intent.target
    if intent is not None and intent.tty:
        event["tty"] = True
    if interrupted:
        event["interrupted"] = True
    store.append_outbox(conn, "tool.called", event)
