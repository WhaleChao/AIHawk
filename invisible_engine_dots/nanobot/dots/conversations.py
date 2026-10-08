"""The Dot's past conversations as files on its own computer, which it searches when it needs what was said.

The transcripts live in the engine's database, which the model's commands cannot read, and the model is
given a long chat's older part only as a summary. So what the chat and every task said is written to
CONVERSATIONS_DIR as well, where the Dot greps it like any file:

- `chat/<day>.md`, one file a day of the chat;
- `tasks/<day>-<task id>.md`, one file a task, named by the day it started.

A file holds what the person and the Dot said, with the time, and a line for each call the Dot made, not
what the calls returned (that is on the computer, or can be looked at again). It is written whole from
the transcript each time, so it is the same however many turns made it. Searching files with grep is
what remembers best in the published measures (the agent-memory benchmarks of LongMemEval, the "Is Grep
All You Need?" study): better than notes a model extracts, a graph, or embeddings.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from nanobot.dots.transcript_outbox import APPROVAL_ID, INBOUND_ID, message_metadata, message_text
from nanobot.session.history_visibility import is_hidden_history_message

CONVERSATIONS_DIR = "/home/dot/conversations"

# dots_kv: how many messages of the chat are already in its files (absent: nothing written yet, not even
# the tasks, which the first write after an upgrade writes too).
KV_CHAT_WRITTEN = "conversations_chat_written"

Target = Callable[[str, Any], str | None]


def day_of(message: Mapping[str, Any]) -> str | None:
    stamp = message.get("timestamp")
    return stamp[:10] if isinstance(stamp, str) and len(stamp) >= 10 else None


def _time_of(message: Mapping[str, Any]) -> str:
    stamp = message.get("timestamp")
    return stamp[11:16] if isinstance(stamp, str) and len(stamp) >= 16 else ""


def _speaker(message: Mapping[str, Any], first: bool, task: bool) -> str | None:
    """Who a user message is from, or None for one the engine wrote to steer the model."""
    metadata = message_metadata(message)
    if APPROVAL_ID in metadata:
        return "approval"
    if INBOUND_ID in metadata:
        return "automation" if message_text(message).startswith("[Automation ") else "the person"
    if task and first:
        return "the task"
    return None


def _calls(message: Mapping[str, Any], target: Target) -> list[str]:
    lines = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") if isinstance(call, Mapping) else None
        if not isinstance(function, Mapping) or not isinstance(function.get("name"), str):
            continue
        name = function["name"]
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except (TypeError, ValueError):
            arguments = None
        what = target(name, arguments)
        lines.append(f"- called {name}: {what}" if what else f"- called {name}")
    return lines


def render(title: str, messages: Sequence[Mapping[str, Any]], *, task: bool, target: Target) -> str:
    """The file of a day of the chat or of a task: `title`, then each message said, oldest first."""
    parts = [f"# {title}"]
    for index, message in enumerate(messages):
        if is_hidden_history_message(message):
            continue
        role = message.get("role")
        if role == "user":
            speaker = _speaker(message, index == 0, task)
            text = message_text(message)
            if speaker is None or not text:
                continue
            parts.append(f"## {_time_of(message)} {speaker}\n\n{text}")
        elif role == "assistant":
            text = message_text(message)
            calls = _calls(message, target)
            if not text and not calls:
                continue
            body = "\n\n".join(part for part in (text, "\n".join(calls)) if part)
            parts.append(f"## {_time_of(message)} you\n\n{body}")
    return "\n\n".join(parts) + "\n"


def chat_files(
    messages: Sequence[Mapping[str, Any]], since: int, *, target: Target
) -> dict[str, str]:
    """The chat's files that messages[since:] added to or changed, by path under CONVERSATIONS_DIR."""
    days = sorted({day for message in messages[since:] if (day := day_of(message))})
    return {
        f"chat/{day}.md": render(
            f"Chat, {day}", [m for m in messages if day_of(m) == day], task=False, target=target
        )
        for day in days
    }


def task_file(
    task_id: str, status: str, messages: Sequence[Mapping[str, Any]], *, target: Target
) -> dict[str, str]:
    """A task's file, by path under CONVERSATIONS_DIR (nothing for a task with no messages yet)."""
    day = next((day for message in messages if (day := day_of(message))), None)
    if day is None:
        return {}
    return {f"tasks/{day}-{task_id}.md": render(f"Task {task_id} ({status}), {day}", messages, task=True, target=target)}

