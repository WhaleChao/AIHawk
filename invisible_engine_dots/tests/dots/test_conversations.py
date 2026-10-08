"""The Dot's past conversations as files on its computer: what a file holds, and when a turn writes it."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from typing import Any

from fakes.scripted_provider import call, calls, says
from fakes.turn_harness import Harness

from nanobot.agent.transcript_metadata import METADATA_KEY
from nanobot.dots import conversations
from nanobot.dots import store as s
from nanobot.dots.permissions import tool_target
from nanobot.dots.transcript_outbox import APPROVAL_ID, INBOUND_ID
from nanobot.dots.turns import OpeningMessage, TurnUnit
from nanobot.session.history_visibility import HIDDEN_HISTORY_META
from nanobot.session.summary import SUMMARY_CONTINUATION_TEXT

MakeHarness = Callable[..., Harness]
CHAT = s.CHAT_SESSION_KEY


def person(text: str, at: str, inbound_id: str = "in") -> dict[str, Any]:
    return {"timestamp": at, "role": "user", "content": text, METADATA_KEY: {INBOUND_ID: inbound_id}}


def dot(text: str, at: str, *calls: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    message: dict[str, Any] = {"timestamp": at, "role": "assistant", "content": text}
    if calls:
        message["tool_calls"] = [
            {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
            for i, (name, arguments) in enumerate(calls)
        ]
    return message


def no_target(name: str, arguments: Any) -> str | None:
    return None


class TestAFile:
    def test_the_chat_is_a_file_a_day_with_who_said_what_and_when(self) -> None:
        messages = [
            person("my cat is called Luna", "2023-05-20T09:15:00"),
            dot("Nice name.", "2023-05-20T09:15:30"),
            person("what was her name?", "2023-05-21T18:02:00"),
            dot("Luna.", "2023-05-21T18:02:10"),
        ]

        files = conversations.chat_files(messages, 0, target=no_target)

        assert files == {
            "chat/2023-05-20.md": "# Chat, 2023-05-20\n\n## 09:15 the person\n\nmy cat is called Luna\n\n## 09:15 you\n\nNice name.\n",
            "chat/2023-05-21.md": "# Chat, 2023-05-21\n\n## 18:02 the person\n\nwhat was her name?\n\n## 18:02 you\n\nLuna.\n",
        }

    def test_only_the_days_of_new_messages_are_written_each_whole(self) -> None:
        messages = [
            person("one", "2023-05-20T09:00:00"),
            dot("a", "2023-05-20T09:00:01"),
            person("two", "2023-05-21T09:00:00"),
            dot("b", "2023-05-21T09:00:01"),
            person("three", "2023-05-21T10:00:00"),
        ]

        files = conversations.chat_files(messages, 3, target=no_target)

        assert list(files) == ["chat/2023-05-21.md"]
        assert "two" in files["chat/2023-05-21.md"] and "three" in files["chat/2023-05-21.md"]

    def test_a_call_is_a_line_and_what_it_returned_is_left_out(self) -> None:
        messages = [
            person("list it", "2023-05-20T09:00:00"),
            dot("", "2023-05-20T09:00:01", ("exec", {"command": "ls -la /srv"}), ("read_file", {"path": "notes.md"})),
            {"timestamp": "2023-05-20T09:00:02", "role": "tool", "tool_call_id": "c0", "content": "SECRET OUTPUT"},
            dot("Two files.", "2023-05-20T09:00:03"),
        ]

        (text,) = conversations.chat_files(messages, 0, target=tool_target).values()

        assert "- called exec: ls -la /srv\n- called read_file: notes.md" in text
        assert "SECRET OUTPUT" not in text
        assert text.endswith("## 09:00 you\n\nTwo files.\n")

    def test_what_the_engine_tells_the_model_is_left_out_and_decisions_and_automations_are_named(self) -> None:
        messages = [
            {"timestamp": "2023-05-20T09:00:00", "role": "user", "content": "Please provide your response."},
            {"timestamp": "2023-05-20T09:00:00", "role": "user", "content": SUMMARY_CONTINUATION_TEXT, HIDDEN_HISTORY_META: True},
            {
                "timestamp": "2023-05-20T09:01:00",
                "role": "user",
                "content": "[The user approved your exec call (ap1).]",
                METADATA_KEY: {APPROVAL_ID: "ap1"},
            },
            person('[Automation "fares" fired] check the fares', "2023-05-20T09:02:00"),
        ]

        (text,) = conversations.chat_files(messages, 0, target=no_target).values()

        assert "Please provide" not in text
        assert SUMMARY_CONTINUATION_TEXT not in text
        assert "## 09:01 approval\n\n[The user approved your exec call (ap1).]" in text
        assert '## 09:02 automation\n\n[Automation "fares" fired] check the fares' in text

    def test_a_task_is_a_file_named_by_the_day_it_started(self) -> None:
        messages = [
            {"timestamp": "2023-05-20T23:59:00", "role": "user", "content": "find the cheapest fare"},
            dot("The cheapest is 120 EUR.", "2023-05-21T00:03:00"),
        ]

        files = conversations.task_file("t1", "completed", messages, target=no_target)

        assert files == {
            "tasks/2023-05-20-t1.md": "# Task t1 (completed), 2023-05-20\n\n## 23:59 the task\n\nfind the cheapest fare\n\n"
            "## 00:03 you\n\nThe cheapest is 120 EUR.\n"
        }
        assert conversations.task_file("t2", "queued", [], target=no_target) == {}


def read(h: Harness, path: str) -> str | None:
    local = h.computer._local(f"{conversations.CONVERSATIONS_DIR}/{path}")
    return local.read_text(encoding="utf-8") if local.exists() else None


def chat_unit(text: str, inbound_id: str) -> TurnUnit:
    return TurnUnit(CHAT, None, (OpeningMessage(text, {INBOUND_ID: inbound_id}),))


class TestATurnWritesThem:
    async def test_a_chat_turn_writes_its_day_and_the_next_one_adds_to_it(self, make_harness: MakeHarness) -> None:
        h = make_harness([says("Noted: Luna."), says("Luna.")])
        today = datetime.now().strftime("%Y-%m-%d")
        h.accept("in1", "my cat is called Luna")
        await h.run(chat_unit("my cat is called Luna", "in1"))
        h.accept("in2", "her name?")
        await h.run(chat_unit("her name?", "in2"))

        text = read(h, f"chat/{today}.md")
        assert text is not None
        assert [line for line in text.splitlines() if line and not line.startswith("#")] == [
            "my cat is called Luna",
            "Noted: Luna.",
            "her name?",
            "Luna.",
        ]
        assert h.store.read(lambda conn: s.read_kv(conn, conversations.KV_CHAT_WRITTEN)) == 4

    async def test_a_task_turn_writes_the_task(self, make_harness: MakeHarness) -> None:
        h = make_harness([calls(call("c1", "list_dir", path=".")), says("It is empty.")])
        session_key = h.start_task("t1")

        await h.run(TurnUnit(session_key, "t1", (OpeningMessage("look at the workspace"),)))

        today = datetime.now().strftime("%Y-%m-%d")
        text = read(h, f"tasks/{today}-t1.md")
        assert text is not None
        assert text.startswith("# Task t1 (completed)")
        assert "the task\n\nlook at the workspace" in text and "- called list_dir: ." in text and "It is empty." in text

    async def test_the_first_turn_after_an_upgrade_writes_the_past_chat_and_tasks(self, make_harness: MakeHarness) -> None:
        h = make_harness([says("Hello again.")])
        old_chat = [person("remember the blue door", "2023-01-02T10:00:00"), dot("I will.", "2023-01-02T10:00:05")]
        h.store.write(lambda conn: s.append_messages(conn, CHAT, old_chat, final_index=None))
        h.store.write(lambda conn: s.enqueue_task(conn, task_id="old", description="paint it", priority=0))
        old_task = [{"timestamp": "2023-01-03T08:00:00", "role": "user", "content": "paint it"}, dot("Done.", "2023-01-03T08:30:00")]
        h.store.write(lambda conn: s.append_messages(conn, s.task_session_key("old"), old_task, final_index=None))
        h.accept("in1", "hi")

        await h.run(chat_unit("hi", "in1"))

        assert "remember the blue door" in (read(h, "chat/2023-01-02.md") or "")
        assert "paint it" in (read(h, "tasks/2023-01-03-old.md") or "")

    async def test_a_file_that_cannot_be_written_is_written_by_a_later_turn(self, make_harness: MakeHarness) -> None:
        h = make_harness([says("one"), says("two")])
        today = datetime.now().strftime("%Y-%m-%d")
        blocked = h.computer._local(f"{conversations.CONVERSATIONS_DIR}/chat/{today}.md")
        blocked.mkdir(parents=True)
        h.accept("in1", "first")
        await h.run(chat_unit("first", "in1"))
        assert h.store.read(lambda conn: s.read_kv(conn, conversations.KV_CHAT_WRITTEN)) is None

        blocked.rmdir()
        h.accept("in2", "second")
        await h.run(chat_unit("second", "in2"))

        text = read(h, f"chat/{today}.md") or ""
        assert "first" in text and "second" in text

    async def test_the_prompt_says_where_they_are(self, make_harness: MakeHarness) -> None:
        h = make_harness([says("hi")])
        h.accept("in1", "hello")

        await h.run(chat_unit("hello", "in1"))

        system = h.provider.requests[0]["messages"][0]["content"]
        assert f"kept in {conversations.CONVERSATIONS_DIR}" in system
