"""The engine: what runs when, and what the end of a turn means.

Every test runs the real engine on a real store with the real tools on a local
computer. Only the model is scripted. A turn that must not end is held at its
model request with a `Gate`; the tests wait on events, never on a clock.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Callable
from typing import Any

import pytest
from fakes.browser_manager import mcp_home
from fakes.dot_config import ALLOW_ALL, runtime_config_body
from fakes.engine_harness import (
    EngineHarness,
    decision,
    inbound,
    task_cancelled,
    task_created,
    user_message,
)
from fakes.fake_mcp_server import write_control
from fakes.scripted_provider import Gate, call, calls, says

from nanobot.agent.transcript_metadata import METADATA_KEY
from nanobot.cron.types import CronJob, CronJobState, CronPayload, CronSchedule
from nanobot.dots import engine as engine_module
from nanobot.dots import store as s
from nanobot.dots import turns as turns_module
from nanobot.dots.engine import (
    RESUME_TASK_NOTE,
    Engine,
    EngineStopped,
    approval_granted_continuation,
    approval_rejected_continuation,
)
from nanobot.dots.protocol import PREPARE_SLEEP_TIMEOUT_S, DotsConfigError
from nanobot.dots.transcript_outbox import CLOSED, CLOSED_INTERRUPTED, INBOUND_ID
from nanobot.providers.base import LLMResponse, ModelLimits

CHAT = s.CHAT_SESSION_KEY
MakeEngine = Callable[..., EngineHarness]


def cfg(permissions: dict[str, str] | None = None, **overrides: Any) -> dict[str, Any]:
    return runtime_config_body(permissions=ALLOW_ALL if permissions is None else permissions, **overrides)


def started(h: EngineHarness, permissions: dict[str, str] | None = None) -> EngineHarness:
    """The engine started and configured; it works as soon as there is something to do."""
    h.engine.start()
    h.configure(cfg(permissions))
    return h


def open_call(name: str = "exec", call_id: str = "c1") -> dict[str, Any]:
    """An assistant message whose one tool call has no result yet."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}],
    }


def park(
    h: EngineHarness,
    session_key: str,
    task_id: str | None,
    *,
    tool: str = "exec",
    call_id: str = "call-1",
    arguments: dict[str, Any] | None = None,
    now_ms: int | None = None,
) -> str:
    """An approval of a call the gate parked, as the gate records it."""
    approval, _ = h.store.write(
        lambda conn: s.request_approval(
            conn,
            session_key=session_key,
            task_id=task_id,
            tool_call_id=call_id,
            tool=tool,
            permission="computer.exec",
            arguments=arguments or {"command": "make clean", "cwd": "/home/dot/workspace"},
            now_ms=now_ms,
        )
    )
    return approval.approval_id


def user_texts(request: dict[str, Any]) -> list[str]:
    return [m["content"] for m in request["messages"] if m["role"] == "user"]


class TestStarting:
    async def test_announces_the_start_says_idle_and_reports_the_calls_the_last_process_stopped_during(
        self, make_engine: MakeEngine
    ) -> None:
        h = make_engine()
        session = s.task_session_key("t1")
        h.store.write(lambda c: s.enqueue_task(c, task_id="t1", description="x", priority=0))
        h.store.write(lambda c: s.start_task(c, "t1"))
        h.store.write(
            lambda c: s.append_messages(c, session, [{"role": "user", "content": "x"}, open_call()], final_index=None)
        )
        h.store.write(lambda c: s.record_tool_intent(c, s.ToolIntent("c1", "exec", session, "t1", 1)))

        h.engine.start()

        events = h.events()
        assert [e["type"] for e in events] == ["agent.started", "tool.called", "agent.state"]
        assert events[1]["data"] == {
            "task_id": "t1",
            "tool": "exec",
            "permission": "computer.exec",
            "decision": "allow",
            "ok": False,
            "duration_ms": 0,
            "interrupted": True,
        }
        assert events[2]["data"] == {"state": "IDLE"}
        closed = h.messages(session)[-1]
        assert (closed["role"], closed["tool_call_id"], closed[METADATA_KEY][CLOSED]) == (
            "tool",
            "c1",
            CLOSED_INTERRUPTED,
        )
        assert h.count("dots_tool_intents") == 0

    async def test_an_intent_whose_call_is_not_in_any_transcript_is_swept_silently(
        self, make_engine: MakeEngine
    ) -> None:
        h = make_engine()
        h.store.write(lambda c: s.record_tool_intent(c, s.ToolIntent("ghost", "exec", CHAT, None, 1)))

        h.engine.start()

        assert h.types() == ["agent.started", "agent.state"]
        assert h.count("dots_tool_intents") == 0

    async def test_a_call_of_the_chat_that_never_started_is_closed_without_an_event(
        self, make_engine: MakeEngine
    ) -> None:
        h = make_engine()
        h.store.write(
            lambda c: s.append_messages(c, CHAT, [{"role": "user", "content": "go"}, open_call("read_file")], final_index=None)
        )

        h.engine.start()

        assert h.types() == ["agent.started", "agent.state"]
        closed = h.messages()[-1]
        assert (closed["role"], closed["tool_call_id"], closed["name"]) == ("tool", "c1", "read_file")
        assert "unit ended before this call ran" in closed["content"]

    async def test_the_config_the_host_pushed_is_the_one_a_restarted_engine_works_with(
        self, make_engine: MakeEngine
    ) -> None:
        h = make_engine()
        h.engine.start()
        body = cfg(name="fare-watch")
        h.configure(body)
        # Stored as the host sent it: a field it left out is not null in the store.
        assert h.store.read(lambda c: s.read_kv(c, s.KV_RUNTIME_CONFIG)) == body

        restarted = h.restart()
        assert restarted.config is None
        restarted.start()

        assert restarted.config is not None and restarted.config.name == "fare-watch"
        assert h.types().count("agent.started") == 2

    async def test_a_stored_config_that_no_longer_parses_is_ignored_until_the_host_pushes_one(
        self, make_engine: MakeEngine
    ) -> None:
        h = make_engine()
        h.store.write(lambda c: s.write_kv(c, s.KV_RUNTIME_CONFIG, {"name": "BAD NAME"}))

        h.engine.start()

        assert h.engine.config is None
        assert h.engine.started


class TestTheChat:
    async def test_waits_for_the_config_and_the_key_then_answers_each_message_once(
        self, make_engine: MakeEngine
    ) -> None:
        h = make_engine([says("hi")], key=False)
        h.engine.start()

        assert h.engine.accept(user_message("m1")) is True
        assert h.engine.accept(user_message("m1")) is False
        await h.idle()
        assert h.asked() == 0
        h.configure(cfg())
        await h.idle()
        assert h.asked() == 0
        h.give_key()
        await h.idle()
        h.engine.kick()
        await h.idle()

        assert h.asked() == 1
        assert h.events_of("message.assistant") == [{"text": "hi", "in_reply_to": "m1", "spent_usd": 0.0}]
        assert h.inbound_state("m1") == "applied"

    async def test_every_message_accepted_before_the_turn_starts_opens_the_same_turn(
        self, make_engine: MakeEngine
    ) -> None:
        h = make_engine([says("both")], key=False)
        h.engine.start()
        h.configure(cfg())
        h.engine.accept(user_message("m1", "first"))
        h.engine.accept(user_message("m2", "second"))

        h.give_key()
        await h.idle()

        assert h.asked() == 1
        assert user_texts(h.provider.requests[0]) == ["first\n\nsecond"]
        assert h.events_of("message.assistant") == [{"text": "both", "in_reply_to": "m2", "spent_usd": 0.0}]

    async def test_a_turn_that_failed_still_answers_with_why(self, make_engine: MakeEngine) -> None:
        h = started(make_engine([LLMResponse(content="rate limited", finish_reason="error")]))

        h.engine.accept(user_message("m1"))
        await h.idle()

        assert h.events_of("message.assistant") == [{"text": "I could not answer: rate limited", "in_reply_to": "m1", "spent_usd": 0.0}]
        assert h.inbound_state("m1") == "applied"
        assert h.engine.state == "IDLE"
        # Nothing is owed any more: no turn follows.
        h.engine.kick()
        await h.idle()
        assert h.asked() == 1

    async def test_a_failure_with_nothing_owed_is_not_reported_as_an_unanswered_message(
        self, make_engine: MakeEngine
    ) -> None:
        h = started(make_engine([LLMResponse(content="rate limited", finish_reason="error")]))
        approval_id = park(h, CHAT, None)

        h.engine.accept(decision("d1", approval_id, "reject"))
        await h.idle()

        assert h.events_of("message.assistant") == []
        assert h.approval(approval_id).status == "done"

    async def test_follows_runs_and_tools_in_the_agent_state(self, make_engine: MakeEngine) -> None:
        h = started(make_engine([calls(call("c1", "list_dir", path=".")), says("done")]))

        h.engine.accept(user_message("m1"))
        await h.idle()

        assert h.states() == ["IDLE", "THINKING", "EXECUTING", "THINKING", "DONE", "IDLE"]
        (called,) = h.events_of("tool.called")
        assert (called["tool"], called["decision"], called["ok"]) == ("list_dir", "allow", True)

    async def test_records_the_intent_of_a_call_before_it_runs_and_clears_it_with_the_result(
        self, make_engine: MakeEngine
    ) -> None:
        h = started(make_engine([calls(call("c1", "exec", command="sleep 30"))]))

        h.engine.accept(user_message("m1"))
        await h.wait_until(lambda: "EXECUTING" in h.states())

        intents = h.store.read(s.list_tool_intents)
        assert [(i.tool_call_id, i.tool, i.session_key) for i in intents] == [("c1", "exec", CHAT)]
        await h.engine.suspend()

    async def test_a_message_that_arrives_during_a_turn_is_taken_in_and_answered_with_the_rest(
        self, make_engine: MakeEngine
    ) -> None:
        gate = Gate()
        h = started(make_engine([gate.holds(calls(call("c1", "list_dir", path="."))), says("both")]))
        h.engine.accept(user_message("m1", "first"))
        await gate.wait_reached()

        h.engine.accept(user_message("m2", "second"))
        gate.release.set()
        await h.idle()

        assert h.asked() == 2
        assert user_texts(h.provider.requests[1]) == ["first", "second"]
        assert h.events_of("message.assistant") == [{"text": "both", "in_reply_to": "m2", "spent_usd": 0.0}]
        assert (h.inbound_state("m1"), h.inbound_state("m2")) == ("applied", "applied")
        injected = [m for m in h.messages() if m["role"] == "user"][1]
        assert injected[METADATA_KEY] == {INBOUND_ID: "m2"}

    async def test_a_chat_turn_and_a_task_turn_run_at_the_same_time(self, make_engine: MakeEngine) -> None:
        task_gate = Gate()
        h = started(make_engine([task_gate.holds(says("task done")), says("chat done")]))
        h.engine.accept(task_created("a"))
        await task_gate.wait_reached()

        h.engine.accept(user_message("m1"))
        await h.wait_until(lambda: bool(h.events_of("message.assistant")))

        assert h.events_of("message.assistant") == [{"text": "chat done", "in_reply_to": "m1", "spent_usd": 0.0}]
        assert h.task("a").status == "running"  # type: ignore[union-attr]
        task_gate.release.set()
        await h.idle()
        assert h.task("a").status == "completed"  # type: ignore[union-attr]


class TestTheAnswerAChatOwes:
    def put_in_transcript(self, h: EngineHarness, inbound_id: str = "m1", text: str = "hello") -> None:
        event = {"id": inbound_id, "type": "user.message", "ts": "2026-10-04T10:00:00.000Z", "data": {"text": text}}
        h.store.write(lambda c: s.record_inbound(c, event, "accepted"))
        h.store.write(
            lambda c: s.append_messages(
                c, CHAT, [{"role": "user", "content": text, METADATA_KEY: {INBOUND_ID: inbound_id}}], final_index=None
            )
        )

    async def test_after_a_restart_the_message_in_the_transcript_is_answered_without_being_sent_again(
        self, make_engine: MakeEngine
    ) -> None:
        h = make_engine([says("sorry for the wait")])
        self.put_in_transcript(h)
        assert h.inbound_state("m1") == "in_transcript"

        started(h)
        await h.idle()

        assert h.events_of("message.assistant") == [{"text": "sorry for the wait", "in_reply_to": "m1", "spent_usd": 0.0}]
        assert [m["role"] for m in h.messages()] == ["user", "assistant"]
        assert user_texts(h.provider.requests[0]) == ["hello"]

    async def test_it_waits_while_an_approval_holds_the_chat(self, make_engine: MakeEngine) -> None:
        h = make_engine([])
        self.put_in_transcript(h)
        park(h, CHAT, None)

        started(h)
        await h.idle()

        assert h.asked() == 0
        assert h.inbound_state("m1") == "in_transcript"

    async def test_a_prepare_sleep_that_cut_a_turn_leaves_the_message_to_be_answered_when_work_resumes(
        self, make_engine: MakeEngine
    ) -> None:
        gate = Gate()
        h = started(make_engine([gate.holds(says("never sent")), says("back again")]))
        h.engine.accept(user_message("m1"))
        await gate.wait_reached()

        await h.engine.suspend()
        await h.idle()

        assert h.events_of("message.assistant") == []
        assert h.inbound_state("m1") == "in_transcript"
        # No work while sleeping, even when something kicks.
        h.engine.kick()
        await h.idle()
        assert h.asked() == 1

        h.give_key()
        await h.idle()

        assert h.events_of("message.assistant") == [{"text": "back again", "in_reply_to": "m1", "spent_usd": 0.0}]
        assert user_texts(h.provider.requests[1]) == ["hello"]


class TestAutomations:
    def job(self, run_at_ms: int = 1000) -> CronJob:
        return CronJob(
            id="j1",
            name="daily-fares",
            schedule=CronSchedule(kind="every", every_ms=1000),
            payload=CronPayload(message="check the fares"),
            state=CronJobState(next_run_at_ms=run_at_ms),
        )

    async def test_a_firing_is_a_chat_turn_answered_without_a_reply_to(self, make_engine: MakeEngine) -> None:
        h = started(make_engine([says("fares checked")]))

        await h.engine.automation_fired(self.job())
        await h.idle()

        assert user_texts(h.provider.requests[0]) == ['[Automation "daily-fares" fired] check the fares']
        assert h.events_of("message.assistant") == [{"text": "fares checked", "spent_usd": 0.0}]
        assert h.inbound_state("cron:j1:1000") == "applied"
        row = h.messages()[0]
        assert row[METADATA_KEY] == {INBOUND_ID: "cron:j1:1000"}

    async def test_the_same_firing_twice_is_one_turn(self, make_engine: MakeEngine) -> None:
        h = started(make_engine([says("once")]))

        await h.engine.automation_fired(self.job())
        await h.engine.automation_fired(self.job())
        await h.idle()
        await h.engine.automation_fired(self.job())
        await h.idle()

        assert h.asked() == 1
        assert h.count("dots_inbound") == 1

    async def test_a_firing_and_a_message_together_are_answered_in_reply_to_the_message(
        self, make_engine: MakeEngine
    ) -> None:
        h = make_engine([says("both handled")], key=False)
        started(h)
        await h.engine.automation_fired(self.job())
        h.engine.accept(user_message("m1", "and this"))

        h.give_key()
        await h.idle()

        assert user_texts(h.provider.requests[0]) == [
            '[Automation "daily-fares" fired] check the fares\n\nand this'
        ]
        assert h.events_of("message.assistant") == [{"text": "both handled", "in_reply_to": "m1", "spent_usd": 0.0}]

    async def test_a_firing_during_a_prepare_sleep_waits_for_the_sleep_to_end(self, make_engine: MakeEngine) -> None:
        h = started(make_engine([says("woken")]))
        await h.engine.suspend()

        await h.engine.automation_fired(self.job())
        await h.idle()
        assert h.asked() == 0
        assert h.inbound_state("cron:j1:1000") == "accepted"

        h.give_key()
        await h.idle()
        assert h.events_of("message.assistant") == [{"text": "woken", "spent_usd": 0.0}]

    async def test_a_stopped_engine_records_nothing(self, make_engine: MakeEngine) -> None:
        h = started(make_engine())
        await h.engine.stop()

        await h.engine.automation_fired(self.job())

        assert h.count("dots_inbound") == 0


class TestTheNextRunReport:
    """The host wakes a stopped computer for an automation, so it is told when the earliest one is due."""

    async def test_the_first_time_is_reported_and_the_same_time_again_is_not(self, make_engine: MakeEngine) -> None:
        h = started(make_engine())

        h.engine.automations_next_run(1_790_000_000_000)
        h.engine.automations_next_run(1_790_000_000_000)

        assert h.events_of("automation.next_run") == [{"next_run_at_ms": 1_790_000_000_000}]

    async def test_each_change_is_reported_and_nothing_due_is_reported_as_none(self, make_engine: MakeEngine) -> None:
        h = started(make_engine())

        h.engine.automations_next_run(1_790_000_000_000)
        h.engine.automations_next_run(1_790_000_060_000)
        h.engine.automations_next_run(None)
        h.engine.automations_next_run(None)

        assert h.events_of("automation.next_run") == [
            {"next_run_at_ms": 1_790_000_000_000},
            {"next_run_at_ms": 1_790_000_060_000},
            {"next_run_at_ms": None},
        ]

    async def test_a_computer_that_never_had_an_automation_reports_nothing(self, make_engine: MakeEngine) -> None:
        h = started(make_engine())

        h.engine.automations_next_run(None)

        assert h.events_of("automation.next_run") == []

    async def test_a_restart_does_not_say_again_what_the_host_already_heard(self, make_engine: MakeEngine) -> None:
        h = started(make_engine())
        h.engine.automations_next_run(1_790_000_000_000)

        h.restart()
        started(h)
        h.engine.automations_next_run(1_790_000_000_000)
        h.engine.automations_next_run(None)

        assert h.events_of("automation.next_run") == [{"next_run_at_ms": 1_790_000_000_000}, {"next_run_at_ms": None}]

    async def test_the_report_is_an_event_of_its_own_and_starts_no_turn(self, make_engine: MakeEngine) -> None:
        h = started(make_engine([says("never asked")]))

        h.engine.automations_next_run(1_790_000_000_000)
        await h.idle()

        assert h.asked() == 0
        assert h.states() == ["IDLE"]

    async def test_a_stopped_engine_reports_nothing(self, make_engine: MakeEngine) -> None:
        h = started(make_engine())
        await h.engine.stop()

        h.engine.automations_next_run(1_790_000_000_000)

        assert h.events_of("automation.next_run") == []


class TestTasks:
    async def test_runs_one_at_a_time_each_in_its_own_session(self, make_engine: MakeEngine) -> None:
        gate = Gate()
        h = started(make_engine([gate.holds(says("first done")), says("second done")]))

        h.engine.accept(task_created("low", "later", 0))
        h.engine.accept(task_created("high", "sooner", 5))
        await gate.wait_reached()
        h.engine.kick()

        # The first kick started "low" before "high" arrived; only one runs.
        assert h.asked() == 1
        assert h.task_events() == [("task.started", "low")]
        gate.release.set()
        await h.idle()

        assert h.task_events() == [
            ("task.started", "low"),
            ("task.completed", "low"),
            ("task.started", "high"),
            ("task.completed", "high"),
        ]
        assert [m["content"] for m in h.messages(s.task_session_key("high"))] == ["sooner", "second done"]
        assert [m["content"] for m in h.messages(s.task_session_key("low"))] == ["later", "first done"]
        assert h.events_of("task.completed")[0] == {"task_id": "low", "summary": "first done", "spent_usd": 0.0}

    async def test_the_highest_priority_goes_first(self, make_engine: MakeEngine) -> None:
        h = make_engine([says("a"), says("b")], key=False)
        started(h)
        h.engine.accept(task_created("low", "later", 0))
        h.engine.accept(task_created("high", "sooner", 5))

        h.give_key()
        await h.idle()

        assert [e[1] for e in h.task_events() if e[0] == "task.started"] == ["high", "low"]
        assert user_texts(h.provider.requests[0]) == ["sooner"]

    async def test_fails_a_task_whose_run_failed_and_one_whose_run_ended_without_an_answer(
        self, make_engine: MakeEngine
    ) -> None:
        h = started(
            make_engine([LLMResponse(content="provider down", finish_reason="error"), says(""), says(""), says("")])
        )

        h.engine.accept(task_created("a", "x", 9))
        await h.idle()
        h.engine.accept(task_created("b", "y", 0))
        await h.idle()

        assert h.events_of("task.failed") == [
            {"task_id": "a", "error": "provider down", "spent_usd": 0.0},
            {"task_id": "b", "error": "the run ended without an answer", "spent_usd": 0.0},
        ]
        assert h.task("a").status == "failed"  # type: ignore[union-attr]
        assert h.engine.state == "IDLE"

    async def test_a_task_that_reaches_the_step_limit_fails_with_the_limit(self, make_engine: MakeEngine) -> None:
        body = cfg(limits={"max_steps_per_task": 1, "max_cost_per_task_usd": 1})
        h = make_engine([calls(call("c1", "list_dir", path="."))])
        h.engine.start()
        h.configure(body)

        h.engine.accept(task_created("a"))
        await h.idle()

        assert h.events_of("task.failed") == [
            {"task_id": "a", "error": "stopped: the task reached limits.max_steps_per_task (1)", "spent_usd": 0.0}
        ]

    async def test_a_task_that_spent_the_cap_fails_with_what_it_spent_and_its_calls_are_closed(
        self, make_engine: MakeEngine
    ) -> None:
        body = cfg(limits={"max_steps_per_task": 60, "max_cost_per_task_usd": 1})
        h = make_engine(
            [
                calls(call("c1", "list_dir", path="."), cost=0.6),
                calls(call("c2", "list_dir", path="."), cost=0.6),
                says("never asked", cost=0.6),
            ]
        )
        h.engine.start()
        h.configure(body)

        h.engine.accept(task_created("a"))
        await h.idle()

        assert h.events_of("task.failed") == [
            {"task_id": "a", "error": "stopped: the task reached limits.max_cost_per_task_usd (spent 1.2000 USD of 1.00)", "spent_usd": 1.2}
        ]
        assert h.asked() == 2
        assert h.task("a").status == "failed"  # type: ignore[union-attr]
        assert [m["role"] for m in h.messages("task:a")] == ["user", "assistant", "tool", "assistant", "tool"]
        assert h.engine.state == "IDLE"

    async def test_a_chat_turn_that_spent_the_cap_answers_with_what_it_spent(self, make_engine: MakeEngine) -> None:
        body = cfg(limits={"max_steps_per_task": 60, "max_cost_per_task_usd": 0.01})
        h = make_engine(
            [calls(call("c1", "list_dir", path="."), cost=0.006), calls(call("c2", "list_dir", path="."), cost=0.006)]
        )
        h.engine.start()
        h.configure(body)

        h.engine.accept(user_message("m1"))
        await h.idle()

        assert h.events_of("message.assistant") == [
            {
                "text": "I could not answer: stopped: the turn reached limits.max_cost_per_task_usd (spent 0.0120 USD of 0.01)",
                "in_reply_to": "m1",
                "spent_usd": 0.012,
            }
        ]
        assert h.asked() == 2
        assert h.inbound_state("m1") == "applied"

    async def test_the_cap_still_holds_for_a_task_after_the_process_was_killed(self, make_engine: MakeEngine) -> None:
        body = cfg(limits={"max_steps_per_task": 60, "max_cost_per_task_usd": 1})
        h = make_engine(
            [
                calls(call("c1", "exec", command="sleep 30"), cost=0.6),
                calls(call("c2", "list_dir", path="."), cost=0.6),
                says("never asked", cost=0.6),
            ]
        )
        h.engine.start()
        h.configure(body)
        h.engine.accept(task_created("a"))
        await h.wait_until(lambda: "EXECUTING" in h.states())
        await h.engine.suspend()
        await h.idle()
        assert h.store.read(lambda c: s.get_spend(c, "task:a")) == 0.6

        # A new process: nothing in memory, the task is left running.
        restarted = h.restart()
        restarted.start()
        h.configure(body)
        await h.idle()

        assert h.events_of("task.failed") == [
            {"task_id": "a", "error": "stopped: the task reached limits.max_cost_per_task_usd (spent 1.2000 USD of 1.00)", "spent_usd": 1.2}
        ]
        assert h.asked() == 2

    async def test_a_task_resumed_after_a_kill_reports_what_it_spent_before_and_after(
        self, make_engine: MakeEngine
    ) -> None:
        h = make_engine([calls(call("c1", "exec", command="sleep 30"), cost=0.6), says("finished", cost=0.25)])
        h.engine.start()
        h.configure(cfg())
        h.engine.accept(task_created("a"))
        await h.wait_until(lambda: "EXECUTING" in h.states())
        await h.engine.suspend()
        await h.idle()

        restarted = h.restart()
        restarted.start()
        h.configure(cfg())
        await h.idle()

        assert h.events_of("task.completed") == [{"task_id": "a", "summary": "finished", "spent_usd": 0.85}]

    async def test_every_chat_answer_reports_the_spend_of_its_own_turn(self, make_engine: MakeEngine) -> None:
        h = started(make_engine([says("one", cost=0.25), says("two", cost=0.5)]))
        h.engine.accept(user_message("m1"))
        await h.idle()
        h.engine.accept(user_message("m2"))
        await h.idle()

        assert [e["spent_usd"] for e in h.events_of("message.assistant")] == [0.25, 0.5]

    async def test_a_task_found_running_with_the_cap_already_spent_is_failed_before_any_request(
        self, make_engine: MakeEngine
    ) -> None:
        h = make_engine([says("never asked")])
        h.store.write(lambda c: s.enqueue_task(c, task_id="a", description="x", priority=0))
        h.store.write(lambda c: s.start_task(c, "a"))
        h.store.write(lambda c: s.add_spend(c, "task:a", 1.5))

        started(h)
        await h.idle()

        assert h.asked() == 0
        assert h.events_of("task.failed") == [
            {"task_id": "a", "error": "stopped: the task reached limits.max_cost_per_task_usd (spent 1.5000 USD of 1.00)", "spent_usd": 1.5}
        ]

    async def test_cancels_a_running_task_its_turn_ends_and_nothing_more_is_reported(
        self, make_engine: MakeEngine
    ) -> None:
        gate = Gate()
        h = started(make_engine([gate.holds(says("never"))]))
        h.engine.accept(task_created("a"))
        await gate.wait_reached()

        h.engine.accept(task_cancelled("a"))
        await h.idle()

        assert h.task("a").status == "cancelled"  # type: ignore[union-attr]
        assert h.task_events() == [("task.started", "a")]
        assert h.engine.state == "IDLE"
        assert h.engine.state_answer().current_task_id is None

    async def test_cancelling_a_task_in_the_middle_of_a_call_reports_the_call_as_interrupted(
        self, make_engine: MakeEngine
    ) -> None:
        h = started(make_engine([calls(call("c1", "exec", command="sleep 30"))]))
        h.engine.accept(task_created("a"))
        await h.wait_until(lambda: "EXECUTING" in h.states())

        h.engine.accept(task_cancelled("a"))
        await h.idle()

        (called,) = h.events_of("tool.called")
        assert (called["task_id"], called["tool"], called["ok"], called["interrupted"]) == ("a", "exec", False, True)
        assert h.count("dots_tool_intents") == 0
        assert h.task_events() == [("task.started", "a")]

    async def test_cancelling_a_queued_task_or_an_unknown_one_changes_nothing_else(
        self, make_engine: MakeEngine
    ) -> None:
        h = make_engine([], key=False)
        started(h)
        h.engine.accept(task_created("a"))

        assert h.engine.accept(task_cancelled("a")) is True
        assert h.engine.accept(task_cancelled("nope")) is True

        assert h.task("a").status == "cancelled"  # type: ignore[union-attr]
        assert h.task_events() == []

    async def test_resumes_a_task_the_last_process_left_running(self, make_engine: MakeEngine) -> None:
        h = make_engine([says("finished it")])
        h.store.write(lambda c: s.enqueue_task(c, task_id="a", description="build it", priority=0))
        h.store.write(lambda c: s.start_task(c, "a"))

        started(h)
        await h.idle()

        assert h.task("a").attempts == 2  # type: ignore[union-attr]
        (opening,) = user_texts(h.provider.requests[0])
        assert opening == f"build it\n\n{RESUME_TASK_NOTE}"
        assert "interrupted by a restart" in opening
        assert "task.started" not in h.types()
        assert h.task_events() == [("task.completed", "a")]

    async def test_gives_up_on_a_task_after_three_starts(self, make_engine: MakeEngine) -> None:
        h = make_engine([])
        h.store.write(lambda c: s.enqueue_task(c, task_id="b", description="x", priority=0))
        for _ in range(3):
            h.store.write(lambda c: s.start_task(c, "b"))

        started(h)
        await h.idle()

        assert h.asked() == 0
        assert h.events_of("task.failed") == [{"task_id": "b", "error": "stopped: the task was interrupted 3 times", "spent_usd": 0.0}]
        assert h.task("b").status == "failed"  # type: ignore[union-attr]

    async def test_a_failed_task_does_not_hold_back_the_next_one(self, make_engine: MakeEngine) -> None:
        h = make_engine([says("c done")])
        for task_id in ("b", "c"):
            h.store.write(lambda c, t=task_id: s.enqueue_task(c, task_id=t, description=t, priority=0))
        for _ in range(3):
            h.store.write(lambda c: s.start_task(c, "b"))

        started(h)
        await h.idle()

        assert h.task_events() == [("task.failed", "b"), ("task.started", "c"), ("task.completed", "c")]

    async def test_a_prepare_sleep_does_not_count_the_attempt_and_nothing_starts_until_work_resumes(
        self, make_engine: MakeEngine
    ) -> None:
        gate = Gate()
        h = started(make_engine([gate.holds(says("never sent")), says("done after the sleep")]))
        h.engine.accept(task_created("a", "the work"))
        await gate.wait_reached()

        await h.engine.suspend()
        await h.idle()

        task = h.task("a")
        assert (task.status, task.attempts) == ("running", 0)  # type: ignore[union-attr]
        h.engine.kick()
        await h.idle()
        assert h.asked() == 1

        h.give_key()
        await h.idle()

        assert h.asked() == 2
        assert h.task("a").status == "completed"  # type: ignore[union-attr]
        assert h.task("a").attempts == 1  # type: ignore[union-attr]
        # The first attempt's description is in the transcript already; the resume says it again with the note.
        assert user_texts(h.provider.requests[1]) == [f"the work\n\nthe work\n\n{RESUME_TASK_NOTE}"]


class TestConfig:
    async def test_refuses_a_config_that_is_not_a_dot_config_and_keeps_the_one_it_had(
        self, make_engine: MakeEngine
    ) -> None:
        h = started(make_engine())

        with pytest.raises(DotsConfigError, match="invalid Dot config"):
            h.engine.set_config({**cfg(), "model": {"provider": "openai", "id": "x"}})

        assert h.engine.config is not None and h.engine.config.model.provider == "openrouter"

    async def test_the_same_config_again_writes_nothing_and_disturbs_no_running_turn(
        self, make_engine: MakeEngine
    ) -> None:
        gate = Gate()
        h = started(make_engine([gate.holds(says("done"))]))
        h.engine.accept(user_message("m1"))
        await gate.wait_reached()
        before = h.store.read(lambda c: s.read_kv(c, s.KV_RUNTIME_CONFIG))

        h.engine.set_config(cfg())
        h.engine.set_config(cfg())
        gate.release.set()
        await h.idle()

        assert h.store.read(lambda c: s.read_kv(c, s.KV_RUNTIME_CONFIG)) == before
        assert h.events_of("message.assistant") == [{"text": "done", "in_reply_to": "m1", "spent_usd": 0.0}]

    async def test_a_new_model_applies_to_the_next_turn_only(self, make_engine: MakeEngine) -> None:
        gate = Gate()
        h = started(make_engine([gate.holds(says("one")), says("two")]))
        h.engine.accept(user_message("m1"))
        await gate.wait_reached()

        h.engine.set_config(cfg(model={"provider": "openrouter", "id": "other/model"}))
        gate.release.set()
        await h.idle()
        h.engine.accept(user_message("m2"))
        await h.idle()

        assert [asked[1] for asked in h.providers.asked] == ["z-ai/glm-5.3-flash", "other/model"]

    async def test_a_new_summary_role_applies_to_the_next_turn_only(
        self, make_engine: MakeEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def four_characters_a_token(provider: Any, model: str, messages: list[dict[str, Any]], tools: Any) -> Any:
            # The thread the test sizes, not the engine's own prompt, which grows with what it teaches the model.
            return sum(len(json.dumps(message)) for message in messages if message.get("role") != "system") // 4, "test"

        for module in ("nanobot.agent.context_governance", "nanobot.agent.memory"):
            monkeypatch.setattr(f"{module}.estimate_prompt_tokens_chain", four_characters_a_token)

        def small_window(role_model: str) -> dict[str, Any]:
            return cfg(models={"summary": role_model})

        def outgrow_the_window() -> None:
            # Over the request budget of a 6000 token window (3976 tokens), the second time with the summary of
            # the first turn in it.
            old: list[dict[str, Any]] = []
            for index in range(11):
                old += [
                    {"role": "user", "content": f"question {index} " + "x" * 700},
                    {"role": "assistant", "content": f"answer {index} " + "y" * 700},
                ]
            h.store.write(lambda conn: s.append_messages(conn, CHAT, old, final_index=None))

        gate = Gate()
        h = make_engine([gate.holds(says("summary one")), says("one"), says("summary two"), says("two")])
        h.provider.default_limits = ModelLimits(context_tokens=6000, answer_tokens=1000)
        h.engine.start()
        h.configure(small_window("first/summarizer"))
        outgrow_the_window()
        h.engine.accept(user_message("m1"))
        await gate.wait_reached()

        h.engine.set_config(small_window("second/summarizer"))
        gate.release.set()
        await h.idle()
        outgrow_the_window()
        h.engine.accept(user_message("m2"))
        await h.idle()

        assert [request["model"] for request in h.provider.requests] == [
            "first/summarizer",
            "z-ai/glm-5.3-flash",
            "second/summarizer",
            "z-ai/glm-5.3-flash",
        ]

    async def test_the_permissions_pushed_between_two_calls_of_a_turn_apply_to_the_second(
        self, make_engine: MakeEngine
    ) -> None:
        h = started(make_engine([]))

        def first(_: Any) -> LLMResponse:
            h.engine.set_config(cfg({"files.read": "deny"}))
            return calls(call("c1", "read_file", path="notes.txt"))

        h.provider.script = [first, says("understood")]
        h.engine.accept(user_message("m1"))
        await h.idle()

        (called,) = h.events_of("tool.called")
        assert (called["decision"], called["ok"]) == ("deny", False)


class TestApprovals:
    async def test_records_the_decision_in_the_transaction_that_accepts_it_and_ignores_one_for_an_unknown_approval(
        self, make_engine: MakeEngine
    ) -> None:
        h = make_engine([], key=False)
        h.engine.start()
        approval_id = park(h, CHAT, None)
        assert h.engine.state_answer().pending_approval == approval_id

        h.engine.accept(decision("d1", approval_id, "reject", "not now"))

        approval = h.approval(approval_id)
        assert (approval.status, approval.note) == ("rejected", "not now")
        assert h.engine.state_answer().pending_approval is None
        assert h.inbound_state("d1") == "applied"
        assert h.engine.accept(decision("d2", "nope", "approve")) is True
        # Decided twice: the first decision stands.
        h.engine.accept(decision("d3", approval_id, "approve"))
        assert h.approval(approval_id).status == "rejected"

    async def test_tells_the_chat_to_make_an_approved_call_again_granted_for_the_length_of_that_turn(
        self, make_engine: MakeEngine
    ) -> None:
        h = started(make_engine([]))
        arguments = {"command": "rm -rf build", "cwd": "/home/dot/workspace"}
        approval_id = park(h, CHAT, None, arguments=arguments)
        seen: list[str] = []

        def ask(_: Any) -> LLMResponse:
            seen.append(h.approval(approval_id).status)
            return says("understood")

        h.provider.script = [ask]
        h.engine.accept(decision("d1", approval_id, "approve", "go"))
        await h.idle()

        (text,) = user_texts(h.provider.requests[0])
        assert f'approved your exec call ({approval_id}). The user\'s note: "go".' in text
        assert text.endswith(f"it will run once: {json.dumps(arguments, separators=(',', ':'))}]")
        assert seen == ["granted"]
        assert h.approval(approval_id).status == "done"
        # The decision's public line goes with it, for the conversation files: the call and its target, no arguments.
        assert h.messages()[0][METADATA_KEY] == {
            "dots_approval_id": approval_id,
            "dots_approval_line": 'The person approved the exec call (rm -rf build). The user\'s note: "go".',
        }

    async def test_the_continuation_texts_are_the_contract_the_model_reads(self, make_engine: MakeEngine) -> None:
        h = make_engine()
        approval_id = park(h, CHAT, None, arguments={"command": "ls"})
        approval = h.approval(approval_id)

        assert approval_granted_continuation(approval) == (
            f"[The user approved your exec call ({approval_id}). "
            "It has not run yet. Call exec again now with exactly these arguments, and it will run once: "
            '{"command":"ls"}]'
        )
        assert approval_rejected_continuation(approval) == (
            f"[The user rejected your exec call ({approval_id}). "
            "It did not run. Do not try it again unless the user asks you to.]"
        )

    async def test_an_approved_call_made_again_runs_once_and_the_answer_goes_to_the_message_that_asked(
        self, make_engine: MakeEngine
    ) -> None:
        arguments = {"path": "a.txt", "content": "x"}
        h = started(
            make_engine(
                [
                    calls(call("c1", "write_file", **arguments)),
                    calls(call("c2", "write_file", **arguments)),
                    says("written"),
                ]
            ),
            {**ALLOW_ALL, "files.write": "ask"},
        )
        h.engine.accept(user_message("m1", "write it"))
        await h.idle()
        (approval,) = h.pending_approvals()
        assert h.engine.state == "WAITING_APPROVAL"
        assert h.events_of("message.assistant") == []
        # The chat owes an answer, and no turn starts while the approval holds it.
        h.engine.kick()
        await h.idle()
        assert h.asked() == 1

        h.engine.accept(decision("d1", approval.approval_id, "approve"))
        await h.idle()

        assert (h.tmp_path / "home" / "dot" / "workspace" / "a.txt").read_text(encoding="utf-8") == "x"
        assert h.approval(approval.approval_id).status == "done"
        (called,) = h.events_of("tool.called")
        assert (called["tool"], called["decision"], called["ok"]) == ("write_file", "ask", True)
        assert h.events_of("message.assistant") == [{"text": "written", "in_reply_to": "m1", "spent_usd": 0.0}]
        assert h.engine.state == "IDLE"

    async def test_the_answer_of_a_chat_that_waited_for_an_approval_reports_what_it_spent_before_and_after(
        self, make_engine: MakeEngine
    ) -> None:
        arguments = {"path": "a.txt", "content": "x"}
        h = started(
            make_engine(
                [
                    calls(call("c1", "write_file", **arguments), cost=0.25),
                    calls(call("c2", "write_file", **arguments), cost=0.125),
                    says("written", cost=0.0625),
                    says("again", cost=0.5),
                ]
            ),
            {**ALLOW_ALL, "files.write": "ask"},
        )
        h.engine.accept(user_message("m1", "write it"))
        await h.idle()
        (approval,) = h.pending_approvals()
        h.engine.accept(decision("d1", approval.approval_id, "approve"))
        await h.idle()

        assert h.events_of("message.assistant") == [{"text": "written", "in_reply_to": "m1", "spent_usd": 0.4375}]
        # What the answer took is spent once: the next answer starts from nothing.
        h.engine.accept(user_message("m2", "again"))
        await h.idle()
        assert [e["spent_usd"] for e in h.events_of("message.assistant")] == [0.4375, 0.5]

    async def test_a_chat_that_spent_the_cap_before_an_approval_does_not_start_again_with_it(
        self, make_engine: MakeEngine
    ) -> None:
        arguments = {"path": "a.txt", "content": "x"}
        h = started(
            make_engine([calls(call("c1", "write_file", **arguments), cost=1.5), says("never asked")]),
            {**ALLOW_ALL, "files.write": "ask"},
        )
        h.engine.accept(user_message("m1", "write it"))
        await h.idle()
        (approval,) = h.pending_approvals()
        h.engine.accept(decision("d1", approval.approval_id, "approve"))
        await h.idle()

        assert h.asked() == 1
        (answer,) = h.events_of("message.assistant")
        assert answer["text"].startswith("I could not answer: stopped: the turn reached limits.max_cost_per_task_usd")
        assert answer["spent_usd"] == 1.5

    async def test_an_approved_call_runs_once_even_when_the_model_makes_it_a_second_time(
        self, make_engine: MakeEngine
    ) -> None:
        arguments = {"path": "a.txt", "content": "x"}
        h = started(
            make_engine(
                [
                    calls(call("c1", "write_file", **arguments)),
                    calls(call("c2", "write_file", **arguments)),
                    calls(call("c3", "write_file", **arguments)),
                ]
            ),
            {**ALLOW_ALL, "files.write": "ask"},
        )
        h.engine.accept(user_message("m1"))
        await h.idle()
        (first,) = h.pending_approvals()

        h.engine.accept(decision("d1", first.approval_id, "approve"))
        await h.idle()

        assert h.approval(first.approval_id).status == "done"
        (second,) = h.pending_approvals()
        assert second.tool_call_id == "c3"

    async def test_parks_a_task_on_a_pending_approval_instead_of_failing_or_resuming_it_then_continues_it_with_the_rejection(
        self, make_engine: MakeEngine
    ) -> None:
        h = started(
            make_engine([calls(call("c1", "exec", command="rm -rf build")), says("understood")]),
            {**ALLOW_ALL, "computer.exec": "ask"},
        )
        h.engine.accept(task_created("a", "clean up"))
        await h.idle()
        (approval,) = h.pending_approvals()
        assert (h.task("a").status, h.engine.state) == ("running", "WAITING_APPROVAL")  # type: ignore[union-attr]
        assert "task.failed" not in h.types()
        h.engine.kick()
        await h.idle()
        assert h.asked() == 1
        seen: list[str] = []

        def ask(_: Any) -> LLMResponse:
            seen.append(h.approval(approval.approval_id).status)
            return says("understood")

        h.provider.script = [ask]
        h.engine.accept(decision("d1", approval.approval_id, "reject", "keep it"))
        await h.idle()

        (continuation,) = user_texts(h.provider.requests[1])[-1:]
        assert (
            f'rejected your exec call ({approval.approval_id}). The user\'s note: "keep it". It did not run.'
            in continuation
        )
        assert seen == ["told"]
        assert h.approval(approval.approval_id).status == "done"
        assert h.task_events() == [("task.started", "a"), ("task.completed", "a")]
        assert h.events_of("tool.called") == []

    async def test_a_task_continued_after_an_approval_that_ends_without_an_answer_fails(
        self, make_engine: MakeEngine
    ) -> None:
        h = started(
            make_engine([calls(call("c1", "exec", command="ls")), says(""), says(""), says("")]),
            {**ALLOW_ALL, "computer.exec": "ask"},
        )
        h.engine.accept(task_created("a"))
        await h.idle()
        (approval,) = h.pending_approvals()

        h.engine.accept(decision("d1", approval.approval_id, "reject"))
        await h.idle()

        assert h.approval(approval.approval_id).status == "done"
        assert h.events_of("task.failed") == [{"task_id": "a", "error": "the run ended without an answer", "spent_usd": 0.0}]

    async def test_an_approved_call_of_a_task_runs_once_and_the_task_completes(self, make_engine: MakeEngine) -> None:
        arguments = {"path": "b.txt", "content": "y"}
        h = started(
            make_engine(
                [calls(call("c1", "write_file", **arguments)), calls(call("c2", "write_file", **arguments)), says("done")]
            ),
            {**ALLOW_ALL, "files.write": "ask"},
        )
        h.engine.accept(task_created("a"))
        await h.idle()
        (approval,) = h.pending_approvals()

        h.engine.accept(decision("d1", approval.approval_id, "approve"))
        await h.idle()

        assert (h.tmp_path / "home" / "dot" / "workspace" / "b.txt").read_text(encoding="utf-8") == "y"
        (called,) = h.events_of("tool.called")
        assert (called["task_id"], called["decision"], called["ok"]) == ("a", "ask", True)
        assert h.task_events() == [("task.started", "a"), ("task.completed", "a")]
        assert h.approval(approval.approval_id).status == "done"

    async def test_an_approval_of_a_task_that_ended_meanwhile_is_closed_and_tells_nobody(
        self, make_engine: MakeEngine
    ) -> None:
        h = make_engine([])
        h.store.write(lambda c: s.enqueue_task(c, task_id="a", description="x", priority=0))
        h.store.write(lambda c: s.start_task(c, "a"))
        approval_id = park(h, s.task_session_key("a"), "a")
        h.engine.start()
        h.engine.accept(task_cancelled("a"))
        h.engine.accept(decision("d1", approval_id, "approve"))

        h.configure(cfg())
        await h.idle()

        assert h.approval(approval_id).status == "done"
        assert h.asked() == 0

    async def test_a_task_cancelled_while_its_call_waits_leaves_no_approval_waiting(
        self, make_engine: MakeEngine
    ) -> None:
        # The host expires the approval of a cancelled task, so no decision for it ever arrives.
        h = started(
            make_engine([calls(call("c1", "write_file", path="b.txt", content="y")), says("hi")]),
            {**ALLOW_ALL, "files.write": "ask"},
        )
        h.engine.accept(task_created("a"))
        await h.idle()
        (approval,) = h.pending_approvals()

        assert h.states()[-1] == "WAITING_APPROVAL"

        h.engine.accept(task_cancelled("a"))
        await h.idle()

        # Idle again, so the host shows the Dot ready and may put it to sleep.
        assert h.approval(approval.approval_id).status == "done"
        assert h.engine.state_answer().pending_approval is None
        assert h.states()[-1] == "IDLE"
        h.engine.accept(user_message("m1"))
        await h.idle()
        assert h.states()[-1] == "IDLE"
        assert not (h.tmp_path / "home" / "dot" / "workspace" / "b.txt").exists()

    async def test_a_task_cancelled_while_another_call_waits_keeps_the_agent_waiting(
        self, make_engine: MakeEngine
    ) -> None:
        h = make_engine([])
        h.store.write(lambda c: s.enqueue_task(c, task_id="a", description="x", priority=0))
        h.store.write(lambda c: s.start_task(c, "a"))
        park(h, s.task_session_key("a"), "a")
        kept = park(h, CHAT, None, call_id="call-2")
        h.engine.start()
        h.store.write(lambda c: s.record_agent_state(c, "WAITING_APPROVAL"))

        h.engine.accept(task_cancelled("a"))

        assert [a.approval_id for a in h.pending_approvals()] == [kept]
        assert h.engine.state == "WAITING_APPROVAL"

    async def test_after_a_stop_ends_an_approval_whose_call_was_running_and_tells_a_cut_turn_again(
        self, make_engine: MakeEngine
    ) -> None:
        h = make_engine([says("understood")])
        ran = park(h, CHAT, None)
        cut = park(h, CHAT, None, call_id="call-2", arguments={"command": "ls"})
        h.store.write(lambda c: s.advance_approval(c, ran, "pending", "approved"))
        h.store.write(lambda c: s.advance_approval(c, ran, "approved", "granted"))
        h.store.write(lambda c: s.advance_approval(c, ran, "granted", "running", run_tool_call_id="call-9"))
        h.store.write(lambda c: s.advance_approval(c, cut, "pending", "approved"))
        h.store.write(lambda c: s.advance_approval(c, cut, "approved", "granted"))

        h.engine.start()

        assert h.approval(ran).status == "done"
        assert h.approval(cut).status == "approved"
        h.configure(cfg())
        await h.idle()
        assert h.asked() == 1
        assert f"({cut})" in user_texts(h.provider.requests[0])[0]
        assert h.approval(cut).status == "done"

    async def test_the_answer_to_a_decision_and_the_end_of_that_approval_commit_together(
        self, make_engine: MakeEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        h = started(make_engine([says("understood")]))
        approval_id = park(h, CHAT, None)

        def killed(engine: Engine, unit: Any, outcome: Any) -> None:
            raise RuntimeError("the process died before the end of the turn was recorded")

        monkeypatch.setattr(Engine, "_turn_ended", killed)
        h.engine.accept(decision("d1", approval_id, "approve"))
        await h.idle()

        # Nothing of the turn's end was written, yet the approval is over with the answer: a restart
        # does not tell the session again.
        assert h.events_of("message.assistant") == [{"text": "understood", "spent_usd": 0.0}]
        assert h.approval(approval_id).status == "done"

    async def test_a_kill_between_the_gate_letting_an_approved_call_through_and_its_start_gives_the_approval_back(
        self, make_engine: MakeEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        arguments = {"path": "a.txt", "content": "x"}
        h = started(
            make_engine(
                [
                    calls(call("c1", "write_file", **arguments)),
                    calls(call("c2", "write_file", **arguments)),
                    calls(call("c3", "write_file", **arguments)),
                    says("written"),
                ]
            ),
            {**ALLOW_ALL, "files.write": "ask"},
        )
        h.engine.accept(user_message("m1", "write it"))
        await h.idle()
        (approval,) = h.pending_approvals()
        # The process dies where the gate has let c2 through and its intent is not recorded yet.
        reached = asyncio.Event()
        real = turns_module.DotsTurnHook.before_execute_tool

        async def dies_here(self: Any, context: Any, tool_call: Any, tool: Any, params: Any) -> None:
            if tool_call.id == "c2":
                reached.set()
                await asyncio.Event().wait()
            await real(self, context, tool_call, tool, params)

        monkeypatch.setattr(turns_module.DotsTurnHook, "before_execute_tool", dies_here)
        h.engine.accept(decision("d1", approval.approval_id, "approve"))
        await asyncio.wait_for(reached.wait(), 10)
        assert h.approval(approval.approval_id).status == "running"
        assert h.store.read(s.list_tool_intents) == []

        restarted = h.restart()
        restarted.start()

        # The approval is not used up: the session is told again, and the call runs once.
        await h.idle()
        assert (h.tmp_path / "home" / "dot" / "workspace" / "a.txt").read_text(encoding="utf-8") == "x"
        assert h.approval(approval.approval_id).status == "done"
        (called,) = h.events_of("tool.called")
        assert (called["tool"], called["decision"], called["ok"]) == ("write_file", "ask", True)
        assert h.events_of("message.assistant") == [{"text": "written", "in_reply_to": "m1", "spent_usd": 0.0}]
        (closed,) = [m for m in h.messages() if m["role"] == "tool" and m["tool_call_id"] == "c2"]
        assert closed["content"] == "Not executed: the unit ended before this call ran."

    async def test_a_sleep_during_the_turn_that_tells_a_decision_tells_it_again_when_work_resumes(
        self, make_engine: MakeEngine
    ) -> None:
        gate = Gate()
        h = started(make_engine([gate.holds(says("never sent")), says("understood")]))
        approval_id = park(h, CHAT, None)
        h.engine.accept(decision("d1", approval_id, "approve"))
        await gate.wait_reached()
        assert h.approval(approval_id).status == "granted"

        await h.engine.suspend()
        await h.idle()

        assert h.approval(approval_id).status == "approved"
        h.give_key()
        await h.idle()

        assert h.approval(approval_id).status == "done"
        assert h.asked() == 2


class TestPrepareSleep:
    async def test_a_tool_that_finishes_within_the_grace_records_its_result_and_the_turn_stops_before_the_next_request(
        self, make_engine: MakeEngine
    ) -> None:
        h = make_engine([calls(call("c1", "exec", command="sleep 0.3")), says("after the sleep")], stop_grace_s=20.0)
        started(h)
        h.engine.accept(user_message("m1"))
        await h.wait_until(lambda: "EXECUTING" in h.states())

        await h.engine.suspend()
        await h.idle()

        assert h.asked() == 1
        (called,) = h.events_of("tool.called")
        assert (called["tool"], called["ok"], "interrupted" in called) == ("exec", True, False)
        assert [m["role"] for m in h.messages()] == ["user", "assistant", "tool"]
        assert h.events_of("message.assistant") == []
        assert h.count("dots_tool_intents") == 0

        # Work resumes: the history ends with the tool's result and the chat still owes its answer.
        h.give_key()
        await h.idle()
        assert h.events_of("message.assistant") == [{"text": "after the sleep", "in_reply_to": "m1", "spent_usd": 0.0}]

    async def test_a_tool_still_running_when_the_grace_ends_is_cut_and_the_next_start_reports_it_interrupted(
        self, make_engine: MakeEngine
    ) -> None:
        h = started(make_engine([calls(call("c1", "exec", command="sleep 30")), says("recovered")]))
        h.engine.accept(user_message("m1"))
        await h.wait_until(lambda: "EXECUTING" in h.states())

        await h.engine.suspend()
        await h.idle()

        assert [i.tool_call_id for i in h.store.read(s.list_tool_intents)] == ["c1"]
        assert h.events_of("tool.called") == []

        restarted = h.restart()
        restarted.start()

        (called,) = h.events_of("tool.called")
        assert (called["tool"], called["ok"], called["interrupted"], called["duration_ms"]) == ("exec", False, True, 0)
        h.configure(cfg())
        await h.idle()
        assert h.events_of("message.assistant") == [{"text": "recovered", "in_reply_to": "m1", "spent_usd": 0.0}]
        closed = [m for m in h.messages() if m["role"] == "tool"][0]
        assert "interrupted before its result was recorded" in closed["content"]

    async def test_a_kill_between_the_progress_text_and_its_tool_reports_the_progress_exactly_once(
        self, make_engine: MakeEngine
    ) -> None:
        h = started(
            make_engine(
                [calls(call("c1", "exec", command="sleep 30"), text="Starting the long job."), says("recovered")]
            )
        )
        h.engine.accept(task_created("a"))
        await h.wait_until(lambda: "EXECUTING" in h.states())
        assert h.events_of("task.progress") == [{"task_id": "a", "text": "Starting the long job.", "spent_usd": 0.0}]

        await h.engine.suspend()
        await h.idle()
        restarted = h.restart()
        restarted.start()
        h.configure(cfg())
        await h.idle()

        assert h.events_of("task.progress") == [{"task_id": "a", "text": "Starting the long job.", "spent_usd": 0.0}]
        assert h.types().index("task.progress") < h.types().index("tool.called") < h.types().index("task.completed")
        assert h.task_events() == [("task.started", "a"), ("task.progress", "a"), ("task.completed", "a")]

    async def test_two_prepare_sleeps_at_once_are_one_suspend_and_give_the_attempt_back_once(
        self, make_engine: MakeEngine
    ) -> None:
        gate = Gate()
        h = started(make_engine([gate.holds(says("never sent"))]))
        h.engine.accept(task_created("a"))
        await gate.wait_reached()
        # A task that was resumed before: it has two attempts counted.
        h.store.write(lambda c: c.execute("UPDATE dots_tasks SET attempts = 2 WHERE task_id = 'a'"))
        flushes = 0
        real_checkpoint = h.store.checkpoint

        def checkpoint() -> None:
            nonlocal flushes
            flushes += 1
            real_checkpoint()

        h.store.checkpoint = checkpoint  # type: ignore[method-assign]

        await asyncio.gather(h.engine.suspend(), h.engine.suspend())

        assert h.task("a").attempts == 1  # type: ignore[union-attr]
        assert flushes == 1

    async def test_a_suspend_after_the_last_one_ended_is_a_new_one(self, make_engine: MakeEngine) -> None:
        h = started(make_engine([]))
        flushes = 0
        real_checkpoint = h.store.checkpoint

        def checkpoint() -> None:
            nonlocal flushes
            flushes += 1
            real_checkpoint()

        h.store.checkpoint = checkpoint  # type: ignore[method-assign]

        await h.engine.suspend()
        await h.engine.suspend()

        assert flushes == 2

    async def test_flushes_the_write_ahead_log(self, make_engine: MakeEngine, tmp_path: Any) -> None:
        h = started(make_engine([says("hi")]))
        h.engine.accept(user_message("m1"))
        await h.idle()
        wal = tmp_path / "state" / "engine.sqlite-wal"
        assert wal.exists() and wal.stat().st_size > 0

        await h.engine.suspend()

        assert not wal.exists() or wal.stat().st_size == 0

    async def test_a_new_event_or_the_key_ends_the_sleep(self, make_engine: MakeEngine) -> None:
        h = started(make_engine([says("up again")]))
        await h.engine.suspend()
        assert h.engine.is_suspending()

        h.engine.accept(user_message("m1"))
        await h.idle()

        assert not h.engine.is_suspending()
        assert h.events_of("message.assistant") == [{"text": "up again", "in_reply_to": "m1", "spent_usd": 0.0}]


class TestStopping:
    async def test_a_stopped_engine_takes_no_event_and_starts_no_turn(self, make_engine: MakeEngine) -> None:
        h = started(make_engine([]))

        await h.engine.stop()

        with pytest.raises(EngineStopped, match="shutting down"):
            h.engine.accept(user_message("m1"))
        h.engine.kick()
        assert h.asked() == 0

    async def test_stopping_ends_a_turn_in_flight_and_leaves_its_message_owed(self, make_engine: MakeEngine) -> None:
        gate = Gate()
        h = started(make_engine([gate.holds(says("never"))]))
        h.engine.accept(user_message("m1"))
        await gate.wait_reached()

        await h.engine.stop()

        assert h.inbound_state("m1") == "in_transcript"
        assert h.events_of("message.assistant") == []

    async def test_stopping_is_bounded_when_a_turn_ignores_its_cancellation(
        self, make_engine: MakeEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "_CANCEL_WAIT_S", 0.1)
        reached = asyncio.Event()
        release = asyncio.Event()

        async def stubborn(provider: Any) -> LLMResponse:
            reached.set()
            while not release.is_set():
                try:
                    await asyncio.sleep(0.01)
                except asyncio.CancelledError:
                    continue
            return says("late")

        h = started(make_engine([stubborn]))
        h.engine.accept(user_message("m1"))
        await asyncio.wait_for(reached.wait(), 10)

        try:
            # The shutdown goes on to close the database even though the turn is still there.
            async with asyncio.timeout(3):
                await h.engine.stop()
        finally:
            release.set()
            await h.idle()


class TestStoppingWithBrowsersOpen:
    async def test_stopping_closes_the_open_browsers_and_keeps_their_profiles(self, make_engine: MakeEngine) -> None:
        h = started(make_engine([]))
        identity = await h.browser.create("open one")
        await h.browser.launch(identity.id)

        await h.engine.stop()

        assert h.browser.open_count == 0
        assert h.events_of("browser.identity.closed") == [{"identity_id": identity.id, "name": "open one"}]
        assert h.browser.get(identity.id) is not None

    async def test_a_browser_still_closing_when_the_stop_ends_is_recorded_closed_while_the_store_takes_it(
        self, make_engine: MakeEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # On a loaded machine Firefox can take longer to flush than the stop gives it. The engine then ends, and
        # its exit ends the browser: the identity is closed now, not by a close that would finish after the store.
        monkeypatch.setattr(engine_module, "_CLOSE_BROWSERS_ON_STOP_S", 0.2)
        h = started(make_engine([]))
        identity = await h.browser.create("slow one")
        write_control(mcp_home(h.tmp_path, identity.id), slow_close_s=1.0)
        await h.browser.launch(identity.id)

        await h.engine.stop()

        assert h.events_of("browser.identity.closed") == [{"identity_id": identity.id, "name": "slow one"}]
        assert h.browser.open_count == 0
        # The close that was cut short ends later and records nothing more.
        await asyncio.sleep(1.5)
        assert len(h.events_of("browser.identity.closed")) == 1

    async def test_a_server_still_ending_when_the_stop_ends_has_its_identity_recorded_closed(
        self, make_engine: MakeEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # browser_close answered, the session was dropped, and its process is slow to end: the close is past the
        # point where the stop finds it among the open ones, and records `closed` only once the process is gone.
        monkeypatch.setattr(engine_module, "_CLOSE_BROWSERS_ON_STOP_S", 0.3)
        h = started(make_engine([]))
        identity = await h.browser.create("slow exit")
        write_control(mcp_home(h.tmp_path, identity.id), slow_exit_s=1.5)
        await h.browser.launch(identity.id)

        await h.engine.stop()

        assert h.events_of("browser.identity.closed") == [{"identity_id": identity.id, "name": "slow exit"}]
        await asyncio.sleep(2.5)
        assert len(h.events_of("browser.identity.closed")) == 1

    async def test_a_browser_that_does_not_close_does_not_hold_the_stop(
        self, make_engine: MakeEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "_CLOSE_BROWSERS_ON_STOP_S", 0.1)
        h = started(make_engine([]))
        hung = asyncio.Event()

        async def never_closes() -> None:
            hung.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(h.browser, "close_all", never_closes)

        async with asyncio.timeout(3):
            await h.engine.stop()

        assert hung.is_set()

    async def test_a_prepare_sleep_waits_for_a_slow_close_that_a_stop_would_cut(
        self, make_engine: MakeEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Only the host's 60 s wait on a prepare-sleep, while systemd's 30 s bound a stop.
        monkeypatch.setattr(engine_module, "_CLOSE_BROWSERS_ON_STOP_S", 0.1)
        monkeypatch.setattr(engine_module, "_CLOSE_BROWSERS_ON_SLEEP_S", 5.0)
        h = started(make_engine([]))
        closed: list[str] = []

        async def slow_close() -> None:
            await asyncio.sleep(0.5)
            closed.append("closed")

        monkeypatch.setattr(h.browser, "close_all", slow_close)

        await h.engine.suspend()
        assert closed == ["closed"]

        h.engine.key_received()
        closed.clear()
        await h.engine.stop()
        assert closed == []

    def test_the_steps_of_a_prepare_sleep_fit_inside_the_hosts_wait_with_a_margin(self) -> None:
        # The host gives up on a prepare-sleep after PREPARE_SLEEP_TIMEOUT_S and stops the guest, which would cut
        # the browsers mid-close: the grace, the wait for the cancelled turns and the closes have to end before
        # that, with room for the flush of the state and the answer.
        steps = (
            engine_module.STOP_GRACE_S + engine_module._CANCEL_WAIT_S + engine_module._CLOSE_BROWSERS_ON_SLEEP_S
        )
        margin_s = 5
        assert steps + margin_s <= PREPARE_SLEEP_TIMEOUT_S


class TestAChatTurnThatFailsBeforeItsOpeningIsStored:
    @staticmethod
    def full_disk(monkeypatch: pytest.MonkeyPatch) -> None:
        def refuse(conn: sqlite3.Connection, session_key: str) -> None:
            raise sqlite3.OperationalError("database or disk is full")

        monkeypatch.setattr(turns_module, "close_open_calls", refuse)

    async def test_answers_its_own_inputs_instead_of_being_started_again(
        self, make_engine: MakeEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.full_disk(monkeypatch)
        h = started(make_engine([]))

        h.engine.accept(user_message("m1"))
        async with asyncio.timeout(5):
            await h.idle()

        assert h.events_of("message.assistant") == [
            {"text": "I could not answer: database or disk is full", "in_reply_to": "m1", "spent_usd": 0.0}
        ]
        assert h.inbound_state("m1") == "applied"
        assert h.asked() == 0

    async def test_is_not_started_again_from_its_own_end_when_that_end_could_not_be_recorded(
        self, make_engine: MakeEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        h = started(make_engine([says("back")]))
        launches: list[str] = []
        launch = h.engine._launch

        def counting(slot: str, unit: Any, **options: Any) -> None:
            launches.append(slot)
            launch(slot, unit, **options)

        monkeypatch.setattr(h.engine, "_launch", counting)
        with monkeypatch.context() as broken:
            self.full_disk(broken)

            def refuse(conn: sqlite3.Connection, unit: Any, reason: str) -> None:
                raise sqlite3.OperationalError("database or disk is full")

            broken.setattr(Engine, "_answer_failure", staticmethod(refuse))
            h.engine.accept(user_message("m1"))
            async with asyncio.timeout(5):
                await h.idle()
            assert (launches, h.inbound_state("m1")) == (["chat"], "accepted")

        # The next kick takes it up again, now that the disk has room.
        h.engine.kick()
        await h.idle()
        assert launches == ["chat", "chat"]
        assert h.events_of("message.assistant") == [{"text": "back", "in_reply_to": "m1", "spent_usd": 0.0}]


class TestWhatTheHostSees:
    async def test_the_state_answer_names_the_running_task_and_the_oldest_pending_approval(
        self, make_engine: MakeEngine
    ) -> None:
        gate = Gate()
        h = started(make_engine([gate.holds(says("never"))]))
        h.engine.accept(task_created("a"))
        await gate.wait_reached()
        first = park(h, CHAT, None, now_ms=1000)
        park(h, CHAT, None, call_id="call-2", now_ms=2000)

        answer = h.engine.state_answer()

        assert (answer.state, answer.current_task_id, answer.pending_approval) == ("THINKING", "a", first)

    async def test_the_outbox_is_read_in_order_after_a_seq_and_listeners_hear_every_append(
        self, make_engine: MakeEngine
    ) -> None:
        h = make_engine([says("hi")])
        heard: list[int] = []
        remove = h.engine.on_append(lambda: heard.append(1))
        started(h)

        h.engine.accept(user_message("m1"))
        await h.idle()
        remove()

        events = h.engine.read_outbox_after(0, 100)
        assert [e["seq"] for e in events] == sorted(e["seq"] for e in events)
        assert [e["type"] for e in h.engine.read_outbox_after(events[1]["seq"], 100)] == [
            e["type"] for e in events[2:]
        ]
        assert len(heard) >= 3

    async def test_a_system_event_that_is_not_a_cancel_is_only_recorded(
        self, make_engine: MakeEngine
    ) -> None:
        h = started(make_engine([]))

        assert h.engine.accept(inbound("s1", "system.event", {"name": "something.else", "data": {}})) is True

        assert h.inbound_state("s1") == "applied"
        await h.idle()
        assert h.asked() == 0

    async def test_a_turn_that_cannot_start_does_not_spin_the_work_loop(self, make_engine: MakeEngine) -> None:
        h = make_engine([], key=False)
        started(h)
        h.engine.accept(user_message("m1"))
        await asyncio.sleep(0)

        assert h.asked() == 0
        assert h.count("dots_outbox") == 2
