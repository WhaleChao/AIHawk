"""The memory pass: MEMORY.md brought up to date from the conversations that changed, and when the engine runs it."""

from __future__ import annotations

import asyncio
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fakes.dot_config import runtime_config_body
from fakes.engine_harness import EngineHarness, user_message
from fakes.local_computer import LocalComputer
from fakes.scripted_provider import Gate, ScriptedProvider, says
from fakes.turn_harness import KEY, FixedProviders

from nanobot.dots import memory_update as m
from nanobot.dots import store as s
from nanobot.dots.conversations import CONVERSATIONS_DIR
from nanobot.dots.projection import project
from nanobot.dots.protocol import parse_runtime_config
from nanobot.dots.secrets import KeyHolder
from nanobot.dots.store import DotStore
from nanobot.providers.base import LLMResponse

PROFILE = "## The person\n- has a cat named Luna, who sheds (2023-05-20)"


class Pass:
    """A memory updater on a real store, a local computer and a scripted model."""

    def __init__(self, tmp_path: Path, store: DotStore, script: list[Any], **config: Any) -> None:
        self.store = store
        self.computer = LocalComputer(tmp_path, tmp_path / "home" / "dot" / "workspace")
        self.provider = ScriptedProvider(script)
        keys = KeyHolder()
        keys.set(KEY)
        settings = project(parse_runtime_config(runtime_config_body(**config)), workspace="/home/dot/workspace", openrouter_base_url=None)
        self.updater = m.MemoryUpdater(
            store=store,
            computer=self.computer,
            providers=FixedProviders(self.provider),
            key_holder=keys,
            settings_getter=lambda: settings,
        )
        self._clock = 1_700_000_000

    def conversation(self, name: str, text: str, source: str = "chat") -> Path:
        """Write a conversation file, each one later than the last."""
        path = self.computer._local(f"{CONVERSATIONS_DIR}/{source}/{name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        self._clock += 60
        os.utime(path, (self._clock, self._clock))
        return path

    def memory(self) -> str | None:
        path = self.computer._local(m.MEMORY_PATH)
        return path.read_text(encoding="utf-8") if path.exists() else None

    def write_memory(self, text: str) -> None:
        path = self.computer._local(m.MEMORY_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def events(self) -> list[tuple[str, dict[str, Any]]]:
        return [(e["type"], e["data"]) for e in self.store.read(lambda conn: s.read_outbox_after(conn, 0, 1000))]

    def through(self) -> Any:
        return self.store.read(lambda conn: s.read_kv(conn, m.KV_THROUGH))


@pytest.fixture
def make_pass(tmp_path: Path, dot_store: DotStore):
    return lambda script, **config: Pass(tmp_path, dot_store, script, **config)


DAY = "# Chat, 2023-05-20\n\n## 09:15 the person\n\nmy cat is called Luna and she sheds a lot\n\n## 09:15 you\n\nNoted.\n"


class TestAPass:
    async def test_with_nothing_new_it_asks_nothing(self, make_pass) -> None:
        p = make_pass([])

        assert await p.updater.run() == m.PassOutcome("nothing")
        assert p.provider.requests == [] and p.events() == []

    async def test_the_conversations_and_memory_md_go_in_one_request_and_the_answer_is_memory_md(self, make_pass) -> None:
        p = make_pass([says(PROFILE, cost=0.002)])
        p.write_memory("## The person\n- lives in Turin (2023-05-01)\n")
        p.conversation("2023-05-20.md", DAY)

        assert await p.updater.run() == m.PassOutcome("updated")

        (request,) = p.provider.requests
        prompt = request["messages"][0]["content"]
        assert request["tools"] is None
        assert "lives in Turin (2023-05-01)" in prompt
        assert f'<conversation file="{CONVERSATIONS_DIR}/chat/2023-05-20.md">' in prompt
        assert "my cat is called Luna" in prompt
        assert p.memory() == PROFILE + "\n"
        assert p.events() == [("memory.updated", {"conversations": 1, "changed": True, "spent_usd": 0.002})]

    async def test_it_writes_with_the_summary_model(self, make_pass) -> None:
        p = make_pass([says(PROFILE)], models={"summary": "google/gemini-2.5-flash-lite"})
        p.conversation("2023-05-20.md", DAY)

        await p.updater.run()

        assert p.provider.requests[0]["model"] == "google/gemini-2.5-flash-lite"

    async def test_the_next_pass_takes_only_what_changed_since(self, make_pass) -> None:
        p = make_pass([says(PROFILE), says(PROFILE + "\n- moved to Rome (2023-05-22)")])
        p.conversation("2023-05-20.md", DAY)
        p.conversation("2023-05-21-t1.md", "# Task t1\n\n## 10:00 the task\n\nbook the vet\n", source="tasks")
        await p.updater.run()
        assert await p.updater.run() == m.PassOutcome("nothing")

        p.conversation("2023-05-22.md", "# Chat, 2023-05-22\n\n## 08:00 the person\n\nI moved to Rome\n")
        assert await p.updater.run() == m.PassOutcome("updated")

        first, second = (request["messages"][0]["content"] for request in p.provider.requests)
        assert "book the vet" in first and "my cat is called Luna" in first
        assert "I moved to Rome" in second and "my cat is called Luna" not in second and "book the vet" not in second

    async def test_an_answer_cut_or_empty_writes_nothing_and_takes_the_conversations_again(self, make_pass) -> None:
        p = make_pass([LLMResponse(content="## The pers", finish_reason="length"), says("   "), says(PROFILE)])
        p.write_memory("old\n")
        p.conversation("2023-05-20.md", DAY)

        assert (await p.updater.run()).kind == "failed"
        assert (await p.updater.run()).kind == "failed"
        assert p.memory() == "old\n" and p.through() is None

        assert await p.updater.run() == m.PassOutcome("updated")
        assert p.memory() == PROFILE + "\n"

    async def test_memory_md_the_dot_changed_while_the_request_ran_is_kept(self, make_pass) -> None:
        def the_dot_writes(provider: ScriptedProvider) -> LLMResponse:
            p.write_memory("written by the Dot meanwhile\n")
            return says(PROFILE)

        p = make_pass([the_dot_writes])
        p.conversation("2023-05-20.md", DAY)

        outcome = await p.updater.run()

        assert outcome == m.PassOutcome("failed", "MEMORY.md was changed while the pass ran")
        assert p.memory() == "written by the Dot meanwhile\n" and p.through() is None

    async def test_an_unpriced_answer_is_not_written(self, make_pass) -> None:
        p = make_pass([says(PROFILE)])
        p.provider.default_cost = None
        p.conversation("2023-05-20.md", DAY)

        outcome = await p.updater.run()

        assert outcome.kind == "failed" and "reported no cost" in (outcome.reason or "")
        assert p.memory() is None

    async def test_the_cap_stops_a_pass_before_it_asks(self, make_pass) -> None:
        p = make_pass([], limits={"max_steps_per_task": 60, "max_cost_per_task_usd": 0.01})
        p.store.write(lambda conn: s.add_spend(conn, m.SESSION_KEY, 0.02))
        p.conversation("2023-05-20.md", DAY)

        outcome = await p.updater.run()

        assert outcome.kind == "failed" and "memory pass reached limits.max_cost_per_task_usd" in (outcome.reason or "")
        assert p.provider.requests == []

    async def test_a_failed_pass_s_spend_is_reported_by_the_next_memory_updated(self, make_pass) -> None:
        p = make_pass([LLMResponse(content="cut", finish_reason="length", cost_usd=0.001), says(PROFILE, cost=0.002)])
        p.conversation("2023-05-20.md", DAY)
        await p.updater.run()

        await p.updater.run()

        assert p.events() == [("memory.updated", {"conversations": 1, "changed": True, "spent_usd": 0.003})]
        assert p.store.read(lambda conn: s.get_spend(conn, m.SESSION_KEY)) == 0

    async def test_conversations_too_long_for_one_request_go_in_several_each_written(self, make_pass) -> None:
        answers = [says(f"{PROFILE}\n- step {i}") for i in range(20)]
        p = make_pass(answers)
        p.provider.default_limits = replace(p.provider.default_limits, context_tokens=40_000, answer_tokens=1000)
        long_day = "# Chat, 2023-05-20\n\n" + "".join(f"## 09:{i:02d} the person\n\nfact number {i} " + "word " * 900 + "\n\n" for i in range(30))
        p.conversation("2023-05-20.md", long_day)
        p.conversation("2023-05-21.md", DAY)

        assert await p.updater.run() == m.PassOutcome("updated")

        prompts = [request["messages"][0]["content"] for request in p.provider.requests]
        assert len(prompts) > 1
        everything = "".join(prompts)
        assert all(f"fact number {i} " in everything for i in range(30)) and "my cat is called Luna" in everything
        # Each request builds on the MEMORY.md the one before wrote.
        assert f"{PROFILE}\n- step 0" in prompts[1]
        assert p.memory() == f"{PROFILE}\n- step {len(prompts) - 1}\n"


    async def test_a_pass_cut_between_the_pieces_of_a_file_takes_the_whole_file_again(self, make_pass) -> None:
        cut = LLMResponse(content="cut", finish_reason="length")
        p = make_pass([says(f"{PROFILE}\n- step 0"), cut, *(says(f"{PROFILE}\n- again {i}") for i in range(20))])
        p.provider.default_limits = replace(p.provider.default_limits, context_tokens=40_000, answer_tokens=1000)
        long_day = "# Chat, 2023-05-20\n\n" + "".join(f"## 09:{i:02d} the person\n\nfact number {i} " + "word " * 900 + "\n\n" for i in range(30))
        p.conversation("2023-05-20.md", long_day)

        assert (await p.updater.run()).kind == "failed"
        assert p.through() is None

        assert await p.updater.run() == m.PassOutcome("updated")
        again = p.provider.requests[2]["messages"][0]["content"]
        assert "fact number 0 " in again

    async def test_a_secret_the_model_copies_into_memory_md_is_masked(self, make_pass) -> None:
        proxy = "http://shopper:Pw7c1dSecret@10.0.0.5:8099"
        p = make_pass([says(f"{PROFILE}\n- shops through the proxy {proxy} (2023-05-20)")])
        p.store.write(lambda conn: s.insert_identity(conn, identity_id="id1", name="shopping", proxy=proxy))
        p.conversation("2023-05-20.md", DAY)

        assert await p.updater.run() == m.PassOutcome("updated")

        memory = p.memory() or ""
        assert "shops through the proxy *** (2023-05-20)" in memory and "Pw7c1dSecret" not in memory

    async def test_a_window_too_small_for_memory_md_fails_without_asking(self, make_pass) -> None:
        p = make_pass([])
        p.provider.default_limits = replace(p.provider.default_limits, context_tokens=6_000, answer_tokens=1000)
        p.conversation("2023-05-20.md", DAY)

        outcome = await p.updater.run()

        assert outcome.kind == "failed" and "cannot hold MEMORY.md" in (outcome.reason or "")
        assert p.provider.requests == []


class TestTheEngine:
    async def test_a_pass_runs_once_the_dot_has_been_quiet_after_a_turn(self, tmp_path: Path, dot_store: DotStore) -> None:
        h = EngineHarness(tmp_path, dot_store, [says("hi"), says(PROFILE)], memory_quiet_s=0.2)
        h.configure(runtime_config_body())
        h.engine.start()
        h.engine.accept(user_message("in1", "my cat is called Luna"))
        await h.idle()
        assert "memory.updated" not in h.types()

        await h.wait_until(lambda: "memory.updated" in h.types())

        assert h.events_of("memory.updated")[0]["conversations"] == 1
        assert "my cat is called Luna" in h.provider.requests[1]["messages"][0]["content"]
        assert (tmp_path / "home" / "dot" / "memory" / "MEMORY.md").read_text(encoding="utf-8") == PROFILE + "\n"
        await h.engine.stop()

    async def test_no_pass_runs_while_a_turn_does(self, tmp_path: Path, dot_store: DotStore) -> None:
        gate = Gate()
        h = EngineHarness(tmp_path, dot_store, [gate.holds(says("done"))], memory_quiet_s=0.05)
        h.configure(runtime_config_body())
        h.engine.start()
        h.engine.accept(user_message("in1"))
        await gate.wait_reached()

        await asyncio.sleep(0.3)

        assert h.engine._memory_pass is None and h.asked() == 1
        gate.release.set()
        await h.engine.stop()

    async def test_a_sleep_cuts_a_pass_and_the_next_quiet_spell_takes_it_up(self, tmp_path: Path, dot_store: DotStore) -> None:
        gate = Gate()
        h = EngineHarness(tmp_path, dot_store, [gate.holds(says(PROFILE)), says(PROFILE)], memory_quiet_s=0.05)
        h.configure(runtime_config_body())
        day = h.computer._local(f"{CONVERSATIONS_DIR}/chat/2023-05-20.md")
        day.parent.mkdir(parents=True)
        day.write_text(DAY, encoding="utf-8")
        h.engine.start()
        await gate.wait_reached()

        await h.engine.suspend()

        assert h.engine._memory_pass is None and "memory.updated" not in h.types()
        h.engine.key_received()
        await h.wait_until(lambda: "memory.updated" in h.types())
        assert h.asked() == 2
        await h.engine.stop()

    async def test_a_start_with_conversations_from_before_takes_them_in(self, tmp_path: Path, dot_store: DotStore) -> None:
        h = EngineHarness(tmp_path, dot_store, [says(PROFILE)], memory_quiet_s=0.05)
        h.configure(runtime_config_body())
        day = h.computer._local(f"{CONVERSATIONS_DIR}/chat/2023-05-20.md")
        day.parent.mkdir(parents=True)
        day.write_text(DAY, encoding="utf-8")

        h.engine.start()

        await h.wait_until(lambda: "memory.updated" in h.types())
        await h.engine.stop()

