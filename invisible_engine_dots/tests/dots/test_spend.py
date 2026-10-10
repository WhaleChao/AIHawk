"""What a turn's requests cost: the meter around the provider, the ledger and the cap's check."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fakes.scripted_provider import ScriptedProvider, says
from openai.types.chat import ChatCompletionChunk

from nanobot.dots import store as s
from nanobot.dots.spend import CostCapReached, MeteredProvider, TurnSpend
from nanobot.dots.store import DotStore
from nanobot.providers.base import LLMResponse
from nanobot.providers.openai_compat_provider import OpenAICompatProvider
from nanobot.providers.registry import OPENROUTER

MESSAGES: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]


def spent(store: DotStore, session_key: str = "chat") -> float:
    return store.read(lambda conn: s.get_spend(conn, session_key))


def metered(store: DotStore, script: list[Any], cap: float = 1.0, scope: Any = "turn") -> tuple[TurnSpend, MeteredProvider]:
    spend = TurnSpend(store, "chat", cap, scope)
    # No default cost: a response says what it cost, or it reported none.
    return spend, spend.meter(ScriptedProvider(script, default_cost=None))


async def ask(provider: MeteredProvider) -> LLMResponse:
    return await provider.chat_stream_with_retry(messages=MESSAGES)


class TestTheMeter:
    async def test_every_response_adds_its_cost_to_the_session(self, dot_store: DotStore) -> None:
        _, provider = metered(dot_store, [says("a", cost=0.25), says("b", cost=0.5)])

        first = await ask(provider)
        second = await ask(provider)

        assert (first.content, second.content) == ("a", "b")
        assert spent(dot_store) == pytest.approx(0.75)

    async def test_the_response_is_handed_on_as_it_is(self, dot_store: DotStore) -> None:
        response = says("a", cost=0.25)
        _, provider = metered(dot_store, [response])

        assert await ask(provider) is response

    async def test_a_request_that_failed_costs_nothing_and_is_not_a_missing_cost(self, dot_store: DotStore) -> None:
        spend, provider = metered(dot_store, [LLMResponse(content="rate limited", finish_reason="error")])

        await ask(provider)

        assert spent(dot_store) == 0
        spend.check()

    async def test_a_response_that_cost_nothing_is_priced(self, dot_store: DotStore) -> None:
        spend, provider = metered(dot_store, [says("free", cost=0.0)])

        await ask(provider)

        assert spent(dot_store) == 0
        spend.check()

    async def test_the_spend_of_a_session_is_not_the_spend_of_another(self, dot_store: DotStore) -> None:
        scripted = ScriptedProvider([says("a", cost=0.25), says("b", cost=0.5)])
        chat = TurnSpend(dot_store, "chat", 1.0, "turn").meter(scripted)
        task = TurnSpend(dot_store, "task:t1", 1.0, "task").meter(scripted)

        await ask(chat)
        await ask(task)

        assert (spent(dot_store, "chat"), spent(dot_store, "task:t1")) == (0.25, 0.5)

    async def test_everything_else_is_the_providers_own(self, dot_store: DotStore) -> None:
        scripted = ScriptedProvider([], max_tokens=321)
        provider = TurnSpend(dot_store, "chat", 1.0, "turn").meter(scripted)

        assert provider.generation.max_tokens == 321
        assert provider.get_default_model() == "scripted/model"
        assert await provider.model_limits("m") == await scripted.model_limits("m")


class TestTheRetriesOfARequest:
    """The meter sits above `chat_stream_with_retry`: a request that was retried is one response."""

    async def test_a_request_that_failed_once_and_then_answered_counts_the_answer_once(
        self, dot_store: DotStore
    ) -> None:
        final = ChatCompletionChunk.model_validate(
            {
                "id": "gen-1",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "m",
                "choices": [{"index": 0, "delta": {"content": "pong"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18, "cost": 0.03},
            }
        )

        class Stream:
            def __init__(self) -> None:
                self.chunks = [final]

            def __aiter__(self) -> Stream:
                return self

            async def __anext__(self) -> Any:
                if self.chunks:
                    return self.chunks.pop(0)
                raise StopAsyncIteration

        provider = OpenAICompatProvider(
            api_key="sk-or-v1-0123456789abcdef", api_base="https://example.com/v1",
            default_model="m", spec=OPENROUTER, provider_name=OPENROUTER.name,
        )
        create = AsyncMock(side_effect=[ConnectionError("connection reset"), Stream()])
        provider._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))  # type: ignore[assignment]
        provider._CHAT_RETRY_DELAYS = (0,)
        wrapped = TurnSpend(dot_store, "chat", 1.0, "turn").meter(provider)

        response = await wrapped.chat_stream_with_retry(messages=MESSAGES, max_tokens=10)

        assert (response.content, create.await_count) == ("pong", 2)
        assert spent(dot_store) == pytest.approx(0.03)


class TestTheCheck:
    async def test_a_session_under_the_cap_may_ask_again(self, dot_store: DotStore) -> None:
        spend, provider = metered(dot_store, [says("a", cost=0.99)], cap=1.0)

        await ask(provider)

        spend.check()

    @pytest.mark.parametrize("scope, noun", [("task", "task"), ("turn", "turn")])
    async def test_a_session_at_the_cap_may_not_and_the_text_names_what_it_spent(
        self, dot_store: DotStore, scope: str, noun: str
    ) -> None:
        spend, provider = metered(dot_store, [says("a", cost=1.0423)], cap=1.0, scope=scope)

        await ask(provider)

        with pytest.raises(CostCapReached) as stopped:
            spend.check()
        assert str(stopped.value) == f"stopped: the {noun} reached limits.max_cost_per_task_usd (spent 1.0423 USD of 1.00)"

    async def test_the_cap_is_met_when_the_spend_equals_it(self, dot_store: DotStore) -> None:
        spend, provider = metered(dot_store, [says("a", cost=0.5), says("b", cost=0.5)], cap=1.0)

        await ask(provider)
        spend.check()
        await ask(provider)

        with pytest.raises(CostCapReached, match=r"spent 1\.0000 USD of 1\.00"):
            spend.check()

    @pytest.mark.parametrize("cap, written", [(0.01, "0.01"), (100.0, "100.00"), (0.015, "0.015"), (2.5, "2.50")])
    async def test_the_cap_is_written_with_two_decimals_and_more_when_it_has_more(
        self, dot_store: DotStore, cap: float, written: str
    ) -> None:
        spend, provider = metered(dot_store, [says("a", cost=1000.0)], cap=cap)

        await ask(provider)

        with pytest.raises(CostCapReached, match=f"of {written}\\)"):
            spend.check()

    async def test_the_spend_of_an_earlier_turn_counts(self, dot_store: DotStore) -> None:
        first, provider = metered(dot_store, [says("a", cost=0.7)], scope="task")
        await ask(provider)

        # A new turn on the same session: nothing in memory, the ledger holds the spend.
        later = TurnSpend(dot_store, "chat", 0.5, "task")

        with pytest.raises(CostCapReached, match=r"spent 0\.7000 USD of 0\.50"):
            later.check()
        first.check()

    async def test_a_response_that_reports_no_cost_fails_the_next_check_closed(self, dot_store: DotStore) -> None:
        spend, provider = metered(dot_store, [says("a")])

        await ask(provider)

        with pytest.raises(CostCapReached) as stopped:
            spend.check()
        assert str(stopped.value) == (
            "stopped: OpenRouter reported no cost for a request, so limits.max_cost_per_task_usd cannot be enforced"
        )

    async def test_a_response_that_reports_no_cost_fails_the_checks_of_a_turn_that_starts_after_a_restart(
        self, dot_store: DotStore
    ) -> None:
        _, provider = metered(dot_store, [says("a")])
        await ask(provider)

        # A new turn on the same session (the engine was killed in between): nothing in memory,
        # the ledger knows that a request went unmetered.
        restarted = TurnSpend(dot_store, "chat", 1.0, "turn")

        with pytest.raises(CostCapReached, match="reported no cost"):
            restarted.check()

    async def test_the_answer_of_a_turn_whose_last_request_had_no_cost_is_not_a_priced_one(
        self, dot_store: DotStore
    ) -> None:
        spend, provider = metered(dot_store, [says("final answer")])
        await ask(provider)

        with pytest.raises(CostCapReached, match="reported no cost"):
            spend.ensure_priced()

    async def test_a_turn_that_met_the_cap_with_a_priced_answer_has_nothing_to_complain_of(
        self, dot_store: DotStore
    ) -> None:
        spend, provider = metered(dot_store, [says("a", cost=2.0)], cap=1.0)
        await ask(provider)

        # An answer that crosses the cap is delivered (the cap stops further requests only).
        spend.ensure_priced()

    async def test_a_response_that_reports_no_cost_is_not_counted_as_free_spend(self, dot_store: DotStore) -> None:
        _, provider = metered(dot_store, [says("a", cost=0.25), says("b")])

        await ask(provider)
        await ask(provider)

        assert spent(dot_store) == 0.25
