"""What a turn's model requests cost, and the cap on it (`limits.max_cost_per_task_usd`).

OpenRouter reports the cost of a request in the usage of its final stream chunk, and the
provider puts it on `LLMResponse.cost_usd`. A turn gets one `TurnSpend`, bound to its
session:

- `TurnSpend.meter` wraps the turn's provider. The wrapper sits above `chat_stream_with_retry`,
  the one entry every model request of the engine goes through (the runner's, its retries
  and the summary requests of the consolidator), so every response is counted once, in its
  own store transaction, before the runner sees it. A failed request has no cost and counts
  for nothing.
- `TurnSpend.check` runs before every iteration of the runner and raises `CostCapReached`
  when the session has spent the cap: the turn ends as a failure whose text is the exception's.
- `TurnSpend.ensure_priced` runs before anything is done on the word of a response: before the
  tool calls of an assistant message are written (so they never run) and before the turn's final
  answer is written (which delivers it and completes a task). The last response of a turn is held
  to the same rule as the others, and so is one that asks for tools.

The spend lives in `dots_spend`, keyed by the session. A task's row is never reset, so the
cap holds across a restart, an approval and a resume. The chat's row is emptied by the answer
that reports it (`store.append_outbox_spent`), so the chat is capped between one answer and the
next: a call parked for approval, a restart or a sleep does not start the count again.

A request cannot be priced before it is answered, so the cap stops the turn from starting
another request: it may be exceeded by the last one, and an answer that crosses the cap is
delivered. A response that reports no cost leaves the cap unenforceable, so the turn fails
rather than carry on unmetered: before the response is acted on (its tool calls do not run, its
answer is not delivered and the task does not complete) or, for a request the runner does not
commit (a summary), at the next check. The note lives in the same ledger row as the money,
written in the same transaction, so a restart does not forget it; it goes where the row goes
(the chat's answer takes it).

What the ledger cannot hold is the cost of a request that never completed: OpenRouter reports
the cost in the last chunk of a stream, so a stream that stalled or was cut has none to count
(architecture 8.2 states the bound).
"""

from __future__ import annotations

from typing import Any, Literal

from nanobot.dots import store as dots_store
from nanobot.dots.store import DotStore
from nanobot.providers.base import LLMResponse

# What the cap is a cap of, in the text that says it was reached: a task, the chat's next answer, or a pass that
# brings MEMORY.md up to date (memory_update.py).
Scope = Literal["task", "turn", "memory pass"]


class CostCapReached(Exception):
    """The turn stops here: the money it may spend is gone, or can no longer be told."""


_UNPRICED_TEXT = (
    "stopped: OpenRouter reported no cost for a request, so limits.max_cost_per_task_usd cannot be enforced"
)


def _cap_text(cap_usd: float) -> str:
    """The cap as the person wrote it: two decimals, more only when the cap has more."""
    return f"{cap_usd:.2f}" if round(cap_usd, 2) == cap_usd else f"{cap_usd:g}"


class TurnSpend:
    """The spend of one turn on one session: meters its requests and says when the cap holds."""

    def __init__(self, store: DotStore, session_key: str, cap_usd: float, scope: Scope) -> None:
        self._store = store
        self._session_key = session_key
        self._cap_usd = cap_usd
        self._scope = scope

    def meter(self, provider: Any) -> MeteredProvider:
        """`provider` with every response of it counted against this session."""
        return MeteredProvider(provider, self)

    def record(self, response: LLMResponse) -> None:
        """Count one response: its cost into the ledger, or the note that it had none."""
        if response.finish_reason == "error":
            # The request failed (and may be retried) and reported no cost: nothing is counted. A stream that was cut
            # after tokens were streamed may have been billed, and OpenRouter reports its cost only in the last chunk,
            # so that cost is not known here (architecture section 8.2).
            return
        if response.cost_usd is None:
            self._store.write(lambda conn: dots_store.note_unpriced(conn, self._session_key))
        elif response.cost_usd > 0:
            usd = response.cost_usd
            self._store.write(lambda conn: dots_store.add_spend(conn, self._session_key, usd))

    def check(self) -> None:
        """Raise `CostCapReached` unless the turn may ask the model once more."""
        spent, unpriced = self._store.read(
            lambda conn: (
                dots_store.get_spend(conn, self._session_key),
                dots_store.has_unpriced(conn, self._session_key),
            )
        )
        if spent >= self._cap_usd:
            raise CostCapReached(
                f"stopped: the {self._scope} reached limits.max_cost_per_task_usd "
                f"(spent {spent:.4f} USD of {_cap_text(self._cap_usd)})"
            )
        if unpriced:
            raise CostCapReached(_UNPRICED_TEXT)

    def ensure_priced(self) -> None:
        """Raise `CostCapReached` when a request of the session reported no cost.

        `check` runs before a request, so a response is never checked by it before it is acted on;
        the cap itself does not apply here, because an answer that crosses it is delivered.
        """
        if self._store.read(lambda conn: dots_store.has_unpriced(conn, self._session_key)):
            raise CostCapReached(_UNPRICED_TEXT)


class MeteredProvider:
    """A provider that counts what its responses cost; everything else is the provider's own."""

    def __init__(self, provider: Any, spend: TurnSpend) -> None:
        self._provider = provider
        self._spend = spend

    async def chat_stream_with_retry(self, *args: Any, **kwargs: Any) -> LLMResponse:
        response: LLMResponse = await self._provider.chat_stream_with_retry(*args, **kwargs)
        self._spend.record(response)
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provider, name)
