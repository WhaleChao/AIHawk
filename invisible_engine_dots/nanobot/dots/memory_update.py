"""Brings the Dot's MEMORY.md up to date from its conversations, in the background.

MEMORY.md (in the Dot's memory directory) is given to the Dot in every prompt: what it should always know about
the person. The Dot may write it while it works, but a model often does not think to, so when the Dot has been
quiet for a while the engine takes in the conversation files that changed since the last pass
(`conversations.py`) and has the summary model rewrite MEMORY.md from them, in one request with no tools
(templates/agent/memory_update.md). Measured on LongMemEval (tests/bench/README.md): a profile written this way
raised the answers that need what the person likes from 65% to 83%, as well as the best of the open-source
memory systems tried, at about a seventeenth of their cost; an agent doing the same with file tools did worse
(75%) at five times the cost.

A pass takes the files whose time is after the newest it took last (`KV_THROUGH`), oldest first, as many to a
request as the model's window holds; a file too long for one request is cut between its messages. Each request
answers with the whole new MEMORY.md, written before the next one starts, so a pass cut short keeps what it
did. A request that is cut, empty or unpriced writes nothing; so does one whose MEMORY.md the Dot changed while
it ran (the next pass takes the files again). Its spend has a ledger of its own (`SESSION_KEY`), capped like a
task's, and is reported by the `memory.updated` that ends a pass.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from loguru import logger

from nanobot.agent.context_governance import answer_limit, prompt_budget
from nanobot.dots import conversations
from nanobot.dots import store as dots_store
from nanobot.dots.computer import Computer, ComputerError, FileTooLargeError
from nanobot.dots.projection import EngineSettings
from nanobot.dots.provider import OpenRouterProviders
from nanobot.dots.secrets import KeyHolder
from nanobot.dots.spend import CostCapReached, TurnSpend
from nanobot.dots.store import DotStore
from nanobot.dots.turns import MEMORY_DIR, MEMORY_INDEX, MEMORY_INDEX_MAX_CHARS, limits_of
from nanobot.utils.helpers import estimate_prompt_tokens_chain
from nanobot.utils.prompt_templates import render_template

# How long the Dot must have been quiet (no turn running or ended) before a pass: a conversation is taken in
# once it pauses, not after every message.
QUIET_S = 5 * 60
# The ledger of what passes spend, and the session key their requests are metered under.
SESSION_KEY = "memory"
# dots_kv: the time of the newest conversation file a pass took in.
KV_THROUGH = "memory_updated_through"
# The model role that writes MEMORY.md: the one that summarises a long thread.
MODEL_ROLE = "summary"
MEMORY_PATH = f"{MEMORY_DIR}/{MEMORY_INDEX}"
# The sub-directories of the conversations a pass reads.
SOURCES = ("chat", "tasks")


@dataclass(frozen=True)
class PassOutcome:
    """How a pass ended: nothing new to take in, MEMORY.md brought up to date, or a failure and why."""

    kind: Literal["nothing", "updated", "failed"]
    reason: str | None = None


@dataclass(frozen=True)
class _Conversation:
    """A conversation file, or a piece of one; `last` is false on every piece of a file but its last."""

    path: str
    mtime: datetime
    text: str
    last: bool = True


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _pieces(conversation: _Conversation, fits: Callable[[list[_Conversation]], bool]) -> list[_Conversation]:
    """`conversation` as one piece, or cut between its messages (`## ` headings) into pieces that fit a request.

    A single message too long for a request is cut by characters: nothing of it is left out.
    """
    if fits([conversation]):
        return [conversation]

    def piece(text: str) -> _Conversation:
        return _Conversation(conversation.path, conversation.mtime, text, last=False)

    blocks = conversation.text.split("\n## ")
    pieces: list[_Conversation] = []
    current = ""
    for index, block in enumerate(blocks):
        part = block if index == 0 else f"## {block}"
        candidate = f"{current}\n{part}" if current else part
        if fits([piece(candidate)]):
            current = candidate
            continue
        if current:
            pieces.append(piece(current))
        current = part
        while not fits([piece(current)]):
            cut = len(current) // 2
            while cut > 1 and not fits([piece(current[:cut])]):
                cut //= 2
            pieces.append(piece(current[:cut]))
            current = current[cut:]
    if current:
        pieces.append(piece(current))
    return [*pieces[:-1], _Conversation(conversation.path, conversation.mtime, pieces[-1].text)]


def batches(conversations_: Sequence[_Conversation], fits: Callable[[list[_Conversation]], bool]) -> list[list[_Conversation]]:
    """The conversations, oldest first, as few requests as `fits` allows, a long one cut into pieces."""
    out: list[list[_Conversation]] = []
    current: list[_Conversation] = []
    for conversation in conversations_:
        for piece in _pieces(conversation, fits):
            if current and fits([*current, piece]):
                current.append(piece)
            else:
                if current:
                    out.append(current)
                current = [piece]
    if current:
        out.append(current)
    return out


def render(memory_md: str, conversations_: Sequence[_Conversation]) -> str:
    return render_template(
        "agent/memory_update.md",
        memory_md=memory_md,
        conversations=[{"path": c.path, "text": c.text} for c in conversations_],
    )


class MemoryUpdater:
    """Runs one pass at a time, for the engine: it decides when."""

    def __init__(
        self,
        *,
        store: DotStore,
        computer: Computer,
        providers: OpenRouterProviders,
        key_holder: KeyHolder,
        settings_getter: Callable[[], EngineSettings | None],
    ) -> None:
        self._store = store
        self._computer = computer
        self._providers = providers
        self._key_holder = key_holder
        self._settings_getter = settings_getter

    async def run(self) -> PassOutcome:
        settings = self._settings_getter()
        if settings is None or not self._key_holder.configured:
            return PassOutcome("failed", "the Dot has no configuration or no OpenRouter key yet")
        try:
            return await self._run(settings)
        except CostCapReached as exc:
            return PassOutcome("failed", str(exc))
        except (ComputerError, FileTooLargeError) as exc:
            return PassOutcome("failed", f"the computer did not answer: {exc}")

    async def _run(self, settings: EngineSettings) -> PassOutcome:
        through_text = self._store.read(lambda conn: dots_store.read_kv(conn, KV_THROUGH))
        through = _parse_time(through_text) if isinstance(through_text, str) else None
        changed = await self._changed(through)
        if not changed:
            return PassOutcome("nothing")

        spend = TurnSpend(self._store, SESSION_KEY, settings.max_cost_usd, "memory pass")
        provider = spend.meter(self._providers.current(settings, self._key_holder.require()))
        model = settings.model_for(MODEL_ROLE)
        limits = await limits_of(provider, model)
        budget = prompt_budget(limits.context_tokens, limits.answer_tokens)
        # MEMORY.md changes from one request to the next: a request is sized with it at its longest, in prose.
        longest_memory = ("- a fact about the person, with the day it was said (2026-01-01)\n" * MEMORY_INDEX_MAX_CHARS)[
            :MEMORY_INDEX_MAX_CHARS
        ]

        def tokens_of(prompt: str) -> int:
            return estimate_prompt_tokens_chain(provider, model, [{"role": "user", "content": prompt}], None)[0]

        def fits(group: list[_Conversation]) -> bool:
            return not budget or tokens_of(render(longest_memory, group)) <= budget

        if not fits([]):
            return PassOutcome("failed", f"the window of {model} cannot hold MEMORY.md and the instructions for it")
        before = await self._read_memory()
        memory = before
        for group in batches(changed, fits):
            spend.check()
            prompt = render(memory, group)
            response = await provider.chat_stream_with_retry(
                messages=[{"role": "user", "content": prompt}],
                tools=None,
                model=model,
                max_tokens=answer_limit(limits.context_tokens, limits.answer_tokens, tokens_of(prompt)),
            )
            text = (response.content or "").strip()
            if response.finish_reason != "stop" or not text:
                return PassOutcome("failed", f"the model's answer ended with {response.finish_reason!r} and {len(text)} characters")
            spend.ensure_priced()
            # MEMORY.md is under /home/dot, which the host's file routes serve: a secret the model copied goes masked,
            # as in the conversation files.
            secrets = [*self._store.write(conversations.known_secrets), self._key_holder.require()]
            text = conversations.redact(text, secrets)
            if await self._read_memory() != memory:
                # The Dot wrote MEMORY.md while this request ran: its version stands, and the next pass takes these
                # conversations in again.
                return PassOutcome("failed", "MEMORY.md was changed while the pass ran")
            await self._computer.write_bytes(MEMORY_PATH, f"{text}\n".encode())
            memory = text
            # Only a file taken in whole moves the mark: a pass cut between the pieces of one takes it again.
            whole = [c.mtime for c in group if c.last]
            if whole:
                newest = max(whole).isoformat()
                self._store.write(lambda conn: dots_store.write_kv(conn, KV_THROUGH, newest))

        changed_memory = memory != before
        self._store.write(
            lambda conn: dots_store.append_outbox_spent(
                conn, "memory.updated", {"conversations": len(changed), "changed": changed_memory}, SESSION_KEY
            )
        )
        logger.info("memory pass took in {} conversation files; MEMORY.md changed={}", len(changed), changed_memory)
        return PassOutcome("updated")

    async def _changed(self, through: datetime | None) -> list[_Conversation]:
        """The conversation files whose time is after `through`, oldest first, with their text."""
        found: list[tuple[datetime, str]] = []
        for source in SOURCES:
            directory = f"{conversations.CONVERSATIONS_DIR}/{source}"
            for entry in await self._computer.list_dir(directory) or []:
                if entry.type != "file" or not entry.name.endswith(".md"):
                    continue
                mtime = _parse_time(entry.mtime)
                if through is None or mtime > through:
                    found.append((mtime, f"{directory}/{entry.name}"))
        out: list[_Conversation] = []
        for mtime, path in sorted(found):
            data = await self._computer.read_bytes(path)
            if data is not None:
                out.append(_Conversation(path, mtime, data.decode("utf-8", errors="replace")))
        return out

    async def _read_memory(self) -> str:
        data = await self._computer.read_bytes(MEMORY_PATH)
        return (data or b"").decode("utf-8", errors="replace").strip()

