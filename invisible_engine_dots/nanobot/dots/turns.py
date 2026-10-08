"""The turn runner: the one place a model turn starts.

Every turn of the Dot (a chat turn, a task, the continuation after an approval, a
resume after a restart) is one `TurnRunner.run`. It makes the Dot's durable
transcript the source of the model's context, hands nanobot's `AgentRunner` the
Dot's policy and its commit callback, and says how the run ended. What an outcome
means (answering the chat, failing a task, the agent state) belongs to the engine.

Commit points of a turn, each one store transaction with the outbox rows that
describe it:
- the opening: closing the calls the previous unit left open, the opening messages, and for a
  chat turn the start of its spend;
- every message the runner adds, one at a time, before it goes on (`_commit`);
- a call's intent with the EXECUTING state (`DotsTurnHook`, through the host);
- the summary checkpoint, at the end.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from typing import Any, Literal, Protocol, cast

from loguru import logger

from nanobot.agent.context import ContextBuilder, TranscriptInput
from nanobot.agent.context_governance import ContextWindowExceededError, prompt_budget
from nanobot.agent.hook import AgentHook, AgentHookContext, AgentRunHookContext
from nanobot.agent.memory import Consolidator, recent_user_message_tokens
from nanobot.agent.runner import AgentRunner, AgentRunResult, AgentRunSpec
from nanobot.agent.tools.context import RequestContext, bind_request_context, reset_request_context
from nanobot.agent.tools.file_state import FileStateStore, bind_file_states, reset_file_states
from nanobot.agent.tools.gate_types import ToolGate
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.transcript_metadata import METADATA_KEY
from nanobot.dots import conversations
from nanobot.dots import store as dots_store
from nanobot.dots.computer import Computer, ComputerError, Entry
from nanobot.dots.gate import close_open_calls
from nanobot.dots.images import TurnImages, bind_turn_images, reset_turn_images
from nanobot.dots.permissions import tool_starts_terminal, tool_target
from nanobot.dots.projection import EngineSettings
from nanobot.dots.provider import OpenRouterProviders
from nanobot.dots.secrets import KeyHolder
from nanobot.dots.skills import all_skills
from nanobot.dots.spend import CostCapReached, TurnSpend
from nanobot.dots.store import CHAT_SESSION_KEY, DotStore, ToolIntent
from nanobot.providers.base import ModelLimits, ToolCallRequest
from nanobot.session.manager import Session
from nanobot.session.summary import session_summary_from_metadata
from nanobot.utils.llm_runtime import LLMRuntime

# The Dot's long-term notes: one file per note on its own computer, kept by the Dot with the file tools.
MEMORY_DIR = "/home/dot/memory"
# How many of the most recently changed memory notes the prompt names.
MEMORY_NOTES_LISTED = 20


class TurnAbandoned(Exception):
    """The turn stops here without an outcome of its own: the engine is suspending."""


@dataclass(frozen=True)
class OpeningMessage:
    """A user message that opens a turn: the text, and the metadata the engine reads back.

    Metadata by kind: an inbound user.message or automation.fired carries
    {"dots_inbound_id": id}, an approval continuation {"dots_approval_id": id}; a
    task description or a resume carries none.
    """

    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


# The runner's checkpoint phases that act on a model response, so an unpriced one must stop the turn first.
_ACTS_ON_A_RESPONSE = frozenset({"assistant_tool_calls", "final_response"})

# What the engine hands a running chat turn: the inputs accepted since the last look, oldest first.
InjectionSource = Callable[[], Awaitable[Sequence[OpeningMessage]]]


@dataclass(frozen=True)
class TurnUnit:
    """One turn's worth of work.

    session_key: the chat or the task whose transcript the turn continues.
    task_id: the task, for a task session.
    opening: the user messages to append before the model is asked.
    approval_id: the approval this turn tells the session about, for a continuation.
    """

    session_key: str
    task_id: str | None
    opening: tuple[OpeningMessage, ...] = ()
    approval_id: str | None = None


@dataclass(frozen=True)
class TurnOutcome:
    """How a turn ended.

    completed: a final answer was committed (the transcript hook already answered the
    chat or completed the task); parked: a call waits for the user's approval;
    failed: the turn ended without an answer, `reason` says why; abandoned: the
    engine suspended the turn.
    """

    kind: Literal["completed", "parked", "failed", "abandoned"]
    reason: str | None = None

    @classmethod
    def failed(cls, reason: str) -> TurnOutcome:
        return cls("failed", reason)


class TurnHost(Protocol):
    """What the engine does for a turn: it owns the agent state machine.

    A turn that ends while a tool still runs (cancelled) never reports `tool_ended`;
    the engine drops its own counters when the run ends.
    """

    def is_suspending(self) -> bool:
        """Whether the engine is stopping work (prepare-sleep, SIGTERM)."""
        ...

    def run_started(self, unit: TurnUnit) -> None:
        """The runner asks the model: THINKING."""
        ...

    def tool_started(self, intent: ToolIntent) -> None:
        """A call is about to run: record its intent and EXECUTING in one transaction."""
        ...

    def tool_ended(self, session_key: str) -> None:
        """A call of the turn on `session_key` returned or failed: THINKING again when no tool runs."""
        ...


class DotsTurnHook(AgentHook):
    """The runner's lifecycle seen by the Dot: agent state, intents, suspension, the cost cap."""

    def __init__(self, unit: TurnUnit, host: TurnHost, spend: TurnSpend) -> None:
        super().__init__()
        self._unit = unit
        self._host = host
        self._spend = spend

    async def before_run(self, context: AgentRunHookContext) -> None:
        self._host.run_started(self._unit)

    async def before_iteration(self, context: AgentHookContext) -> None:
        if self._host.is_suspending():
            raise TurnAbandoned()
        self._spend.check()

    async def before_execute_tool(
        self,
        context: AgentHookContext,
        tool_call: ToolCallRequest,
        tool: Any,
        params: Any,
    ) -> None:
        self._host.tool_started(
            ToolIntent(
                tool_call_id=tool_call.id,
                tool=tool_call.name,
                session_key=self._unit.session_key,
                task_id=self._unit.task_id,
                started_at=dots_store.clock_ms(),
                target=tool_target(tool_call.name, params),
                tty=tool_starts_terminal(tool_call.name, params),
            )
        )

    async def after_execute_tool(
        self,
        context: AgentHookContext,
        tool_call: ToolCallRequest,
        tool: Any,
        params: Any,
        result: Any,
    ) -> None:
        self._host.tool_ended(self._unit.session_key)

    async def on_execute_tool_error(
        self,
        context: AgentHookContext,
        tool_call: ToolCallRequest,
        tool: Any,
        params: Any,
        error: Any,
    ) -> None:
        self._host.tool_ended(self._unit.session_key)


def _stamped(message: Mapping[str, Any]) -> dict[str, Any]:
    """The message as stored: with the time it was written, which the summary of old messages cites."""
    return {"timestamp": datetime.now().isoformat(), **message}


def _mtime(entry: Entry) -> datetime:
    try:
        return datetime.fromisoformat(entry.mtime)
    except ValueError:
        return datetime.fromtimestamp(0).astimezone()


class TurnRunner:
    """Runs one turn at a time per call; the engine decides which and when."""

    def __init__(
        self,
        *,
        store: DotStore,
        computer: Computer,
        base_registry: ToolRegistry,
        providers: OpenRouterProviders,
        key_holder: KeyHolder,
        settings_getter: Callable[[], EngineSettings | None],
        gate: ToolGate,
        consolidator: Consolidator,
        host: TurnHost,
        injection_source: InjectionSource | None = None,
    ) -> None:
        self._store = store
        self._computer = computer
        self._base_registry = base_registry
        self._providers = providers
        self._key_holder = key_holder
        self._settings_getter = settings_getter
        self._gate = gate
        self._consolidator = consolidator
        self._host = host
        self._injection_source = injection_source
        self._runner = AgentRunner()
        # What the model read and wrote, per session, for the tools' read-before-write checks.
        self._file_states = FileStateStore()

    async def run(self, unit: TurnUnit) -> TurnOutcome:
        settings = self._settings_getter()
        if settings is None:
            return TurnOutcome.failed("the Dot has no configuration yet")
        if not self._key_holder.configured:
            return TurnOutcome.failed("the OpenRouter key has not arrived yet")
        logger.debug(
            "turn start session={} task={} approval={}", unit.session_key, unit.task_id, unit.approval_id,
        )
        try:
            outcome = await self._run(unit, settings)
        except TurnAbandoned:
            outcome = TurnOutcome("abandoned")
        except CostCapReached as exc:
            outcome = TurnOutcome.failed(str(exc))
        except ContextWindowExceededError as exc:
            # Said in words: the error's own text ("6371/0 via tiktoken") reached the chat as it was. The window is
            # the model's own, so what is left to change is the model, or what the request carries.
            outcome = TurnOutcome.failed(
                f"the request needs {exc.estimated_tokens} tokens and the model {settings.model_id} takes "
                f"{exc.input_budget} once the room for its answer is kept, even with the older part of the thread "
                "summarised; choose a model with a larger context window"
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.opt(exception=True).error("turn failed for {}", unit.session_key)
            outcome = TurnOutcome.failed(str(exc) or type(exc).__name__)
        await self._write_conversations(unit)
        return outcome

    async def _run(self, unit: TurnUnit, settings: EngineSettings) -> TurnOutcome:
        session_key = unit.session_key

        def open_turn(conn: sqlite3.Connection) -> Session:
            close_open_calls(conn, session_key)
            if unit.opening:
                dots_store.append_messages(
                    conn,
                    session_key,
                    [_stamped(_opening_message(opening)) for opening in unit.opening],
                    final_index=None,
                )
            return dots_store.load_session(conn, session_key)

        session = self._store.write(open_turn)
        history = session.get_history()
        rows_at_start = len(session.messages)

        spend = TurnSpend(
            self._store, session_key, settings.max_cost_usd, "turn" if session_key == CHAT_SESSION_KEY else "task"
        )
        provider = spend.meter(self._providers.current(settings, self._key_holder.require()))
        # Every request uses the whole of what its model can do: its own context window and longest answer.
        runtime = LLMRuntime.at_model_limits(provider, settings.model_id, await _limits_of(provider, settings.model_id))
        # The summary of an outgrown thread may be written by another model (the `summary` role), through
        # the same metered provider, so its cost counts; it works within its own limits.
        summary_model = settings.model_for("summary")
        summary_runtime = LLMRuntime.at_model_limits(provider, summary_model, await _limits_of(provider, summary_model))
        tools = self._base_registry.view(settings.offered_tools)
        # What the tools of this turn returned for the model to look at, and the transcript does not keep.
        images = TurnImages()
        builder = ContextBuilder(
            settings.dot_prompt,
            workspace=settings.workspace,
            memory_dir=MEMORY_DIR,
            memory_notes=await self._recent_notes(),
            now=datetime.now().astimezone(),
            skills=await all_skills(self._computer),
        )

        async def commit(payload: dict[str, Any]) -> None:
            final_index = 0 if payload["phase"] == "final_response" else None
            if payload["phase"] in _ACTS_ON_A_RESPONSE:
                # These writes act on the response: the answer is delivered (and a task completes), the
                # tool calls run next, and no check of the hook comes in between. A turn that sent a
                # request it could not price ends here, before anything is done on its word.
                spend.ensure_priced()
            self._store.write(
                lambda conn: dots_store.append_messages(
                    conn, session_key, [_stamped(payload["message"])], final_index=final_index
                )
            )

        spec = AgentRunSpec(
            tools=tools,
            runtime=runtime,
            max_iterations=settings.max_iterations,
            max_tool_result_chars=settings.max_tool_result_chars,
            transcript_input=TranscriptInput(
                history=history,
                current_message=None,
                session_summary=session_summary_from_metadata(
                    session.metadata, fallback_last_active=session.updated_at
                ),
            ),
            transcript_builder=builder.build_transcript,
            hook=DotsTurnHook(unit, self._host, spend),
            concurrent_tools=False,
            # No spill files: a long result is cut to the limit, the Dot's computer is not the engine's disk.
            workspace=None,
            session_key=session_key,
            checkpoint_callback=commit,
            consolidate_history=partial(
                self._consolidator.summarize_transcript,
                runtime=summary_runtime,
                session_key=session_key,
                # The turn's own model sends the tools it was given, which keeps its prompt cache; another
                # model may not take tool definitions at all.
                tools=tools.get_definitions() if summary_model == settings.model_id else [],
                # The summary goes into the requests of the turn's own model: what it keeps whole fits that window.
                recent_user_tokens=recent_user_message_tokens(
                    prompt_budget(runtime.context_window_tokens, runtime.generation.max_tokens)
                ),
            ),
            injection_callback=self._injected if session_key == CHAT_SESSION_KEY else None,
            gate=self._gate,
            request_attachments=images.attach,
        )

        request_token = bind_request_context(RequestContext(session_key=session_key))
        file_states_token = bind_file_states(self._file_states.for_session(session_key))
        images_token = bind_turn_images(images)
        try:
            result = await self._runner.run(spec)
        finally:
            reset_turn_images(images_token)
            reset_file_states(file_states_token)
            reset_request_context(request_token)

        self._commit_summary(session_key, result, rows_at_start, len(history))
        return _outcome(result, settings)

    async def _injected(self) -> list[dict[str, Any]]:
        """The inputs a running chat turn takes in, as messages the runner commits."""
        if self._injection_source is None:
            return []
        return [_opening_message(opening) for opening in await self._injection_source()]

    def _commit_summary(
        self, session_key: str, result: AgentRunResult, rows_at_start: int, history_length: int
    ) -> None:
        """Store the summary checkpoint the run made, at the boundary it covers.

        The runner counts its transcript as the system prompt, the replayed history and
        what it added; every message it added is the next row of the session, so the
        boundary in rows is where the history ended plus how far past it the run got.
        """
        checkpoint = result.summary_checkpoint
        if checkpoint is None:
            return
        if checkpoint.transcript_boundary < 1 + history_length:
            logger.warning("ignoring the summary checkpoint of {}: it ends inside the replayed history", session_key)
            return
        boundary = rows_at_start + checkpoint.transcript_boundary - 1 - history_length
        try:
            self._store.write(
                lambda conn: dots_store.commit_summary_checkpoint(
                    conn, session_key, checkpoint.summary, boundary
                )
            )
        except ValueError as exc:
            logger.warning("ignoring the summary checkpoint of {}: {}", session_key, exc)

    async def _write_conversations(self, unit: TurnUnit) -> None:
        """Write what the turn said to the Dot's conversation files (`conversations`).

        The chat's files from the first message not yet written, a task's whole file; the first write
        after an upgrade writes the chat from its start and every task. A file that cannot be written now is
        written by a later turn: the count of the chat's written messages moves only once all are.
        """

        def read(conn: sqlite3.Connection) -> tuple[dict[str, str], int]:
            written = dots_store.read_kv(conn, conversations.KV_CHAT_WRITTEN)
            files: dict[str, str] = {}
            tasks = (
                dots_store.list_tasks(conn)
                if written is None
                else [task for task in [dots_store.get_task_by_session(conn, unit.session_key)] if task]
            )
            for task in tasks:
                messages = dots_store.load_session(conn, task.session_key).messages
                files.update(conversations.task_file(task.task_id, task.status, messages, target=tool_target))
            chat = dots_store.load_session(conn, CHAT_SESSION_KEY).messages
            since = written or 0
            files.update(conversations.chat_files(chat, since, target=tool_target))
            return files, len(chat)

        files, chat_length = self._store.read(read)
        try:
            for path, text in files.items():
                await self._computer.write_bytes(f"{conversations.CONVERSATIONS_DIR}/{path}", text.encode("utf-8"))
        except (ComputerError, OSError) as exc:
            logger.warning("could not write the conversation files: {}", exc)
            return
        self._store.write(lambda conn: dots_store.write_kv(conn, conversations.KV_CHAT_WRITTEN, chat_length))

    async def _recent_notes(self) -> list[str]:
        """Names of the most recently changed memory notes, newest first."""
        try:
            entries = await self._computer.list_dir(MEMORY_DIR)
        except ComputerError:
            logger.warning("could not list {}", MEMORY_DIR)
            return []
        notes = sorted(
            (entry for entry in entries or [] if entry.type == "file"), key=_mtime, reverse=True
        )
        return [entry.name for entry in notes[:MEMORY_NOTES_LISTED]]


async def _limits_of(provider: Any, model: str) -> ModelLimits:
    """The limits OpenRouter publishes for `model`; a turn cannot run without them."""
    try:
        return cast(ModelLimits, await provider.model_limits(model))
    except Exception as exc:
        raise RuntimeError(f"could not read the limits of the model {model} from OpenRouter: {exc}") from exc


def _opening_message(opening: OpeningMessage) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "user", "content": opening.text}
    if opening.metadata:
        message[METADATA_KEY] = dict(opening.metadata)
    return message


def _outcome(result: AgentRunResult, settings: EngineSettings) -> TurnOutcome:
    match result.stop_reason:
        case "completed":
            return TurnOutcome("completed")
        case "parked":
            return TurnOutcome("parked")
        case "empty_final_response":
            return TurnOutcome.failed("the run ended without an answer")
        case "max_iterations":
            return TurnOutcome.failed(
                f"stopped: the task reached limits.max_steps_per_task ({settings.max_iterations})"
            )
        case _:
            return TurnOutcome.failed(result.error or f"the run ended: {result.stop_reason}")
