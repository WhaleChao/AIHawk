"""Context builder for assembling the Dot's prompts."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

from nanobot.dots.conversations import CONVERSATIONS_DIR
from nanobot.dots.skills import DOT_SKILLS_DIR, Skill
from nanobot.session.summary import SessionSummary
from nanobot.utils.prompt_templates import render_template


@dataclass(frozen=True, slots=True)
class TranscriptInput:
    """Raw turn inputs from which ``ContextBuilder`` assembles a transcript."""

    history: list[dict[str, Any]]
    current_message: str | None
    current_role: str = "user"
    session_summary: SessionSummary | None = None


class ContextBuilder:
    """Builds the context (system prompt + messages) of one turn of the Dot."""

    def __init__(
        self,
        dot_prompt: str,
        *,
        workspace: str,
        memory_dir: str,
        memory_notes: Sequence[str],
        now: datetime,
        skills: Sequence[Skill] = (),
    ) -> None:
        """`dot_prompt` says whose Dot this is and what it is for (projection.py).

        `memory_notes` are the names of the most recently changed notes in `memory_dir`; `skills` are the Dot's
        skills (nanobot/dots/skills.py), named in the prompt with their descriptions and paths.
        """
        self.dot_prompt = dot_prompt
        self.workspace = workspace
        self.memory_dir = memory_dir
        self.memory_notes = memory_notes
        self.skills = skills
        self.now = now

    def build_system_prompt(self, *, session_summary: SessionSummary | None = None) -> str:
        """Build the system prompt: the Dot, the tool contract, its computer and the summary."""
        parts = [
            self.dot_prompt,
            render_template("agent/tool_contract.md"),
            render_template(
                "agent/platform.md",
                workspace=self.workspace,
                memory_dir=self.memory_dir,
                memory_notes=list(self.memory_notes),
                conversations_dir=CONVERSATIONS_DIR,
                skills=list(self.skills),
                dot_skills_dir=DOT_SKILLS_DIR,
                now=self.now.strftime("%Y-%m-%d %H:%M %Z").strip(),
            ),
        ]
        if session_summary and session_summary["text"] != "(nothing)":
            parts.append(
                "[Archived Context Summary]\n\n"
                # Codex's summary prefix (Apache-2.0: github.com/openai/codex, codex-rs/prompts/templates/compact).
                "Another model started this work and wrote this summary of it before its context was compacted "
                f"(last active {session_summary['last_active']}). The tools and files it used are still yours: "
                "build on what it did and do not repeat work already done.\n\n"
                f"{session_summary['text']}"
            )
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _merge_message_content(left: Any, right: Any) -> str | list[dict[str, Any]]:
        if isinstance(left, str) and isinstance(right, str):
            if not left:
                return right
            if not right:
                return left
            return f"{left}\n\n{right}"

        def _to_blocks(value: Any) -> list[dict[str, Any]]:
            if isinstance(value, list):
                return [
                    cast(dict[str, Any], item)
                    if isinstance(item, dict)
                    else {"type": "text", "text": str(item)}
                    for item in cast(list[Any], value)
                ]
            if value is None:
                return []
            return [{"type": "text", "text": str(value)}]

        return _to_blocks(left) + _to_blocks(right)

    def build_transcript(self, transcript: TranscriptInput) -> list[dict[str, Any]]:
        """Build a model transcript while preserving the fresh-turn boundary."""
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": self.build_system_prompt(session_summary=transcript.session_summary),
            },
            *transcript.history,
        ]
        if transcript.current_message is not None:
            messages.append({"role": transcript.current_role, "content": transcript.current_message})
        return messages
