"""The Dot's prompts: its system prompt around the transcript."""

from __future__ import annotations

from datetime import datetime, timezone

from nanobot.agent.context import ContextBuilder, TranscriptInput
from nanobot.dots.skills import Skill

NOW = datetime(2026, 10, 5, 14, 30, tzinfo=timezone.utc)
DOT = 'You are the Dot "fare-watch".'


def builder(memory_notes: tuple[str, ...] = (), now: datetime = NOW) -> ContextBuilder:
    return ContextBuilder(
        DOT,
        workspace="/home/dot/workspace",
        memory_dir="/home/dot/memory",
        memory_notes=memory_notes,
        now=now,
    )


def test_the_system_prompt_is_the_dot_the_tool_contract_and_its_computer() -> None:
    prompt = builder().build_system_prompt()

    sections = prompt.split("\n\n---\n\n")
    assert sections[0] == DOT
    assert sections[1].startswith("# Tool Usage Notes")
    assert "You run on your own Linux computer." in sections[2]
    assert "run as the user dot" in sections[2]
    assert "Your workspace is /home/dot/workspace." in sections[2]
    assert "Today is 2026-10-05 (Monday) UTC. For the time, run `date`." in sections[2]
    assert "untrusted external data" in sections[2]
    assert len(sections) == 3


def test_the_system_prompt_is_the_same_all_day() -> None:
    # A prompt that changed every turn would miss the provider's cache and its count of the prompt.
    morning = builder(now=NOW.replace(hour=0, minute=1)).build_system_prompt()

    assert builder(now=NOW.replace(hour=23, minute=59)).build_system_prompt() == morning
    assert builder(now=NOW.replace(day=6, hour=0, minute=1)).build_system_prompt() != morning


def test_nothing_of_the_upstream_assistants_identity_or_platform_is_left() -> None:
    prompt = builder().build_system_prompt()

    # Skills are the Dot's own now (nanobot/dots/skills.py), not upstream's bundled ones.
    for leftover in ("nanobot", "Windows", "POSIX", "channel", "SOUL", "AGENTS", "HEARTBEAT", "clawhub"):
        assert leftover not in prompt


def test_the_memory_section_says_the_dot_keeps_its_notes_itself_with_the_file_tools() -> None:
    prompt = builder().build_system_prompt()

    assert "Your long-term memory is /home/dot/memory, one note per file, and you keep it yourself" in prompt
    for tool in ("grep", "find_files", "read_file", "write_file", "edit_file"):
        assert tool in prompt
    assert "Most recently changed notes" not in prompt


def test_the_memory_section_names_the_notes_it_was_given() -> None:
    prompt = builder(("b.md", "a.md")).build_system_prompt()

    assert "Most recently changed notes: b.md, a.md." in prompt


def test_the_skills_section_names_each_skill_with_its_description_and_file_and_says_how_to_write_one() -> None:
    skills = (
        Skill("invisible-playwright", "Use the browser.", "/opt/engine/skills/invisible-playwright/SKILL.md", "builtin", ""),
        Skill("shop-login", "Log in to the shop.", "/home/dot/skills/shop-login/SKILL.md", "dot", ""),
    )
    prompt = ContextBuilder(DOT, workspace="/home/dot/workspace", memory_dir="/home/dot/memory", memory_notes=(), now=NOW, skills=skills).build_system_prompt()

    assert "- invisible-playwright: Use the browser. (/opt/engine/skills/invisible-playwright/SKILL.md)" in prompt
    assert "- shop-login: Log in to the shop. (/home/dot/skills/shop-login/SKILL.md)" in prompt
    assert "read its file with read_file" in prompt
    assert "/home/dot/skills/<name>/SKILL.md" in prompt


def test_a_session_summary_is_added_and_a_nothing_summary_is_not() -> None:
    summary = {"text": "the fares were checked", "last_active": "2026-10-04T10:00:00+00:00"}

    prompt = builder().build_system_prompt(session_summary=summary)

    assert prompt.endswith(
        "[Archived Context Summary]\n\n"
        "Another model started this work and wrote this summary of it before its context was compacted "
        "(last active 2026-10-04T10:00:00+00:00). The tools and files it used are still yours: "
        "build on what it did and do not repeat work already done.\n\n"
        "the fares were checked"
    )
    nothing = {"text": "(nothing)", "last_active": "2026-10-04T10:00:00+00:00"}
    assert "Archived Context Summary" not in builder().build_system_prompt(session_summary=nothing)


def test_the_transcript_is_the_system_prompt_the_history_and_the_current_message() -> None:
    history = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "answer"}]

    messages = builder().build_transcript(TranscriptInput(history=history, current_message="now"))

    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[0]["content"] == builder().build_system_prompt()
    assert messages[1:3] == history
    assert messages[3] == {"role": "user", "content": "now"}


def test_with_no_current_message_the_transcript_ends_with_the_history() -> None:
    history = [{"role": "user", "content": "already in the transcript"}]

    messages = builder().build_transcript(TranscriptInput(history=history, current_message=None))

    assert messages[1:] == history
