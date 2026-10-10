"""The model's commands run through `dot-agentd relay`, never in the engine's own shell."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fakes.local_computer import LocalComputer

from nanobot.agent.tools.exec_session import MAX_YIELD_MS, ExecSessionManager, ExecSessionTool
from nanobot.agent.tools.shell import ExecTool


def _records(log: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture()
def computer(tmp_path: Path) -> LocalComputer:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return LocalComputer(tmp_path, workspace, relay_log=tmp_path / "relay.log")


@pytest.fixture(autouse=True)
def no_local_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    """A command the engine ran through a shell of its own would not be in the relay log."""

    async def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the engine started a shell of its own")

    monkeypatch.setattr(asyncio, "create_subprocess_shell", refuse)


async def test_exec_runs_the_command_through_the_relay_in_a_login_shell(
    computer: LocalComputer, tmp_path: Path
) -> None:
    result = await ExecTool(computer).execute(command="echo through the relay")

    assert "through the relay" in result
    assert "Exit code: 0" in result
    (record,) = _records(tmp_path / "relay.log")
    assert record["program"] == ["/bin/bash", "-lc", "echo through the relay"]
    assert record["cwd"] == str(tmp_path / "workspace")
    assert record["tty"] is False
    assert record["env"] == []


async def test_exec_resolves_a_relative_working_dir_against_the_workspace(
    computer: LocalComputer, tmp_path: Path
) -> None:
    (tmp_path / "workspace" / "project").mkdir()

    result = await ExecTool(computer).execute(command="pwd", working_dir="project")

    assert str(tmp_path / "workspace" / "project") in result
    (record,) = _records(tmp_path / "relay.log")
    assert record["cwd"] == str(tmp_path / "workspace" / "project")


async def test_exec_keeps_an_absolute_working_dir(computer: LocalComputer, tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    result = await ExecTool(computer).execute(command="pwd", working_dir=str(elsewhere))

    assert str(elsewhere) in result


async def test_no_variable_of_the_engine_reaches_the_command(
    computer: LocalComputer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_SECRET", "super_secret_token")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret")

    result = await ExecTool(computer).execute(
        command='echo "secret=${FAKE_SECRET:-absent} key=${OPENROUTER_API_KEY:-absent}"'
    )

    assert "secret=absent key=absent" in result


async def test_every_exec_is_one_relay_call(computer: LocalComputer, tmp_path: Path) -> None:
    tool = ExecTool(computer)

    await tool.execute(command="echo one")
    await tool.execute(command="echo two", timeout=5)
    await tool.execute(cmd="echo three")

    commands = [record["program"][2] for record in _records(tmp_path / "relay.log")]  # type: ignore[index]
    assert commands == ["echo one", "echo two", "echo three"]


async def test_exec_sessions_run_through_the_relay_too(
    computer: LocalComputer, tmp_path: Path
) -> None:
    manager = ExecSessionManager()
    tool = ExecTool(computer, session_manager=manager)
    try:
        started = await tool.execute(command="cat", yield_time_ms=200)
        assert "session_id:" in started
        session_id = started.split("session_id:")[1].split()[0]
        # Until cat exits, which it does once its input closes: no window the machine's speed decides.
        answer = await ExecSessionTool(manager=manager).execute(
            session_id=session_id, input="hello\n", close_stdin=True, until_exit=True,
        )
        assert "hello" in answer
    finally:
        await manager.close_all()

    (record,) = _records(tmp_path / "relay.log")
    assert record["program"] == ["/bin/bash", "-lc", "cat"]
    assert record["cwd"] == str(tmp_path / "workspace")


async def test_exec_with_a_tty_asks_the_relay_for_one_and_runs_as_a_session(
    computer: LocalComputer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TERM", raising=False)
    manager = ExecSessionManager()
    tool = ExecTool(computer, session_manager=manager)
    try:
        started = await tool.execute(command='echo "term=$TERM"; cat', tty=True, yield_time_ms=300)
        assert "session_id:" in started
        session_id = started.split("session_id:")[1].split()[0]
        answer = await ExecSessionTool(manager=manager).execute(
            session_id=session_id, input="hello\n", close_stdin=True, until_exit=True,
        )
        # What the session printed, in its first answer or after it: a slow login shell may print after 300 ms.
        assert "term=xterm-256color" in started + answer
        assert "hello" in answer
    finally:
        await manager.close_all()

    (record,) = _records(tmp_path / "relay.log")
    assert record["tty"] is True
    assert record["program"] == ["/bin/bash", "-lc", 'echo "term=$TERM"; cat']
    assert record["cwd"] == str(tmp_path / "workspace")


async def test_a_tty_without_yield_time_still_starts_a_session(
    computer: LocalComputer, tmp_path: Path
) -> None:
    manager = ExecSessionManager()
    try:
        started = await ExecTool(computer, session_manager=manager).execute(command="cat", tty=True)
        assert "Process running. session_id:" in started
    finally:
        await manager.close_all()

    (record,) = _records(tmp_path / "relay.log")
    assert record["tty"] is True


async def test_a_tty_command_that_ends_at_once_answers_in_the_same_call(
    computer: LocalComputer,
) -> None:
    # The answer comes as the command ends; the long yield is only how long that may take on a slow machine.
    result = await ExecTool(computer, session_manager=ExecSessionManager()).execute(
        command="echo quick", tty=True, yield_time_ms=MAX_YIELD_MS
    )

    assert "quick" in result
    assert "Exit code: 0" in result
    assert "session_id" not in result


async def test_a_tty_keeps_the_terminal_type_the_engine_was_given(
    computer: LocalComputer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TERM", "vt100")

    result = await ExecTool(computer, session_manager=ExecSessionManager()).execute(
        command='echo "term=$TERM"', tty=True, yield_time_ms=MAX_YIELD_MS
    )

    assert "term=vt100" in result


async def test_without_a_tty_the_terminal_type_of_the_engine_does_not_reach_the_command(
    computer: LocalComputer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TERM", "vt100")

    result = await ExecTool(computer).execute(command='echo "term=${TERM:-absent}"', tty=False)

    assert "term=vt100" not in result  # the engine's TERM stays with the engine (bash itself says dumb)
    (record,) = _records(tmp_path / "relay.log")
    assert record["tty"] is False


async def test_a_timeout_ends_the_command_started_through_the_relay(
    computer: LocalComputer, tmp_path: Path
) -> None:
    marker = tmp_path / "survived"

    result = await ExecTool(computer).execute(
        command=f"sleep 3 && touch {marker}", timeout=1
    )

    assert "timed out" in result.lower()
    await asyncio.sleep(3.5)
    assert not marker.exists()
