from pathlib import Path, PureWindowsPath
from unittest.mock import patch

import anyio
import msgspec
import pytest

from takopi.model import ActionEvent, CompletedEvent, ResumeToken, StartedEvent
from takopi.runners.pi import (
    ENGINE,
    PiRunner,
    PiStreamState,
    _default_session_dir,
    translate_pi_event,
)
from takopi.schemas import pi as pi_schema


def _load_fixture(name: str) -> list[pi_schema.PiEvent]:
    path = Path(__file__).parent / "fixtures" / name
    events: list[pi_schema.PiEvent] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            decoded = pi_schema.decode_event(line)
        except Exception as exc:
            raise AssertionError(f"{name} contained unparseable line: {line}") from exc
        events.append(decoded)
    return events


def test_pi_resume_format_and_extract(tmp_path: Path) -> None:
    runner = PiRunner(
        extra_args=[],
        model=None,
        provider=None,
    )
    session_path = tmp_path / "session.jsonl"
    token = ResumeToken(engine=ENGINE, value=str(session_path))

    assert runner.format_resume(token) == f"`pi --session {session_path}`"
    assert runner.extract_resume(f"`pi --session {session_path}`") == token
    assert runner.extract_resume(f'pi --session "{session_path}"') == token
    assert runner.extract_resume("`codex resume sid`") is None

    spaced_path = tmp_path / "pi session.jsonl"
    spaced = ResumeToken(engine=ENGINE, value=str(spaced_path))
    assert runner.format_resume(spaced) == f'`pi --session "{spaced_path}"`'
    assert runner.extract_resume(f'`pi --session "{spaced_path}"`') == spaced


def test_translate_success_fixture() -> None:
    state = PiStreamState(resume=ResumeToken(engine=ENGINE, value="session.jsonl"))
    events: list = []
    for event in _load_fixture("pi_stream_success.jsonl"):
        events.extend(translate_pi_event(event, title="pi", meta=None, state=state))

    assert isinstance(events[0], StartedEvent)
    started = next(evt for evt in events if isinstance(evt, StartedEvent))
    assert started.meta is None

    action_events = [evt for evt in events if isinstance(evt, ActionEvent)]
    assert len(action_events) == 4

    started_actions = {
        (evt.action.id, evt.phase): evt
        for evt in action_events
        if evt.phase == "started"
    }
    assert started_actions[("tool_1", "started")].action.kind == "command"
    write_action = started_actions[("tool_2", "started")].action
    assert write_action.kind == "file_change"
    assert write_action.detail["changes"][0]["path"] == "notes.md"

    completed_actions = {
        (evt.action.id, evt.phase): evt
        for evt in action_events
        if evt.phase == "completed"
    }
    assert completed_actions[("tool_1", "completed")].ok is True
    assert completed_actions[("tool_2", "completed")].ok is True

    completed = next(evt for evt in events if isinstance(evt, CompletedEvent))
    assert events[-1] == completed
    assert completed.ok is True
    assert completed.resume == started.resume
    assert completed.answer == "Done. Added notes.md."


def test_translate_error_fixture() -> None:
    state = PiStreamState(resume=ResumeToken(engine=ENGINE, value="session.jsonl"))
    events: list = []
    for event in _load_fixture("pi_stream_error.jsonl"):
        events.extend(translate_pi_event(event, title="pi", meta=None, state=state))

    completed = next(evt for evt in events if isinstance(evt, CompletedEvent))
    assert completed.ok is False
    assert completed.error == "Upstream error"
    assert completed.answer == "Request failed."


def _assistant_message(
    text: str, *, stop_reason: str, error: str | None = None
) -> dict:
    message = {
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "stopReason": stop_reason,
    }
    if error is not None:
        message["errorMessage"] = error
    return message


def _translate_sequence(state: PiStreamState, events: list[pi_schema.PiEvent]) -> list:
    translated: list = []
    for event in events:
        translated.extend(translate_pi_event(event, title="pi", meta=None, state=state))
    return translated


def test_translate_auto_retry_success_completes_on_settled() -> None:
    state = PiStreamState(resume=ResumeToken(engine=ENGINE, value="session.jsonl"))
    failed = _assistant_message(
        "First attempt failed.", stop_reason="error", error="Rate limited"
    )
    succeeded = _assistant_message("Retry succeeded.", stop_reason="stop")

    before_settled = _translate_sequence(
        state,
        [
            pi_schema.MessageEnd(message=failed),
            pi_schema.AgentEnd(messages=[failed], willRetry=True),
            pi_schema.MessageEnd(message=succeeded),
            pi_schema.AgentEnd(messages=[succeeded], willRetry=False),
        ],
    )
    settled = _translate_sequence(
        state, [pi_schema.AgentSettled(), pi_schema.AgentSettled()]
    )

    assert not any(isinstance(event, CompletedEvent) for event in before_settled)
    completed = [event for event in settled if isinstance(event, CompletedEvent)]
    assert len(completed) == 1
    assert completed[0].ok is True
    assert completed[0].answer == "Retry succeeded."
    assert completed[0].error is None


def test_translate_modern_agent_end_allows_continuation_before_settled() -> None:
    state = PiStreamState(resume=ResumeToken(engine=ENGINE, value="session.jsonl"))
    first = _assistant_message("Before compaction.", stop_reason="stop")
    final = _assistant_message("After continuation.", stop_reason="stop")

    events = _translate_sequence(
        state,
        [
            pi_schema.AgentEnd(messages=[first], willRetry=False),
            pi_schema.MessageEnd(message=final),
            pi_schema.AgentEnd(messages=[final], willRetry=False),
            pi_schema.AgentSettled(),
        ],
    )

    completed = [event for event in events if isinstance(event, CompletedEvent)]
    assert len(completed) == 1
    assert completed[0].ok is True
    assert completed[0].answer == "After continuation."


def test_translate_legacy_agent_end_completes_and_settled_does_not_duplicate() -> None:
    state = PiStreamState(resume=ResumeToken(engine=ENGINE, value="session.jsonl"))
    succeeded = _assistant_message("Legacy success.", stop_reason="stop")
    legacy = pi_schema.decode_event(
        msgspec.json.encode({"type": "agent_end", "messages": [succeeded]})
    )

    events = _translate_sequence(state, [legacy, pi_schema.AgentSettled()])

    completed = [event for event in events if isinstance(event, CompletedEvent)]
    assert len(completed) == 1
    assert completed[0].ok is True
    assert completed[0].answer == "Legacy success."


def test_translate_retry_exhausted_completes_on_settled() -> None:
    state = PiStreamState(resume=ResumeToken(engine=ENGINE, value="session.jsonl"))
    first_error = _assistant_message(
        "First failure.", stop_reason="error", error="Temporary error"
    )
    final_error = _assistant_message(
        "Final failure.", stop_reason="error", error="Retries exhausted"
    )

    events = _translate_sequence(
        state,
        [
            pi_schema.AgentEnd(messages=[first_error], willRetry=True),
            pi_schema.AgentEnd(messages=[final_error], willRetry=False),
            pi_schema.AgentSettled(),
        ],
    )

    completed = [event for event in events if isinstance(event, CompletedEvent)]
    assert len(completed) == 1
    assert completed[0].ok is False
    assert completed[0].answer == "Final failure."
    assert completed[0].error == "Retries exhausted"


def test_empty_final_assistant_clears_stale_answer_and_usage() -> None:
    state = PiStreamState(resume=ResumeToken(engine=ENGINE, value="session.jsonl"))
    previous = _assistant_message("Stale answer.", stop_reason="stop")
    previous["usage"] = {"totalTokens": 10}
    empty = {"role": "assistant", "content": [], "stopReason": "stop"}

    events = _translate_sequence(
        state,
        [
            pi_schema.AgentEnd(messages=[previous], willRetry=False),
            pi_schema.AgentEnd(messages=[empty], willRetry=False),
            pi_schema.AgentSettled(),
        ],
    )

    completed = [event for event in events if isinstance(event, CompletedEvent)]
    assert len(completed) == 1
    assert completed[0].ok is True
    assert completed[0].answer == ""
    assert completed[0].usage is None


def test_session_id_promotion_from_stdout() -> None:
    state = PiStreamState(
        resume=ResumeToken(engine=ENGINE, value="session.jsonl"),
        allow_id_promotion=True,
    )
    events = translate_pi_event(
        pi_schema.SessionHeader(
            id="ccd569e0-4e1b-4c7d-a981-637ed4107310",
            version=3,
            timestamp="2026-01-13T00:33:34.702Z",
            cwd="/tmp",
        ),
        title="pi",
        meta=None,
        state=state,
    )
    started = next(evt for evt in events if isinstance(evt, StartedEvent))
    assert started.resume.value == "ccd569e0"


def test_extract_resume_keeps_session_path(tmp_path: Path) -> None:
    session_path = tmp_path / "session.jsonl"
    runner = PiRunner(
        extra_args=[],
        model=None,
        provider=None,
    )
    token = runner.extract_resume(f"pi --session {session_path}")
    assert token is not None
    assert token.value == str(session_path)


@pytest.mark.anyio
async def test_run_keeps_resume_path(tmp_path: Path) -> None:
    session_path = tmp_path / "session.jsonl"
    runner = PiRunner(
        extra_args=[],
        model=None,
        provider=None,
    )
    seen_resume: ResumeToken | None = None

    async def run_stub(_prompt: str, resume: ResumeToken | None):
        nonlocal seen_resume
        seen_resume = resume
        yield CompletedEvent(
            engine=ENGINE,
            resume=resume,
            ok=True,
            answer="ok",
        )

    runner.run_impl = run_stub  # type: ignore[assignment]
    resume = ResumeToken(engine=ENGINE, value=str(session_path))
    async for _event in runner.run("test", resume):
        pass
    assert seen_resume is not None
    assert seen_resume.value == str(session_path)


@pytest.mark.anyio
async def test_run_serializes_same_session() -> None:
    runner = PiRunner(
        extra_args=[],
        model=None,
        provider=None,
    )
    gate = anyio.Event()
    in_flight = 0
    max_in_flight = 0

    async def run_stub(*_args, **_kwargs):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        try:
            await gate.wait()
            yield CompletedEvent(
                engine=ENGINE,
                resume=ResumeToken(engine=ENGINE, value="session.jsonl"),
                ok=True,
                answer="ok",
            )
        finally:
            in_flight -= 1

    runner.run_impl = run_stub  # type: ignore[assignment]

    async def drain(prompt: str, resume: ResumeToken | None) -> None:
        async for _event in runner.run(prompt, resume):
            pass

    token = ResumeToken(engine=ENGINE, value="session.jsonl")
    async with anyio.create_task_group() as tg:
        tg.start_soon(drain, "a", token)
        tg.start_soon(drain, "b", token)
        await anyio.sleep(0)
        gate.set()
    assert max_in_flight == 1


def test_session_path_prefers_run_base_dir(tmp_path: Path) -> None:
    runner = PiRunner(
        extra_args=[],
        model=None,
        provider=None,
    )
    project_cwd = Path("/project")
    session_root = tmp_path / "sessions"

    with (
        patch("takopi.runners.pi.get_run_base_dir", return_value=project_cwd),
        patch(
            "takopi.runners.pi._default_session_dir",
            return_value=session_root,
        ) as default_session_dir,
    ):
        session_path = runner._new_session_path()

    default_session_dir.assert_called_once_with(project_cwd)
    assert str(session_root) in session_path


def test_session_path_sanitizes_windows_separators() -> None:
    cwd = PureWindowsPath("C:\\foo\\bar")
    session_dir = _default_session_dir(cwd)
    name = session_dir.name
    assert "\\" not in name
    assert ":" not in name
