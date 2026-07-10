import anyio
import pytest

from takopi.model import ResumeToken, TitleChangedEvent
from takopi.runner_bridge import RunningTask, run_runner_with_cancel
from takopi.runners.mock import Emit, Return, ScriptRunner


class _Edits:
    async def on_event(self, _event) -> None:
        pass


def test_title_changed_event() -> None:
    event = TitleChangedEvent(engine="pi", title="New Title")
    assert event.type == "title_changed"
    assert event.title == "New Title"


@pytest.mark.anyio
async def test_prompt_fallback_title_is_only_generated_for_new_session() -> None:
    token = ResumeToken(engine="pi", value="existing-session")
    runner = ScriptRunner(
        [Return("ok")],
        engine="pi",
        resume_value=token.value,
    )
    task = RunningTask()
    callback_titles: list[str | None] = []

    async def on_thread_known(_token: ResumeToken, _done: anyio.Event) -> None:
        callback_titles.append(task.title)

    await run_runner_with_cancel(
        runner,
        prompt="This must not become a new topic title",
        resume_token=token,
        edits=_Edits(),
        running_task=task,
        on_thread_known=on_thread_known,
    )

    assert task.title is None
    assert callback_titles == [None]


@pytest.mark.anyio
async def test_pi_title_can_still_rename_resumed_session() -> None:
    token = ResumeToken(engine="pi", value="existing-session")
    runner = ScriptRunner(
        [
            Emit(TitleChangedEvent(engine="pi", title="Stable session title")),
            Return("ok"),
        ],
        engine="pi",
        resume_value=token.value,
    )
    task = RunningTask()
    callback_titles: list[str | None] = []

    async def on_thread_known(_token: ResumeToken, _done: anyio.Event) -> None:
        callback_titles.append(task.title)

    await run_runner_with_cancel(
        runner,
        prompt="A later message",
        resume_token=token,
        edits=_Edits(),
        running_task=task,
        on_thread_known=on_thread_known,
    )

    assert task.title == "Stable session title"
    assert callback_titles == [None, "Stable session title"]
