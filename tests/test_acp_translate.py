from takopi_acp_engine.runner import AcpRunner, AcpStreamState
from takopi_acp_engine.backend import DroidConfig


def _runner() -> AcpRunner:
    return AcpRunner(DroidConfig())


def test_run_start_and_complete() -> None:
    runner = _runner()
    state = AcpStreamState()
    started = runner.translate(
        {
            "type": "run.created",
            "run": {"session_id": "sess-123"},
        },
        state=state,
        resume=None,
        found_session=None,
    )
    assert started
    assert started[0].type == "started"
    assert state.session_id == "sess-123"

    runner.translate(
        {
            "type": "message.part",
            "part": {"content_type": "text/plain", "content": "hello"},
        },
        state=state,
        resume=None,
        found_session=None,
    )

    completed = runner.translate(
        {
            "type": "run.completed",
            "run": {"session_id": "sess-123"},
        },
        state=state,
        resume=None,
        found_session=None,
    )
    assert completed
    assert completed[0].type == "completed"
    assert completed[0].answer == "hello"


def test_trajectory_action() -> None:
    runner = _runner()
    state = AcpStreamState()
    events = runner.translate(
        {
            "type": "message.part",
            "part": {
                "content_type": "text/plain",
                "content": "",
                "metadata": {
                    "kind": "trajectory",
                    "tool_name": "Read",
                    "tool_input": {"file_path": "README.md"},
                    "tool_output": "ok",
                },
            },
        },
        state=state,
        resume=None,
        found_session=None,
    )
    assert events
    assert events[0].type == "action"
    assert events[0].action.kind == "tool"
    assert events[0].action.title == "Read"


def test_message_completed_overrides_text() -> None:
    runner = _runner()
    state = AcpStreamState()
    runner.translate(
        {
            "type": "message.part",
            "part": {"content_type": "text/plain", "content": "partial"},
        },
        state=state,
        resume=None,
        found_session=None,
    )
    runner.translate(
        {
            "type": "message.completed",
            "message": {
                "role": "agent/droid",
                "parts": [{"content_type": "text/plain", "content": "final"}],
            },
        },
        state=state,
        resume=None,
        found_session=None,
    )
    assert state.last_text == "final"


def test_run_failed_emits_error() -> None:
    runner = _runner()
    state = AcpStreamState(session_id="sess-err")
    events = runner.translate(
        {
            "type": "run.failed",
            "run": {"session_id": "sess-err", "error": {"message": "boom"}},
        },
        state=state,
        resume=None,
        found_session=None,
    )
    assert events
    completed = [evt for evt in events if evt.type == "completed"]
    assert completed
    assert completed[0].ok is False
    assert completed[0].error == "boom"


def test_invalid_json_events_ignores_sse_lines() -> None:
    runner = _runner()
    state = AcpStreamState()
    assert (
        runner.invalid_json_events(
            raw="event: message",
            line="event: message",
            state=state,
        )
        == []
    )
    assert (
        runner.invalid_json_events(
            raw=": keepalive",
            line=": keepalive",
            state=state,
        )
        == []
    )
