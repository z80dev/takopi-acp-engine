from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import anyio
from anyio.streams.buffered import BufferedByteReceiveStream

from takopi.logging import get_logger
from takopi.model import Action, ActionEvent, CompletedEvent, ResumeToken, StartedEvent
from takopi.runner import JsonlRunState, JsonlSubprocessRunner, ResumeTokenMixin

from .settings import DroidConfig

ENGINE = "droid"

_RESUME_RE = re.compile(
    r"(?im)^\s*`?droid\s+resume\s+(?P<token>[A-Za-z0-9\-]{8,})`?\s*$"
)


@dataclass(slots=True)
class AcpStreamState(JsonlRunState):
    session_id: str | None = None
    last_text: str = ""
    emitted_started: bool = False
    action_seq: int = 0


class AcpRunner(ResumeTokenMixin, JsonlSubprocessRunner):
    engine = ENGINE
    resume_re = _RESUME_RE

    def __init__(self, config: DroidConfig) -> None:
        self.config = config
        self.logger = get_logger(__name__)

    def command(self) -> str:
        return self.config.cmd

    def build_args(
        self,
        prompt: str,
        resume: ResumeToken | None,
        *,
        state: Any,
    ) -> list[str]:
        _ = prompt
        _ = resume
        _ = state
        args = ["exec", "--output-format", "acp"]
        if self.config.model:
            args += ["--model", self.config.model]
        if self.config.reasoning_effort:
            args += ["--reasoning-effort", self.config.reasoning_effort]
        if self.config.auto:
            args += ["--auto", self.config.auto]
        if self.config.enabled_tools:
            args += ["--enabled-tools", ",".join(self.config.enabled_tools)]
        if self.config.disabled_tools:
            args += ["--disabled-tools", ",".join(self.config.disabled_tools)]
        if self.config.cwd:
            args += ["--cwd", self.config.cwd]
        return args

    def stdin_payload(
        self,
        prompt: str,
        resume: ResumeToken | None,
        *,
        state: Any,
    ) -> bytes | None:
        _ = state
        message = {
            "role": "user",
            "parts": [{"content_type": "text/plain", "content": prompt}],
        }
        payload: dict[str, Any] = {
            "agent_name": self.config.agent_name,
            "input": [message],
            "mode": "stream",
        }
        if resume is not None:
            payload["session_id"] = resume.value

        data = json.dumps(payload).encode("utf-8")
        if not self.config.lsp_framing:
            return data + b"\n"

        header = f"Content-Length: {len(data)}\r\n\r\n".encode("utf-8")
        return header + data

    def new_state(self, prompt: str, resume: ResumeToken | None) -> Any:
        _ = prompt
        _ = resume
        return AcpStreamState()

    def pipes_error_message(self) -> str:
        return "droid failed to open subprocess pipes"

    def decode_jsonl(self, *, line: bytes) -> Any | None:
        text = line.decode("utf-8", errors="replace").strip()
        if not text:
            return None
        if text.startswith("data:"):
            text = text[5:].strip()
        if text in {"[DONE]", "done"}:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def invalid_json_events(
        self,
        *,
        raw: str,
        line: str,
        state: Any,
    ) -> list[Any]:
        _ = state
        stripped = line.strip()
        if not stripped:
            return []
        lowered = stripped.lower()
        if lowered.startswith((":", "event:", "id:", "retry:")):
            return []
        return super().invalid_json_events(raw=raw, line=line, state=state)

    async def iter_json_lines(self, stream: Any) -> anyio.AsyncIterator[bytes]:
        buffered = BufferedByteReceiveStream(stream)
        while True:
            try:
                line = await buffered.receive_until(b"\n", 1024 * 1024)
            except anyio.IncompleteRead:
                return

            if line.strip().lower().startswith(b"content-length:"):
                headers = [line]
                content_length = None
                while True:
                    header_line = await buffered.receive_until(b"\n", 1024 * 1024)
                    headers.append(header_line)
                    if header_line in (b"\r\n", b"\n"):
                        break
                for header in headers:
                    if header.strip().lower().startswith(b"content-length:"):
                        value = header.split(b":", 1)[1].strip()
                        try:
                            content_length = int(value)
                        except ValueError:
                            content_length = None
                        break
                if content_length is None:
                    continue
                try:
                    payload = await buffered.receive_exactly(content_length)
                except anyio.IncompleteRead:
                    return
                yield payload
                continue

            yield line.rstrip(b"\n")

    def translate(
        self,
        data: Any,
        *,
        state: Any,
        resume: ResumeToken | None,
        found_session: ResumeToken | None,
    ) -> list[Any]:
        _ = found_session
        if not isinstance(state, AcpStreamState):
            return []
        if not isinstance(data, dict):
            return []

        evt_type = data.get("type")
        if not isinstance(evt_type, str):
            return []

        if evt_type.startswith("run."):
            return self._handle_run_event(evt_type, data, state, resume)

        if evt_type == "message.part":
            part = data.get("part")
            if isinstance(part, dict):
                content_type = part.get("content_type")
                content = part.get("content")
                if content_type == "text/plain" and isinstance(content, str):
                    state.last_text += content

                metadata = part.get("metadata")
                if isinstance(metadata, dict) and metadata.get("kind") == "trajectory":
                    return [self._trajectory_action(metadata, state)]
            return []

        if evt_type in {"message.created", "message.completed"}:
            message = data.get("message")
            if isinstance(message, dict):
                role = message.get("role")
                if isinstance(role, str) and role.startswith("agent"):
                    parts = message.get("parts")
                    if isinstance(parts, list):
                        text = self._parts_to_text(parts)
                        if text:
                            state.last_text = text
            return []

        if evt_type == "error":
            error = data.get("error")
            message = None
            if isinstance(error, dict):
                message = error.get("message")
            elif isinstance(error, str):
                message = error
            return [
                CompletedEvent(
                    engine=self.engine,
                    ok=False,
                    answer=state.last_text,
                    resume=self._resume_from_state(state, resume),
                    error=message,
                )
            ]

        return []

    def _handle_run_event(
        self,
        evt_type: str,
        data: dict[str, Any],
        state: AcpStreamState,
        resume: ResumeToken | None,
    ) -> list[Any]:
        run = data.get("run")
        started_event: StartedEvent | None = None
        if isinstance(run, dict):
            session_id = run.get("session_id")
            if isinstance(session_id, str) and session_id:
                state.session_id = session_id
            if not state.emitted_started and state.session_id:
                state.emitted_started = True
                started_event = StartedEvent(
                    engine=self.engine,
                    resume=ResumeToken(engine=self.engine, value=state.session_id),
                    title="Droid",
                )

        events: list[Any] = []
        if evt_type == "run.completed":
            events.append(
                CompletedEvent(
                    engine=self.engine,
                    ok=True,
                    answer=state.last_text,
                    resume=self._resume_from_state(state, resume),
                )
            )
        elif evt_type in {"run.failed", "run.cancelled"}:
            error = None
            if isinstance(run, dict):
                err = run.get("error")
                if isinstance(err, dict):
                    error = err.get("message")
                elif isinstance(err, str):
                    error = err
            events.append(
                CompletedEvent(
                    engine=self.engine,
                    ok=False,
                    answer=state.last_text,
                    resume=self._resume_from_state(state, resume),
                    error=error,
                )
            )

        if started_event is None:
            return events
        return [started_event, *events] if events else [started_event]

    def _resume_from_state(
        self, state: AcpStreamState, resume: ResumeToken | None
    ) -> ResumeToken | None:
        if state.session_id:
            return ResumeToken(engine=self.engine, value=state.session_id)
        return resume

    def _parts_to_text(self, parts: list[Any]) -> str:
        chunks: list[str] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            if part.get("content_type") != "text/plain":
                continue
            content = part.get("content")
            if isinstance(content, str) and content:
                chunks.append(content)
        return "".join(chunks)

    def _trajectory_action(self, metadata: dict[str, Any], state: AcpStreamState) -> ActionEvent:
        state.action_seq += 1
        message = metadata.get("message")
        tool_name = metadata.get("tool_name")
        tool_input = metadata.get("tool_input")
        tool_output = metadata.get("tool_output")

        if isinstance(tool_name, str) and tool_name:
            title = tool_name
            kind = "tool"
            detail = {
                "tool_name": tool_name,
                "tool_input": tool_input,
                "tool_output": tool_output,
            }
        else:
            title = "trajectory"
            kind = "note"
            detail = {"message": message}

        action = Action(
            id=f"acp.trajectory.{state.action_seq}",
            kind=kind,
            title=title,
            detail=detail,
        )
        return ActionEvent(
            engine=self.engine,
            action=action,
            phase="completed",
            ok=True,
            message=message if isinstance(message, str) else None,
            level="info",
        )
