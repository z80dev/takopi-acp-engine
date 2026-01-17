from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DroidConfig:
    cmd: str = "droid"
    agent_name: str = "droid"
    model: str | None = None
    reasoning_effort: str | None = None
    auto: str | None = None
    enabled_tools: tuple[str, ...] | None = None
    disabled_tools: tuple[str, ...] | None = None
    cwd: str | None = None
    lsp_framing: bool = True
    fallback_to_text: bool = True
