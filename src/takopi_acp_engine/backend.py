from __future__ import annotations

from pathlib import Path
from typing import Any

from takopi.api import EngineBackend, EngineConfig, Runner
from takopi.config import ConfigError

from .runner import AcpRunner
from .settings import DroidConfig


def _get_str(config: EngineConfig, key: str) -> str | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"Invalid droid config for {key!r}; expected a string.")
    return value


def _get_bool(config: EngineConfig, key: str, *, default: bool) -> bool:
    value = config.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ConfigError(f"Invalid droid config for {key!r}; expected a boolean.")


def _get_str_list(config: EngineConfig, key: str) -> tuple[str, ...] | None:
    value = config.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        items = [item for item in value.replace(",", " ").split() if item]
        return tuple(items) if items else None
    if isinstance(value, list):
        if not all(isinstance(item, str) for item in value):
            raise ConfigError(
                f"Invalid droid config for {key!r}; expected a list of strings."
            )
        items = [item for item in value if item]
        return tuple(items) if items else None
    raise ConfigError(
        f"Invalid droid config for {key!r}; expected a list or string."
    )


def _load_config(config: EngineConfig) -> DroidConfig:
    return DroidConfig(
        cmd=_get_str(config, "cmd") or "droid",
        agent_name=_get_str(config, "agent_name") or "droid",
        model=_get_str(config, "model"),
        reasoning_effort=_get_str(config, "reasoning_effort"),
        auto=_get_str(config, "auto"),
        enabled_tools=_get_str_list(config, "enabled_tools"),
        disabled_tools=_get_str_list(config, "disabled_tools"),
        cwd=_get_str(config, "cwd"),
        lsp_framing=_get_bool(config, "lsp_framing", default=True),
    )


def build_runner(config: EngineConfig, config_path: Path) -> Runner:
    _ = config_path
    settings = _load_config(config)
    return AcpRunner(settings)


BACKEND = EngineBackend(
    id="droid",
    build_runner=build_runner,
    cli_cmd="droid",
    install_cmd="curl -fsSL https://app.factory.ai/cli | sh",
)
