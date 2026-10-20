"""Small local configuration helpers with no third-party dotenv dependency."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


def load_local_env(path: str | Path | None = None) -> dict[str, str]:
    """Load simple KEY=value entries without overriding process variables.

    The file is intentionally limited to local configuration and is ignored by
    Git. Values already supplied by the shell always take precedence.
    """
    env_path = Path(path) if path else Path.cwd() / ".env"
    if not env_path.exists():
        return {}
    loaded: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
            if raw_line.strip().split("=", 1)[1].strip().startswith('"'):
                value = value.replace('\\"', '"').replace("\\\\", "\\")
        if key:
            loaded[key] = value
            os.environ.setdefault(key, value)
    return loaded


def _quote_env(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_llm_env(path: str | Path, *, model: str, api_key: str,
                  parser: str = "deepseek") -> Path:
    """Create/update local LLM settings while preserving unrelated entries."""
    env_path = Path(path)
    existing = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    updates = {
        "DEEPSEEK_API_KEY": api_key,
        "DEEPSEEK_MODEL": model,
        "FAULT_AGENT_PARSER": parser,
    }
    seen: set[str] = set()
    output: list[str] = []
    for line in existing:
        stripped = line.strip()
        key = stripped.split("=", 1)[0].strip() if "=" in stripped and not stripped.startswith("#") else ""
        if key in updates:
            output.append(f"{key}={_quote_env(updates[key])}")
            seen.add(key)
        else:
            output.append(line)
    if output and output[-1].strip():
        output.append("")
    for key, value in updates.items():
        if key not in seen:
            output.append(f"{key}={_quote_env(value)}")
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("\n".join(output).rstrip() + "\n", encoding="utf-8")
    return env_path
