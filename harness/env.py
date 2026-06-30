from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"
LOGS_DIR = REPO_ROOT / "logs"
RESULTS_DIR = REPO_ROOT / "results"


def load_repo_env() -> dict[str, str]:
    values: dict[str, str] = {}
    if not ENV_FILE.exists():
        return values
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, raw_value = stripped.partition("=")
        if not separator:
            continue
        values[key.strip()] = raw_value.strip().strip('"').strip("'")
    return values


def sudo_password() -> str | None:
    password = os.environ.get("SUDO_PASSWORD") or load_repo_env().get("SUDO_PASSWORD")
    if password:
        return password
    return None