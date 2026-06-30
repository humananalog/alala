"""Locate and import mlx_lm when not installed in the active interpreter."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from env import load_repo_env
from errors import HarnessError

MLX_PYTHON_CANDIDATES = (
    "/Users/alexclaw/Projects/mineru-api/.venv/bin/python",
    "/opt/homebrew/bin/python3.12",
)


def resolve_mlx_python() -> Path | None:
    for candidate in (
        os.environ.get("MLX_PYTHON"),
        load_repo_env().get("MLX_PYTHON"),
        *MLX_PYTHON_CANDIDATES,
    ):
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    return None


def bootstrap_mlx_imports() -> None:
    try:
        import mlx_lm  # noqa: F401
        return
    except ImportError:
        pass

    python = resolve_mlx_python()
    if python is None:
        raise HarnessError(
            "mlx_lm is not installed. Install mlx-lm or set MLX_PYTHON in .env to a venv with mlx-lm."
        )

    site_packages = subprocess.check_output(
        [str(python), "-c", "import site; print(site.getsitepackages()[0])"],
        text=True,
    ).strip()
    if site_packages not in sys.path:
        sys.path.insert(0, site_packages)

    try:
        import mlx_lm  # noqa: F401
    except ImportError as exc:
        raise HarnessError(f"Failed to import mlx_lm from {site_packages}") from exc