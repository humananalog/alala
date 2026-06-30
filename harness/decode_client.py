"""Invoke MLX decode in a compatible Python subprocess."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from decode import DecodeResult
from errors import HarnessError
from mlx_bootstrap import resolve_mlx_python

WORKER = Path(__file__).resolve().parent / "decode_worker.py"


def _run_worker(command: str, args: list[str], timeout: int | None = None) -> dict:
    python = resolve_mlx_python()
    if python is None:
        raise HarnessError("Set MLX_PYTHON in .env to a Python with mlx_lm installed.")

    result = subprocess.run(
        [str(python), str(WORKER), command, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "decode worker failed"
        raise HarnessError(message)

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise HarnessError("decode worker returned no output")
    payload = json.loads(lines[-1])
    if "error" in payload:
        raise HarnessError(payload["error"])
    return payload


class DecodeRunner:
    def __init__(self, model_id: str) -> None:
        self._model_id = model_id

    def run_context_step(
        self,
        context_length: int,
        duration_s: int,
        steady_window_s: int,
        decode_tokens: int,
        kv_bits: int | None = None,
        orchestration_delay_ms: int = 0,
    ) -> DecodeResult:
        args = [
            "--model",
            self._model_id,
            "--context-length",
            str(context_length),
            "--duration",
            str(duration_s),
            "--steady-window",
            str(steady_window_s),
            "--decode-tokens",
            str(decode_tokens),
            "--orchestration-delay-ms",
            str(orchestration_delay_ms),
        ]
        if kv_bits is not None:
            args.extend(["--kv-bits", str(kv_bits)])
        payload = _run_worker("context-step", args, timeout=duration_s + 900)
        return DecodeResult(
            context_length=int(payload["context_length"]),
            tokens_generated=int(payload["tokens_generated"]),
            tokens_per_second=float(payload["tokens_per_second"]),
            tokens_per_second_sustained=float(payload["tokens_per_second_sustained"]),
            peak_memory_gb=float(payload["peak_memory_gb"]),
            decode_rounds=int(payload["decode_rounds"]),
        )