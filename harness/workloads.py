"""Sustained workloads for Phase 0 harness modes."""

from __future__ import annotations

import threading
from typing import Callable


class SustainedLoad:
    def __init__(self, worker: Callable[[], None]) -> None:
        self._worker = worker
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="alala-sustained-load", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            self._worker()


def mlx_matmul_load(matrix_size: int = 4096) -> SustainedLoad:
    """GPU-sustained load via MLX matmul (interim until ANE decode workload lands)."""

    def _work() -> None:
        import mlx.core as mx

        a = mx.random.uniform((matrix_size, matrix_size))
        b = mx.random.uniform((matrix_size, matrix_size))
        mx.eval(mx.matmul(a, b))

    return SustainedLoad(_work)


def cpu_spin_load() -> SustainedLoad:
    """CPU-only sustained load for harness validation without MLX."""

    def _work() -> None:
        total = 0
        for i in range(1_000_000):
            total += i * i
        if total < 0:  # pragma: no cover - keep loop side-effect visible
            print(total)

    return SustainedLoad(_work)