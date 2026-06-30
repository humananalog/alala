"""Phase 0 workloads for m4_energy_harness — MLX when available, CPU fallback."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False


@dataclass
class WorkloadResult:
    tokens_generated: int = 0
    forward_passes: int = 0
    tokens_per_second_sustained: float | None = None
    ane_compute_fraction_pct: float | None = None
    notes: str = ""


def cpu_sustained_load(stop: threading.Event, **_kwargs: Any) -> WorkloadResult:
    """CPU fallback when MLX unavailable."""
    x = 1.0001
    passes = 0
    t0 = time.monotonic()
    while not stop.is_set():
        for _ in range(10_000):
            x *= 1.0000001
        if x > 2.0:
            x = 1.0001
        passes += 1
    elapsed = max(time.monotonic() - t0, 1e-6)
    return WorkloadResult(
        forward_passes=passes,
        tokens_per_second_sustained=round(passes / elapsed, 2),
        notes="cpu_sustained_stub",
    )


def mlx_matmul_sustained(stop: threading.Event, *, size: int = 1024) -> WorkloadResult:
    """Minimal MLX matmul loop — exercises Apple Silicon GPU path on M4."""
    if not HAS_MLX:
        return cpu_sustained_load(stop)
    a = mx.random.uniform((size, size))
    b = mx.random.uniform((size, size))
    passes = 0
    t0 = time.monotonic()
    while not stop.is_set():
        _ = mx.matmul(a, b)
        mx.eval(_)
        passes += 1
    elapsed = max(time.monotonic() - t0, 1e-6)
    return WorkloadResult(
        forward_passes=passes,
        tokens_per_second_sustained=round(passes / elapsed, 2),
        notes=f"mlx_matmul_{size}x{size}",
    )


def mlx_context_scaled_load(stop: threading.Event, *, context_length: int) -> WorkloadResult:
    """Proxy load scaling with context — matrix dim grows with sqrt(context)."""
    dim = min(4096, max(256, int((context_length / 512) ** 0.5 * 512)))
    return mlx_matmul_sustained(stop, size=dim)


def run_in_thread(
    fn,
    duration_s: float,
    **kwargs,
) -> WorkloadResult:
    stop = threading.Event()

    def target() -> None:
        nonlocal result
        result = fn(stop, **kwargs)

    result = WorkloadResult()
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    time.sleep(duration_s)
    stop.set()
    thread.join(timeout=5)
    return result


def resolve_workload_name(preferred: str = "auto") -> str:
    if preferred != "auto":
        return preferred
    return "mlx" if HAS_MLX else "cpu"
