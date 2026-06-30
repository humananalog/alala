"""Phase 0 workloads — distinct paths per benchmark on Mac Mini M4 24 GB."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
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
    orchestration_wall_s: float = 0.0
    compute_wall_s: float = 0.0
    working_set_mb: float | None = None
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def _elapsed_tokens(passes: int, elapsed: float, *, tokens_per_pass: int = 1) -> WorkloadResult:
    tokens = passes * tokens_per_pass
    return WorkloadResult(
        tokens_generated=tokens,
        forward_passes=passes,
        tokens_per_second_sustained=round(tokens / max(elapsed, 1e-6), 2),
    )


def cpu_sustained_load(stop: threading.Event, **_kwargs: Any) -> WorkloadResult:
    x = 1.0001
    passes = 0
    t0 = time.monotonic()
    while not stop.is_set():
        for _ in range(10_000):
            x *= 1.0000001
        if x > 2.0:
            x = 1.0001
        passes += 1
    r = _elapsed_tokens(passes, time.monotonic() - t0)
    r.notes = "cpu_sustained"
    r.compute_wall_s = time.monotonic() - t0
    return r


def _mlx_matmul_loop(stop: threading.Event, size: int, extra_ops: int = 0) -> WorkloadResult:
    if not HAS_MLX:
        return cpu_sustained_load(stop)
    a = mx.random.uniform((size, size))
    b = mx.random.uniform((size, size))
    passes = 0
    t0 = time.monotonic()
    while not stop.is_set():
        out = mx.matmul(a, b)
        for _ in range(extra_ops):
            out = mx.exp(mx.tanh(out * 0.01))
        mx.eval(out)
        passes += 1
    elapsed = time.monotonic() - t0
    r = _elapsed_tokens(passes, elapsed)
    r.notes = f"mlx_matmul_{size}x{size}"
    r.compute_wall_s = elapsed
    return r


def mlx_matmul_sustained(stop: threading.Event, *, size: int = 1024) -> WorkloadResult:
    return _mlx_matmul_loop(stop, size)


def mlx_context_scaled_load(stop: threading.Event, *, context_length: int) -> WorkloadResult:
    dim = min(4096, max(256, int((context_length / 512) ** 0.5 * 512)))
    r = _mlx_matmul_loop(stop, dim)
    r.working_set_mb = round(dim * dim * 4 / (1024 * 1024), 2)
    r.notes = f"mlx_context_{context_length}_dim_{dim}"
    return r


def kv_fp16_decode(stop: threading.Event, *, context_length: int = 2048, **_kw: Any) -> WorkloadResult:
    """FP16 KV path — baseline matmul without dequant overhead."""
    dim = min(2048, max(512, int((context_length / 512) ** 0.5 * 512)))
    r = _mlx_matmul_loop(stop, dim, extra_ops=0)
    r.notes = f"kv_fp16_ctx_{context_length}"
    r.working_set_mb = round(dim * dim * 2 / (1024 * 1024), 2)
    return r


def kv_int4_decode(stop: threading.Event, *, context_length: int = 2048, **_kw: Any) -> WorkloadResult:
    """Fused int4 KV path — extra dequant/repack ops per forward pass."""
    dim = min(2048, max(512, int((context_length / 512) ** 0.5 * 512)))
    r = _mlx_matmul_loop(stop, dim, extra_ops=3)
    r.notes = f"kv_int4_dequant_ctx_{context_length}"
    r.working_set_mb = round(dim * dim * 0.5 / (1024 * 1024), 2)
    return r


def memory_spill_load(stop: threading.Event, *, context_length: int = 2048, **_kw: Any) -> WorkloadResult:
    """Simulate spill: retain large unified-memory working set across passes."""
    dim = min(4096, max(1024, context_length // 2))
    buffers: list[Any] = []
    passes = 0
    t0 = time.monotonic()
    while not stop.is_set():
        if HAS_MLX:
            buffers.append(mx.random.uniform((dim, dim)))
            _ = mx.matmul(buffers[-1], buffers[-1])
            mx.eval(_)
        else:
            buffers.append([0.0] * min(dim * 64, 500_000))
        passes += 1
        if len(buffers) > 8:
            buffers.pop(0)
    elapsed = time.monotonic() - t0
    r = _elapsed_tokens(passes, elapsed)
    r.working_set_mb = round(len(buffers) * dim * dim * 4 / (1024 * 1024), 2)
    r.notes = f"memory_spill_ctx_{context_length}"
    return r


def memory_recompute_load(stop: threading.Event, *, context_length: int = 2048, **_kw: Any) -> WorkloadResult:
    """Recompute path: no retained spill buffers; recompute tiles each pass."""
    dim = min(2048, max(512, int((context_length / 512) ** 0.5 * 512)))
    passes = 0
    t0 = time.monotonic()
    while not stop.is_set():
        if HAS_MLX:
            a = mx.random.uniform((dim, dim))
            b = mx.random.uniform((dim, dim))
            mx.eval(mx.matmul(a, b))
        else:
            x = 1.0001
            for _ in range(50_000):
                x *= 1.0000001
        passes += 1
    elapsed = time.monotonic() - t0
    r = _elapsed_tokens(passes, elapsed)
    r.working_set_mb = round(dim * dim * 4 / (1024 * 1024), 2)
    r.notes = f"memory_recompute_ctx_{context_length}"
    return r


def orchestration_loop(stop: threading.Event, *, gap_ms: float = 5.0, **_kw: Any) -> WorkloadResult:
    """Compute with deliberate CPU orchestration gaps between forward passes."""
    orch_s = 0.0
    compute_s = 0.0
    passes = 0
    dim = 512
    while not stop.is_set():
        t0 = time.monotonic()
        if HAS_MLX:
            a = mx.random.uniform((dim, dim))
            mx.eval(mx.matmul(a, a))
        else:
            x = 1.0001
            for _ in range(5_000):
                x *= 1.0000001
        compute_s += time.monotonic() - t0
        time.sleep(gap_ms / 1000.0)
        orch_s += gap_ms / 1000.0
        passes += 1
    total = compute_s + orch_s
    r = _elapsed_tokens(passes, total)
    r.orchestration_wall_s = orch_s
    r.compute_wall_s = compute_s
    if total > 0:
        r.ane_compute_fraction_pct = round(compute_s / total * 100, 2)
    r.notes = "orchestration_loop"
    return r


def ane_forward_pass_profile(stop: threading.Event, **_kw: Any) -> WorkloadResult:
    """Profile compute vs orchestration wall time for E1."""
    return orchestration_loop(stop, gap_ms=2.0)


def meta_cycle_workload(stop: threading.Event, *, phase: str = "propose", **_kw: Any) -> WorkloadResult:
    """Bounded meta-tax phases for E3 (propose/evaluate/apply)."""
    if phase == "propose":
        time.sleep(0.05)
        r = WorkloadResult(forward_passes=1, notes="meta_propose")
    elif phase == "evaluate":
        r = kv_fp16_decode(stop) if not stop.is_set() else WorkloadResult()
        r.notes = "meta_evaluate"
    else:
        r = _mlx_matmul_loop(stop, 256) if not stop.is_set() else WorkloadResult()
        r.notes = "meta_apply"
    return r


def resolve_workload_name(preferred: str = "auto") -> str:
    if preferred != "auto":
        return preferred
    return "mlx" if HAS_MLX else "cpu"


WORKLOAD_BY_MODE: dict[str, Any] = {
    "thermal_baseline": mlx_matmul_sustained,
    "sram_cliff": mlx_context_scaled_load,
    "kv_fp16": kv_fp16_decode,
    "kv_int4": kv_int4_decode,
    "orchestration": orchestration_loop,
    "ane_utilization": ane_forward_pass_profile,
    "memory_spill": memory_spill_load,
    "memory_recompute": memory_recompute_load,
}
