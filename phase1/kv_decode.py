"""Stateful decode loops for Phase 1 MLX and Core ML backends."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

NUM_LAYERS = 24
KV_HEADS = 2
HEAD_DIM = 64
DEFAULT_DECODE_PT = Path("models/qwen2.5-0.5b-decode-kv.pt")


@dataclass
class DecodeRunResult:
    context_length: int
    tokens_generated: int
    tokens_per_second: float
    tokens_per_second_sustained: float
    peak_memory_gb: float | None
    kv_cache_active: bool
    backend: str
    decode_runtime: str | None = None
    kv_cache_mode: str = "linear"
    ring_size: int | None = None
    re_prefill_count: int = 0
    kv_io_bytes_per_step: int | None = None
    active_kv_slots: int | None = None


def _greedy_token(logits: np.ndarray) -> int:
    flat = logits.reshape(-1)
    return int(flat.argmax())


def run_mlx_decode_loop(
    runner,
    *,
    context_length: int,
    duration_s: int,
    steady_window_s: int,
    decode_tokens: int,
) -> DecodeRunResult:
    """Real MLX autoregressive decode via harness DecodeRunner (KV inside mlx_lm)."""
    result = runner.run_context_step(
        context_length=context_length,
        duration_s=duration_s,
        steady_window_s=steady_window_s,
        decode_tokens=decode_tokens,
    )
    return DecodeRunResult(
        context_length=context_length,
        tokens_generated=result.tokens_generated,
        tokens_per_second=result.tokens_per_second,
        tokens_per_second_sustained=result.tokens_per_second_sustained,
        peak_memory_gb=result.peak_memory_gb,
        kv_cache_active=True,
        backend="mlx",
        decode_runtime="mlx_lm",
    )


def _causal_mask(end_step: int) -> np.ndarray:
    """4D causal mask for one decode query attending to `end_step` cached keys."""
    return np.zeros((1, 1, 1, end_step), dtype=np.float16)


def _fixed_causal_mask(max_ctx: int) -> np.ndarray:
    """Fixed-width causal mask for torch.export decode (static-friendly)."""
    return np.zeros((1, 1, 1, max_ctx), dtype=np.float16)


def _decode_input_names(model) -> set[str]:
    return {i.name for i in model.get_spec().description.input}


def _prefill_seq_len(model) -> int:
    spec = model.get_spec()
    for feature in spec.description.input:
        if feature.name != "inputIds":
            continue
        shape = feature.type.multiArrayType.shape
        if len(shape) >= 2:
            return int(shape[1])
    raise ValueError("Could not infer prefill sequence length from inputIds shape")


def _decode_kv_seq_len(model) -> int | None:
    spec = model.get_spec()
    for feature in spec.description.input:
        if feature.name != "keyCache":
            continue
        shape = feature.type.multiArrayType.shape
        if len(shape) >= 4:
            return int(shape[3])
    return None


def _slice_kv_to_decode(key_cache: np.ndarray, value_cache: np.ndarray, decode_seq_len: int):
    if key_cache.shape[3] == decode_seq_len:
        return key_cache, value_cache
    if key_cache.shape[3] < decode_seq_len:
        raise ValueError(
            f"Prefill KV seq len {key_cache.shape[3]} < decode seq len {decode_seq_len}"
        )
    return (
        np.ascontiguousarray(key_cache[:, :, :, :decode_seq_len, :], dtype=np.float16),
        np.ascontiguousarray(value_cache[:, :, :, :decode_seq_len, :], dtype=np.float16),
    )


def _resolve_decode_backend(
    decode_path: Path,
    decode_pt_path: Path | None,
    *,
    compute_units: str = "all",
    log_model_load: bool = True,
):
    """Prefer Core ML decode mlpackage; fall back to TorchScript .pt."""
    if decode_path.exists():
        from coreml_instrumentation import load_coreml_model, log_load_info

        model, info = load_coreml_model(
            decode_path,
            role="decode_kv",
            compute_units=compute_units,
            capture_compute_plan=False,
        )
        if model is None:
            raise RuntimeError(f"Failed to load decode model {decode_path}: {info.load_error}")
        if log_model_load:
            log_load_info(info)
        input_names = _decode_input_names(model)
        if "cachePosition" in input_names:
            return "coreml", model, None, "torch_export"
        return "coreml", model, model.make_state(), "mlstate"

    pt_path = decode_pt_path or DEFAULT_DECODE_PT
    if pt_path.exists():
        import torch

        return "torchscript", torch.jit.load(str(pt_path)).eval(), None, "torchscript"

    raise RuntimeError(
        f"No decode artifact found. Expected {decode_path} or {pt_path}. "
        "Run: phase1/.venv/bin/python phase1/coreml_kv_convert.py"
    )


def run_coreml_decode_loop(
    *,
    prefill_path: Path,
    decode_path: Path,
    context_length: int,
    max_ctx: int,
    duration_s: int,
    steady_window_s: int,
    decode_tokens: int,
    decode_pt_path: Path | None = None,
    compute_units: str = "all",
    log_model_load: bool = True,
    kv_cache_mode: str = "linear",
    ring_size: int = 512,
) -> DecodeRunResult:
    """Autoregressive decode with explicit KV cache hand-off between steps.

    1. Prefill (Core ML): prompt → logits, keyCache, valueCache
    2. Decode loop: feed (token, caches, cacheSeqLen) → updated caches + logits
       Uses Core ML decode mlpackage when available, else TorchScript .pt (CPU/MPS).
    3. Greedy argmax on logits each step
    """
    try:
        import coremltools as ct
    except ImportError as exc:
        raise RuntimeError("coremltools required for Core ML decode") from exc

    from coreml_instrumentation import load_coreml_model, log_load_info, log_runtime_environment
    from ring_buffer_kv import (
        RingBufferConfig,
        active_kv_slots,
        ring_causal_mask,
        should_re_prefill,
    )

    rb_config = RingBufferConfig(
        mode="ring" if kv_cache_mode == "ring" else "linear",
        ring_size=ring_size,
        max_ctx=max_ctx,
        num_layers=NUM_LAYERS,
        kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
    )

    if log_model_load:
        log_runtime_environment()

    prefill_model, prefill_info = load_coreml_model(
        prefill_path,
        role="prefill_kv",
        compute_units=compute_units,
        capture_compute_plan=False,
    )
    if prefill_model is None:
        raise RuntimeError(f"Failed to load prefill model {prefill_path}: {prefill_info.load_error}")
    if log_model_load:
        log_load_info(prefill_info)

    prefill_max_ctx = _prefill_seq_len(prefill_model)
    decode_max_ctx = max_ctx

    decode_runtime, decode_runner, decode_state, decode_style = _resolve_decode_backend(
        decode_path,
        decode_pt_path,
        compute_units=compute_units,
        log_model_load=log_model_load,
    )
    if decode_style == "torch_export":
        inferred = _decode_kv_seq_len(decode_runner)
        if inferred is not None:
            decode_max_ctx = inferred

    if context_length > prefill_max_ctx:
        raise ValueError(f"context_length {context_length} > prefill_max_ctx {prefill_max_ctx}")

    # Leave at least one KV slot for autoregressive steps when prompt would fill the decode cache.
    active_context = min(context_length, decode_max_ctx - 1)
    if active_context < 1:
        raise ValueError(
            f"decode_max_ctx {decode_max_ctx} too small for context_length {context_length}"
        )
    if active_context != context_length:
        logger.warning(
            "Capping prefill prompt from %d to %d tokens (decode_max_ctx=%d needs decode headroom)",
            context_length,
            active_context,
            decode_max_ctx,
        )

    rng = np.random.default_rng(context_length)
    prompt = np.zeros((1, prefill_max_ctx), dtype=np.int32)
    prompt[0, :active_context] = rng.integers(1, 5000, size=active_context, dtype=np.int32)

    def _seed_decode_state(key_cache: np.ndarray, value_cache: np.ndarray) -> None:
        if decode_runtime == "coreml" and decode_style == "mlstate" and decode_state is not None:
            # Core ML state write accepts fp32 copies of prefill cache tensors.
            decode_state.write_state(
                "keyCache", np.ascontiguousarray(key_cache, dtype=np.float32)
            )
            decode_state.write_state(
                "valueCache", np.ascontiguousarray(value_cache, dtype=np.float32)
            )

    prefill_out = prefill_model.predict({"inputIds": prompt})
    key_cache = np.array(prefill_out["keyCache"], dtype=np.float16)
    value_cache = np.array(prefill_out["valueCache"], dtype=np.float16)
    key_cache, value_cache = _slice_kv_to_decode(key_cache, value_cache, decode_max_ctx)
    cache_seq_len = active_context

    logits = np.array(prefill_out["logits"])
    next_token = _greedy_token(logits[:, active_context - 1 : active_context, :])

    _seed_decode_state(key_cache, value_cache)
    use_ring_mask = decode_style == "torch_export" and rb_config.mode == "ring"
    fixed_mask = (
        None
        if use_ring_mask
        else (_fixed_causal_mask(decode_max_ctx) if decode_style == "torch_export" else None)
    )
    re_prefill_count = 0
    if log_model_load:
        logger.info(
            "decode_loop config: compute_units=%s decode_style=%s kv_cache_mode=%s "
            "ring_size=%s context_length=%d prefill_max_ctx=%d decode_max_ctx=%d "
            "kv_io_bytes_per_step=%d",
            compute_units,
            decode_style,
            rb_config.mode,
            ring_size if rb_config.mode == "ring" else None,
            context_length,
            prefill_max_ctx,
            decode_max_ctx,
            rb_config.kv_io_bytes_per_step,
        )

    start = time.monotonic()
    steady_start = start + max(0, duration_s - steady_window_s)
    total_tokens = 0
    steady_tokens = 0

    while time.monotonic() - start < duration_s:
        # Linear mode re-prefills when KV is full; ring mode wraps in-place.
        if should_re_prefill(cache_seq_len, rb_config):
            prefill_out = prefill_model.predict({"inputIds": prompt})
            key_cache = np.array(prefill_out["keyCache"], dtype=np.float16)
            value_cache = np.array(prefill_out["valueCache"], dtype=np.float16)
            key_cache, value_cache = _slice_kv_to_decode(key_cache, value_cache, decode_max_ctx)
            cache_seq_len = active_context
            _seed_decode_state(key_cache, value_cache)
            logits = np.array(prefill_out["logits"])
            next_token = _greedy_token(logits[:, active_context - 1 : active_context, :])
            re_prefill_count += 1
            continue

        end_step = cache_seq_len + 1
        # Ring export models apply (logical % ring_size) inside the graph; pass logical position.
        cache_position_input = cache_seq_len
        step_mask = (
            ring_causal_mask(cache_seq_len, rb_config)
            if use_ring_mask
            else fixed_mask
        )
        if decode_runtime == "coreml":
            if decode_style == "torch_export":
                decode_out = decode_runner.predict(
                    {
                        "inputIds": np.array([[next_token]], dtype=np.int32),
                        "keyCache": np.ascontiguousarray(key_cache, dtype=np.float16),
                        "valueCache": np.ascontiguousarray(value_cache, dtype=np.float16),
                        "cachePosition": np.array([cache_position_input], dtype=np.int32),
                        "causalMask": step_mask,
                    }
                )
                key_out_name = "keyCacheOut" if "keyCacheOut" in decode_out else "keyCache"
                value_out_name = "valueCacheOut" if "valueCacheOut" in decode_out else "valueCache"
                key_cache = np.array(decode_out[key_out_name], dtype=np.float16)
                value_cache = np.array(decode_out[value_out_name], dtype=np.float16)
            else:
                decode_out = decode_runner.predict(
                    {
                        "inputIds": np.array([[next_token]], dtype=np.int32),
                        "causalMask": _causal_mask(end_step),
                    },
                    state=decode_state,
                )
            step_logits = np.array(decode_out["logits"])
        else:
            import torch as th

            with th.no_grad():
                out = decode_runner(
                    th.tensor([[next_token]], dtype=th.int32),
                    th.tensor(_causal_mask(end_step)),
                )
            step_logits = out.numpy() if isinstance(out, th.Tensor) else out[0].numpy()

        cache_seq_len += 1
        total_tokens += 1
        if time.monotonic() >= steady_start:
            steady_tokens += 1

        next_token = _greedy_token(step_logits)

    elapsed = time.monotonic() - start
    steady_elapsed = min(steady_window_s, elapsed)
    tps = total_tokens / elapsed if elapsed > 0 else 0.0
    tps_sustained = steady_tokens / steady_elapsed if steady_elapsed > 0 else 0.0

    runtime_label = f"{decode_runtime}:{decode_style}" if decode_runtime else decode_style
    if rb_config.mode == "ring":
        runtime_label = f"{runtime_label}:ring{ring_size}"

    return DecodeRunResult(
        context_length=context_length,
        tokens_generated=total_tokens,
        tokens_per_second=tps,
        tokens_per_second_sustained=tps_sustained,
        peak_memory_gb=None,
        kv_cache_active=True,
        backend="coreml",
        decode_runtime=runtime_label,
        kv_cache_mode=rb_config.mode,
        ring_size=ring_size if rb_config.mode == "ring" else None,
        re_prefill_count=re_prefill_count,
        kv_io_bytes_per_step=rb_config.kv_io_bytes_per_step,
        active_kv_slots=active_kv_slots(cache_seq_len, rb_config),
    )


def run_coreml_prefill_proxy(
    model_path: Path,
    *,
    context_length: int,
    trace_context_size: int,
    duration_s: int,
    steady_window_s: int,
    compute_units: str = "all",
    log_model_load: bool = True,
) -> DecodeRunResult:
    """Legacy prefill-only proxy (no KV)."""
    from coreml_instrumentation import load_coreml_model, log_load_info

    mlmodel, info = load_coreml_model(
        model_path,
        role="prefill_only_proxy",
        compute_units=compute_units,
        capture_compute_plan=False,
    )
    if mlmodel is None:
        raise RuntimeError(f"Failed to load prefill-only model {model_path}: {info.load_error}")
    if log_model_load:
        log_load_info(info)
    shape = (1, trace_context_size)
    rng = np.random.default_rng(context_length)
    input_ids = np.zeros(shape, dtype=np.int32)
    input_ids[0, :context_length] = rng.integers(1, 1000, size=context_length, dtype=np.int32)

    start = time.monotonic()
    steady_start = start + max(0, duration_s - steady_window_s)
    total = steady = 0
    while time.monotonic() - start < duration_s:
        mlmodel.predict({"inputIds": input_ids})
        total += 1
        if time.monotonic() >= steady_start:
            steady += 1

    elapsed = time.monotonic() - start
    steady_elapsed = min(steady_window_s, elapsed)
    return DecodeRunResult(
        context_length=context_length,
        tokens_generated=total,
        tokens_per_second=total / elapsed if elapsed else 0.0,
        tokens_per_second_sustained=steady / steady_elapsed if steady_elapsed else 0.0,
        peak_memory_gb=None,
        kv_cache_active=False,
        backend="coreml",
        decode_runtime="prefill_proxy",
    )