#!/usr/bin/env python3
"""Export Qwen2.5-0.5B Core ML prefill + decode models with explicit KV cache I/O."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
NUM_LAYERS = 24
KV_HEADS = 2
HEAD_DIM = 64
DEFAULT_MAX_CTX = 1024


def _import_stack():
    import coremltools as ct
    import torch
    from transformers import AutoModelForCausalLM

    return ct, torch, AutoModelForCausalLM


def stack_kv(past_key_values, max_ctx: int, torch):
    k_out = torch.zeros(NUM_LAYERS, 1, KV_HEADS, max_ctx, HEAD_DIM, dtype=torch.float16)
    v_out = torch.zeros(NUM_LAYERS, 1, KV_HEADS, max_ctx, HEAD_DIM, dtype=torch.float16)
    seq_len = past_key_values[0][0].shape[2]
    for i, (pk, pv) in enumerate(past_key_values):
        k_out[i, 0, :, :seq_len, :] = pk[0]
        v_out[i, 0, :, :seq_len, :] = pv[0]
    return k_out, v_out


class TraceFriendlyCache:
    """StaticCache with slice-based update (TorchScript-traceable)."""

    def __init__(self, cache) -> None:
        self._cache = cache

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        from transformers.cache_utils import StaticCache

        assert isinstance(self._cache, StaticCache)
        cache_position = cache_kwargs.get("cache_position")
        k_out = self._cache.key_cache[layer_idx]
        v_out = self._cache.value_cache[layer_idx]
        k_out[:, :, cache_position, :] = key_states
        v_out[:, :, cache_position, :] = value_states
        return k_out, v_out


def _patch_static_cache_update(static_cache) -> None:
    """Monkey-patch StaticCache.update for trace-friendly slice writes."""
    from transformers.cache_utils import StaticCache

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        cache_position = cache_kwargs.get("cache_position")
        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]
        k_out[:, :, cache_position, :] = key_states
        v_out[:, :, cache_position, :] = value_states
        return k_out, v_out

    static_cache.update = update.__get__(static_cache, StaticCache)


def convert_prefill_model(*, hf, ct, torch, output_dir: Path, max_ctx: int) -> Path:
    class PrefillWithKV(torch.nn.Module):
        def __init__(self, inner) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, input_ids: torch.LongTensor):
            out = self.inner(input_ids=input_ids, use_cache=True)
            k, v = stack_kv(out.past_key_values, max_ctx, torch)
            return out.logits, k, v

    prefill_path = output_dir / "qwen2.5-0.5b-prefill-kv.mlpackage"
    prefill = PrefillWithKV(hf).eval()
    prefill_in = (torch.zeros((1, max_ctx), dtype=torch.int32),)
    prefill_exp = torch.export.export(prefill, prefill_in).run_decompositions({})
    prefill_ml = ct.convert(
        prefill_exp,
        inputs=[ct.TensorType(shape=(1, max_ctx), dtype=np.int32, name="inputIds")],
        outputs=[
            ct.TensorType(dtype=np.float16, name="logits"),
            ct.TensorType(dtype=np.float16, name="keyCache"),
            ct.TensorType(dtype=np.float16, name="valueCache"),
        ],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        skip_model_load=True,
    )
    prefill_ml.save(str(prefill_path))
    logger.info("Saved %s", prefill_path)
    return prefill_path


def convert_decode_model(*, hf, ct, torch, output_dir: Path, max_ctx: int) -> dict:
    """Export decode via TorchScript trace. Core ML convert may fail (tensor index ops)."""
    from transformers.cache_utils import StaticCache

    decode_path = output_dir / "qwen2.5-0.5b-decode-kv.mlpackage"
    pt_path = output_dir / "qwen2.5-0.5b-decode-kv.pt"
    trace_sl = max_ctx - 1

    class DecodeWithStaticCache(torch.nn.Module):
        def __init__(self, inner, max_ctx: int) -> None:
            super().__init__()
            self.inner = inner
            self.static_cache = StaticCache(
                config=inner.config,
                max_batch_size=1,
                max_cache_len=max_ctx,
                device=next(inner.parameters()).device,
                dtype=torch.float16,
            )
            _patch_static_cache_update(self.static_cache)

        def forward(self, input_ids, key_cache, value_cache, cache_seq_len):
            for i in range(NUM_LAYERS):
                self.static_cache.key_cache[i].copy_(key_cache[i, 0])
                self.static_cache.value_cache[i].copy_(value_cache[i, 0])
            hidden = self.inner.model(
                input_ids=input_ids.long(),
                past_key_values=self.static_cache,
                use_cache=True,
                cache_position=cache_seq_len.to(torch.long),
            )[0]
            logits = self.inner.lm_head(hidden)
            k_out = torch.stack([self.static_cache.key_cache[i].unsqueeze(0) for i in range(NUM_LAYERS)])
            v_out = torch.stack([self.static_cache.value_cache[i].unsqueeze(0) for i in range(NUM_LAYERS)])
            return logits, k_out, v_out

    static = StaticCache(
        config=hf.config,
        max_batch_size=1,
        max_cache_len=max_ctx,
        device=next(hf.parameters()).device,
        dtype=torch.float16,
    )
    _patch_static_cache_update(static)
    warmup_ids = torch.arange(1, trace_sl + 1).unsqueeze(0)
    hf(
        input_ids=warmup_ids,
        past_key_values=static,
        use_cache=True,
        cache_position=torch.arange(trace_sl),
    )
    k = torch.stack([static.key_cache[i].unsqueeze(0) for i in range(NUM_LAYERS)])
    v = torch.stack([static.value_cache[i].unsqueeze(0) for i in range(NUM_LAYERS)])

    decode = DecodeWithStaticCache(hf, max_ctx).eval()
    decode_in = (
        torch.zeros((1, 1), dtype=torch.int32),
        k,
        v,
        torch.tensor([trace_sl], dtype=torch.int32),
    )
    traced = torch.jit.trace(decode, decode_in, strict=False)
    traced.save(str(pt_path))
    logger.info("Saved TorchScript decode %s", pt_path)

    coreml_ok = False
    coreml_error: str | None = None
    try:
        decode_ml = ct.convert(
            traced,
            inputs=[
                ct.TensorType(shape=(1, 1), dtype=np.int32, name="inputIds"),
                ct.TensorType(
                    shape=(NUM_LAYERS, 1, KV_HEADS, max_ctx, HEAD_DIM),
                    dtype=np.float16,
                    name="keyCache",
                ),
                ct.TensorType(
                    shape=(NUM_LAYERS, 1, KV_HEADS, max_ctx, HEAD_DIM),
                    dtype=np.float16,
                    name="valueCache",
                ),
                ct.TensorType(shape=(1,), dtype=np.int32, name="cacheSeqLen"),
            ],
            outputs=[
                ct.TensorType(dtype=np.float16, name="logits"),
                ct.TensorType(dtype=np.float16, name="keyCache"),
                ct.TensorType(dtype=np.float16, name="valueCache"),
            ],
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS15,
            skip_model_load=True,
        )
        decode_ml.save(str(decode_path))
        coreml_ok = True
        logger.info("Saved Core ML decode %s", decode_path)
    except Exception as exc:
        coreml_error = str(exc)
        logger.warning("Core ML decode convert failed (TorchScript fallback at %s): %s", pt_path, exc)

    return {
        "decode_path": str(decode_path) if coreml_ok else None,
        "decode_pt_path": str(pt_path),
        "coreml_decode_ok": coreml_ok,
        "coreml_decode_error": coreml_error,
    }


def convert_kv_models(
    *,
    model_id: str,
    output_dir: Path,
    max_ctx: int = DEFAULT_MAX_CTX,
    prefill_only: bool = False,
    decode_only: bool = False,
    skip_validation: bool = False,
) -> dict:
    ct, torch, AutoModelForCausalLM = _import_stack()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s", model_id)
    hf = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, attn_implementation="sdpa"
    ).eval()

    report: dict = {"model_id": model_id, "max_ctx": max_ctx}

    if not decode_only:
        prefill_path = convert_prefill_model(hf=hf, ct=ct, torch=torch, output_dir=output_dir, max_ctx=max_ctx)
        report["prefill_path"] = str(prefill_path)

    if not prefill_only:
        decode_info = convert_decode_model(hf=hf, ct=ct, torch=torch, output_dir=output_dir, max_ctx=max_ctx)
        report.update(decode_info)

    if not skip_validation and report.get("prefill_path"):
        pm = ct.models.MLModel(str(report["prefill_path"]), compute_units=ct.ComputeUnit.ALL)
        ids = np.zeros((1, max_ctx), dtype=np.int32)
        ids[0, :8] = np.arange(1, 9, dtype=np.int32)
        out = pm.predict({"inputIds": ids})
        report["prefill_logits_shape"] = list(np.array(out["logits"]).shape)

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export Core ML prefill+decode KV models.")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--max-ctx", type=int, default=DEFAULT_MAX_CTX)
    parser.add_argument("--prefill-only", action="store_true")
    parser.add_argument("--decode-only", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")
    try:
        report = convert_kv_models(
            model_id=args.model,
            output_dir=args.output_dir,
            max_ctx=args.max_ctx,
            prefill_only=args.prefill_only,
            decode_only=args.decode_only,
            skip_validation=args.skip_validation,
        )
    except Exception:
        logger.exception("KV conversion failed")
        return 1
    print("KV conversion complete:")
    for k, v in report.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())