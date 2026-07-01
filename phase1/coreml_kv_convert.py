#!/usr/bin/env python3
"""Export Qwen2.5-0.5B Core ML prefill + stateful decode models with KV cache."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

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


class SliceUpdateKeyValueCache:
    """KV cache with integer-slice updates (Core ML / TorchScript friendly).

    Shape: (#layers, batch_size, #kv_heads, context_size, head_dim).
    Updates use slice indices derived from causalMask.shape[-1], avoiding
    dynamic tensor indexing that breaks coremltools conversion.
    """

    def __init__(self, shape: Tuple[int, ...], device="cpu", dtype=None) -> None:
        import torch
        from transformers.cache_utils import Cache

        super_init = getattr(Cache, "__init__", None)
        if super_init:
            Cache.__init__(self)
        self.past_seen_tokens: int = 0
        self.k_cache = torch.zeros(shape, dtype=dtype, device=device)
        self.v_cache = torch.zeros(shape, dtype=dtype, device=device)

    def update(
        self,
        k_state,
        v_state,
        layer_idx: int,
        cache_kwargs=None,
    ) -> Tuple:
        if cache_kwargs is None:
            raise ValueError("slice_indices required in cache_kwargs")
        begin, end = cache_kwargs["slice_indices"]
        self.k_cache[layer_idx, :, : k_state.shape[1], begin:end, :] = k_state
        self.v_cache[layer_idx, :, : v_state.shape[1], begin:end, :] = v_state
        k_out = self.k_cache[layer_idx, :, :, :end, :]
        v_out = self.v_cache[layer_idx, :, :, :end, :]
        return k_out, v_out

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self.past_seen_tokens

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        return self.past_seen_tokens


def _patch_qwen2_attention(torch) -> None:
    """Install slice-based SDPA attention for exportable KV cache updates."""
    from transformers.models.qwen2.modeling_qwen2 import QWEN2_ATTENTION_CLASSES, Qwen2SdpaAttention

    def repeat_kv_const(hidden_states, n_rep: int, num_kv_heads: int, num_heads: int):
        if n_rep == 1:
            return hidden_states
        bsz = hidden_states.shape[0]
        slen = hidden_states.shape[2]
        head_dim = hidden_states.shape[3]
        expanded = hidden_states.unsqueeze(2).expand(bsz, num_kv_heads, n_rep, slen, head_dim)
        return expanded.reshape(bsz, num_heads, slen, head_dim)

    def rotate_half(x, half_dim: int):
        return torch.cat((-x[..., half_dim:], x[..., :half_dim]), dim=-1)

    def apply_rotary_pos_emb(q, k, cos, sin, position_ids, half_dim: int, unsqueeze_dim: int = 1):
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q, half_dim) * sin)
        k_embed = (k * cos) + (rotate_half(k, half_dim) * sin)
        return q_embed, k_embed

    class SliceUpdateQwen2SdpaAttention(Qwen2SdpaAttention):
        @torch.no_grad()
        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value=None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
        ):
            bsz, q_len, _ = hidden_states.size()

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            end_step = attention_mask.shape[-1]
            past_kv_len = end_step - q_len
            half_dim = self.head_dim // 2

            cos, sin = self.rotary_emb(value_states, seq_len=self.rotary_emb.max_position_embeddings)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids, half_dim
            )

            if past_key_value is not None:
                key_states, value_states = past_key_value.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                    cache_kwargs={"slice_indices": (past_kv_len, end_step)},
                )

            key_states = repeat_kv_const(
                key_states, self.num_key_value_groups, self.num_key_value_heads, self.num_heads
            )
            value_states = repeat_kv_const(
                value_states, self.num_key_value_groups, self.num_key_value_heads, self.num_heads
            )

            causal_mask = attention_mask
            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

            dtype = query_states.dtype
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states.to(dtype),
                value_states.to(dtype),
                attn_mask=causal_mask.to(dtype),
                dropout_p=0.0,
                is_causal=False,
            )

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)
            return attn_output, None, past_key_value

    QWEN2_ATTENTION_CLASSES["sdpa"] = SliceUpdateQwen2SdpaAttention


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


def convert_decode_model(*, model_id: str, ct, torch, AutoModelForCausalLM, output_dir: Path, max_ctx: int) -> dict:
    """Export stateful decode model with MLState (keyCache / valueCache buffers)."""
    from transformers.cache_utils import Cache

    # Make SliceUpdateKeyValueCache a Cache subclass for HF compatibility
    from transformers.cache_utils import Cache as HFCache

    class SliceKVCache(SliceUpdateKeyValueCache, HFCache):
        pass

    decode_path = output_dir / "qwen2.5-0.5b-decode-kv.mlpackage"
    pt_path = output_dir / "qwen2.5-0.5b-decode-kv.pt"

    _patch_qwen2_attention(torch)

    class StatefulQwen2ForCausalLM(torch.nn.Module):
        def __init__(self, mid: str, max_context_size: int, batch_size: int = 1) -> None:
            super().__init__()
            self.model = AutoModelForCausalLM.from_pretrained(
                mid, torch_dtype=torch.float16, attn_implementation="sdpa"
            )
            config = self.model.config
            head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
            self.kv_cache_shape = (
                config.num_hidden_layers,
                batch_size,
                config.num_key_value_heads,
                max_context_size,
                head_dim,
            )
            self.kv_cache = SliceKVCache(
                shape=self.kv_cache_shape,
                device=next(self.model.parameters()).device,
                dtype=torch.float16,
            )
            self.register_buffer("keyCache", self.kv_cache.k_cache)
            self.register_buffer("valueCache", self.kv_cache.v_cache)

        @torch.no_grad()
        def forward(self, input_ids: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
            self.kv_cache.past_seen_tokens = causal_mask.shape[-1] - input_ids.shape[-1]
            return self.model(
                input_ids.long(),
                attention_mask=causal_mask,
                past_key_values=self.kv_cache,
                use_cache=True,
            ).logits

    torch_model = StatefulQwen2ForCausalLM(model_id, max_context_size=max_ctx).eval()
    kv_cache_shape = torch_model.kv_cache_shape

    # Trace at decode step: 1 token, causal mask width = past_len + 1 (example: 512 + 1)
    example_past = 512
    input_ids = torch.zeros((1, 1), dtype=torch.int32)
    causal_mask = torch.zeros((1, 1, 1, example_past + 1), dtype=torch.float16)
    traced = torch.jit.trace(torch_model, [input_ids, causal_mask], strict=False)
    traced.save(str(pt_path))
    logger.info("Saved TorchScript reference %s", pt_path)

    query_length = ct.RangeDim(lower_bound=1, upper_bound=max_ctx, default=1)
    end_step_dim = ct.RangeDim(lower_bound=1, upper_bound=max_ctx, default=1)
    inputs = [
        ct.TensorType(shape=(1, query_length), dtype=np.int32, name="inputIds"),
        ct.TensorType(
            shape=(1, 1, query_length, end_step_dim),
            dtype=np.float16,
            name="causalMask",
        ),
    ]
    outputs = [ct.TensorType(dtype=np.float16, name="logits")]
    states = [
        ct.StateType(
            wrapped_type=ct.TensorType(shape=kv_cache_shape, dtype=np.float16),
            name="keyCache",
        ),
        ct.StateType(
            wrapped_type=ct.TensorType(shape=kv_cache_shape, dtype=np.float16),
            name="valueCache",
        ),
    ]

    coreml_ok = False
    coreml_error: str | None = None
    try:
        decode_ml = ct.convert(
            traced,
            inputs=inputs,
            outputs=outputs,
            states=states,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS15,
            skip_model_load=True,
        )
        decode_ml.save(str(decode_path))
        coreml_ok = True
        logger.info("Saved stateful Core ML decode %s", decode_path)
    except Exception as exc:
        coreml_error = str(exc)
        logger.warning("Stateful Core ML decode convert failed: %s", exc)

    report = {
        "decode_path": str(decode_path) if coreml_ok else None,
        "decode_pt_path": str(pt_path),
        "coreml_decode_ok": coreml_ok,
        "coreml_decode_error": coreml_error,
        "export_method": "mlstate" if coreml_ok else "failed",
        "kv_cache_shape": list(kv_cache_shape),
    }

    if coreml_ok and not getattr(convert_decode_model, "_skip_validation", False):
        try:
            mlmodel = ct.models.MLModel(str(decode_path), compute_units=ct.ComputeUnit.ALL)
            state = mlmodel.make_state()
            state.write_state("keyCache", np.zeros(kv_cache_shape, dtype=np.float32))
            state.write_state("valueCache", np.zeros(kv_cache_shape, dtype=np.float32))
            out = mlmodel.predict(
                {
                    "inputIds": np.array([[1]], dtype=np.int32),
                    "causalMask": np.zeros((1, 1, 1, 1), dtype=np.float16),
                },
                state=state,
            )
            report["decode_logits_shape"] = list(np.array(out["logits"]).shape)
        except Exception as exc:
            report["validation_error"] = str(exc)
            logger.warning("Decode validation failed: %s", exc)

    del torch_model
    return report


def convert_kv_models(
    *,
    model_id: str,
    output_dir: Path,
    max_ctx: int = DEFAULT_MAX_CTX,
    mode: str = "all",
    skip_validation: bool = False,
) -> dict:
    ct, torch, AutoModelForCausalLM = _import_stack()
    output_dir.mkdir(parents=True, exist_ok=True)

    prefill_only = mode == "prefill"
    decode_only = mode == "decode"

    report: dict = {"model_id": model_id, "max_ctx": max_ctx, "mode": mode}

    hf = None
    if not decode_only:
        logger.info("Loading %s for prefill", model_id)
        hf = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, attn_implementation="sdpa"
        ).eval()
        prefill_path = convert_prefill_model(hf=hf, ct=ct, torch=torch, output_dir=output_dir, max_ctx=max_ctx)
        report["prefill_path"] = str(prefill_path)

    if not prefill_only:
        convert_decode_model._skip_validation = skip_validation  # type: ignore[attr-defined]
        decode_info = convert_decode_model(
            model_id=model_id,
            ct=ct,
            torch=torch,
            AutoModelForCausalLM=AutoModelForCausalLM,
            output_dir=output_dir,
            max_ctx=max_ctx,
        )
        report.update(decode_info)

    if not skip_validation and report.get("prefill_path"):
        pm = ct.models.MLModel(str(report["prefill_path"]), compute_units=ct.ComputeUnit.ALL)
        ids = np.zeros((1, max_ctx), dtype=np.int32)
        ids[0, :8] = np.arange(1, 9, dtype=np.int32)
        out = pm.predict({"inputIds": ids})
        report["prefill_logits_shape"] = list(np.array(out["logits"]).shape)

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export Core ML prefill+stateful decode KV models.")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--max-ctx", type=int, default=DEFAULT_MAX_CTX)
    parser.add_argument(
        "--mode",
        choices=("all", "prefill", "decode"),
        default="all",
        help="Export prefill only, decode only, or both",
    )
    parser.add_argument("--prefill-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--decode-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.prefill_only:
        args.mode = "prefill"
    if args.decode_only:
        args.mode = "decode"
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")
    try:
        report = convert_kv_models(
            model_id=args.model,
            output_dir=args.output_dir,
            max_ctx=args.max_ctx,
            mode=args.mode,
            skip_validation=args.skip_validation,
        )
    except Exception:
        logger.exception("KV conversion failed")
        return 1
    print("KV conversion complete:")
    for k, v in report.items():
        print(f"  {k}: {v}")
    return 0 if report.get("coreml_decode_ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())