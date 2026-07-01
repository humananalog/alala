"""Convert Hugging Face causal LMs to Core ML for ANE-first Phase 1 experiments.

Example:
    python phase1/coreml_convert.py \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --output models/qwen2.5-0.5b-ane.mlpackage

Dependencies (M4 host venv):
    pip install "torch>=2.2" "transformers>=4.44" "coremltools>=8.0" "numpy>=1.26"

ANE-friendly defaults:
- Fixed context length (512, below Phase 0 L_cliff=1024) for traceable static shapes.
- macOS 15+ deployment target → fused scaled_dot_product_attention in Core ML.
- mlprogram + float16 trace, optional 8-bit/4-bit weight compression.
- Validate with ComputeUnit.ALL so the compiler may place ops on ANE where eligible.

Avoid ops that force GPU fallback (dynamic control flow, oversized contexts). Full KV-state
decode export is a follow-up; this script establishes the first convertible forward graph.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_SIZE = 512
DEFAULT_BATCH_SIZE = 1
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_DEPLOYMENT_TARGET = "macOS15"
DEFAULT_COMPUTE_UNITS = "all"

QuantBits = Literal[16, 8, 4]


class ConversionError(RuntimeError):
    """Raised when HF → Core ML conversion or validation fails."""


def _import_stack():
    try:
        import coremltools as ct
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise ConversionError(
            'Install: pip install "torch>=2.2" "transformers>=4.44" "coremltools>=8.0" "numpy>=1.26"'
        ) from exc
    return ct, torch, AutoModelForCausalLM


def _deployment_target(ct, name: str):
    target = getattr(ct.target, name, None)
    if target is None:
        raise ConversionError(f"Unsupported deployment target: {name}")
    return target


def _compute_units(ct, name: str):
    mapping = {
        "all": ct.ComputeUnit.ALL,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
    }
    key = name.lower()
    if key not in mapping:
        raise ConversionError(f"Unknown compute_units={name!r}")
    return mapping[key]


def _apply_quantization(mlmodel, quant_bits: QuantBits):
    if quant_bits == 16:
        return mlmodel
    try:
        from coremltools.optimize.coreml import (
            OpLinearQuantizerConfig,
            OpPalettizerConfig,
            linear_quantize_weights,
            palettize_weights,
        )
    except ImportError as exc:
        raise ConversionError("coremltools.optimize required for weight compression") from exc

    if quant_bits == 8:
        logger.info("Applying 8-bit linear weight quantization")
        return linear_quantize_weights(
            mlmodel,
            config=OpLinearQuantizerConfig(mode="linear_symmetric", weight_threshold=512),
        )
    logger.info("Applying 4-bit palettization")
    return palettize_weights(
        mlmodel,
        config=OpPalettizerConfig(nbits=4, weight_threshold=512),
    )


def validate_coreml_model(
    model_path: str | Path,
    *,
    context_size: int = DEFAULT_CONTEXT_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    compute_units: str = DEFAULT_COMPUTE_UNITS,
) -> dict:
    """Load .mlpackage and run a tiny forward pass."""
    ct, _, _ = _import_stack()
    path = Path(model_path)
    if not path.exists():
        raise ConversionError(f"Core ML model not found: {path}")

    logger.info("Validating %s (compute_units=%s)", path, compute_units)
    mlmodel = ct.models.MLModel(str(path), compute_units=_compute_units(ct, compute_units))
    shape = (batch_size, context_size)
    out = mlmodel.predict({"inputIds": np.zeros(shape, dtype=np.int32)})
    logits = np.array(out["logits"])
    if logits.ndim != 3:
        raise ConversionError(f"Unexpected logits shape: {logits.shape}")

    spec = mlmodel.get_spec()
    return {
        "ok": True,
        "compute_units": compute_units,
        "logits_shape": list(logits.shape),
        "logits_dtype": str(logits.dtype),
        "spec_version": int(spec.specificationVersion),
    }


def convert_to_coreml(
    model_id: str,
    output_path: str,
    *,
    quantize: bool = True,
    quant_bits: QuantBits = 8,
    context_size: int = DEFAULT_CONTEXT_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    minimum_deployment_target: str = DEFAULT_DEPLOYMENT_TARGET,
    compute_units: str = DEFAULT_COMPUTE_UNITS,
    trust_remote_code: bool = False,
    skip_validation: bool = False,
) -> dict:
    """Convert a Hugging Face causal LM to Core ML and optionally validate inference.

    Args:
        model_id: Hugging Face repo id (e.g. Qwen/Qwen2.5-0.5B-Instruct).
        output_path: Destination .mlpackage path.
        quantize: When True, apply weight compression (default 8-bit; override with quant_bits).
    """
    ct, torch, AutoModelForCausalLM = _import_stack()
    out = Path(output_path)
    effective_bits: QuantBits = quant_bits if quantize else 16

    if context_size > 1024:
        logger.warning("context_size=%s exceeds L_cliff=1024; ANE mapping may degrade", context_size)

    logger.info("Loading %s", model_id)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )
    hf_model.eval()

    class ExportableCausalLM(torch.nn.Module):
        def __init__(self, inner) -> None:
            super().__init__()
            self.inner = inner

        @torch.no_grad()
        def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
            # Single-input trace avoids attention_mask ops (new_ones) that break Core ML convert.
            return self.inner(input_ids=input_ids, use_cache=False).logits

    wrapper = ExportableCausalLM(hf_model)
    wrapper.eval()
    input_shape = (batch_size, context_size)
    example = (torch.zeros(input_shape, dtype=torch.int32),)

    logger.info("Exporting static forward (shape=%s)", input_shape)
    try:
        exported = torch.export.export(wrapper, example).run_decompositions({})
        convert_source = exported
    except Exception as export_exc:
        logger.warning("torch.export failed (%s); falling back to torch.jit.trace", export_exc)
        traced = torch.jit.trace(wrapper, example, strict=False)
        traced.eval()
        convert_source = traced

    logger.info("Converting to Core ML (target=%s, quant_bits=%s)", minimum_deployment_target, effective_bits)
    mlmodel = ct.convert(
        convert_source,
        inputs=[ct.TensorType(shape=input_shape, dtype=np.int32, name="inputIds")],
        outputs=[ct.TensorType(dtype=np.float16, name="logits")],
        convert_to="mlprogram",
        minimum_deployment_target=_deployment_target(ct, minimum_deployment_target),
        skip_model_load=True,
    )
    mlmodel = _apply_quantization(mlmodel, effective_bits)

    out.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving %s", out)
    mlmodel.save(str(out))

    report = {
        "model_id": model_id,
        "output_path": str(out),
        "context_size": context_size,
        "quantize": quantize,
        "quant_bits": effective_bits,
        "minimum_deployment_target": minimum_deployment_target,
    }
    if not skip_validation:
        report["validation"] = validate_coreml_model(
            out,
            context_size=context_size,
            batch_size=batch_size,
            compute_units=compute_units,
        )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert HF causal LM → Core ML for ANE Phase 1.")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="Hugging Face model id")
    parser.add_argument("--output", required=True, help="Output .mlpackage path")
    parser.add_argument("--context-size", type=int, default=DEFAULT_CONTEXT_SIZE)
    parser.add_argument("--no-quantize", action="store_true", help="Keep float16 weights (no compression)")
    parser.add_argument(
        "--quant-bits",
        type=int,
        choices=(8, 4),
        default=8,
        help="Weight compression when quantize enabled (default: 8)",
    )
    parser.add_argument("--minimum-deployment-target", default=DEFAULT_DEPLOYMENT_TARGET)
    parser.add_argument(
        "--compute-units",
        default=DEFAULT_COMPUTE_UNITS,
        choices=("all", "cpu_and_ne", "cpu_and_gpu", "cpu_only"),
    )
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")
    try:
        report = convert_to_coreml(
            args.model,
            args.output,
            quantize=not args.no_quantize,
            quant_bits=args.quant_bits,
            context_size=args.context_size,
            minimum_deployment_target=args.minimum_deployment_target,
            compute_units=args.compute_units,
            trust_remote_code=args.trust_remote_code,
            skip_validation=args.skip_validation,
        )
    except ConversionError as exc:
        logger.error("%s", exc)
        return 1
    except Exception:
        logger.exception("Conversion failed")
        return 1

    print("Conversion complete:")
    for key, value in report.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())