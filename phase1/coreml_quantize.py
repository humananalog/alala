#!/usr/bin/env python3
"""Post-export weight quantization for Phase 1 Core ML models."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_INPUT = Path("models/qwen2.5-0.5b-decode-kv-torch-export.mlpackage")
DEFAULT_OUTPUT = Path("models/qwen2.5-0.5b-decode-kv-torch-export-int4.mlpackage")

def _import_stack():
    import coremltools as ct
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig,
        OptimizationConfig,
        linear_quantize_weights,
    )

    return ct, OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights


def quantize_decode_weights(
    *,
    input_path: Path,
    output_path: Path,
    quant_bits: int = 4,
    weight_threshold: int = 512,
    skip_validation: bool = False,
) -> dict:
    """Apply post-export linear weight quantization to a Core ML mlprogram."""
    import numpy as np

    ct, OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights = _import_stack()

    if not input_path.exists():
        raise FileNotFoundError(f"Input model not found: {input_path}")

    dtype = "int4" if quant_bits == 4 else "int8"
    logger.info("Loading %s", input_path)
    mlmodel = ct.models.MLModel(str(input_path), compute_units=ct.ComputeUnit.CPU_ONLY)

    op_config = OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype=dtype,
        weight_threshold=weight_threshold,
    )
    # linear_quantize_weights only compresses linear layer weights; activation
    # I/O (keyCache, valueCache, logits, etc.) remain fp16.
    config = OptimizationConfig(global_config=op_config)
    logger.info("Applying %s linear weight quantization (weights only; KV I/O stays fp16)", dtype)
    quantized = linear_quantize_weights(mlmodel, config=config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantized.save(str(output_path))
    logger.info("Saved %s", output_path)

    report = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "quant_bits": quant_bits,
        "dtype": dtype,
        "weight_threshold": weight_threshold,
        "kv_io_preserved_fp16": True,
        "notes": "Weight-only quant; KV cache tensors are activations not quantized",
    }

    if not skip_validation:
        max_ctx = 1024
        kv_shape = (24, 1, 2, max_ctx, 64)
        validated = ct.models.MLModel(str(output_path), compute_units=ct.ComputeUnit.ALL)
        out = validated.predict(
            {
                "inputIds": np.array([[1]], dtype=np.int32),
                "keyCache": np.zeros(kv_shape, dtype=np.float16),
                "valueCache": np.zeros(kv_shape, dtype=np.float16),
                "cachePosition": np.array([0], dtype=np.int32),
                "causalMask": np.zeros((1, 1, 1, max_ctx), dtype=np.float16),
            }
        )
        report["validation_outputs"] = list(out.keys())
        report["logits_shape"] = list(np.array(out["logits"]).shape)

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Post-export int4/int8 weight quantization for Core ML decode models."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quant-bits", type=int, choices=(4, 8), default=4)
    parser.add_argument("--weight-threshold", type=int, default=512)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    try:
        report = quantize_decode_weights(
            input_path=args.input,
            output_path=args.output,
            quant_bits=args.quant_bits,
            weight_threshold=args.weight_threshold,
            skip_validation=args.skip_validation,
        )
    except Exception:
        logger.exception("Quantization failed")
        return 1

    print("Quantization complete:")
    for key, value in report.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())