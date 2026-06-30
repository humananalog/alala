#!/usr/bin/env python3
"""Subprocess MLX decode worker — run with the venv Python that has mlx_lm."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def cmd_warmup(args: argparse.Namespace) -> dict:
    runner = _load_runner(args.model)
    runner.warmup(args.context_length, decode_tokens=min(16, args.decode_tokens))
    return {"status": "ok"}


def cmd_context_step(args: argparse.Namespace) -> dict:
    runner = _load_runner(args.model)
    runner.warmup(args.context_length, decode_tokens=min(16, args.decode_tokens))
    result = runner.run_sustained(
        context_length=args.context_length,
        duration_s=args.duration,
        steady_window_s=args.steady_window,
        decode_tokens=args.decode_tokens,
        kv_bits=args.kv_bits,
        orchestration_delay_ms=args.orchestration_delay_ms,
    )
    return {
        "context_length": result.context_length,
        "tokens_generated": result.tokens_generated,
        "tokens_per_second": result.tokens_per_second,
        "tokens_per_second_sustained": result.tokens_per_second_sustained,
        "peak_memory_gb": result.peak_memory_gb,
        "decode_rounds": result.decode_rounds,
    }


def cmd_sustained(args: argparse.Namespace) -> dict:
    runner = _load_runner(args.model)
    result = runner.run_sustained(
        context_length=args.context_length,
        duration_s=args.duration,
        steady_window_s=args.steady_window,
        decode_tokens=args.decode_tokens,
    )
    return {
        "context_length": result.context_length,
        "tokens_generated": result.tokens_generated,
        "tokens_per_second": result.tokens_per_second,
        "tokens_per_second_sustained": result.tokens_per_second_sustained,
        "peak_memory_gb": result.peak_memory_gb,
        "decode_rounds": result.decode_rounds,
    }


def _load_runner(model: str):
    from decode import DecodeRunner

    return DecodeRunner(model)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    warmup = sub.add_parser("warmup")
    warmup.add_argument("--model", required=True)
    warmup.add_argument("--context-length", type=int, required=True)
    warmup.add_argument("--decode-tokens", type=int, default=16)

    sustained = sub.add_parser("sustained")
    sustained.add_argument("--model", required=True)
    sustained.add_argument("--context-length", type=int, required=True)
    sustained.add_argument("--duration", type=int, required=True)
    sustained.add_argument("--steady-window", type=int, required=True)
    sustained.add_argument("--decode-tokens", type=int, default=32)

    context_step = sub.add_parser("context-step")
    context_step.add_argument("--model", required=True)
    context_step.add_argument("--context-length", type=int, required=True)
    context_step.add_argument("--duration", type=int, required=True)
    context_step.add_argument("--steady-window", type=int, required=True)
    context_step.add_argument("--decode-tokens", type=int, default=32)
    context_step.add_argument("--kv-bits", type=int, default=None)
    context_step.add_argument("--orchestration-delay-ms", type=int, default=0)

    args = parser.parse_args()
    started = time.time()
    try:
        if args.command == "warmup":
            payload = cmd_warmup(args)
        elif args.command == "context-step":
            payload = cmd_context_step(args)
        else:
            payload = cmd_sustained(args)
        payload["elapsed_s"] = round(time.time() - started, 3)
        print(json.dumps(payload))
        return 0
    except Exception as exc:  # noqa: BLE001 - worker must report errors to parent
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())