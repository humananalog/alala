"""MLX LM sustained decode for Phase 0 SRAM cliff sweeps."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from errors import HarnessError


@dataclass
class DecodeResult:
    context_length: int
    tokens_generated: int
    tokens_per_second: float
    tokens_per_second_sustained: float
    peak_memory_gb: float
    decode_rounds: int


class DecodeRunner:
    def __init__(self, model_id: str) -> None:
        import mlx.core as mx
        from mlx_lm import load, stream_generate

        self._mx = mx
        self._stream_generate = stream_generate
        self._model, self._tokenizer, self._config = load(
            model_id,
            return_config=True,
            tokenizer_config={"trust_remote_code": True},
        )
        self._tokenizer._eos_token_ids = {}
        self._model_id = model_id

    def _vocab_size(self) -> int:
        if "vocab_size" in self._config:
            return int(self._config["vocab_size"])
        return int(self._config["text_config"]["vocab_size"])

    def _prompt_for_context(self, context_length: int) -> list[int]:
        vocab_size = self._vocab_size()
        self._mx.random.seed(context_length)
        return self._mx.random.randint(0, vocab_size, (context_length,)).tolist()

    def warmup(self, context_length: int, decode_tokens: int = 16) -> None:
        prompt = self._prompt_for_context(context_length)
        for _ in self._stream_generate(
            self._model,
            self._tokenizer,
            prompt,
            max_tokens=decode_tokens,
        ):
            pass

    def _generate_kwargs(self, kv_bits: int | None) -> dict:
        if kv_bits is None:
            return {}
        return {
            "kv_bits": kv_bits,
            "kv_group_size": 64,
            "quantized_kv_start": 0,
        }

    def run_sustained(
        self,
        context_length: int,
        duration_s: int,
        steady_window_s: int,
        decode_tokens: int,
        should_stop: Callable[[], bool] | None = None,
        kv_bits: int | None = None,
        orchestration_delay_ms: int = 0,
    ) -> DecodeResult:
        if steady_window_s >= duration_s:
            raise HarnessError("--steady-window must be less than --step-duration.")

        prompt = self._prompt_for_context(context_length)
        start = time.monotonic()
        steady_start = start + (duration_s - steady_window_s)
        total_tokens = 0
        steady_tokens = 0
        peak_memory_gb = 0.0
        decode_rounds = 0

        generate_kwargs = self._generate_kwargs(kv_bits)
        bookkeeping: list[int] = []

        while time.monotonic() - start < duration_s:
            if should_stop and should_stop():
                break

            if orchestration_delay_ms > 0:
                time.sleep(orchestration_delay_ms / 1000.0)
                bookkeeping.append(len(bookkeeping))

            last = None
            for response in self._stream_generate(
                self._model,
                self._tokenizer,
                prompt,
                max_tokens=decode_tokens,
                **generate_kwargs,
            ):
                last = response
                now = time.monotonic()
                total_tokens += 1
                peak_memory_gb = max(peak_memory_gb, response.peak_memory)
                if now >= steady_start:
                    steady_tokens += 1

            decode_rounds += 1
            if last is None:
                raise HarnessError(f"Decode produced no tokens at context_length={context_length}")

        elapsed = time.monotonic() - start
        steady_elapsed = min(steady_window_s, max(0.0, elapsed - (duration_s - steady_window_s)))
        if steady_elapsed <= 0:
            steady_elapsed = steady_window_s

        tokens_per_second = total_tokens / elapsed if elapsed > 0 else 0.0
        tokens_per_second_sustained = steady_tokens / steady_elapsed if steady_elapsed > 0 else 0.0

        return DecodeResult(
            context_length=context_length,
            tokens_generated=total_tokens,
            tokens_per_second=tokens_per_second,
            tokens_per_second_sustained=tokens_per_second_sustained,
            peak_memory_gb=peak_memory_gb,
            decode_rounds=decode_rounds,
        )