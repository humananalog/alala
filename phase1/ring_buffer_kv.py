"""Ring-buffer KV cache helpers for fixed-shape Core ML decode.

Maps a logical sequence position to a physical cache slot and builds a
fixed-width causal attention mask for torch.export decode (mask int4 path).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

KvCacheMode = Literal["linear", "ring"]

# fp16 minimum — masks invalid KV slots in SDPA (0 = attend).
_MASK_NEG_INF = np.float16(np.finfo(np.float16).min)


@dataclass(frozen=True)
class RingBufferConfig:
    mode: KvCacheMode = "linear"
    ring_size: int = 512
    max_ctx: int = 1024
    num_layers: int = 24
    kv_heads: int = 2
    head_dim: int = 64

    def __post_init__(self) -> None:
        if self.ring_size < 1:
            raise ValueError("ring_size must be >= 1")
        if self.ring_size > self.max_ctx:
            raise ValueError("ring_size must be <= max_ctx")

    @property
    def kv_io_bytes_per_step(self) -> int:
        """Explicit in+out KV tensor copies per decode step (fp16)."""
        elems = self.num_layers * 1 * self.kv_heads * self.max_ctx * self.head_dim
        return 2 * 2 * elems * 2  # key+value, in+out, fp16


def write_slot(seq_len: int, config: RingBufferConfig) -> int:
    """Physical cache index for the next KV write.

    ``seq_len`` is the logical position of the incoming token (0-based), equal
    to the number of tokens already in the sequence before this decode step.
    """
    if config.mode == "linear":
        return seq_len
    return seq_len % config.ring_size


def ring_causal_mask(seq_len: int, config: RingBufferConfig) -> np.ndarray:
    """Build ``(1, 1, 1, max_ctx)`` mask: 0 = attend, neg_inf = ignore."""
    max_ctx = config.max_ctx
    mask = np.full((1, 1, 1, max_ctx), _MASK_NEG_INF, dtype=np.float16)

    if config.mode == "linear":
        # Attend to all cached keys [0, seq_len).
        if seq_len > 0:
            mask[0, 0, 0, : min(seq_len, max_ctx)] = np.float16(0.0)
        return mask

    ring = config.ring_size
    window_start = max(0, seq_len - ring)
    for logical in range(window_start, seq_len):
        physical = logical % ring
        if physical < max_ctx:
            mask[0, 0, 0, physical] = np.float16(0.0)
    return mask


def linear_causal_mask(seq_len: int, max_ctx: int) -> np.ndarray:
    """All-zero mask over active prefix (legacy torch.export path)."""
    mask = np.zeros((1, 1, 1, max_ctx), dtype=np.float16)
    if seq_len < max_ctx:
        mask[0, 0, 0, seq_len:] = _MASK_NEG_INF
    return mask


def should_re_prefill(seq_len: int, config: RingBufferConfig) -> bool:
    """Whether linear mode must re-run prefill (ring mode never does)."""
    if config.mode == "ring":
        return False
    return seq_len >= config.max_ctx


def active_kv_slots(seq_len: int, config: RingBufferConfig) -> int:
    """Count of KV slots attended to this step (proxy for effective KV working set)."""
    if config.mode == "linear":
        return min(seq_len, config.max_ctx)
    return min(seq_len, config.ring_size)