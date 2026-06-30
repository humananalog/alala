"""Phase 0 model registry for harness decode benchmarks."""

from __future__ import annotations

MODEL_REGISTRY: dict[str, str] = {
    "baseline": "mlx-community/Qwen2.5-7B-Instruct-4bit",
}


def resolve_model_id(name: str) -> str:
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]
    return name