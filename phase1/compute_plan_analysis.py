#!/usr/bin/env python3
"""Per-op MLComputePlan breakdown for planner vs runtime gap analysis."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PHASE1_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PHASE1_DIR))

from coreml_instrumentation import load_coreml_model, resolve_compute_unit  # noqa: E402

# Ops tied to mask-based KV slot write and cache rebuild.
MASK_KV_OPS = frozenset(
    {"ios18.equal", "tile", "ios18.greater_equal", "select", "ios18.gather"}
)
CACHE_REBUILD_OPS = frozenset({"ios18.concat"})
INT4_DEQUANT_OPS = frozenset({"ios18.constexpr_blockwise_shift_scale"})
ATTENTION_OPS = frozenset({"ios18.scaled_dot_product_attention"})
LINEAR_OPS = frozenset({"ios18.linear"})


def _device_class(device) -> str:
    name = type(device).__name__
    if "NeuralEngine" in name:
        return "ANE"
    if "GPU" in name:
        return "GPU"
    if "CPU" in name:
        return "CPU"
    return name


def analyze_model(path: Path, *, compute_units: str = "all") -> dict:
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan

    cu = resolve_compute_unit(ct, compute_units)
    model = ct.models.MLModel(str(path), compute_units=cu)
    compiled = model.get_compiled_model_path()
    plan = MLComputePlan.load_from_path(compiled, compute_units=cu)
    program = plan.model_structure.program
    if program is None:
        return {"error": "no_ml_program", "path": str(path)}

    per_op_total: Counter[str] = Counter()
    per_op_pref: dict[str, Counter[str]] = defaultdict(Counter)
    per_op_supported: dict[str, Counter[str]] = defaultdict(Counter)
    gpu_only_ops: Counter[str] = Counter()
    gpu_despite_ane: Counter[str] = Counter()
    unplaced_ops: Counter[str] = Counter()

    for _func_name, function in program.functions.items():
        for op in function.block.operations:
            name = op.operator_name
            per_op_total[name] += 1
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if usage is None:
                unplaced_ops[name] += 1
                continue
            pref = _device_class(usage.preferred_compute_device)
            per_op_pref[name][pref] += 1
            supported = {_device_class(d) for d in usage.supported_compute_devices}
            for dev in supported:
                per_op_supported[name][dev] += 1
            if pref == "GPU" and "ANE" not in supported:
                gpu_only_ops[name] += 1
            elif pref == "GPU" and "ANE" in supported:
                gpu_despite_ane[name] += 1

    total = sum(per_op_total.values())
    placed = total - sum(unplaced_ops.values())
    pref_totals = Counter()
    for counts in per_op_pref.values():
        pref_totals.update(counts)

    def _category_counts(ops: frozenset[str]) -> dict:
        total_cat = sum(per_op_total[o] for o in ops)
        pref_cat = Counter()
        for op in ops:
            pref_cat.update(per_op_pref.get(op, {}))
        return {
            "total_ops": total_cat,
            "preferred": dict(pref_cat),
            "ane_fraction_of_placed": round(
                pref_cat.get("ANE", 0) / max(sum(pref_cat.values()), 1), 4
            ),
        }

    def _top_rows(counter: Counter, n: int = 15) -> list[dict]:
        rows = []
        for op, count in counter.most_common(n):
            pref = dict(per_op_pref.get(op, {}))
            rows.append(
                {
                    "op": op,
                    "count": count,
                    "preferred": pref,
                    "ane_eligible_unplaced": unplaced_ops.get(op, 0),
                }
            )
        return rows

    ane_count = pref_totals.get("ANE", 0)
    return {
        "path": str(path),
        "compute_units": compute_units,
        "total_operations": total,
        "placed_operations": placed,
        "unplaced_operations": sum(unplaced_ops.values()),
        "unplaced_by_op": unplaced_ops.most_common(12),
        "preferred_device_class": dict(pref_totals),
        "ane_preferred_fraction_all_ops": round(ane_count / max(total, 1), 4),
        "ane_preferred_fraction_placed_only": round(ane_count / max(placed, 1), 4),
        "gpu_preferred_fraction_all_ops": round(
            pref_totals.get("GPU", 0) / max(total, 1), 4
        ),
        "gpu_only_ops": gpu_only_ops.most_common(20),
        "gpu_despite_ane_support": gpu_despite_ane.most_common(20),
        "categories": {
            "mask_kv": _category_counts(MASK_KV_OPS),
            "cache_rebuild": _category_counts(CACHE_REBUILD_OPS),
            "int4_dequant": _category_counts(INT4_DEQUANT_OPS),
            "attention": _category_counts(ATTENTION_OPS),
            "linear": _category_counts(LINEAR_OPS),
        },
        "top_ops_by_count": _top_rows(per_op_total),
        "kv_io_note": (
            "keyCache/valueCache/keyCacheOut/valueCacheOut are explicit I/O tensors; "
            "copy cost is outside MLComputePlan op counts."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Path to .mlpackage")
    parser.add_argument("--compute-units", default="all")
    parser.add_argument("--output", type=Path, help="Write JSON report")
    args = parser.parse_args()

    report = analyze_model(args.model, compute_units=args.compute_units)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())