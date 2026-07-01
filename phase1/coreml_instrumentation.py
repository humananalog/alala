"""Core ML load-time instrumentation and compute-plan introspection for Phase 1."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

COMPUTE_UNIT_CHOICES = ("all", "cpu_and_ne", "cpu_and_gpu", "cpu_only")


@dataclass
class CoreMLLoadInfo:
    path: str
    role: str
    compute_units: str
    compute_unit_enum: str
    spec_version: int
    source_dialect: str | None
    inputs: list[dict[str, str]]
    outputs: list[dict[str, str]]
    states: list[str] = field(default_factory=list)
    compiled_model_path: str | None = None
    load_error: str | None = None
    compute_plan: dict[str, Any] | None = None


def resolve_compute_unit(ct, name: str):
    mapping = {
        "all": ct.ComputeUnit.ALL,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unknown compute_units={name!r}; expected one of {COMPUTE_UNIT_CHOICES}")
    return mapping[key]


def _feature_type_name(feature) -> str:
    which = feature.type.WhichOneof("Type")
    return which or "unknown"


def _summarize_compute_plan(ct, compiled_path: str, compute_units) -> dict[str, Any]:
    from coremltools.models.compute_plan import MLComputePlan

    plan = MLComputePlan.load_from_path(compiled_path, compute_units=compute_units)
    program = plan.model_structure.program
    if program is None:
        return {"error": "no_ml_program"}

    preferred_class = Counter()
    supported_class = Counter()
    operator_names = Counter()
    ane_ops: list[str] = []
    gpu_ops: list[str] = []
    cpu_ops: list[str] = []

    def _device_class(device) -> str:
        name = type(device).__name__
        if "NeuralEngine" in name:
            return "ANE"
        if "GPU" in name:
            return "GPU"
        if "CPU" in name:
            return "CPU"
        return name

    for _func_name, function in program.functions.items():
        for op in function.block.operations:
            operator_names[op.operator_name] += 1
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if usage is None:
                continue
            pref_class = _device_class(usage.preferred_compute_device)
            preferred_class[pref_class] += 1
            if pref_class == "ANE":
                ane_ops.append(op.operator_name)
            elif pref_class == "GPU":
                gpu_ops.append(op.operator_name)
            else:
                cpu_ops.append(op.operator_name)
            for device in usage.supported_compute_devices:
                supported_class[_device_class(device)] += 1

    total_ops = sum(operator_names.values())
    ane_count = preferred_class.get("ANE", 0)
    return {
        "total_operations": total_ops,
        "preferred_device_class": dict(preferred_class),
        "supported_device_class": dict(supported_class),
        "top_operator_names": operator_names.most_common(12),
        "ane_preferred_operator_sample": sorted(set(ane_ops))[:20],
        "gpu_preferred_operator_sample": sorted(set(gpu_ops))[:20],
        "cpu_preferred_operator_sample": sorted(set(cpu_ops))[:20],
        "ane_preferred_fraction": round(ane_count / max(total_ops, 1), 4),
        "gpu_preferred_fraction": round(preferred_class.get("GPU", 0) / max(total_ops, 1), 4),
    }


def load_coreml_model(
    path: Path,
    *,
    role: str,
    compute_units: str = "all",
    capture_compute_plan: bool = True,
):
    import coremltools as ct

    cu = resolve_compute_unit(ct, compute_units)
    info = CoreMLLoadInfo(
        path=str(path),
        role=role,
        compute_units=compute_units,
        compute_unit_enum=cu.name,
        spec_version=0,
        source_dialect=None,
        inputs=[],
        outputs=[],
    )
    try:
        model = ct.models.MLModel(str(path), compute_units=cu)
        spec = model.get_spec()
        info.spec_version = int(spec.specificationVersion)
        md = spec.description.metadata
        info.source_dialect = md.userDefined.get("com.github.apple.coremltools.source_dialect")
        info.inputs = [{"name": i.name, "type": _feature_type_name(i)} for i in spec.description.input]
        info.outputs = [{"name": o.name, "type": _feature_type_name(o)} for o in spec.description.output]
        states = getattr(spec.description, "state", None)
        if states:
            info.states = [s.name for s in states]
        info.compiled_model_path = model.get_compiled_model_path()
        if capture_compute_plan and info.compiled_model_path:
            try:
                info.compute_plan = _summarize_compute_plan(ct, info.compiled_model_path, cu)
            except Exception as exc:
                info.compute_plan = {"error": str(exc)}
        return model, info
    except Exception as exc:
        info.load_error = str(exc)
        return None, info


def log_load_info(info: CoreMLLoadInfo, *, stream=None) -> None:
    stream = stream or sys.stdout
    print(f"[coreml] role={info.role} path={Path(info.path).name}", file=stream)
    print(
        f"[coreml]   compute_units={info.compute_units} ({info.compute_unit_enum})",
        file=stream,
    )
    if info.load_error:
        print(f"[coreml]   LOAD_ERROR: {info.load_error}", file=stream)
        return
    print(f"[coreml]   spec_version={info.spec_version} dialect={info.source_dialect}", file=stream)
    print(f"[coreml]   inputs={info.inputs}", file=stream)
    print(f"[coreml]   outputs={info.outputs}", file=stream)
    if info.states:
        print(f"[coreml]   states={info.states}", file=stream)
    if info.compute_plan:
        cp = info.compute_plan
        if "error" in cp:
            print(f"[coreml]   compute_plan_error={cp['error']}", file=stream)
        else:
            print(
                f"[coreml]   compute_plan_ops={cp['total_operations']} "
                f"ane_fraction={cp.get('ane_preferred_fraction')} "
                f"gpu_fraction={cp.get('gpu_preferred_fraction')} "
                f"preferred_class={cp.get('preferred_device_class')}",
                file=stream,
            )


def log_runtime_environment(*, stream=None) -> dict[str, str | None]:
    stream = stream or sys.stdout
    env_keys = (
        "COREML_VERBOSE",
        "COREML_DEBUG",
        "MLTOOLS_VERBOSE",
        "E5RT_DEBUG",
        "E5RT_LOG_LEVEL",
    )
    observed = {k: os.environ.get(k) for k in env_keys}
    print("[coreml] runtime_environment:", file=stream)
    for key, value in observed.items():
        print(f"[coreml]   {key}={value!r}", file=stream)
    return observed


def dump_load_report(
    infos: list[CoreMLLoadInfo],
    output_path: Path,
    *,
    environment: dict[str, str | None] | None = None,
) -> None:
    payload = {
        "models": [asdict(i) for i in infos],
        "environment": environment or {},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")