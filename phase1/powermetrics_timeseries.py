#!/usr/bin/env python3
"""Summarize per-window ANE proxy from powermetrics plist logs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "harness"))

from powermetrics_log import parse_powermetrics_file  # noqa: E402


def summarize(path: Path, *, window_s: float = 5.0) -> dict:
    samples = parse_powermetrics_file(path)
    if not samples:
        return {"error": "no_samples", "path": str(path)}

    windows: list[dict] = []
    bucket_cpu = bucket_gpu = bucket_ane = 0.0
    bucket_start_idx = 0
    elapsed_s = 0.0

    for i, sample in enumerate(samples):
        bucket_cpu += sample.cpu_mj
        bucket_gpu += sample.gpu_mj
        bucket_ane += sample.ane_mj
        elapsed_s += max(sample.elapsed_ns / 1_000_000_000.0, 1.0)

        if elapsed_s >= window_s or i == len(samples) - 1:
            total_j = (bucket_cpu + bucket_gpu + bucket_ane) / 1000.0
            ane_j = bucket_ane / 1000.0
            proxy = (ane_j / total_j * 100.0) if total_j > 0 else 0.0
            windows.append(
                {
                    "window_index": len(windows),
                    "sample_start": bucket_start_idx,
                    "sample_end": i,
                    "duration_s": round(elapsed_s, 2),
                    "ane_proxy_pct": round(proxy, 3),
                    "ane_joules": round(ane_j, 4),
                    "gpu_joules": round(bucket_gpu / 1000.0, 4),
                    "cpu_joules": round(bucket_cpu / 1000.0, 4),
                    "package_power_w_avg": round(
                        sum(
                            s.package_mw
                            for s in samples[bucket_start_idx : i + 1]
                        )
                        / max(i - bucket_start_idx + 1, 1)
                        / 1000.0,
                        2,
                    ),
                    "temp_c_max": max(
                        (s.temp_c for s in samples[bucket_start_idx : i + 1] if s.temp_c),
                        default=None,
                    ),
                }
            )
            bucket_cpu = bucket_gpu = bucket_ane = 0.0
            elapsed_s = 0.0
            bucket_start_idx = i + 1

    proxies = [w["ane_proxy_pct"] for w in windows]
    return {
        "path": str(path),
        "sample_count": len(samples),
        "window_s": window_s,
        "windows": windows,
        "ane_proxy_pct_mean": round(sum(proxies) / len(proxies), 3),
        "ane_proxy_pct_min": min(proxies),
        "ane_proxy_pct_max": max(proxies),
        "ane_proxy_pct_p50": sorted(proxies)[len(proxies) // 2],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("powermetrics_log", type=Path)
    parser.add_argument("--window-s", type=float, default=5.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = summarize(args.powermetrics_log, window_s=args.window_s)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())