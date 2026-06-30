# How to Run the First Micro-Benchmark on M4 — Alalā

**Version**: 1.0  
**Target**: Mac Mini M4 24GB  
**Purpose**: Establish baseline measurement capability for IPJ, ANE utilization, thermal behavior, and SRAM limits.

## Prerequisites

- macOS with `powermetrics` access (usually requires `sudo` or proper entitlements).
- Python 3.11+ with `numpy`, `matplotlib` (optional for plotting).
- The energy logging harness: `harness/m4_energy_harness.py`.
- Stable thermal state (machine idled for 10+ minutes).

## Step-by-Step Instructions

### 1. Prepare Environment

```bash
cd alala
mkdir -p experiments logs harness results
cp harness/m4_energy_harness.py harness/
chmod +x harness/m4_energy_harness.py
```
