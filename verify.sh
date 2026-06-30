#!/usr/bin/env bash
# Alalā repository verification — run before every commit.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

errors=0

fail() {
  echo "ERROR: $*" >&2
  errors=$((errors + 1))
}

warn() {
  echo "WARN: $*" >&2
}

echo "=== Alalā verify.sh ==="

# 1. Reject known placeholder stub strings
stub_pattern='^(Content of |Full content of |Binary (zip|image) content)$'
while IFS= read -r -d '' file; do
  if grep -qE "$stub_pattern" "$file" 2>/dev/null; then
    fail "Placeholder stub found in: $file"
  fi
done < <(find . -type f \
  ! -path './.git/*' \
  \( -name '*.md' -o -name '*.zip' -o -name '*.png' \) \
  -print0)

# 2. README must be substantive
if [ ! -s README.md ] || [ "$(wc -c < README.md)" -lt 200 ]; then
  fail "README.md is missing or too small"
fi

# 3. All 19 Project Index documents must exist and be substantive
required_docs=(
  "Project_Index_Alalā.md"
  "Alalā_Physics_Corrected_Foundation.md"
  "Alalā_Core_Invariant_Specification_HCA.md"
  "OSLab_Program_Board.md"
  "OSLab_Execution_Playbook.md"
  "IPJ_Measurement_Protocol_Alalā.md"
  "Phase0_AI_Coder_Task_List.md"
  "Phase0_Microbenchmark_Suite_Plan.md"
  "Phase0_Week1_2_Task_Breakdown.md"
  "Risk_Register.md"
  "Revised_Phase0_2_Systems_Plan_Alalā.md"
  "Hierarchical_Memory_Architecture_Alalā.md"
  "Memory_Access_Pattern_Guidelines_Alalā.md"
  "Compiler_Passes_Skeleton_Alalā.md"
  "Alalā_Improvement_Playbook.md"
  "Alalā_Experimentation_Framework.md"
  "Meta_Controller_Skeleton_Alalā.md"
  "How_to_Run_First_Micro_Benchmark_on_M4_Alalā.md"
  "AI_Coder_Rules_Guidelines_Alalā.md"
)

for doc in "${required_docs[@]}"; do
  path="docs/$doc"
  if [ ! -f "$path" ]; then
    fail "Missing required doc: $path"
  elif [ "$(wc -c < "$path")" -lt 200 ]; then
    fail "Doc too small (<200 bytes): $path"
  fi
done

# 4. Staging/meta stub files must not exist
for stale in docs/filename.md docs/OtherKeyFiles.md; do
  if [ -f "$stale" ]; then
    fail "Remove stale meta file: $stale"
  fi
done

# 5. Broken placeholder archives must not exist
if [ -f alala_docs.zip ]; then
  if ! file alala_docs.zip | grep -q 'Zip archive'; then
    fail "alala_docs.zip is not a valid zip archive (remove or replace with real zip)"
  fi
fi

# 6. Broken placeholder images must not exist
if [ -f assets/alala_logo.png ]; then
  if ! file assets/alala_logo.png | grep -q 'PNG'; then
    fail "assets/alala_logo.png is not a valid PNG (remove or replace with real image)"
  fi
fi

# 7. LICENSE must exist
if [ ! -f LICENSE ]; then
  fail "LICENSE file missing"
fi

# 8. Optional: warn if Phase 0 harness not yet present
if [ ! -f harness/m4_energy_harness.py ]; then
  warn "harness/m4_energy_harness.py not yet implemented (Phase 0 pending)"
fi

echo "=== Summary ==="
if [ "$errors" -gt 0 ]; then
  echo "FAILED: $errors error(s)"
  exit 1
fi

echo "PASSED: documentation and repository structure checks OK"
exit 0
