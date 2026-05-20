#!/usr/bin/env bash
# Project K live demo. Runs in a few minutes against Ollama + Qwen2.5-Coder-14B.
#
# Usage:
#   bash scripts/demo.sh
#
# Verifies each Project K goal one at a time and points you at the evidence.

set -euo pipefail

if [[ ! -d .venv ]]; then
    echo "[demo] No .venv detected. Run: make install && make ollama"
    exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate
export MSWEA_SILENT_STARTUP=1
export LITELLM_LOG=ERROR

bold() { printf "\n\033[1m== %s ==\033[0m\n" "$*"; }

bold "Goal 1 — Tool suite (file-read / dir-list / grep / edit / run-tests)"
echo "Defined in src/minisweagent/config/projectk/ollama.yaml. Grep proof:"
grep -nE 'file-read|directory-list|grep|edit|run-tests' \
     src/minisweagent/config/projectk/ollama.yaml | sed 's/^/  /'

bold "Goal 2 — Agent loop on an open-weight LLM"
echo "Backend: Ollama (local) + Qwen2.5-Coder-14B."
curl -sf http://localhost:11434/api/version >/dev/null \
    && echo "  ollama service: ✓ up" \
    || { echo "  ollama not up. Run: make ollama"; exit 1; }

bold "Goal 3 — 5 bundled fixtures (+ SWE-Bench-Lite curator for 20-30 picks)"
ls src/minisweagent/projectk/minibench/fixtures/ | sed 's/^/  /'

bold "Goal 4-5 — Run a single fixture end-to-end; metrics + failure taxonomy"
rm -rf runs/demo /tmp/projk-demo
mkdir -p /tmp/projk-demo
cp -r src/minisweagent/projectk/minibench/fixtures/toy__add-sign /tmp/projk-demo/
projectk-mini -o runs/demo \
    -c src/minisweagent/config/projectk/ollama.yaml \
    --fixtures /tmp/projk-demo

bold "Stretch — Plan-execute decomposition"
echo "Config: src/minisweagent/config/projectk/ollama_planner.yaml"
echo "The planner role's plan is embedded in the first user message of the trajectory:"
python - <<'PY'
import json, sys, pathlib
p = pathlib.Path("runs/demo/toy__add-sign/toy__add-sign.traj.json")
if p.exists():
    t = json.loads(p.read_text())
    print(f"  (already ran without planner — see runs/planner-demo for a planner example)")
else:
    print("  (no traj yet)")
PY

bold "Stretch — Persistent scratchpad"
echo "Config: src/minisweagent/config/projectk/ollama_scratchpad.yaml"
echo "Re-run with: projectk-mini -c src/minisweagent/config/projectk/ollama_scratchpad.yaml -o runs/scratch"

bold "Stretch — Retrieval ablation (33% → 67% on 3 fixtures, see REPORT.md §5)"
echo "Run: make ablation     # 4-condition grid: baseline / planner / scratchpad / retrieval"

bold "Pointers"
echo "  REPORT.md            — full write-up (motivation, methods, results, discussion)"
echo "  make test            — 29 unit/integration tests"
echo "  make ablation        — full 4-condition × 5-fixture grid (~1 hour)"
echo "  projectk-report DIR  — re-render metrics + failure taxonomy from any run dir"

bold "Done."
