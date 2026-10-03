#!/usr/bin/env bash
# Regenerate every result in out/ from the public tau2-bench trajectories, then verify.
# Usage: ./reproduce.sh [path-to-tau2-bench-checkout]
set -euo pipefail
cd "$(dirname "$0")"
TAU2=${1:-tau2-bench}
COMMIT=5bfa7e37b36656b37dc6d022156be6563c1007f3
if [ ! -d "$TAU2" ]; then
  git clone https://github.com/sierra-research/tau2-bench.git "$TAU2"
  git -C "$TAU2" checkout "$COMMIT"
fi
D="$TAU2/data/tau2/results/final"
A="$D/claude-3-7-sonnet-20250219_telecom_default_gpt-4.1-2025-04-14_4trials.json"
B="$D/gpt-4.1-2025-04-14_telecom-workflow_default_gpt-4.1-2025-04-14_4trials.json"
OUT=regen; mkdir -p "$OUT"
python3 runlogic/exp1_rederive.py "$A" "$OUT/exp1.json"      # rule checker, support stand-in, gate replay
python3 runlogic/exp1_rederive.py "$B" "$OUT/exp1_B.json"
python3 runlogic/exp2_fragments.py "$OUT/exp1.json" "$OUT/exp2.json"
OUT="$OUT" python3 monitor.py "$A"                           # outcome-trained prefix monitor
OUT="$OUT" python3 runlogic/exp3_relabel.py "$A" "$OUT/exp1.json" "$OUT/exp3.json"
OUT="$OUT" python3 runlogic/exp3_relabel.py "$B" "$OUT/exp1_B.json" "$OUT/exp3_B.json"
python3 runlogic/exp3b_folds.py "$A" "$OUT/exp1.json" "$OUT/exp3b.json"
python3 runlogic/exp4_change.py "$OUT/exp1.json" "$OUT/exp1_B.json" "$OUT/exp4.json"
python3 runlogic/exp5_compose.py "$OUT/exp1.json" "$OUT/monitor.json" "$OUT/exp5.json"
python3 verify_counts.py "$OUT"
