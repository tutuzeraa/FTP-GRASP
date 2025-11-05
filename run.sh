#!/usr/bin/env bash
set -euo pipefail

INST_DIR="instances"
ROOT=1          # 1-based
ALPHA=0.2
SEED=123
MAX_ITERS=3000
MAX_TIME=1    # seconds or comment out to disable
NO_IMPROV=400

for f in "$INST_DIR"/*.tsp; do
  [[ -f "$f" ]] || continue
  echo ">>> $f"
  python3 grasp_ftp.py "$f" \
    --root "$ROOT" \
    --alpha "$ALPHA" --seed "$SEED" \
    --lambda-central 0.25 --k-central 15 \
    --lambda-congest 0.5 \
    --max-iters "$MAX_ITERS" --max-time "$MAX_TIME" --max-no-improv "$NO_IMPROV" \
    --save-tree --save-csv
done
