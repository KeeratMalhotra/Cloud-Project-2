#!/usr/bin/env bash
# Train the EdgePlus baseline policy for every load and seed (9 runs), several at a time.
# Safe to re-run: finished runs are skipped and interrupted runs resume from their last checkpoint.
#
#   bash scripts/train_all.sh            # uses all CPU cores
#   JOBS=4 bash scripts/train_all.sh     # limit parallel runs (each run uses about 1 GB RAM and 1 core)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
JOBS=${JOBS:-$(nproc 2>/dev/null || echo 4)}
DATASET=${DATASET:-brasov}
LOADS=${LOADS:-"150000 100000 50000"}
SEEDS=${SEEDS:-"0 1 2"}
mkdir -p runs
for load in $LOADS; do
  for seed in $SEEDS; do
    echo "$load $seed"
  done
done | xargs -P "$JOBS" -n 2 sh -c \
  "$PY -m edgeplus.train --dataset $DATASET --load \$0 --seed \$1 >> runs/${DATASET}_\$((\$0 / 1000))k_s\$1.out 2>&1; echo finished load=\$0 seed=\$1"
echo "all runs finished"
