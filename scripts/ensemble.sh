#!/usr/bin/env bash
# Train several seeds IN PARALLEL, then build the averaged-score submission zip.
#
#   scripts/ensemble.sh <config.yaml>                    # seeds 1 2 3 4, all at once
#   scripts/ensemble.sh <config.yaml> -s "1 2 3 4 5 6"   # custom seeds
#   scripts/ensemble.sh <config.yaml> -j 2               # at most 2 at a time
#   scripts/ensemble.sh <config.yaml> -n                 # train only, no submission
#
# With uv:  uv run scripts/ensemble.sh <config.yaml>
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_common.sh

[ $# -ge 1 ] || { sed -n '2,10p' "$0"; exit 1; }
CFG="$1"; shift
SEEDS="1 2 3 4"; JOBS=""; EVAL=1
while getopts "s:j:n" o; do
  case $o in s) SEEDS="$OPTARG";; j) JOBS="$OPTARG";; n) EVAL=0;; *) exit 1;; esac
done
read -ra SEED_ARR <<< "$SEEDS"
JOBS="${JOBS:-${#SEED_ARR[@]}}"
NAME=$(run_name "$CFG")

printf '%s\n' "${SEED_ARR[@]}" | xargs -P "$JOBS" -I{} bash -c 'train_one "$0" "$1"' "$CFG" {}

echo; echo "Best val EER per seed:"
$PYTHON - "$NAME" "${SEED_ARR[@]}" <<'PY'
import json, os, sys
name, seeds = sys.argv[1], sys.argv[2:]
v = []
for s in seeds:
    f = f"output/{name}/s{s}/meta.json"
    if os.path.exists(f):
        e = json.load(open(f))["best_val_eer"] * 100
        v.append(e); print(f"  s{s}: {e:.2f}%")
    else:
        print(f"  s{s}: MISSING (check logs/{name}/s{s}.log)")
if v: print(f"  mean: {sum(v)/len(v):.2f}%  (n={len(v)})")
PY

if [ "$EVAL" = 1 ]; then
  $PYTHON evaluate.py --config "$CFG" --run "output/$NAME" --seeds "${SEED_ARR[@]}"
fi
