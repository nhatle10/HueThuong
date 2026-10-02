#!/usr/bin/env bash
# Train many configs x seeds IN PARALLEL and print a val-EER summary table.
#
#   scripts/sweep.sh "configs/exp06_*.yaml"                  # seeds 1 2 3, 6 jobs at once
#   scripts/sweep.sh "configs/exp06_*.yaml" -s "1 2 3 4" -j 8
#   scripts/sweep.sh "configs/exp05_*.yaml configs/exp01_paeff.yaml"
#
# With uv:  uv run scripts/sweep.sh "configs/exp06_*.yaml"
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_common.sh

[ $# -ge 1 ] || { sed -n '2,9p' "$0"; exit 1; }
PATTERN="$1"; shift
SEEDS="1 2 3"; JOBS=6
while getopts "s:j:" o; do
  case $o in s) SEEDS="$OPTARG";; j) JOBS="$OPTARG";; *) exit 1;; esac
done

# shellcheck disable=SC2206
CFGS=($PATTERN)
[ -e "${CFGS[0]}" ] || { echo "no configs match: $PATTERN" >&2; exit 1; }

for c in "${CFGS[@]}"; do for s in $SEEDS; do echo "$c $s"; done; done |
  xargs -P "$JOBS" -L1 bash -c 'train_one "$0" "$1"'

echo; echo "Summary (best val EER, seeds: $SEEDS)"
$PYTHON - "$SEEDS" "${CFGS[@]}" <<'PY'
import json, os, re, sys
seeds, cfgs = sys.argv[1].split(), sys.argv[2:]
def run_name(p):
    return re.search(r'^\s+name:\s*"?([^"\s]+)"?', open(p).read(), re.M).group(1)
rows = []
for c in cfgs:
    n = run_name(c)
    v = []
    for s in seeds:
        f = f"output/{n}/s{s}/meta.json"
        if os.path.exists(f):
            v.append(json.load(open(f))["best_val_eer"] * 100)
    if v:
        rows.append((sum(v)/len(v), n, v))
print(f"{'run':<34}{'mean':>8}   per-seed")
for m, n, v in sorted(rows):
    print(f"{n:<34}{m:>7.2f}%   {[round(x, 2) for x in v]}")
PY
