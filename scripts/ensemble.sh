#!/usr/bin/env bash
# Train N seeds of a config, then build the averaged-score submission.
# Usage: scripts/ensemble.sh <config.yaml> [seeds...]     (default seeds: 1 2 3 4)
set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -lt 1 ]; then
  echo "Usage: $0 <config.yaml> [seeds...]" >&2
  exit 1
fi

CFG="$1"; shift
if [ $# -gt 0 ]; then SEEDS=("$@"); else SEEDS=(1 2 3 4); fi

# run name comes from the config's first `name:` line (run.name)
NAME=$(grep -m1 -E '^\s+name:' "$CFG" | sed -E 's/.*name:\s*"?([^" ]+)"?.*/\1/')
RUN="output/$NAME"

for s in "${SEEDS[@]}"; do
  if [ -f "$RUN/s$s/meta.json" ]; then
    echo ">> seed $s already finished, skipping"
  else
    uv run python main.py --config "$CFG" --seed "$s"
  fi
done

uv run python evaluate.py --config "$CFG" --run "$RUN" --seeds "${SEEDS[@]}"
