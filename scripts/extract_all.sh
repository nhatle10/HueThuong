#!/usr/bin/env bash
# Extract all extra backbone features, one encoder after another (safe to leave running in tmux).
#
#   scripts/extract_all.sh                         # all encoders
#   scripts/extract_all.sh ecapa2 clip_l14         # only these
#
# - finished encoders are skipped (marker: .../train_set/features_<enc>/.done), so re-running resumes
# - a failing encoder is logged and the next one still runs
# - logs: logs/extract_<enc>.log ; summary printed at the end
cd "$(dirname "$0")/.."

PY=".venv/bin/python"
ROOT="data/FLAG_Grand_Challenge_2027_15_09_2027/train_set_extracted/train_set"
if [ $# -gt 0 ]; then ENCODERS=("$@"); else ENCODERS=(agegender_vit clip_l14 clip_h14 siglip2 ecapa2 resnet293 wavlm_large); fi
mkdir -p logs

declare -A STATUS
for enc in "${ENCODERS[@]}"; do
  if [ -f "$ROOT/features_$enc/.done" ]; then
    echo "[skip]  $enc (already done)"; STATUS[$enc]="skipped (done earlier)"; continue
  fi
  echo "[start] $enc  $(date '+%H:%M:%S')  -> logs/extract_$enc.log"
  start=$(date +%s)
  if $PY feature_extraction/extract_features.py --encoder "$enc" > "logs/extract_$enc.log" 2>&1; then
    STATUS[$enc]="OK in $(( ($(date +%s) - start) / 60 )) min"
  else
    STATUS[$enc]="FAILED after $(( ($(date +%s) - start) / 60 )) min (see logs/extract_$enc.log)"
  fi
  echo "[end]   $enc  ${STATUS[$enc]}"
done

echo; echo "=== summary  $(date '+%Y-%m-%d %H:%M:%S')"
for enc in "${ENCODERS[@]}"; do printf '  %-14s %s\n' "$enc" "${STATUS[$enc]}"; done
grep -h "FAILED " logs/extract_*.log 2>/dev/null | head -20
