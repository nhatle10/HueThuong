# Shared helpers, sourced by ensemble.sh and sweep.sh

# Pick the Python interpreter once (override with PYTHON=...): project venv, then PATH, then uv.
if [ -z "${PYTHON:-}" ]; then
  if   [ -x .venv/bin/python ];      then PYTHON=".venv/bin/python"
  elif command -v python  >/dev/null; then PYTHON="python"
  elif command -v python3 >/dev/null; then PYTHON="python3"
  else PYTHON="uv run python"; fi
fi
export PYTHON
echo "[python] $PYTHON"
run_name() { grep -m1 -E '^\s+name:' "$1" | sed -E 's/.*name:\s*"?([^" ]+)"?.*/\1/'; }

# train_one <config> <seed>: skips finished seeds, logs to logs/<run>/s<seed>.log
train_one() {
  local cfg="$1" seed="$2" name
  name=$(run_name "$cfg")
  if [ -f "output/$name/s$seed/meta.json" ]; then
    echo "[skip]  $name s$seed (already finished)"; return 0
  fi
  mkdir -p "logs/$name"
  echo "[start] $name s$seed -> logs/$name/s$seed.log"
  $PYTHON main.py --config "$cfg" --seed "$seed" > "logs/$name/s$seed.log" 2>&1 \
    && echo "[done]  $name s$seed" || echo "[FAIL]  $name s$seed (see log)"
}
export -f run_name train_one

# one thread per process so parallel jobs don't fight over CPU cores
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
