#!/bin/bash
# VNNCOMP prepare_instance.sh for vibecheck. UNTIMED, runs once before
# the timed run.  args: <version> <category> <onnx> <vnnlib>
# Does: narrow stale-proc cleanup + GPU sanity. (No pkl pre-parse: vc2's
# net-cache measured SLOWER than conversion on the pickled tensor caches, so
# the timed run parses directly; the graph load is a few seconds even on vgg.)
set -u
VERSION_STRING=v1
[ "$1" != "$VERSION_STRING" ] && { echo "bad version '$1'"; exit 1; }
CATEGORY=$2; ONNX_FILE=$3; VNNLIB_FILE=$4
TOOL_DIR=$(dirname "$(dirname "$(realpath "$0")")")
PY="${VNNCOMP_PYTHON_PATH:-$TOOL_DIR/.venv/bin}/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH="${VC2_SRC:-$TOOL_DIR/src}"
echo "[vc2:prepare] BEGIN category=$CATEGORY onnx=$ONNX_FILE vnnlib=$VNNLIB_FILE"

# narrow stale-proc kill (match ONLY vc2's verifier cmdline; never a broad kill)
stale() { pgrep -f 'vibecheck\.verify' >/dev/null 2>&1; }
pkill -f 'vibecheck\.verify' 2>/dev/null || true
w=0; while stale && [ $w -lt 20 ]; do pkill -9 -f 'vibecheck\.verify' 2>/dev/null || true; sleep 1; w=$((w+1)); done

# GPU sanity: a missing/blind GPU means every timed run falls to CPU and times
# out, so make it LOUD (non-fatal: the run still gets a chance).
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
	echo "## GPU ALARM: nvidia-smi missing/failing -- runs will time out on CPU"
elif ! "$PY" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
	echo "## GPU ALARM: torch.cuda.is_available() False (torch/CUDA mismatch)"
fi
echo "[vc2:prepare] END status=ok category=$CATEGORY"
exit 0
