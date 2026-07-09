#!/bin/bash
# VNNCOMP run_instance.sh for vibecheck2 (vc2).
#   args: <version> <category> <onnx> <vnnlib> <results_file> <timeout>
# Writes the authoritative verdict to <results_file> (line 1: unsat/sat/
# unknown/timeout; for sat the counterexample s-expression follows).
set -u
VERSION_STRING=vc2
[ "$1" != "$VERSION_STRING" ] && { echo "bad version '$1'"; exit 1; }
CATEGORY=$2; ONNX_FILE=$3; VNNLIB_FILE=$4; RESULTS_FILE=$5; TIMEOUT=$6
TOOL_DIR=$(dirname "$(dirname "$(realpath "$0")")")
PY="${VNNCOMP_PYTHON_PATH:-$TOOL_DIR/.venv/bin}/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH="${VC2_SRC:-$TOOL_DIR/src}"
rm -f "$RESULTS_FILE"
echo "[vc2:run] BEGIN category=$CATEGORY timeout=${TIMEOUT}s onnx=$ONNX_FILE"
T0=$(date +%s.%N)
NOATK=""; [ "${VC2_NO_ATTACK:-0}" = "1" ] && NOATK="--no-attack"
"$PY" -m vibecheck2.verify \
	--net "$ONNX_FILE" --spec "$VNNLIB_FILE" \
	--timeout "$TIMEOUT" --device "${VC2_DEVICE:-cuda}" \
	$NOATK --results-file "$RESULTS_FILE"
RC=$?
EL=$(awk "BEGIN{printf \"%.2f\", $(date +%s.%N) - $T0}")
V=$(head -n1 "$RESULTS_FILE" 2>/dev/null | tr -d '[:space:]'); [ -z "$V" ] && V=unknown
echo "[vc2:run] END verdict=$V elapsed=${EL}s rc=$RC category=$CATEGORY"
exit 0
