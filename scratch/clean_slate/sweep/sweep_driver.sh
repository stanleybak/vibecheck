#!/bin/bash
# Durable vc2 sweep driver. Runs a BATCH manifest to completion, resumable
# and idempotent: skips any instance already in results.csv, so re-launching
# after a disconnect / kill / fix just continues. Survives ssh drop (launched
# via setsid; only the per-instance gpulock run is foreground).
#
#   sweep_driver.sh <batch_manifest> <bench_root> <results_csv> <ce_dir> <scripts_dir>
# manifest lines: category,version,onnx_rel,vnnlib_rel,timeout
# results.csv rows: category,onnx_rel,vnnlib_rel,version,verdict,time
set -u
MANIFEST=$1; BENCH=$2; RESULTS=$3; CEDIR=$4; SCRIPTS=$5
GPULOCK="${GPULOCK:-/home/ubuntu/gpulock}"
# the box venv is ~/vibe, NOT $TOOL_DIR/.venv -- run_instance.sh resolves
# PY from VNNCOMP_PYTHON_PATH, so export it (a missing python was a silent
# 0.3s 'error' on every instance until this was set).
export VNNCOMP_PYTHON_PATH="${VNNCOMP_PYTHON_PATH:-/home/ubuntu/vibe/bin}"
export VC2_SRC="${VC2_SRC:-/home/ubuntu/vc2/src}"
PREP="$SCRIPTS/prepare_instance.sh"; RUN="$SCRIPTS/run_instance.sh"
mkdir -p "$(dirname "$RESULTS")" "$CEDIR"
touch "$RESULTS"
PROG="$(dirname "$RESULTS")/progress.txt"

done_key() { grep -Fq ",$1,$2," "$RESULTS"; }   # (onnx_rel,vnnlib_rel) present?

n=0; total=$(grep -c . "$MANIFEST")
while IFS=, read -r cat ver onnx vnnlib to; do
	[ -z "$cat" ] && continue
	n=$((n+1))
	if done_key "$onnx" "$vnnlib"; then continue; fi
	VN="$BENCH/$vnnlib"
	case "$onnx" in
	PAIR\|*)   # network-pair: onnx = PAIR|<f_rel>|<g_rel>
		F=$(echo "$onnx" | cut -d'|' -f2); G=$(echo "$onnx" | cut -d'|' -f3)
		ON="[('f', '$BENCH/$F'), ('g', '$BENCH/$G')]"
		# files may be staged as .onnx.gz (vc2 resolves the .gz sibling from
		# the plain path); accept either form in the existence check.
		ex() { [ -f "$1" ] || [ -f "$1.gz" ]; }
		ex "$BENCH/$F" && ex "$BENCH/$G" && ex "$VN" || {
			echo "$cat,$onnx,$vnnlib,$ver,missing_file,0" >> "$RESULTS"; continue; } ;;
	*)
		ON="$BENCH/$onnx"
		[ -f "$ON" ] && [ -f "$VN" ] || {
			echo "$cat,$onnx,$vnnlib,$ver,missing_file,0" >> "$RESULTS"; continue; } ;;
	esac
	res=$(mktemp)
	sudo rm -f /tmp/idle_since 2>/dev/null || true
	T0=$(date +%s.%N)
	# prepare (untimed) then run (timed) -- faithful to VNNCOMP; gpulock
	# serializes the timed run on the shared GPU.
	VC2_SRC="${VC2_SRC:-/home/ubuntu/vc2/src}" bash "$PREP" vc2 "$cat" "$ON" "$VN" >/dev/null 2>&1 || true
	if [ -n "$GPULOCK" ] && [ -x "$GPULOCK" ]; then
		VC2_SRC="${VC2_SRC:-/home/ubuntu/vc2/src}" "$GPULOCK" run "vc2-sweep: $cat ${onnx##*/}" -- \
			bash "$RUN" vc2 "$cat" "$ON" "$VN" "$res" "$to" >/dev/null 2>&1 || true
	else
		VC2_SRC="${VC2_SRC:-$PWD/src}" bash "$RUN" vc2 "$cat" "$ON" "$VN" "$res" "$to" >/dev/null 2>&1 || true
	fi
	EL=$(awk "BEGIN{printf \"%.2f\", $(date +%s.%N) - $T0}")
	V=$(head -n1 "$res" 2>/dev/null | tr -d '[:space:]')
	# empty result = the python process died without writing a verdict
	# (OS OOM-kill / SIGKILL on a huge net -- no Python handler runs). That
	# is an honest 'unknown' (vc2 could not complete in memory/time), not a
	# tool 'error'. A genuine Python crash writes 'error\n<msg>' itself.
	if [ -z "$V" ]; then V=unknown; echo "  (empty result -> unknown; likely OOM-kill) $onnx" >> "$PROG"; fi
	echo "$cat,$onnx,$vnnlib,$ver,$V,$EL" >> "$RESULTS"
	# capture a counterexample (sat rows carry the s-expr after line 1)
	if [ "$V" = "sat" ] && [ "$(wc -l < "$res")" -gt 1 ]; then
		h=$(echo "$onnx$vnnlib" | md5sum | cut -c1-16)
		tail -n +2 "$res" > "$CEDIR/$h.counterexample"
	fi
	rm -f "$res"
	echo "$(date +%H:%M:%S) [$n/$total] $cat ${onnx##*/} / ${vnnlib##*/} -> $V (${EL}s)" >> "$PROG"
done < "$MANIFEST"
echo "BATCH_DONE $(date +%H:%M:%S)" >> "$PROG"
