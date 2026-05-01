#!/usr/bin/env bash
#
# Run the four MFEM example inputs at serial-refinement levels 0..4, capture
# the perf_graph output, then call plot_mfem_timings.py to produce log-log
# plots of timing vs. number of true DoFs for each example.
#
# Usage:
#   ./run_benchmark.sh [results_dir]
#
# Environment overrides:
#   MOOSE_ROOT     — root of the moose checkout (default: parent of scripts/)
#   MOOSE_EXEC     — path to moose_test executable (default: $MOOSE_ROOT/test/moose_test-dbg)
#   REFINEMENTS    — space-separated list of serial_refine values (default: "0 1 2 3 4")
#   DEVICES        — space-separated list of Executioner/device values
#                    (default: "cpu ceed-cpu hip ceed-hip")
#
# After completion, results_dir contains:
#   <example>_<device>_r<N>.log   — raw stdout of each run
#   <example>_timings.png         — log-log timing plot per example
#                                    (one curve per section × device)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOOSE_ROOT="${MOOSE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
MOOSE_EXEC="${MOOSE_EXEC:-$MOOSE_ROOT/test/moose_test-dbg}"
TEST_DIR="$MOOSE_ROOT/test"
RESULTS_DIR="${1:-$MOOSE_ROOT/mfem_benchmark_results}"
REFINEMENTS="${REFINEMENTS:-0 1 2 3 4}"
DEVICES="${DEVICES:-cpu ceed-cpu hip ceed-hip}"

if [[ ! -x "$MOOSE_EXEC" ]]; then
  echo "ERROR: $MOOSE_EXEC not executable. Set MOOSE_EXEC to the right binary." >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR"

# tag <-> input file (relative to $TEST_DIR)
EXAMPLES=(
  "diffusion:tests/mfem/kernels/diffusion.i"
  "curlcurl:tests/mfem/kernels/curlcurl.i"
  "complex:tests/mfem/complex/complex.i"
  "heattransfer:tests/mfem/kernels/heattransfer.i"
)

cd "$TEST_DIR"

failures=()
for entry in "${EXAMPLES[@]}"; do
  tag="${entry%%:*}"
  inp="${entry#*:}"
  for device in $DEVICES; do
    for r in $REFINEMENTS; do
      log="$RESULTS_DIR/${tag}_${device}_r${r}.log"
      echo ">>> $tag  device=$device  serial_refine=$r"
      if ! "$MOOSE_EXEC" -i "$inp" \
          Mesh/serial_refine="$r" \
          Executioner/device="$device" \
          Outputs/perf_graph=true \
          > "$log" 2>&1
      then
        echo "    FAILED — see $log"
        failures+=("$tag device=$device r=$r")
      fi
    done
  done
done

echo
if (( ${#failures[@]} > 0 )); then
  echo "Some runs failed:"
  printf '  %s\n' "${failures[@]}"
fi

echo
echo "Generating plots..."
python3 "$SCRIPT_DIR/plot_mfem_timings.py" "$RESULTS_DIR"
