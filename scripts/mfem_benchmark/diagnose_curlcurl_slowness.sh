#!/usr/bin/env bash
#
# Short diagnostic for the slow Krylov iterations of curlcurl with legacy assembly on
# HIP: about 0.5 s per GMRES + Jacobi iteration at 1752 true DoFs (order 1,
# parallel_refine=0, 8 ranks), against about 0.5 ms per CG + Jacobi iteration for
# diffusion at a similar size.
#
# Every run is capped at MAX_ITS iterations and the summary reports the solve time per
# iteration. Each run differs from the reference (curlcurl, GMRES + Jacobi, order 1,
# parallel_refine=0, legacy, DEVICE, RANKS ranks) in one respect, to tell where the
# time goes:
#   cg         CG instead of GMRES             -> is it GMRES (about 25 global dot products
#                                                 per iteration, against 2 for CG)?
#   one_rank   a single MPI rank               -> is it communication between ranks?
#   cpu        device=cpu                      -> is it the GPU path?
#   diffusion  diffusion instead of curlcurl   -> is it the curl-curl operator?
#   order3     order 3 (fast in results3)      -> does the order dependence reproduce?
# The reference runs first and last, to show whether the cost drifts during the job.
#
# Usage:
#   ./diagnose_curlcurl_slowness.sh [results_dir]
#
# Environment overrides:
#   MOOSE_ROOT, MOOSE_EXEC, MPI_LAUNCHER, GPU_BIND, TIMEOUT - as in
#                run_gpu_aware_mpi_benchmark.sh (TIMEOUT default here: 600)
#   DEVICE     - Executioner/device of every run but "cpu" (default: hip)
#   RANKS      - MPI rank count of every run but "one_rank" (default: 8)
#   MAX_ITS    - Krylov iteration cap (default: 150, three GMRES restart cycles)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOOSE_ROOT="${MOOSE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
MOOSE_EXEC="${MOOSE_EXEC:-$MOOSE_ROOT/test/moose_test-opt}"
TEST_DIR="$MOOSE_ROOT/test"
RESULTS_DIR="${1:-$SCRIPT_DIR/curlcurl_diagnostic}"
MPI_LAUNCHER="${MPI_LAUNCHER:-srun -n}"
GPU_BIND="${GPU_BIND:-0}"
TIMEOUT="${TIMEOUT:-600}"
DEVICE="${DEVICE:-hip}"
RANKS="${RANKS:-8}"
MAX_ITS="${MAX_ITS:-150}"

read -ra LAUNCHER <<< "$MPI_LAUNCHER"
EXEC=("$MOOSE_EXEC")
if [[ "$GPU_BIND" == 1 ]]; then
  EXEC=("$SCRIPT_DIR/gpu_bind.sh" "$MOOSE_EXEC")
fi

if [[ ! -x "$MOOSE_EXEC" ]]; then
  echo "ERROR: $MOOSE_EXEC not executable. Set MOOSE_EXEC to the right binary." >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR"
cd "$TEST_DIR"

# Solver overrides. The curlcurl input already sets l_tol=1e-12 on its gmres and cg
# entries; diffusion's is set to match.
CURLCURL_GMRES=("Solvers/active=gmres jacobi"
                Solvers/jacobi/type=MFEMOperatorJacobiSmoother
                Solvers/gmres/type=MFEMGMRESSolver Solvers/gmres/preconditioner=jacobi
                Solvers/gmres/l_max_its="$MAX_ITS")
CURLCURL_CG=("Solvers/active=cg jacobi"
             Solvers/jacobi/type=MFEMOperatorJacobiSmoother
             Solvers/cg/preconditioner=jacobi Solvers/cg/l_max_its="$MAX_ITS")
DIFFUSION_GMRES=(Solvers/inactive=boomeramg
                 Solvers/main/type=MFEMGMRESSolver Solvers/main/preconditioner=jacobi
                 Solvers/main/l_tol=1e-12 Solvers/main/l_max_its="$MAX_ITS")
CURLCURL_P1=(FESpaces/HCurlFESpace/fec_order=FIRST FESpaces/HDivFESpace/fec_order=CONSTANT)
CURLCURL_P3=(FESpaces/HCurlFESpace/fec_order=THIRD FESpaces/HDivFESpace/fec_order=SECOND)
DIFFUSION_P1=(FESpaces/H1FESpace/fec_order=FIRST FESpaces/HCurlFESpace/fec_order=FIRST)

RUN_NAMES=()
# run <name> <input> <device> <ranks> <overrides...>
run() {
  local name="$1" inp="$2" device="$3" ranks="$4"
  shift 4
  local log="$RESULTS_DIR/$name.log"
  RUN_NAMES+=("$name")
  echo ">>> $name  (device=$device  ranks=$ranks)"
  # --kill-after sends SIGKILL if the launcher ignores the initial SIGTERM.
  timeout --kill-after=30s "$TIMEOUT" \
      "${LAUNCHER[@]}" "$ranks" "${EXEC[@]}" -i "$inp" \
      Mesh/parallel_refine=0 \
      Executioner/device="$device" \
      Executioner/assembly_level=legacy \
      "$@" \
      Outputs/perf_graph=true \
      > "$log" 2>&1
  local status=$?
  if (( status != 0 )); then
    echo "    FAILED (exit status $status) - see $log"
  fi
}

CURLCURL=tests/mfem/kernels/curlcurl.i
DIFFUSION=tests/mfem/kernels/diffusion.i
run reference "$CURLCURL" "$DEVICE" "$RANKS" "${CURLCURL_P1[@]}" "${CURLCURL_GMRES[@]}"
run cg "$CURLCURL" "$DEVICE" "$RANKS" "${CURLCURL_P1[@]}" "${CURLCURL_CG[@]}"
run one_rank "$CURLCURL" "$DEVICE" 1 "${CURLCURL_P1[@]}" "${CURLCURL_GMRES[@]}"
run cpu "$CURLCURL" cpu "$RANKS" "${CURLCURL_P1[@]}" "${CURLCURL_GMRES[@]}"
run diffusion "$DIFFUSION" "$DEVICE" "$RANKS" "${DIFFUSION_P1[@]}" "${DIFFUSION_GMRES[@]}"
run order3 "$CURLCURL" "$DEVICE" "$RANKS" "${CURLCURL_P3[@]}" "${CURLCURL_GMRES[@]}"
run reference_again "$CURLCURL" "$DEVICE" "$RANKS" "${CURLCURL_P1[@]}" "${CURLCURL_GMRES[@]}"

# The logs contain NUL bytes, which make grep treat them as binary, so strip them.
echo
printf '%-16s %10s %6s %10s %8s\n' run "true DoFs" its "solve [s]" "ms/it"
for name in "${RUN_NAMES[@]}"; do
  text=$(tr -d '\000' < "$RESULTS_DIR/$name.log")
  dofs=$(grep -a -m1 -o "total_true_dofs=[0-9]*" <<< "$text" | cut -d= -f2)
  its=$(grep -a -o "Number of iterations: *[0-9]*" <<< "$text" | tail -1 | grep -o "[0-9]*$")
  # Column 8 of the perf graph row is the section's total time.
  mult=$(grep -a -m1 "SolveWithOperator::Mult" <<< "$text" | awk -F'|' '{print $8 + 0}')
  if [[ -n "$its" && -n "$mult" ]] && (( its > 0 )); then
    printf '%-16s %10s %6s %10.3f %8.2f\n' "$name" "$dofs" "$its" "$mult" \
        "$(awk -v t="$mult" -v n="$its" 'BEGIN {print 1000 * t / n}')"
  else
    printf '%-16s %10s %6s %10s %8s\n' "$name" "${dofs:--}" - - -
  fi
done
