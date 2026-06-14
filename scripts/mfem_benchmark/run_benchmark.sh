#!/usr/bin/env bash
#
# Run the four MFEM example inputs at serial-refinement levels 0..5, capture
# the perf_graph output, then call plot_mfem_timings.py to produce log-log
# plots of timing vs. number of true DoFs for each example.
#
# Usage:
#   ./run_benchmark.sh [results_dir]
#
# Environment overrides:
#   MOOSE_ROOT     — root of the moose checkout (default: parent of scripts/)
#   MOOSE_EXEC     — path to moose_test executable (default: $MOOSE_ROOT/test/moose_test-opt)
#   REFINEMENTS    — space-separated list of serial_refine values (default: "0 1 2 3 4 5")
#   DEVICES        — space-separated list of Executioner/device values
#                    (default: "cpu ceed-cpu hip ceed-hip")
#   ASSEMBLY_LEVELS — space-separated list of Executioner/assembly_level values
#                    (default: "legacy full partial none")
#   MAX_ITS        — Krylov iteration cap (Solver/l_max_its) applied to every
#                    iterative-solver run (default: 100000)
#   ORDERS         — space-separated list of polynomial orders (default: "1 2 3 4")
#   RANKS          — space-separated list of MPI rank counts to run (default: "1")
#   MPI_LAUNCHER   — launcher prefix to which the rank count is appended
#                    (default: "srun -n"; use e.g. "mpirun -np" off SLURM)
#
# After completion, results_dir contains:
#   <example>_<device>_<assembly>_p<P>_n<R>_r<N>.log — raw stdout of each run
#   <example>_timings.png         — log-log timing plot per example
#                                    (one curve per section × device × assembly)

set -uo pipefail

SCRIPT_DIR=/home/dc-roch1/moose_hip/moose/scripts/mfem_benchmark
MOOSE_ROOT=/home/dc-roch1/moose_hip
MOOSE_EXEC="${MOOSE_EXEC:-$MOOSE_ROOT/test/moose_test-opt}"
TEST_DIR="$MOOSE_ROOT/test"
RESULTS_DIR="${1:-$MOOSE_ROOT/mfem_benchmark_results}"
REFINEMENTS="${REFINEMENTS:-0 1 2 3 4 5}"
DEVICES="${DEVICES:-cpu ceed-cpu hip ceed-hip}"
ASSEMBLY_LEVELS="${ASSEMBLY_LEVELS:-legacy full partial none}"
MAX_ITS="${MAX_ITS:-100000}"
ORDERS="${ORDERS:-1 2 3 4}"
RANKS="${RANKS:-1}"
MPI_LAUNCHER="${MPI_LAUNCHER:-srun -n}"

# Split the launcher prefix into words so e.g. "srun -n" stays two tokens; the
# rank count and executable are appended per run below.
read -ra LAUNCHER <<< "$MPI_LAUNCHER"

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

# Per-example solver overrides so the SAME solver/preconditioner pair runs at
# EVERY assembly level (apples-to-apples). plain OperatorJacobiSmoother is the
# only preconditioner usable matrix-free: the LOR-wrapped Hypre solvers
# (AMS/ADS/BoomerAMG) reject the matrix-free operator the driver hands them, and
# the Hypre Krylov solvers require an assembled HypreParMatrix. So every example
# uses native CG/GMRES + jacobi. Stored as a bash array (set by solver_args) so
# values containing spaces (BCs/active) stay a single argument.
#   - heattransfer: convective BC has no partial-assembly boundary integrator,
#     so it is swapped for a Dirichlet BC (same as the LOR regression test).
#   - complex: needs a direct solver and is multi-variable (non-legacy assembly
#     is single-variable only), so it is left to error out at non-legacy levels.
#   - l_max_its is raised to MAX_ITS everywhere: the input-file caps (1000 for
#     heattransfer, the 10000 default elsewhere) were hit without convergence at
#     high refinement levels, making those timing points iteration-capped.
solver_args() {
  case "$1" in
    diffusion)
      SOLVER_ARGS=(Solver/type=MFEMCGSolver Solver/l_max_its="$MAX_ITS"
                   Preconditioner/active=jacobi Solver/preconditioner=jacobi)
      ;;
    heattransfer)
      SOLVER_ARGS=("BCs/active=bottom top_dirichlet"
                   Solver/type=MFEMCGSolver Solver/l_max_its="$MAX_ITS"
                   Preconditioner/active=jacobi Solver/preconditioner=jacobi)
      ;;
    curlcurl)
      SOLVER_ARGS=(Preconditioner/active=jacobi
                   Preconditioner/jacobi/type=MFEMOperatorJacobiSmoother
                   Solver/type=MFEMGMRESSolver Solver/preconditioner=jacobi
                   Solver/l_max_its="$MAX_ITS")
      ;;
    *)
      SOLVER_ARGS=()  # complex: no override, expected to error at non-legacy
      ;;
  esac
}

# Map a numeric polynomial order to the fec_order overrides for each example's
# FESpaces. Auxiliary spaces are scaled with the solve space: diffusion's ND
# space matches the H1 order, and curlcurl's RT space (holding curl of an ND(p)
# field) is one order lower (CONSTANT for p=1, etc.).
ORDER_NAMES=(CONSTANT FIRST SECOND THIRD FOURTH)
order_args() {
  local tag="$1" p="$2"
  local fec="${ORDER_NAMES[$p]}" rt_fec="${ORDER_NAMES[$((p - 1))]}"
  case "$tag" in
    diffusion)
      ORDER_ARGS=(FESpaces/H1FESpace/fec_order="$fec"
                  FESpaces/HCurlFESpace/fec_order="$fec")
      ;;
    curlcurl)
      ORDER_ARGS=(FESpaces/HCurlFESpace/fec_order="$fec"
                  FESpaces/HDivFESpace/fec_order="$rt_fec")
      ;;
    heattransfer | complex)
      ORDER_ARGS=(FESpaces/H1FESpace/fec_order="$fec")
      ;;
    *)
      ORDER_ARGS=()
      ;;
  esac
}

cd "$TEST_DIR"

failures=()
for entry in "${EXAMPLES[@]}"; do
  tag="${entry%%:*}"
  inp="${entry#*:}"
  solver_args "$tag"
  for device in $DEVICES; do
    for asm in $ASSEMBLY_LEVELS; do
      for p in $ORDERS; do
        order_args "$tag" "$p"
        for ranks in $RANKS; do
          for r in $REFINEMENTS; do
            log="$RESULTS_DIR/${tag}_${device}_${asm}_p${p}_n${ranks}_r${r}.log"
            echo ">>> $tag  device=$device  assembly=$asm  order=$p  ranks=$ranks  serial_refine=$r"
            if ! "${LAUNCHER[@]}" "$ranks" "$MOOSE_EXEC" -i "$inp" \
                Mesh/serial_refine="$r" \
                Executioner/device="$device" \
                Executioner/assembly_level="$asm" \
                "${ORDER_ARGS[@]}" \
                "${SOLVER_ARGS[@]}" \
                Outputs/perf_graph=true \
                > "$log" 2>&1
            then
              echo "    FAILED — see $log"
              failures+=("$tag device=$device assembly=$asm p=$p ranks=$ranks r=$r")
            fi
          done
        done
      done
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
