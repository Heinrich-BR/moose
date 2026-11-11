#!/usr/bin/env bash
#
# Variant of run_benchmark.sh that runs every example/parameter combination with
# MFEM's GPU-aware MPI enabled, disabled, or (by default) both, as selected by GAMS.
# When both are selected, it then calls plot_gpu_aware_mpi_speedup.py to plot, for
# each timed subroutine (and their sum), the speedup gained by enabling GPU-aware
# MPI vs. the number of true DoFs.
#
# GPU-aware MPI is toggled through MFEM's MFEM_GPU_AWARE_MPI environment variable
# (read in mfem::Device's constructor): exported -> enabled, unset -> disabled.
# NOTE: this requires MOOSE to NOT force the flag on. If MooseApp::setMFEMDevice
# calls `_mfem_device->SetGPUAwareMPI(true)` unconditionally, remove that line (or
# guard it) so the environment variable is respected.
#
# Each series (example, device, assembly level, order, rank count) starts with an
# untimed warmup run at the lowest refinement level; see the comment at the call.
#
# Usage:
#   ./run_gpu_aware_mpi_benchmark.sh [results_dir]
#
# Environment overrides (identical to run_benchmark.sh):
#   MOOSE_ROOT     — root of the moose checkout (default: parent of scripts/)
#   MOOSE_EXEC     — path to moose_test executable (default: $MOOSE_ROOT/test/moose_test-opt)
#   EXAMPLES       — space-separated list of example names to run, any subset of
#                    "diffusion curlcurl complex heattransfer" (default: all)
#   REFINEMENTS    — space-separated list of parallel_refine values (default: "0 1 2 3 4 5")
#   DEVICES        — space-separated list of Executioner/device values
#                    (default: "cpu ceed-cpu hip ceed-hip")
#   ASSEMBLY_LEVELS — space-separated list of Executioner/assembly_level values
#                    (default: "legacy full element partial none")
#   MAX_ITS        — Krylov iteration cap (l_max_its of the main [Solvers] entry) applied to every
#                    iterative-solver run (default: 100000)
#   ORDERS         — space-separated list of polynomial orders (default: "1 2 3 4")
#   RANKS          — space-separated list of MPI rank counts to run (default: "1")
#                    (GPU-aware MPI only matters for >1 rank; use a single value)
#   MPI_LAUNCHER   — launcher prefix to which the rank count is appended
#                    (default: "srun -n"; use e.g. "mpirun -np" off SLURM)
#   GPU_BIND       - 1 to start each rank through gpu_bind.sh, which restricts it
#                    to its own GPU by node-local rank; needed with mpirun, which
#                    does not bind GPUs, so all ranks would share the first GPU
#                    (default: 0)
#   TIMEOUT        — per-run wall-clock limit in seconds; a run exceeding it is
#                    cancelled and all higher refinement levels for the same
#                    parameters are skipped (default: 600)
#
# Environment override specific to this script:
#   GAMS           - space-separated list of GPU-aware MPI settings to run, in the
#                    order given: 1 = on, 0 = off (default: "1 0")
#
# After completion, results_dir contains:
#   <example>_<device>_<assembly>_p<P>_n<R>_r<N>_gam<0|1>.log — raw stdout of each
#                    run (gam1 = GPU-aware MPI on, gam0 = off)
#   <example>_*_gpu_aware_mpi_speedup.png — speedup-vs-DoFs plot per example, if
#                    GAMS selected both settings

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOOSE_ROOT="${MOOSE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
MOOSE_EXEC="${MOOSE_EXEC:-$MOOSE_ROOT/test/moose_test-opt}"
TEST_DIR="$MOOSE_ROOT/test"
RESULTS_DIR="${1:-$SCRIPT_DIR/mfem_gpu_aware_mpi_results}"
EXAMPLES="${EXAMPLES:-diffusion curlcurl complex heattransfer}"
REFINEMENTS="${REFINEMENTS:-0 1 2 3 4 5}"
DEVICES="${DEVICES:-cpu ceed-cpu hip ceed-hip}"
ASSEMBLY_LEVELS="${ASSEMBLY_LEVELS:-legacy full element partial none}"
MAX_ITS="${MAX_ITS:-100000}"
ORDERS="${ORDERS:-1 2 3 4}"
RANKS="${RANKS:-1}"
MPI_LAUNCHER="${MPI_LAUNCHER:-srun -n}"
TIMEOUT="${TIMEOUT:-600}"
GPU_BIND="${GPU_BIND:-0}"
GAMS="${GAMS:-1 0}"

# Refinement levels run in ascending order, so the first is the cheapest to warm up with.
read -r WARMUP_REFINEMENT _ <<< "$REFINEMENTS"

# A malformed entry would otherwise run with a wrong setting under a misleading log
# name: "0,1" evaluates as true in the (( gam )) test below.
for gam in $GAMS; do
  if [[ "$gam" != 0 && "$gam" != 1 ]]; then
    echo "ERROR: GAMS entries must be 0 or 1, got '$gam'." >&2
    exit 1
  fi
done

# Split the launcher prefix into words so e.g. "srun -n" stays two tokens; the
# rank count and executable are appended per run below.
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
# Made absolute because the runs write their logs after changing to $TEST_DIR.
RESULTS_DIR="$(cd "$RESULTS_DIR" && pwd)"

# tag -> input file (relative to $TEST_DIR). The EXAMPLES variable selects which
# of these tags to run.
declare -A EXAMPLE_INPUTS=(
  [diffusion]="tests/mfem/kernels/diffusion.i"
  [curlcurl]="tests/mfem/kernels/curlcurl.i"
  [complex]="tests/mfem/complex/complex.i"
  [heattransfer]="tests/mfem/kernels/heattransfer.i"
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
#   - every active [Solvers] entry that no other entry references becomes a
#     solver driver, so the inputs' default preconditioners are deactivated.
solver_args() {
  case "$1" in
    diffusion)
      SOLVER_ARGS=(Solvers/inactive=boomeramg
                   Solvers/main/type=MFEMCGSolver Solvers/main/l_max_its="$MAX_ITS"
                   Solvers/main/preconditioner=jacobi)
      ;;
    heattransfer)
      SOLVER_ARGS=("BCs/active=bottom top_dirichlet"
                   Solvers/inactive=boomeramg
                   Solvers/main/type=MFEMCGSolver Solvers/main/l_max_its="$MAX_ITS"
                   Solvers/main/preconditioner=jacobi)
      ;;
    curlcurl)
      SOLVER_ARGS=("Solvers/active=gmres jacobi"
                   Solvers/jacobi/type=MFEMOperatorJacobiSmoother
                   Solvers/gmres/type=MFEMGMRESSolver Solvers/gmres/preconditioner=jacobi
                   Solvers/gmres/l_max_its="$MAX_ITS")
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

# Run the case selected by the sweep loop below (inp, device, asm, ranks,
# ORDER_ARGS, SOLVER_ARGS) at refinement level $1, writing its output to $2.
# Returns timeout's exit status, which is 124 on timeout.
run_case() {
  # --kill-after sends SIGKILL if the launcher ignores the initial
  # SIGTERM (e.g. a hung GPU job) so a stuck run cannot wedge the sweep.
  timeout --kill-after=30s "$TIMEOUT" \
      "${LAUNCHER[@]}" "$ranks" "${EXEC[@]}" -i "$inp" \
      Mesh/parallel_refine="$1" \
      Executioner/device="$device" \
      Executioner/assembly_level="$asm" \
      "${ORDER_ARGS[@]}" \
      "${SOLVER_ARGS[@]}" \
      Outputs/perf_graph=true \
      > "$2" 2>&1
}

cd "$TEST_DIR"

failures=()
for tag in $EXAMPLES; do
  inp="${EXAMPLE_INPUTS[$tag]:-}"
  if [[ -z "$inp" ]]; then
    echo "WARNING: unknown example '$tag' (known: ${!EXAMPLE_INPUTS[*]}) — skipping" >&2
    failures+=("unknown example $tag")
    continue
  fi
  solver_args "$tag"
  for device in $DEVICES; do
    for asm in $ASSEMBLY_LEVELS; do
      for p in $ORDERS; do
        order_args "$tag" "$p"
        for ranks in $RANKS; do
          # The first run of a series pays a one-off setup cost that later runs on
          # the node do not (median 2x in FormLinearSystem on HIP), and it is always
          # the first GAMS setting at the lowest level, which biases that pair. This
          # untimed run absorbs the cost; its output is discarded and a failure here
          # is reported by the timed run at the same level.
          echo ">>> $tag  device=$device  assembly=$asm  order=$p  ranks=$ranks  parallel_refine=$WARMUP_REFINEMENT  (warmup)"
          run_case "$WARMUP_REFINEMENT" /dev/null
          for r in $REFINEMENTS; do
            timed_out=0
            # Run with GPU-aware MPI on (gam=1) and/or off (gam=0). MFEM enables it
            # when MFEM_GPU_AWARE_MPI is set in the environment, so we export it
            # for the "on" run and unset it for the "off" run.
            for gam in $GAMS; do
              if (( gam )); then
                export MFEM_GPU_AWARE_MPI=1
              else
                unset MFEM_GPU_AWARE_MPI
              fi
              log="$RESULTS_DIR/${tag}_${device}_${asm}_p${p}_n${ranks}_r${r}_gam${gam}.log"
              echo ">>> $tag  device=$device  assembly=$asm  order=$p  ranks=$ranks  parallel_refine=$r  gpu_aware_mpi=$gam"
              run_case "$r" "$log"
              status=$?
              if (( status == 124 )); then
                echo "    TIMED OUT (>${TIMEOUT}s)"
                failures+=("$tag device=$device assembly=$asm p=$p ranks=$ranks r=$r gam=$gam (timeout)")
                timed_out=1
              elif (( status != 0 )); then
                echo "    FAILED — see $log"
                failures+=("$tag device=$device assembly=$asm p=$p ranks=$ranks r=$r gam=$gam")
              fi
            done
            # Refinement levels run in ascending order: if either GPU-aware-MPI
            # variant timed out, every higher level for this case will too, so
            # skip the rest by breaking the r loop.
            if (( timed_out )); then
              echo "    skipping higher refinement levels for this case"
              break
            fi
          done
        done
      done
    done
  done
done
unset MFEM_GPU_AWARE_MPI

echo
if (( ${#failures[@]} > 0 )); then
  echo "Some runs failed:"
  printf '  %s\n' "${failures[@]}"
fi

echo
# The speedup plots pair each gam1 log with its gam0 log, and exit with an error if
# there are no pairs.
if [[ " $GAMS " == *" 0 "* && " $GAMS " == *" 1 "* ]]; then
  echo "Generating GPU-aware MPI speedup plots..."
  python3 "$SCRIPT_DIR/plot_gpu_aware_mpi_speedup.py" "$RESULTS_DIR"
else
  echo "Skipping GPU-aware MPI speedup plots: GAMS=\"$GAMS\" selects only one setting."
fi
