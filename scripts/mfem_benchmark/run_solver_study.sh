#!/usr/bin/env bash
#
# Find the fastest linear solver and preconditioner for the diffusion and curlcurl
# examples on one device, for each assembly level, at a fixed relative tolerance.
#
# Each combination solves the same problem once, at one order and refinement level
# chosen to be large for a single node. summarize_solver_study.py then ranks the
# combinations of each case by time to solution: operator assembly
# (EquationSystemProblemOperator::FormSystem) plus preconditioner setup and solve
# (EquationSystemProblemOperator::SolveSystem). It also flags runs that did not
# converge or whose solution functional disagrees with the other runs.
#
# Cases and combinations (solver + preconditioner):
#   diffusion, full:    HypreGMRES, HyprePCG and MFEM CG with BoomerAMG; MFEM CG + Jacobi
#   diffusion, partial: MFEM CG with Jacobi, Chebyshev, or p-multigrid (Chebyshev
#                       smoothers on partially assembled levels, CG + BoomerAMG on the
#                       assembled order 1 level)
#   curlcurl, legacy:   HypreGMRES, HyprePCG and MFEM CG with HypreAMS; MFEM CG + Jacobi
#   curlcurl, partial:  MFEM CG with MatrixFreeAMS, Chebyshev or Jacobi
# curlcurl uses legacy in place of full assembly, which MFEM does not implement for its
# curl-curl integrator. Hypre solvers and preconditioners need an assembled
# HypreParMatrix, so none runs on partial assembly; that includes the LOR-wrapped ones,
# because MOOSE hands every preconditioner the matrix-free high-order operator.
# curlcurl runs on MFEM's default ND basis. AMS builds its discrete gradient by point
# evaluation, which from order 2 matches that basis only: on the GaussLobatto /
# IntegratedGLL basis, AMS needs 6x more iterations already at parallel_refine=2.
# On the CPU, full and partial assembly are much slower than legacy assembly. There,
# diffusion's full-assembly combinations run with legacy assembly, and the
# partial-assembly cases are skipped: their matrix-free preconditioners only stand in
# for the hypre ones, which legacy assembly makes available.
#
# Usage:
#   ./run_solver_study.sh [results_dir]
#
# Environment overrides:
#   MOOSE_ROOT, MOOSE_EXEC, MPI_LAUNCHER, GPU_BIND - as in run_gpu_aware_mpi_benchmark.sh
#   TIMEOUT    - per-run wall-clock limit in seconds (default: 900)
#   DEVICE     - Executioner/device value (default: hip)
#   RANKS      - MPI rank count (default: 8)
#   ORDER      - polynomial order, 2 to 4 (default: 2)
#   REFINE     - Mesh/parallel_refine value (default: 4, giving 82M true DoFs for
#                diffusion and 44M for curlcurl at order 2)
#   TOL        - relative tolerance of every outer solver (default: 1e-12)
#
# Logs are named <example>_<device>_<assembly>_<combination>_p<P>_n<R>_r<N>.log. Run
# once per device into the same results_dir to summarize both devices together.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOOSE_ROOT="${MOOSE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
MOOSE_EXEC="${MOOSE_EXEC:-$MOOSE_ROOT/test/moose_test-opt}"
TEST_DIR="$MOOSE_ROOT/test"
RESULTS_DIR="${1:-$SCRIPT_DIR/solver_study_results}"
MPI_LAUNCHER="${MPI_LAUNCHER:-srun -n}"
GPU_BIND="${GPU_BIND:-0}"
TIMEOUT="${TIMEOUT:-900}"
DEVICE="${DEVICE:-hip}"
RANKS="${RANKS:-8}"
ORDER="${ORDER:-2}"
REFINE="${REFINE:-4}"
TOL="${TOL:-1e-12}"
# High enough that only TOL or TIMEOUT stops a solve.
MAX_ITS=100000

# p-multigrid coarsens to order 1, and ORDER_NAMES below ends at FOURTH.
if (( ORDER < 2 || ORDER > 4 )); then
  echo "ERROR: ORDER must be 2, 3 or 4, got '$ORDER'." >&2
  exit 1
fi

read -ra LAUNCHER <<< "$MPI_LAUNCHER"
EXEC=("$MOOSE_EXEC")
# One thread per rank. hypre and OpenBLAS are OpenMP-threaded in MOOSE's stock PETSc
# build (--with-openmp=1), and otherwise start a thread per core in every rank.
export OMP_NUM_THREADS=1
if [[ "$GPU_BIND" == 1 ]]; then
  EXEC=("$SCRIPT_DIR/gpu_bind.sh" "$MOOSE_EXEC")
fi

if [[ ! -x "$MOOSE_EXEC" ]]; then
  echo "ERROR: $MOOSE_EXEC not executable. Set MOOSE_EXEC to the right binary." >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR/csv"
# Made absolute because the runs write their logs after changing to $TEST_DIR.
RESULTS_DIR="$(cd "$RESULTS_DIR" && pwd)"
source "$SCRIPT_DIR/timed_run.sh"
cd "$TEST_DIR"

ORDER_NAMES=(CONSTANT FIRST SECOND THIRD FOURTH)
FEC="${ORDER_NAMES[$ORDER]}"
RT_FEC="${ORDER_NAMES[$((ORDER - 1))]}"

DIFFUSION=tests/mfem/kernels/diffusion.i
CURLCURL=tests/mfem/kernels/curlcurl.i

# p-multigrid needs the variable on an FE space hierarchy, and the 'fespace' that
# diffusion.i gives it cannot be removed from the command line, so it runs from a copy
# that names the hierarchy instead. The copy lives in RESULTS_DIR, so the mesh path,
# relative to the original input, is passed explicitly.
DIFFUSION_PMG="$RESULTS_DIR/diffusion_pmg.i"
sed 's/fespace = H1FESpace/fespace_hierarchy = h1_hierarchy/' "$DIFFUSION" > "$DIFFUSION_PMG"

# Overrides shared by every combination of an example. The level 2 perf graph shows
# EquationSystemProblemOperator::FormSystem and ::SolveSystem; SolveSystem includes
# preconditioner setups that the level 1 sections miss (MatrixFreeAMS, p-multigrid).
DIFFUSION_ARGS=(FESpaces/H1FESpace/fec_order="$FEC" FESpaces/HCurlFESpace/fec_order="$FEC"
                "Outputs/active=CSV pgraph")
# Order 1 base space, p-refined level by level up to ORDER.
PMG_ARGS=(Mesh/file="$TEST_DIR/tests/mfem/mesh/mug.e"
          FESpaces/H1FESpace/fec_order=FIRST FESpaces/HCurlFESpace/fec_order="$FEC"
          FESpaceHierarchies/h1_hierarchy/type=MFEMFESpaceHierarchy
          FESpaceHierarchies/h1_hierarchy/fespace=H1FESpace
          "FESpaceHierarchies/h1_hierarchy/refinements=$(seq -s ' ' 2 "$ORDER")"
          "Outputs/active=CSV pgraph")
# The L2 error against the exact solution is written to the CSV file, so the summary
# can check that every combination reached the same solution.
CURLCURL_ARGS=(FESpaces/HCurlFESpace/fec_order="$FEC" FESpaces/HDivFESpace/fec_order="$RT_FEC"
               Postprocessors/e_error/type=MFEMVectorL2Error
               Postprocessors/e_error/variable=e_field
               Postprocessors/e_error/function=exact_e_field)

# moose_run <log> <csv_file_base> <assembly> <input> <overrides...>
moose_run() {
  local log="$1" csv="$2" asm="$3" inp="$4"
  shift 4
  timed_run "$TIMEOUT" "$log" \
      "${LAUNCHER[@]}" "$RANKS" "${EXEC[@]}" -i "$inp" \
      Mesh/parallel_refine="$REFINE" \
      Executioner/device="$DEVICE" \
      Executioner/assembly_level="$asm" \
      "$@" \
      Outputs/CSV/file_base="$csv" \
      Outputs/pgraph/type=PerfGraphOutput Outputs/pgraph/level=2
}

failures=()
# run_combo <example> <assembly> <combination> <input> <overrides...>
run_combo() {
  local tag="$1" asm="$2" combo="$3" inp="$4"
  shift 4
  local name="${tag}_${DEVICE}_${asm}_${combo}_p${ORDER}_n${RANKS}_r${REFINE}"
  local log="$RESULTS_DIR/$name.log"
  echo ">>> $name"
  moose_run "$log" "$RESULTS_DIR/csv/$name" "$asm" "$inp" "$@"
  local status=$?
  if (( status == 124 )); then
    echo "    TIMED OUT (>${TIMEOUT}s)"
    # Marks the log for summarize_solver_study.py, which cannot see the exit status.
    echo "TIMED OUT after ${TIMEOUT} s" >> "$log"
    failures+=("$name (timeout)")
  elif (( status != 0 )); then
    echo "    FAILED - see $log"
    failures+=("$name")
  fi
}

# Assembly levels of the assembled-matrix and matrix-free cases; see the header for the
# CPU's.
if [[ "$DEVICE" == cpu ]]; then
  ASSEMBLED=legacy
  RUN_PARTIAL=0
else
  ASSEMBLED=full
  RUN_PARTIAL=1
fi

# Outer solver tolerance and iteration cap for the [Solvers] entry named $1.
tol_args() {
  echo "Solvers/$1/l_tol=$TOL Solvers/$1/l_max_its=$MAX_ITS"
}

# The first run of a job pays a one-off setup cost that later runs on the node do not
# (FormLinearSystem 3x slower on HIP). This untimed run of the first combination below
# absorbs it; its output is discarded and a failure here is reported by the timed run.
echo ">>> warmup"
moose_run /dev/null "$RESULTS_DIR/warmup" "$ASSEMBLED" "$DIFFUSION" "${DIFFUSION_ARGS[@]}" \
    Solvers/main/type=MFEMHypreGMRES Solvers/boomeramg/print_level=0 $(tol_args main)
rm -f "$RESULTS_DIR"/warmup*.csv

# Preconditioners print at level 0: at hypre's default of 2, BoomerAMG and AMS compute
# a residual norm on every application, which costs extra operator applications.
# diffusion.i: [main] is HypreGMRES preconditioned by [boomeramg]; [jacobi] is inactive.

# diffusion, full assembly (legacy on the CPU)
for type in MFEMHypreGMRES:hypregmres MFEMHyprePCG:hyprepcg MFEMCGSolver:cg; do
  run_combo diffusion "$ASSEMBLED" "${type#*:}-amg" "$DIFFUSION" "${DIFFUSION_ARGS[@]}" \
      Solvers/main/type="${type%:*}" Solvers/boomeramg/print_level=0 $(tol_args main)
done
run_combo diffusion "$ASSEMBLED" cg-jacobi "$DIFFUSION" "${DIFFUSION_ARGS[@]}" \
    Solvers/inactive=boomeramg Solvers/main/type=MFEMCGSolver \
    Solvers/main/preconditioner=jacobi $(tol_args main)

# diffusion, partial assembly
if (( RUN_PARTIAL )); then
  run_combo diffusion partial cg-jacobi "$DIFFUSION" "${DIFFUSION_ARGS[@]}" \
      Solvers/inactive=boomeramg Solvers/main/type=MFEMCGSolver \
      Solvers/main/preconditioner=jacobi $(tol_args main)
  run_combo diffusion partial cg-chebyshev "$DIFFUSION" "${DIFFUSION_ARGS[@]}" \
      "Solvers/inactive=boomeramg jacobi" \
      Solvers/chebyshev/type=MFEMOperatorChebyshevSmoother \
      Solvers/main/type=MFEMCGSolver Solvers/main/preconditioner=chebyshev $(tol_args main)
  # One Chebyshev smoother per level above order 1. The order 1 level is assembled
  # (legacy) for BoomerAMG and the levels between it and the finest are partially
  # assembled. The coarse CG + BoomerAMG solve uses the settings of MFEM example 26p.
  pmg_levels=()
  for p in $(seq 2 "$ORDER"); do
    pmg_levels+=("Solvers/chebyshev$p/type=MFEMOperatorChebyshevSmoother")
  done
  run_combo diffusion partial cg-pmg "$DIFFUSION_PMG" "${PMG_ARGS[@]}" \
      Solvers/inactive=jacobi Solvers/boomeramg/print_level=0 \
      Solvers/coarse/type=MFEMCGSolver Solvers/coarse/preconditioner=boomeramg \
      Solvers/coarse/l_tol=1e-2 Solvers/coarse/l_max_its=10 Solvers/coarse/print_level=-1 \
      "${pmg_levels[@]}" \
      Solvers/pmg/type=MFEMGeometricMultigridSolver Solvers/pmg/variable=concentration \
      "Solvers/pmg/smoothers=$(seq -s ' ' -f 'chebyshev%g' 2 "$ORDER")" \
      Solvers/pmg/coarse_solver=coarse \
      "Solvers/pmg/assembly_levels=legacy$(printf ' partial%.0s' $(seq 2 "$ORDER"))" \
      Solvers/main/type=MFEMCGSolver Solvers/main/preconditioner=pmg $(tol_args main)
fi

# curlcurl.i: active are [gmres], HypreGMRES preconditioned by [ams], and [ams];
# inactive are [cg], MFEM CG preconditioned by [matrix_free_ams], and [matrix_free_ams].

# curlcurl, legacy assembly
run_combo curlcurl legacy hypregmres-ams "$CURLCURL" "${CURLCURL_ARGS[@]}" \
    Solvers/ams/print_level=0 $(tol_args gmres)
run_combo curlcurl legacy hyprepcg-ams "$CURLCURL" "${CURLCURL_ARGS[@]}" \
    Solvers/gmres/type=MFEMHyprePCG Solvers/ams/print_level=0 $(tol_args gmres)
run_combo curlcurl legacy cg-ams "$CURLCURL" "${CURLCURL_ARGS[@]}" \
    "Solvers/active=cg ams" Solvers/cg/preconditioner=ams Solvers/ams/print_level=0 \
    $(tol_args cg)
run_combo curlcurl legacy cg-jacobi "$CURLCURL" "${CURLCURL_ARGS[@]}" \
    "Solvers/active=cg jacobi" Solvers/jacobi/type=MFEMOperatorJacobiSmoother \
    Solvers/cg/preconditioner=jacobi $(tol_args cg)

# curlcurl, partial assembly
if (( RUN_PARTIAL )); then
  run_combo curlcurl partial cg-mfams "$CURLCURL" "${CURLCURL_ARGS[@]}" \
      "Solvers/active=cg matrix_free_ams" $(tol_args cg)
  run_combo curlcurl partial cg-chebyshev "$CURLCURL" "${CURLCURL_ARGS[@]}" \
      "Solvers/active=cg chebyshev" Solvers/chebyshev/type=MFEMOperatorChebyshevSmoother \
      Solvers/cg/preconditioner=chebyshev $(tol_args cg)
  run_combo curlcurl partial cg-jacobi "$CURLCURL" "${CURLCURL_ARGS[@]}" \
      "Solvers/active=cg jacobi" Solvers/jacobi/type=MFEMOperatorJacobiSmoother \
      Solvers/cg/preconditioner=jacobi $(tol_args cg)
fi

echo
if (( ${#failures[@]} > 0 )); then
  echo "Some runs failed:"
  printf '  %s\n' "${failures[@]}"
  echo
fi

python3 "$SCRIPT_DIR/summarize_solver_study.py" "$RESULTS_DIR" --tol "$TOL" \
    | tee "$RESULTS_DIR/summary.txt"
