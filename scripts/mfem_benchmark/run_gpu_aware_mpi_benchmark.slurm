#!/bin/bash
# SLURM wrapper for run_gpu_aware_mpi_benchmark.sh: requests the resources, loads
# the environment MOOSE was built with, sets the benchmark parameters through the
# environment overrides documented in run_gpu_aware_mpi_benchmark.sh, and runs it.
#
# Before submitting, replace every <...> placeholder below (<account>,
# <partition> and the <path> values) and adapt the resource requests to your
# cluster. Submit from this directory:
#   cd scripts/mfem_benchmark && sbatch run_gpu_aware_mpi_benchmark.slurm
#
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --mail-type=none
#SBATCH -p <partition>
#SBATCH -A <account>
#SBATCH --job-name=moose_gpu_aware_mpi
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Script that loads the compiler, MPI and GPU environment MOOSE was built with
ENV_SCRIPT="<path>"
# Directory the run logs and plots are written to
RESULTS_DIR="<path>"

if [[ "$ENV_SCRIPT" == "<path>" || "$RESULTS_DIR" == "<path>" ]]; then
  echo "ERROR: replace the <path> placeholders in run_gpu_aware_mpi_benchmark.slurm" \
       "before submitting." >&2
  exit 1
fi

echo "Application running on node ${SLURMD_NODENAME}"

source "$ENV_SCRIPT" || exit 1

export EXAMPLES="diffusion curlcurl complex heattransfer"
export REFINEMENTS="0 1 2 3 4 5"
export DEVICES="ceed-cuda"
export ASSEMBLY_LEVELS="none"
export ORDERS="1 2 3 4"
export RANKS=$SLURM_NTASKS
export GAMS="1 0"
export MPI_LAUNCHER="mpirun -n"
# mpirun does not apply --gpus-per-task, so bind each rank to its own GPU
export GPU_BIND=1
export TIMEOUT=1200

# The results directory is taken from the first argument, not the environment
./run_gpu_aware_mpi_benchmark.sh "$RESULTS_DIR"
