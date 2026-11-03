#!/usr/bin/env bash
#
# Restrict each MPI rank to a single GPU, chosen by its node-local rank, then
# exec the given command. MOOSE configures the MFEM device without a device
# id, so every rank uses its first visible GPU; unless the launcher binds GPUs
# per rank (mpirun does not), all ranks on a node then share one GPU.
#
# Usage (as a prefix to the executable, after the MPI launcher):
#   mpirun -n 8 ./gpu_bind.sh ./moose_test-opt -i input.i
#
# If the scheduler already restricted the visible GPUs (e.g. Slurm setting
# ROCR_VISIBLE_DEVICES=0,1,2,3), rank r gets entry r of that list, wrapping
# around when there are more ranks than GPUs; a list that is already a single
# GPU (srun --gpus-per-task=1) is therefore left as is.
#
# Environment:
#   GPU_VENDOR     - "amd" or "nvidia"; selects ROCR_VISIBLE_DEVICES or
#                    CUDA_VISIBLE_DEVICES (default: detected from rocm-smi /
#                    nvidia-smi being installed)

set -euo pipefail

local_rank="${OMPI_COMM_WORLD_LOCAL_RANK:-${MPI_LOCALRANKID:-${SLURM_LOCALID:-}}}"
if [[ -z "$local_rank" ]]; then
  echo "gpu_bind.sh: cannot determine the node-local MPI rank" >&2
  exit 1
fi

vendor="${GPU_VENDOR:-}"
if [[ -z "$vendor" ]]; then
  if command -v rocm-smi > /dev/null; then
    vendor=amd
  elif command -v nvidia-smi > /dev/null; then
    vendor=nvidia
  else
    echo "gpu_bind.sh: no rocm-smi or nvidia-smi found; set GPU_VENDOR=amd|nvidia" >&2
    exit 1
  fi
fi

case "$vendor" in
  amd) var=ROCR_VISIBLE_DEVICES ;;
  nvidia) var=CUDA_VISIBLE_DEVICES ;;
  *)
    echo "gpu_bind.sh: unknown GPU_VENDOR '$vendor' (expected amd or nvidia)" >&2
    exit 1
    ;;
esac

IFS=, read -ra visible <<< "${!var:-}"
if (( ${#visible[@]} > 0 )); then
  gpu="${visible[$(( local_rank % ${#visible[@]} ))]}"
else
  gpu="$local_rank"
fi
export "$var=$gpu"

echo "gpu_bind.sh: host $(hostname) local rank $local_rank -> $var=$gpu" >&2
exec "$@"
