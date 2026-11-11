#!/usr/bin/env bash
#
# Sourced by the benchmark scripts in this directory; defines timed_run.
#
# timed_run <limit_s> <log> <command...>
#   Runs the command, with its output in <log>, under timeout(1): SIGTERM at the
#   limit, then SIGKILL 30 s later if it has not exited, so that a stuck run cannot
#   wedge a sweep. timeout signals only its own process group, which does not hold
#   the MPI ranks, so a rank that has not exited when mpirun is killed keeps running,
#   orphaned, and slows down every later run on the node. timed_run therefore kills
#   any process the run left behind and waits for it to exit. It finds them by a
#   marker in their environment, which mpirun and srun pass on to the ranks, so it
#   leaves alone the processes of other runs, such as another sweep on the same node.
#   Returns the command's exit status, or 124 if the command hit the limit.

TIMED_RUN_COUNT=0

timed_run() {
  local limit="$1" log="$2"
  shift 2
  # $$ tells apart the runs of scripts running at the same time on the node.
  local marker="TIMED_RUN_ID=$$.$((++TIMED_RUN_COUNT))"
  local start=$SECONDS
  env "$marker" timeout --kill-after=30s "$limit" "$@" > "$log" 2>&1
  local status=$?
  # timeout exits with 124 if the command ends after the SIGTERM, but with 137 if it
  # needs the SIGKILL, which is also the status of a command killed by SIGKILL for
  # another reason, such as the out-of-memory killer.
  if (( status == 137 && SECONDS - start >= limit )); then
    status=124
  fi
  local pids
  mapfile -t pids < <(timed_run_pids "$marker")
  if (( ${#pids[@]} > 0 )); then
    local kill_start=$SECONDS
    kill -KILL "${pids[@]}" 2>/dev/null
    # A process blocked in the kernel, e.g. waiting for memory, dies only once it
    # returns from it.
    while [[ -n "$(timed_run_pids "$marker")" ]]; do
      sleep 1
    done
    echo "    killed ${#pids[@]} leftover process(es) of the run," \
         "which took $((SECONDS - kill_start)) s to exit"
  fi
  return "$status"
}

# PIDs of the live processes whose environment contains the entry $1.
timed_run_pids() {
  grep -lsxzF "$1" /proc/[0-9]*/environ | cut -d/ -f3
}
