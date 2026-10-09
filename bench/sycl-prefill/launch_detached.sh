#!/bin/bash
# Launch a Strata engine run detached from the agent's process group and with a
# high oom_score_adj, so the kernel OOM killer targets the engine (not maki) and
# a tool-call timeout cannot SIGKILL the run.
#
# Usage: launch_detached.sh <logfile> <engine args...>
# Returns immediately; the run continues under setsid+nohup.
set -u
LOG="${1:?logfile}"; shift
cd /home/dwmcqueen/Strata
# Bias the OOM killer toward this process tree (unprivileged raise is allowed).
setsid nohup bash -c 'echo 1000 > /proc/self/oom_score_adj; exec "$@"' _ "$@" \
  >"$LOG" 2>&1 < /dev/null &
echo $!
