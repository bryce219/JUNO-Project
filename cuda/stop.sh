#!/usr/bin/env bash
# Stops the watchdog, the supervisor and the scanner of this copy of JUNO. make stop runs this
set -u
cd "$(dirname "$0")/.."
PROJECT=$(pwd -P)
. ./cuda/procs.sh

scripts() {
    ours -f '^bash \./cuda/watchdog\.sh'
    ours -f '^bash .*supervise\.sh'
}

mkdir -p results
touch results/STOP_WATCHDOG results/STOP_GPU
echo "Stopping..."

# The scripts go first, so nothing starts the scanner up again
for i in $(seq 30); do
    if [ -z "$(scripts)" ]; then
        break
    fi
    sleep 1
done

# SIGTERM makes the scanner finish the batches it's on and save the checkpoint
SCANNERS=$(ours -x scan)
if [ -n "$SCANNERS" ]; then
    kill -TERM $SCANNERS 2>/dev/null
fi
for i in $(seq 120); do
    if [ -z "$(ours -x scan)" ]; then
        break
    fi
    sleep 1
done

if [ -n "$(scripts)$(ours -x scan)" ]; then
    echo "Something is still stopping, check make status in a minute"
    exit 1
fi
echo "Everything is stopped."
