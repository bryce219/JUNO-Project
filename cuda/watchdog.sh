#!/usr/bin/env bash
# Starts supervise.sh whenever it isn't running, unless results/STOP_GPU is there.
# Arguments go to the supervisor and then on to ./scan
# Stop it with: touch results/STOP_WATCHDOG
set -u
cd "$(dirname "$0")/.."
PROJECT=$(pwd -P)
. ./cuda/procs.sh
LOG=results/watchdog.log
mkdir -p results

log() {
    echo "[$(date)] Watchdog: $1" >> "$LOG"
}

# like sleep, but it checks for STOP_WATCHDOG every second
nap() {
    for i in $(seq "$1"); do
        [ -f results/STOP_WATCHDOG ] && return
        sleep 1
    done
}

rm -f results/STOP_WATCHDOG
log "Started"
while [ ! -f results/STOP_WATCHDOG ]; do
    if [ ! -f results/STOP_GPU ]; then
        # is our supervisor running? (a supervisor from another copy doesn't count)
        if [ -z "$(ours -f '^bash .*supervise\.sh')" ]; then
            log "The supervisor isn't running, starting it again..."
            nohup bash ./cuda/supervise.sh ${1+"$@"} >/dev/null 2>&1 &
            nap 60
        fi
    fi
    nap 30
done
log "Found STOP_WATCHDOG, stopping"
