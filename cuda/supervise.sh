#!/usr/bin/env bash
# Keeps the scanner going. If ./scan stops, this starts it again with run (it carries on from the checkpoint) 
# Arguments get passed to ./scan, eg --min-sents 0.91
# To stop it: touch results/STOP_GPU (a scan that's already running keeps going)
set -u
cd "$(dirname "$0")"
PROJECT=$(cd .. && pwd -P)
. ./procs.sh
RESULTS=../results
LOG=$RESULTS/gpu_scan.log

log() {
    echo "[$(date)] Supervisor: $1" >> "$LOG"
}

mkdir -p $RESULTS
rm -f $RESULTS/STOP_GPU
LAST_CHECKPOINT="none"
TRIES=0
WAITING=no
while [ ! -f $RESULTS/STOP_GPU ]; do
    if ! pgrep -x scan >/dev/null; then
        # check STOP_GPU again, make stop might be what killed the scanner
        if [ -f $RESULTS/STOP_GPU ]; then
            break
        fi

        # Give up if the scanner keeps dying and the checkpoint never moves
        CHECKPOINT=$(cat $RESULTS/gpu_progress.txt 2>/dev/null)
        if [ "$CHECKPOINT" = "$LAST_CHECKPOINT" ]; then
            TRIES=$((TRIES + 1))
        else
            TRIES=0
        fi
        if [ $TRIES -ge 3 ]; then
            log "Scanner stopped 3 times in a row without saving any progress, check the messages above! Stopping..."
            touch $RESULTS/STOP_GPU
            exit 1
        fi
        LAST_CHECKPOINT=$CHECKPOINT

        log "Starting the scanner..."
        ./scan ${1+"$@"} run >> "$LOG" 2>&1 &
        WAITING=no
    elif [ -z "$(ours -x scan)" ] && [ $WAITING = no ]; then
        # don't start ours while some other scanner is running, two on the GPU can freeze the computer
        log "Something else called scan is running on this computer, waiting for it to stop..."
        WAITING=yes
    fi

    # Wait 20 seconds, but notice STOP_GPU right away
    for i in $(seq 20); do
        if [ -f $RESULTS/STOP_GPU ]; then
            break
        fi
        sleep 1
    done
done
log "Found STOP_GPU, stopping."
