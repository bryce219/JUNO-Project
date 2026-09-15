#!/usr/bin/env bash
# This is what make status runs. It prints what's running, the progress, the speed and the best hit 
set -u
cd "$(dirname "$0")/.."
PROJECT=$(pwd -P)
. ./cuda/procs.sh

# Makes a big number easier to read (billions, trillions etc)
readable() {
    awk -v number="$1" 'BEGIN {
        if (number >= 1e15) printf "%.1f quadrillion", number / 1e15
        else if (number >= 1e12) printf "%.1f trillion", number / 1e12
        else if (number >= 1e9) printf "%.1f billion", number / 1e9
        else if (number >= 1e6) printf "%.1f million", number / 1e6
        else printf "%d", number
    }'
}

# Look for the processes first, before we run ./scan top ourselves. We only want the ones running in this folder
if [ -n "$(ours -f '^bash \./cuda/watchdog\.sh')" ]; then WATCHDOG="running"; else WATCHDOG="not running"; fi
if [ -n "$(ours -f '^bash .*supervise\.sh')" ]; then SUPERVISOR="running"; else SUPERVISOR="not running"; fi
if [ -n "$(ours -x scan)" ]; then
    SCANNER="running"
elif pgrep -x scan > /dev/null; then
    SCANNER="not running (but something else called scan is running on this computer)"
else
    SCANNER="not running"
fi
echo "Watchdog:    $WATCHDOG"
echo "Supervisor:  $SUPERVISOR"
echo "Scanner:     $SCANNER"

if [ -s results/custom_seed.txt ]; then
    echo "Saved seed:  $(head -n 1 results/custom_seed.txt) (the custom seed new ranges use)"
fi

# The checkpoint has the tag, the stream offset, the start, the count and how many are done
if [ -s results/gpu_progress.txt ]; then
    read -r _ OFFSET START COUNT DONE < results/gpu_progress.txt
    AGE=$(( $(date +%s) - $(stat -c %Y results/gpu_progress.txt) ))
    if [ "$OFFSET" = "0" ]; then
        echo "Stream:      the plain stream"
    fi
    if [ "$COUNT" = "9223372036850581503" ]; then
        echo "Progress:    $(readable "$DONE") indexes done, starting from index $START (this range goes until you stop it)"
    else
        PERCENT=$(awk -v done="$DONE" -v count="$COUNT" 'BEGIN { printf "%.1f", 100 * done / count }')
        echo "Progress:    $(readable "$DONE") of $(readable "$COUNT") indexes done, starting from index $START ($PERCENT%)"
    fi
    echo "             (the checkpoint was saved $AGE seconds ago)"
else
    echo "Progress:    there's no checkpoint yet"
fi

# The scanner saves its speed next to the checkpoint every 30 seconds
if [ "$SCANNER" = "running" ] && [ -s results/gpu_speed.txt ] && [ $(( $(date +%s) - $(stat -c %Y results/gpu_speed.txt) )) -lt 90 ]; then
    echo "Speed:       about $(readable "$(head -n 1 results/gpu_speed.txt)") indexes a second"
elif [ "$SCANNER" = "running" ]; then
    echo "Speed:       not measured yet (give it 30 seconds)"
fi

if [ -s results/gpu_hits.jsonl ]; then
    echo "Hits:        $(grep -c '^{"seed"' results/gpu_hits.jsonl) saved in results/gpu_hits.jsonl"
    BEST=""
    if [ -x cuda/scan ]; then
        BEST=$(cd cuda && ./scan top 1 | grep '^ *1\. ' | sed 's/^ *1\. //')
    fi
    if [ -n "$BEST" ]; then
        echo "Best hit:    $BEST"
    fi
else
    echo "Hits:        none yet"
fi
