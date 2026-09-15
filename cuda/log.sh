#!/usr/bin/env bash
# make log runs this. It shows the new records as the scanner finds them, or with MIN_SENTS / MIN_ARBITRATIONS 
# every new hit with a high enough score. Ctrl+C closes the log but the scan keeps going
set -u
cd "$(dirname "$0")/.."
MIN_SENTS=${MIN_SENTS-}
MIN_ARBITRATIONS=${MIN_ARBITRATIONS-}
mkdir -p results
touch results/gpu_hits.jsonl results/gpu_scan.log

if [ -n "$MIN_SENTS$MIN_ARBITRATIONS" ]; then
    echo "Showing new hits with SENTS >= ${MIN_SENTS:-0} and ARBITRATIONS >= ${MIN_ARBITRATIONS:-0} (Ctrl+C closes the log, JUNO keeps running)"
else
    echo "Showing new records (Ctrl+C closes the log, JUNO keeps running)"
fi

# mawk (Ubuntu's awk) buffers the pipe, -W interactive gets it to read one line at a time
# without this the log just sits there showing nothing, super helpful
AWK="awk"
if awk -W version 2>&1 | grep -q mawk; then
    AWK="awk -W interactive"
fi

# tail starts at the top of both files. The hits that were already in the file only set the records to beat
OLD_HITS=$(wc -l < results/gpu_hits.jsonl)
OLD_LOG=$(wc -l < results/gpu_scan.log)
tail -n +1 -F results/gpu_hits.jsonl results/gpu_scan.log 2>/dev/null | $AWK -v oldHits="$OLD_HITS" -v oldLog="$OLD_LOG" \
    -v minSents="$MIN_SENTS" -v minArbitrations="$MIN_ARBITRATIONS" '
function field(name,    found) {
    if (match($0, "\"" name "\": -?[0-9.]+")) {
        found = substr($0, RSTART, RLENGTH)
        sub(/^[^:]*: /, "", found)
        return found
    }
    return ""
}
function now(    time) {
    "date +%H:%M:%S" | getline time
    close("date +%H:%M:%S")
    return time
}
/^==> .* <==$/ { file = $2; next }
$0 == "" { next }
file ~ /gpu_scan\.log$/ {
    logLines++
    if (logLines > oldLog && $0 ~ /Supervisor:|Stopped|Couldn.t|already running|isn.t/) {
        print
        fflush()
    }
    next
}
{
    hitLines++
    seed = field("seed"); sentsText = field("sents"); arbitrationsText = field("arbitrations")
    if (seed == "" || sentsText == "" || arbitrationsText == "") next
    sents = sentsText + 0; arbitrations = arbitrationsText + 0
    scores = "seed " seed "  SENTS " sentsText "  ARBITRATIONS " arbitrationsText
    if (hitLines <= oldHits) {
        if (sents > bestSents) { bestSents = sents; bestSentsText = sentsText }
        if (arbitrations > bestArbitrations) { bestArbitrations = arbitrations; bestArbitrationsText = arbitrationsText }
        if (hitLines == oldHits) {
            print "Best so far: SENTS " bestSentsText ", ARBITRATIONS " bestArbitrationsText
            fflush()
        }
        next
    }
    if (minSents != "" || minArbitrations != "") {
        if (sents >= minSents + 0 && arbitrations >= minArbitrations + 0) {
            print "[" now() "] " scores
            fflush()
        }
    } else if (sents > bestSents || arbitrations > bestArbitrations) {
        if (sents > bestSents && arbitrations > bestArbitrations) what = "SENTS and ARBITRATIONS"
        else if (sents > bestSents) what = "SENTS"
        else what = "ARBITRATIONS"
        print "[" now() "] New record (" what ")! " scores
    }
    if (sents > bestSents) { bestSents = sents; bestSentsText = sentsText }
    if (arbitrations > bestArbitrations) { bestArbitrations = arbitrations; bestArbitrationsText = arbitrationsText }
    fflush()
}'
