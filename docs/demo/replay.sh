#!/usr/bin/env bash
# replay.sh LOG [SPEED]: print a warmctl progress log as it was written,
# each "[  12.3s] ..." line at its own time, divided by SPEED (default 1).
set -eu
log=$1 speed=${2:-1} start=$(date +%s%N)
while IFS= read -r line; do
  if [[ $line =~ ^\[\ *([0-9]+)\.([0-9])s\] ]]; then
    at=$((start + (BASH_REMATCH[1] * 10 + BASH_REMATCH[2]) * 100000000 / speed))
    wait=$((at - $(date +%s%N)))
    if ((wait > 0)); then sleep "$((wait / 1000000000)).$(printf '%09d' $((wait % 1000000000)))"; fi
  fi
  printf '%s\n' "$line"
done < "$log"
