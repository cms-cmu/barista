#!/usr/bin/env bash
# Wall time, GPU utilisation and per-stage training time of an ORCHARD run directory.
#   software/orchard/summarize.sh /project/community/$USER/barista/runs/<label>
# Sums over all (re)starts of the run. GPU-hours = wall time of the job(s), 1 GPU each.
set -uo pipefail
RUN="${1:?usage: summarize.sh <run dir>}"; LOGS="${RUN}/logs"
echo "== ${RUN}"
awk '
    FNR == 1 { delete t }
    { t[$1] = $2 }
    $1 == "staged"  { stage += t["staged"] - t["start"] }
    $1 == "trained" || $1 == "interrupted" { train += $2 - t["staged"]; total += $2 - t["start"]; n++ }
    END { printf "job runs: %d   staging: %.2f h   training: %.2f h   total: %.2f GPU-h\n", n, stage/3600, train/3600, total/3600 }
' "${LOGS}"/timing_*.txt 2>/dev/null
awk -F', *' '
    $2 ~ /^[0-9.]+$/ { n++; u += $2; if ($2 >= 50) busy++; if ($4 > m) m = $4; p += $5 }
    END { if (n) printf "GPU (%d samples, staging excluded): mean util %.0f %%, >=50 %% util in %.0f %% of samples, peak memory %.1f GB, mean power %.0f W\n", n, u/n, 100*busy/n, m/1024, p/n }
' "${LOGS}"/gpu_*.csv 2>/dev/null
if [ -f "${RUN}/train.log" ]; then
    echo "-- stages (from train.log)"
    grep -a -o -E '[A-Za-z_]+: run [0-9]+ epochs in [0-9:.]+( days?, [0-9:.]+)?' "${RUN}/train.log" | sort | uniq -c
    grep -a -c 'Successfully resumed' "${RUN}/train.log" | sed 's/^/resumes from checkpoint: /'
fi
[ -f "${RUN}/train.done" ] && echo "status: train.done" || echo "status: not finished"
