# runner_monitor.py

Live monitor for parallel `runner.py` jobs submitted via Snakemake + HTCondor.

## Usage

Run from the barista root directory (no container needed — stdlib only):

```bash
python src/tools/runner_monitor.py output/Run3_FeynNet/feynnet_friendtrees/logs/
```

Or search all `output/` subdirectories:

```bash
python src/tools/runner_monitor.py
```

Refreshes every 10 seconds. Press **Ctrl-C** to quit.

## What it shows

```
Dask job monitor  —  10:45:32  (20 jobs: 7 done, 9 running, 4 pending — refresh every 10s)

Job                                                 Progress
----------------------------------------------------------------------
SvBFeynNet_TTTo2L2Nu__2022_EE                       COMPLETED
SvBFeynNet_TTTo2L2Nu__2022_preEE                    COMPLETED
SvBFeynNet_TTTo2L2Nu__2023_BPix                     workers=36(11 busy)  processing=25  waiting=7  memory=77
SvBFeynNet_TTTo2L2Nu__2023_preBPix                  starting / unreachable
SvBFeynNet_TTToHadronic__2022_EE                    workers=12(4 busy)  processing=4  memory=30
SvBFeynNet_mixeddata_all__2023_preBPix              pending
...
```

| Status | Meaning |
|---|---|
| `COMPLETED` (green) | Log contains "JOB EXECUTION COMPLETED SUCCESSFULLY" or "Dask performance report saved" |
| `workers=N(M busy)  processing=...` | Live metrics from the Dask Prometheus endpoint |
| `starting / unreachable` | Job started but dashboard not yet reachable |
| `pending` (grey) | Planned by Snakemake but not yet started (no log file) |

## How it works

1. **Scans log files** in the given directory for:
   - `Dask dashboard: http://host:port` — direct dashboard URL (logged by runner.py after scheduler starts)
   - `Dask dashboard: /proxy/PORT` — proxy URL (used on some LPC nodes); port is combined with the scheduler host to form a direct URL
   - `tcp://host:port` — scheduler TCP address (fallback to connect via `dask.distributed.Client`)
   - Completion markers to mark jobs done without querying a dead scheduler

2. **Reads the most recent `.snakemake/log/*.snakemake.log`** to find all jobs planned by Snakemake (including ones not yet started), and shows them as `pending`.

3. **Queries `http://host:port/metrics`** (Dask's Prometheus endpoint) for live task and worker counts.

## Prerequisites

- Run from the barista root directory so `.snakemake/log/` is found automatically.
- The Dask dashboard URL is logged by `runner.py` after the scheduler starts. Jobs submitted before this logging was added will fall back to TCP scheduler connection or show `starting / unreachable`.
- No extra Python packages needed (uses `urllib` from stdlib). The TCP fallback requires `dask.distributed` but that is available in the analysis container.

## Companion tools

| Tool | Purpose |
|---|---|
| `src/tools/condor_monitor.py` | Job-level condor status (idle/running/held) and worker node stderr tail |
| `src/tools/runner_monitor.py` | Dask task-level progress for each running `runner.py` scheduler |

Use both together: `condor_monitor` tells you how many condor workers are alive, `runner_monitor` tells you how many chunks have been processed.
