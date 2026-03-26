#!/usr/bin/env python3
"""Monitor HTCondor jobs, tailing the last line of stdout for each running job.

Usage:
    condor_monitor.py [grep_string]

Examples:
    condor_monitor.py                # monitor all your jobs
    condor_monitor.py HH4b           # only jobs whose batch name / args contain 'HH4b'
    condor_monitor.py 2294883        # only jobs in cluster 2294883
"""
import json
import os
import subprocess
import sys
import time
from collections import Counter

# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------

def _terminal_size():
    try:
        rows, cols = os.popen("stty size", "r").read().split()
        return int(rows), int(cols)
    except Exception:
        return 40, 120


def _place_cursor(row, col=0):
    """Move cursor to absolute row (1-based) and column."""
    sys.stdout.write(f"\033[{row};{col}H")


def _clear_line():
    sys.stdout.write("\033[K")


def _move_cursor_down(n):
    sys.stdout.write(f"\033[{n}B")


# ---------------------------------------------------------------------------
# Job status mapping
# ---------------------------------------------------------------------------

STATUS = {
    0: "U",   # Unexpanded
    1: "I",   # Idle
    2: "R",   # Running
    3: "X",   # Removed
    4: "C",   # Complete
    5: "H",   # Held
    6: "T",   # Transferring output
    7: "S",   # Suspended
}


# ---------------------------------------------------------------------------
# condor_q helpers
# ---------------------------------------------------------------------------

def get_current_user():
    return os.environ.get("USER") or os.popen("whoami").read().strip()


def query_jobs(grep=""):
    """Return list of job dicts from condor_q -json, filtered by owner and grep."""
    user = get_current_user()
    try:
        out = subprocess.check_output(
            "condor_q -json",
            shell=True,
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        )
        all_jobs = json.loads(out) if out.strip() else []
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return []

    jobs = []
    for j in all_jobs:
        if j.get("Owner") != user:
            continue
        # Skip DAGMan manager jobs
        if "dagman" in str(j.get("Cmd", "")).lower():
            continue
        # Skip removed jobs
        if j.get("JobStatus") == 3:
            continue
        # Apply grep filter against batch name, args, cluster id
        if grep:
            haystack = " ".join([
                str(j.get("JobBatchName", "")),
                str(j.get("Arguments", "")),
                str(j.get("ClusterId", "")),
            ])
            if grep not in haystack:
                continue
        jobs.append(j)
    return jobs


def job_id(j):
    return f"{j['ClusterId']}.{j['ProcId']}"


def schedd(j):
    gid = j.get("GlobalJobId", "")
    return gid.split("#")[0] if gid else "unknown"


# ---------------------------------------------------------------------------
# Per-job tail fetching
# ---------------------------------------------------------------------------

def _fmt_runtime(seconds):
    seconds = int(seconds)
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


class CondorJob:
    def __init__(self, job_dict):
        self.d = job_dict
        self.id = job_id(job_dict)
        self.schedd = schedd(job_dict)
        self.status = job_dict.get("JobStatus", 0)
        self.batch = job_dict.get("JobBatchName", "")
        self.out = job_dict.get("Out", "")
        self.start_time = job_dict.get("JobStartDate")
        self.mem_mb = job_dict.get("MemoryProvisioned")
        self.cpu_s = job_dict.get("CumulativeRemoteUserCpu", 0.0)
        machine = job_dict.get("MachineAttrMachine0", "")
        self.machine = machine.split(".")[0] if machine else ""
        self.restarts = max(0, job_dict.get("NumShadowStarts", 1) - 1)
        self.tail_line = ""
        self._proc = None
        self.fetching = False
        self.done = self.status in (3, 4)  # removed or complete

    @property
    def status_char(self):
        return STATUS.get(self.status, "?")

    @property
    def _null_output(self):
        return self.out in ("/dev/null", "", None)

    def _stats_str(self):
        parts = []
        if self.start_time:
            parts.append(_fmt_runtime(time.time() - self.start_time))
        if self.mem_mb:
            parts.append(f"{self.mem_mb}MB")
        if self.machine:
            parts.append(self.machine)
        if self.restarts:
            parts.append(f"[restart:{self.restarts}]")
        return "  ".join(parts)

    def fetch_tail(self):
        """Kick off an async condor_tail for running jobs with real stdout."""
        if self.done or self.fetching or self._proc is not None:
            return
        if self.status != 2:
            return
        if self._null_output:
            return
        cmd = f"condor_tail -maxbytes 256 -name {self.schedd} {self.id}"
        self._proc = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            executable="/bin/bash", universal_newlines=True,
        )
        self.fetching = True

    def poll_tail(self):
        """Check if the tail subprocess finished; update tail_line if so."""
        if self._proc is None:
            return
        if self._proc.poll() is not None:
            self.fetching = False
            raw = self._proc.stdout.read()
            self._proc = None
            lines = [l for l in raw.split("\n") if l.strip()]
            if lines:
                self.tail_line = lines[-1]

    def display_line(self, cols):
        batch_str = f"[{self.batch}]" if self.batch else ""
        if self.status == 2:
            info = self.tail_line or self._stats_str()
        else:
            info = ""
        line = f"{self.id:>14s} {self.status_char} {batch_str:<30s} {info}"
        if len(line) > cols - 1:
            line = line[:cols - 1]
        return line


# ---------------------------------------------------------------------------
# Main display loop
# ---------------------------------------------------------------------------

def display_jobs(jobs, rows, cols):
    """Print all job lines (used for initial render and cursor-based updates)."""
    counts = Counter(j.status_char for j in jobs)
    header = "  ".join(f"{v}×{k}" for k, v in sorted(counts.items()))
    print(f"{'ID':>14s} S {'Batch':<30s} Last output line")
    print("-" * min(cols, 80))
    for j in jobs:
        print(j.display_line(cols))
    print("-" * min(cols, 80))
    print(header)


def run(grep=""):
    rows, cols = _terminal_size()

    jobs = [CondorJob(j) for j in query_jobs(grep)]
    if not jobs:
        print("No jobs found.")
        return

    # Trim to screen height (leave room for header/footer)
    max_display = rows - 5
    if len(jobs) > max_display:
        print(f"WARNING: {len(jobs)} jobs but only {max_display} rows; showing first {max_display}.")
        jobs = jobs[:max_display]

    n = len(jobs)
    HEADER_ROWS = 2  # header line + separator

    # Initial render
    display_jobs(jobs, rows, cols)

    # Map job index → screen row (1-based). Header takes rows (rows-n-3) to (rows-n-2).
    def job_row(i):
        # Jobs are displayed starting 3 rows from the bottom (2 header + jobs + 2 footer)
        return rows - n - 2 + HEADER_ROWS + i - 1

    MAX_CONCURRENT_FETCHES = 16

    try:
        while True:
            rows, cols = _terminal_size()

            # Kick off tail fetches up to the concurrency limit
            fetching_count = sum(1 for j in jobs if j.fetching)
            for j in jobs:
                if not j.done and not j.fetching and fetching_count < MAX_CONCURRENT_FETCHES:
                    j.fetch_tail()
                    fetching_count += 1

            # Poll completed fetches and redraw changed lines
            for i, j in enumerate(jobs):
                j.poll_tail()
                _place_cursor(job_row(i), 0)
                _clear_line()
                print(j.display_line(cols), end="", flush=True)

            # Status footer
            done_count = sum(1 for j in jobs if j.done)
            fetching_now = sum(1 for j in jobs if j.fetching)
            _place_cursor(rows - 1, 0)
            _clear_line()
            print(
                f"-- {done_count:2d}/{n:2d} done  |  {fetching_now:2d} fetching  |  "
                f"refresh in 10s (Ctrl-C to quit) --",
                end="", flush=True,
            )

            time.sleep(10)

            # Re-query condor_q
            updated = {job_id(j): j.get("JobStatus") for j in query_jobs(grep)}
            for j in jobs:
                new_status = updated.get(j.id)
                if new_status is None:
                    j.status = 4   # job has left the queue → treat as complete
                    j.done = True
                else:
                    j.status = new_status
                    j.done = new_status in (3, 4)

            if all(j.done for j in jobs):
                _place_cursor(rows, 0)
                print(f"\nAll {n} jobs done.")
                break

    except KeyboardInterrupt:
        _move_cursor_down(rows)
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    grep = sys.argv[1] if len(sys.argv) > 1 else ""
    run(grep)
