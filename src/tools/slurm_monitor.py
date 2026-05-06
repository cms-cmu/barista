#!/usr/bin/env python3
"""Monitor Slurm jobs with per-job progress and a workflow-level header.

For each running job, instead of a blind `tail -1`, the monitor reports:
  - chunks completed   (count of regex matches in the log; configurable)
  - age since last log write (stall detector)
  - log size
  - the most recent bracketed-timestamp line (skips per-event spam)

A header row shows snakemake workflow progress (`X of Y steps (Z%) done`)
parsed from the latest `.snakemake/log/*.snakemake.log`.

Usage:
    slurm_monitor.py [grep_string] [--progress-pattern REGEX]
                                   [--snakemake-log PATH]

Examples:
    slurm_monitor.py
    slurm_monitor.py FvT
    slurm_monitor.py --progress-pattern 'chunk \\d+/\\d+'
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Coffea/coffea4bees emit `Loading SvB friend tree` ~twice per chunk; the
# generic alternatives catch other processors. Override with --progress-pattern.
DEFAULT_PROGRESS_PATTERN = (
    r"Loading .* friend tree"
    r"|Processing chunk"
    r"|chunk \d+/\d+"
    r"|Loaded SvB"
)

# Bracketed timestamp emitted by coffea/rich logging, e.g. `[05/01/26 17:03:10]`.
_BRACKET_TS_RE = re.compile(r"^\[\d+/\d+/\d+ [\d:]+\][^\n]*", re.MULTILINE)

# Matches CSI/OSC ANSI escape sequences (tqdm/rich emit these on progress lines).
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*(?:\x07|\x1b\\)")

# Snakemake "N of M steps (P%) done"
_SNAKE_PROGRESS_RE = re.compile(r"(\d+)\s+of\s+(\d+)\s+steps\s+\((\d+)%\)\s+done")

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
    sys.stdout.write(f"\033[{row};{col}H")


def _clear_line():
    sys.stdout.write("\033[K")


def _clear_screen():
    sys.stdout.write("\033[2J\033[H")


# ---------------------------------------------------------------------------
# Slurm state mapping
# ---------------------------------------------------------------------------

STATE_CHAR = {
    "RUNNING":      "R",
    "PENDING":      "PD",
    "COMPLETING":   "CG",
    "CONFIGURING":  "CF",
    "SUSPENDED":    "S",
    "COMPLETED":    "C",
    "CANCELLED":    "X",
    "FAILED":       "F",
    "TIMEOUT":      "TO",
    "NODE_FAIL":    "NF",
    "PREEMPTED":    "PR",
    "BOOT_FAIL":    "BF",
    "DEADLINE":     "DL",
    "OUT_OF_MEMORY":"OOM",
}

DONE_STATES = {
    "COMPLETED", "CANCELLED", "FAILED", "TIMEOUT",
    "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "OUT_OF_MEMORY",
}


# ---------------------------------------------------------------------------
# squeue helpers
# ---------------------------------------------------------------------------

def get_current_user():
    return os.environ.get("USER") or os.popen("whoami").read().strip()


def _unwrap(v):
    if isinstance(v, dict) and "number" in v:
        return v["number"] if v.get("set") else None
    return v


def query_jobs(grep=""):
    user = get_current_user()
    try:
        out = subprocess.check_output(
            ["squeue", "-u", user, "--json"],
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        )
        data = json.loads(out) if out.strip() else {}
    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
        return []

    jobs = []
    for j in data.get("jobs", []):
        if grep:
            haystack = " ".join([
                str(j.get("name", "")),
                str(j.get("comment", "")),
                str(j.get("job_id", "")),
                str(j.get("partition", "")),
            ])
            if grep not in haystack:
                continue
        jobs.append(j)
    return jobs


# ---------------------------------------------------------------------------
# Snakemake workflow log
# ---------------------------------------------------------------------------

def find_snakemake_log(cwd=None):
    """Return the most-recently-modified .snakemake/log/*.snakemake.log, or None."""
    cwd = cwd or os.getcwd()
    pattern = os.path.join(cwd, ".snakemake", "log", "*.snakemake.log")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p))
    return candidates[-1]


def parse_snakemake_progress(path):
    """Return (done, total, pct, age_s) from the latest progress line, or None."""
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            try:
                f.seek(-8192, os.SEEK_END)
            except OSError:
                f.seek(0)
            tail = f.read().decode("utf-8", errors="replace")
        st = os.stat(path)
    except OSError:
        return None
    matches = list(_SNAKE_PROGRESS_RE.finditer(tail))
    if not matches:
        return None
    m = matches[-1]
    age = max(0, int(time.time() - st.st_mtime))
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), age


# ---------------------------------------------------------------------------
# Per-job state
# ---------------------------------------------------------------------------

def _fmt_runtime(seconds):
    seconds = int(seconds)
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


class SlurmJob:
    def __init__(self, job_dict):
        self.d = job_dict
        self.id = str(job_dict.get("job_id"))
        self.state = (job_dict.get("job_state") or ["UNKNOWN"])[0]
        self.name = job_dict.get("name", "") or ""
        self.comment = job_dict.get("comment", "") or ""
        self.label = self.comment or self.name
        self.partition = job_dict.get("partition", "") or ""
        self.out = job_dict.get("standard_output", "") or ""
        self.err = job_dict.get("standard_error", "") or ""
        self.start_time = _unwrap(job_dict.get("start_time"))
        self.mem_mb = _unwrap(job_dict.get("memory_per_node"))
        self.cpus = _unwrap(job_dict.get("cpus"))
        self.node = job_dict.get("batch_host") or job_dict.get("nodes") or ""
        self.restarts = job_dict.get("restart_cnt", 0) or 0
        self.state_reason = job_dict.get("state_reason", "") or ""
        self.done = self.state in DONE_STATES
        # Refreshed each cycle:
        self.size_kb = 0
        self.age_s = 0
        self.chunks = 0
        self.tail_line = ""

    @property
    def status_char(self):
        return STATE_CHAR.get(self.state, "?")

    @property
    def log_path(self):
        return self.out or self.err

    def _stats_str(self):
        parts = []
        if self.start_time:
            parts.append(_fmt_runtime(time.time() - self.start_time))
        if self.mem_mb:
            parts.append(f"{self.mem_mb}MB")
        if self.node:
            parts.append(str(self.node))
        if self.restarts:
            parts.append(f"[restart:{self.restarts}]")
        if self.state == "PENDING" and self.state_reason and self.state_reason != "None":
            parts.append(f"({self.state_reason})")
        return "  ".join(parts)

    def update(self, progress_re):
        """Refresh size / age / chunks / tail_line from the log file."""
        if self.done or self.state != "RUNNING":
            return
        path = self.log_path
        if not path or not os.path.isfile(path):
            return
        try:
            st = os.stat(path)
            self.size_kb = st.st_size // 1024
            self.age_s = max(0, int(time.time() - st.st_mtime))
            with open(path, "rb") as f:
                data = f.read().decode("utf-8", errors="replace")
        except OSError:
            return
        self.chunks = len(progress_re.findall(data))
        bracketed = _BRACKET_TS_RE.findall(data)
        if bracketed:
            self.tail_line = _ANSI_RE.sub("", bracketed[-1]).strip()
        else:
            for line in reversed(data.splitlines()):
                line = _ANSI_RE.sub("", line).strip()
                if line:
                    self.tail_line = line
                    break

    def display_line(self, cols):
        label_str = f"[{self.label}]" if self.label else ""
        if self.state == "RUNNING":
            bits = []
            if self.chunks:
                bits.append(f"{self.chunks:>3d}ch")
            stale = self.age_s > 120
            age_str = f"{self.age_s:>3d}s"
            if stale:
                age_str = f"\033[31m{age_str}\033[0m"
            bits.append(f"{age_str} ago")
            bits.append(f"{self.size_kb:>4d}KB")
            if self.start_time:
                bits.append(_fmt_runtime(time.time() - self.start_time))
            info = "  ".join(bits)
            if self.tail_line:
                info = f"{info}  ·  {self.tail_line}"
        elif self.state == "PENDING":
            info = self._stats_str()
        else:
            info = self.state
        line = f"{self.id:>8s} {self.status_char:<3s} {label_str:<40s} {info}"
        # Truncate using visible-width by stripping ANSI for length-check
        visible = _ANSI_RE.sub("", line)
        if len(visible) > cols - 1:
            # crude truncation; ok as long as ANSI is at end of info
            line = visible[:cols - 1]
        return line


# ---------------------------------------------------------------------------
# Main display loop
# ---------------------------------------------------------------------------

def _render_workflow_header(snakemake_log, cols):
    """Single-line header summarizing snakemake progress."""
    if not snakemake_log:
        return "snakemake: (no .snakemake/log found)"
    prog = parse_snakemake_progress(snakemake_log)
    name = os.path.basename(snakemake_log)
    if prog is None:
        return f"snakemake [{name}]: (no progress lines yet)"
    done, total, pct, age = prog
    bar_w = max(10, min(40, cols - 60))
    filled = int(bar_w * done / total) if total else 0
    bar = "█" * filled + "░" * (bar_w - filled)
    return f"snakemake [{name}]: {bar} {done}/{total} ({pct}%)  · last update {age}s ago"


def run(grep="", progress_pattern=DEFAULT_PROGRESS_PATTERN, snakemake_log=None):
    progress_re = re.compile(progress_pattern)
    if snakemake_log is None:
        snakemake_log = find_snakemake_log()

    rows, cols = _terminal_size()

    jobs = [SlurmJob(j) for j in query_jobs(grep)]
    if not jobs:
        print("No jobs found.")
        return

    max_display = rows - 6
    if len(jobs) > max_display:
        print(f"WARNING: {len(jobs)} jobs but only {max_display} rows; showing first {max_display}.")
        jobs = jobs[:max_display]

    n = len(jobs)

    # Layout (top-anchored):
    #   row 1:      workflow header
    #   row 2:      column header
    #   row 3:      separator
    #   row 4..3+n: job rows
    #   row 4+n:    separator
    #   row 5+n:    footer
    def job_row(i):
        return 4 + i

    sep = "-" * min(cols, 100)

    _clear_screen()
    _place_cursor(1, 0)
    print(_render_workflow_header(snakemake_log, cols))
    _place_cursor(2, 0)
    print(f"{'ID':>8s} {'S':<3s} {'Name/Comment':<40s} progress")
    _place_cursor(3, 0)
    print(sep)
    _place_cursor(4 + n, 0)
    print(sep)

    try:
        while True:
            rows, cols = _terminal_size()

            for j in jobs:
                j.update(progress_re)

            # workflow header
            _place_cursor(1, 0)
            _clear_line()
            print(_render_workflow_header(snakemake_log, cols), end="", flush=True)

            for i, j in enumerate(jobs):
                _place_cursor(job_row(i), 0)
                _clear_line()
                print(j.display_line(cols), end="", flush=True)

            done_count = sum(1 for j in jobs if j.done)
            stalled = sum(1 for j in jobs if j.state == "RUNNING" and j.age_s > 120)
            _place_cursor(5 + n, 0)
            _clear_line()
            footer = (
                f"-- {done_count:2d}/{n:2d} done  |  "
                f"{stalled} stalled (>120s)  |  "
                f"refresh 2s (Ctrl-C to quit) --"
            )
            print(footer, end="", flush=True)

            time.sleep(2)

            # Re-query squeue; jobs that leave the queue are treated as complete.
            updated = {}
            for j in query_jobs(grep):
                jid = str(j.get("job_id"))
                state = (j.get("job_state") or ["UNKNOWN"])[0]
                updated[jid] = state
            for j in jobs:
                new_state = updated.get(j.id)
                if new_state is None:
                    j.state = "COMPLETED"
                    j.done = True
                else:
                    j.state = new_state
                    j.done = new_state in DONE_STATES

            if all(j.done for j in jobs):
                _place_cursor(6 + n, 0)
                print(f"All {n} jobs done.")
                break

    except KeyboardInterrupt:
        _place_cursor(6 + n, 0)
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Monitor Slurm jobs with chunk-level progress and snakemake header.",
    )
    p.add_argument("grep", nargs="?", default="",
                   help="filter substring (matched against name/comment/id/partition)")
    p.add_argument("--progress-pattern", default=DEFAULT_PROGRESS_PATTERN,
                   help="regex; matches in the log are counted as 'chunks'")
    p.add_argument("--snakemake-log", default=None,
                   help="path to snakemake .log (default: auto-detect latest in ./.snakemake/log/)")
    args = p.parse_args()
    run(grep=args.grep,
        progress_pattern=args.progress_pattern,
        snakemake_log=args.snakemake_log)


if __name__ == "__main__":
    main()
