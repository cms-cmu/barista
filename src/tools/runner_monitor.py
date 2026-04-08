#!/usr/bin/env python3
"""Monitor Dask scheduler progress across multiple parallel runner.py jobs.

Discovers dashboard URLs (or TCP scheduler addresses) from Snakemake log
files, then periodically queries each scheduler's Prometheus metrics endpoint
for task progress.

For jobs started before the dashboard URL was logged, falls back to
connecting via the TCP scheduler address using dask.distributed.Client.

Usage:
    runner_monitor.py [log_dir_or_glob]

Examples:
    runner_monitor.py                                              # searches output/ recursively
    runner_monitor.py output/Run3_FeynNet/feynnet_friendtrees/logs/
"""

import glob
import os
import re
import sys
import time
import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# Terminal helpers (shared style with condor_monitor.py)
# ---------------------------------------------------------------------------

def _terminal_size():
    try:
        rows, cols = os.popen("stty size", "r").read().split()
        return int(rows), int(cols)
    except Exception:
        return 40, 120


def _clear_line():
    sys.stdout.write("\033[K")


# ---------------------------------------------------------------------------
# Log file discovery
# ---------------------------------------------------------------------------

DASHBOARD_RE  = re.compile(r"Dask dashboard:\s+(http://\S+)")
PROXY_RE      = re.compile(r"Dask dashboard:\s+/proxy/(\d+)")  # /proxy/PORT/status
SCHEDULER_RE  = re.compile(r"'tcp://([^']+)'")   # matches tcp://host:port inside Client repr
COMPLETE_RE   = re.compile(r"JOB EXECUTION COMPLETED SUCCESSFULLY|Dask performance report saved")
SMKLOG_RE     = re.compile(r"^\s+log:\s+(\S+\.log)")


def scan_logs(search_root):
    """
    Walk log files and extract dashboard URLs, scheduler TCP addresses,
    and completion status.

    Returns:
        {job_name: {'dashboard': str, 'scheduler': str, 'done': bool}}
        Any key may be missing if not found in the log.
    """
    if os.path.isdir(search_root):
        pattern = os.path.join(search_root, "**", "*.log")
    else:
        pattern = search_root

    jobs = {}
    for path in sorted(glob.glob(pattern, recursive=True)):
        name = os.path.basename(path).replace(".log", "")
        info = {}
        try:
            with open(path) as f:
                for line in f:
                    if not info.get('dashboard'):
                        m = DASHBOARD_RE.search(line)
                        if m:
                            info['dashboard'] = m.group(1).rstrip("/status").rstrip("/")
                        else:
                            m = PROXY_RE.search(line)
                            if m:
                                info['proxy_port'] = m.group(1)
                    if not info.get('scheduler'):
                        m = SCHEDULER_RE.search(line)
                        if m:
                            info['scheduler'] = f"tcp://{m.group(1)}"
                    if COMPLETE_RE.search(line):
                        info['done'] = True
        except OSError:
            pass
        if info:
            jobs[name] = info
    return jobs


# ---------------------------------------------------------------------------
# Dashboard URL resolution via TCP scheduler (fallback)
# ---------------------------------------------------------------------------

_dashboard_cache = {}   # scheduler_addr -> dashboard_url


def resolve_dashboard_via_scheduler(scheduler_addr, timeout=3):
    """Connect to a running Dask scheduler and retrieve its dashboard URL."""
    if scheduler_addr in _dashboard_cache:
        return _dashboard_cache[scheduler_addr]
    try:
        from distributed import Client
        c = Client(scheduler_addr, timeout=timeout, set_as_default=False)
        url = c.dashboard_link.rstrip("/status").rstrip("/")
        c.close()
        _dashboard_cache[scheduler_addr] = url
        return url
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Prometheus metrics parsing
# ---------------------------------------------------------------------------

TASK_METRIC_RE   = re.compile(r'dask_scheduler_tasks\{state="(\w+)"\}\s+([\d.]+)')
WORKER_METRIC_RE = re.compile(r'dask_scheduler_workers\{state="(\w+)"\}\s+([\d.]+)')


def query_metrics(base_url, timeout=2):
    """
    Fetch /metrics from the Dask dashboard and return task/worker counts.

    Returns dict like:
        {'processing': 25, 'waiting': 7, 'memory': 77, 'erred': 0,
         'workers': 36, 'workers_busy': 11}
    or None if unreachable.
    """
    url = f"{base_url}/metrics"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError):
        return None

    counts = {}
    for m in TASK_METRIC_RE.finditer(text):
        counts[m.group(1)] = int(float(m.group(2)))

    worker_states = {}
    for m in WORKER_METRIC_RE.finditer(text):
        worker_states[m.group(1)] = int(float(m.group(2)))
    if worker_states:
        counts['workers'] = sum(worker_states.values())
        counts['workers_busy'] = (worker_states.get('partially_saturated', 0)
                                  + worker_states.get('saturated', 0))

    return counts if counts else None


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _fmt_counts(done, counts, pending=False):
    if pending:
        return "\033[90mpending\033[0m"
    if done:
        return "\033[32mCOMPLETED\033[0m"
    if counts is None:
        return "starting / unreachable"
    parts = []
    for state in ("processing", "waiting", "memory", "erred"):
        v = counts.get(state, 0)
        if state == "erred" and v > 0:
            parts.append(f"\033[31merred={v}\033[0m")
        elif v > 0:
            parts.append(f"{state}={v}")
    w = counts.get("workers", "?")
    busy = counts.get("workers_busy", 0)
    worker_str = f"workers={w}({busy} busy)"
    summary = "  ".join(parts) if parts else "idle"
    return f"{worker_str}  {summary}"


def display(job_state, cols):
    header = f"{'Job':<50s}  {'Progress'}"
    sep = "-" * min(cols, 110)
    print(header)
    print(sep)
    for name, (dashboard_url, done, counts, pending) in sorted(job_state.items()):
        progress = _fmt_counts(done, counts, pending)
        line = f"{name:<50s}  {progress}"
        if len(line) > cols - 1:
            line = line[:cols - 1]
        print(line)
    print(sep)


# ---------------------------------------------------------------------------
# Expected jobs from Snakemake log
# ---------------------------------------------------------------------------

def expected_log_files(snakemake_log_dir=".snakemake/log"):
    """
    Parse the most recent Snakemake log file and return the set of
    log file paths declared for all submitted jobs.
    """
    logs = sorted(glob.glob(os.path.join(snakemake_log_dir, "*.snakemake.log")))
    if not logs:
        return set()
    paths = set()
    try:
        with open(logs[-1]) as f:
            for line in f:
                m = SMKLOG_RE.match(line)
                if m:
                    paths.add(m.group(1))
    except OSError:
        pass
    return paths


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(search_root="output/"):
    print(f"Searching for log files under: {search_root}")
    print("Resolving dashboard URLs from running schedulers (may take a moment)...")
    print("Press Ctrl-C to quit.\n")

    job_state = {}  # {name: (dashboard_url, counts)}

    try:
        while True:
            rows, cols = _terminal_size()

            jobs = scan_logs(search_root)

            # Add pending jobs (in Snakemake plan but no log file yet)
            for log_path in expected_log_files():
                name = os.path.basename(log_path).replace(".log", "")
                if name not in jobs and not os.path.exists(log_path):
                    jobs[name] = {'pending': True}

            for name, info in jobs.items():
                if info.get('pending'):
                    job_state[name] = (None, False, None, True)
                    continue

                done = info.get('done', False)
                dashboard_url = info.get('dashboard')

                # Proxy URL (/proxy/PORT): reconstruct direct URL from scheduler host
                if not dashboard_url and info.get('proxy_port') and info.get('scheduler'):
                    host = info['scheduler'].split("://")[1].split(":")[0]
                    dashboard_url = f"http://{host}:{info['proxy_port']}"

                # Fallback: connect to TCP scheduler to get dashboard URL
                if not dashboard_url and not done and info.get('scheduler'):
                    dashboard_url = resolve_dashboard_via_scheduler(info['scheduler'])

                counts = query_metrics(dashboard_url) if (dashboard_url and not done) else None
                job_state[name] = (dashboard_url, done, counts, False)

            os.system("clear")
            n_done    = sum(1 for _, done, _, _ in job_state.values() if done)
            n_pending = sum(1 for _, _, _, pending in job_state.values() if pending)
            n_running = len(job_state) - n_done - n_pending
            print(f"Dask job monitor  —  {time.strftime('%H:%M:%S')}  "
                  f"({len(job_state)} jobs: {n_done} done, {n_running} running, "
                  f"{n_pending} pending — refresh every 10s, Ctrl-C to quit)")
            print()

            if job_state:
                display(job_state, cols)
            else:
                print("No jobs found yet — waiting for Snakemake to start...")

            time.sleep(10)

    except KeyboardInterrupt:
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "output/"
    run(root)
