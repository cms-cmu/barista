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
import subprocess
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

DASHBOARD_RE    = re.compile(r"Dask dashboard:\s+(http://\S+)")
PROXY_RE        = re.compile(r"Dask dashboard:\s+/proxy/(\d+)")  # /proxy/PORT/status
SCHEDULER_RE    = re.compile(r"'tcp://([^']+)'")   # matches tcp://host:port inside Client repr
SCHED_HOST_RE   = re.compile(r"Dask scheduler host:\s+(\S+)")  # explicit host logged by runner.py
# rich's log formatter right-aligns a source locator like "runner.py:500" and
# wraps a long message's value onto the next line.  Used to reject that token
# and to recognise a bare hostname on the wrapped continuation line.
_SRC_LOC_RE     = re.compile(r"^\w+\.py:\d+$")
_HOSTNAME_RE    = re.compile(r"^[A-Za-z0-9][\w.\-]*$")  # plausible bare hostname token
COMPLETE_RE     = re.compile(r"JOB EXECUTION COMPLETED SUCCESSFULLY|Dask performance report saved")
SMKLOG_RE       = re.compile(r"^\s+log:\s+(\S+\.log)")
WORKER_LOG_DIR_RE = re.compile(r"Condor worker log directory: (\S+)")


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
        pending_host = False  # saw "Dask scheduler host:" whose value wrapped
        try:
            with open(path) as f:
                for line in f:
                    m = DASHBOARD_RE.search(line)
                    if m:
                        info['dashboard'] = m.group(1).rstrip("/status").rstrip("/")
                        info.pop('proxy_port', None)
                        info.pop('done', None)  # new run started — clear stale completion
                    else:
                        m = PROXY_RE.search(line)
                        if m:
                            info['proxy_port'] = m.group(1)
                            info.pop('done', None)  # new run started — clear stale completion
                    m = SCHEDULER_RE.search(line)
                    if m:
                        info['scheduler'] = f"tcp://{m.group(1)}"
                    m = SCHED_HOST_RE.search(line)
                    if m:
                        val = m.group(1)
                        if _SRC_LOC_RE.match(val):
                            # rich wrapped the hostname onto the next line; the
                            # token captured here is the "runner.py:NNN" source
                            # locator.  Grab the real host from the next line.
                            pending_host = True
                        else:
                            info['scheduler_host'] = val
                            pending_host = False
                    elif pending_host:
                        tok = line.split()
                        if tok and _HOSTNAME_RE.match(tok[0]) and not _SRC_LOC_RE.match(tok[0]):
                            info['scheduler_host'] = tok[0]
                        pending_host = False
                    m = WORKER_LOG_DIR_RE.search(line)
                    if m:
                        info['worker_log_dir'] = m.group(1)
                    if COMPLETE_RE.search(line):
                        info['done'] = True
        except OSError:
            pass
        if info:
            info['log_path'] = os.path.abspath(path)
            jobs[name] = info
    return jobs


def dashboard_url_from_info(info):
    """Build a Dask dashboard base URL from a scan_logs info dict, or None.

    Resolution order for the host of a /proxy/PORT dashboard:
      1. `scheduler_host` — explicit hostname logged by runner.py (works
         cross-node; preferred).
      2. host parsed from a `tcp://host:port` scheduler repr, if present.
      3. `localhost` — last resort; only correct when the monitor runs on the
         same node as the scheduler.

    A full `dashboard` URL (older non-proxy logs) is returned as-is.
    """
    url = info.get('dashboard')
    if url:
        return url
    port = info.get('proxy_port')
    if not port:
        return None
    host = info.get('scheduler_host')
    if not host and info.get('scheduler'):
        host = info['scheduler'].split('://')[1].split(':')[0]
    if not host:
        host = 'localhost'
    return f"http://{host}:{port}"


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


def _parse_metrics(text):
    """Parse Prometheus /metrics text into task/worker counts, or None."""
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
        counts['workers_paused'] = worker_states.get('paused', 0)
    return counts if counts else None


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
    return _parse_metrics(text)


# SSH ControlMaster socket path — %h is expanded by SSH to the remote hostname,
# so each host gets its own persistent socket at /tmp/barista_ssh_ctl_<host>.
_SSH_CTL_FMT = "/tmp/barista_ssh_ctl_%h"


def _classify_ssh_failure(stderr):
    """Map ssh stderr to a short, human-friendly reason (or None if it doesn't
    look like an ssh-level failure)."""
    e = (stderr or "").lower()
    if ("permission denied" in e or "gssapi" in e or "no kerberos" in e
            or "credentials cache" in e or "not found in keytab" in e):
        return "ssh auth — kinit?"
    if "control" in e and ("socket" in e or "path" in e or "multiplex" in e):
        return "ssh socket — stale"
    if ("connection refused" in e or "connection timed out" in e
            or "no route to host" in e or "connect to host" in e
            or "operation timed out" in e or "name or service not known" in e):
        return "ssh no route"
    return None


def query_metrics_remote_diag(base_url, timeout=5):
    """Like query_metrics_remote but also returns a diagnostic reason.

    Returns ``(counts, reason)`` where ``counts`` is the metrics dict (or None
    on failure) and ``reason`` is None on success or a short string explaining
    why the probe failed, e.g. ``"ssh auth — kinit?"``, ``"ssh no route"``,
    ``"ssh socket — stale"``, ``"dashboard down"``, ``"bad url"``.

    Resolution: try direct HTTP first; if unreachable, fall back to SSH (which
    is what reveals *why* — direct HTTP can't distinguish a firewall from a
    dead dashboard, but the ssh attempt's stderr usually can).
    """
    counts = query_metrics(base_url, timeout=2)
    if counts is not None:
        return counts, None

    host_match = re.match(r'https?://([^/:]+):(\d+)', base_url)
    if not host_match:
        return None, "bad url"
    host, port = host_match.group(1), host_match.group(2)

    try:
        result = subprocess.run(
            [
                "ssh",
                "-o", "ControlMaster=auto",
                "-o", f"ControlPath={_SSH_CTL_FMT}",
                "-o", "ControlPersist=120",
                "-o", "StrictHostKeyChecking=no",
                "-o", "BatchMode=yes",
                "-o", f"ConnectTimeout={timeout}",
                host,
                f"curl -sf --max-time {timeout} http://localhost:{port}/metrics",
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 3,
        )
    except subprocess.TimeoutExpired:
        return None, "ssh timeout"
    except OSError:
        return None, "ssh error"

    # returncode 255 (or negative/signal) is an ssh-level failure; any other
    # non-zero code is the *remote* command (curl) failing, i.e. the dashboard
    # itself is unreachable on the scheduler node.
    if result.returncode == 255 or result.returncode < 0:
        return None, (_classify_ssh_failure(result.stderr) or "ssh failed")
    if result.returncode != 0 or not result.stdout:
        return None, "dashboard down"

    counts = _parse_metrics(result.stdout)
    return (counts, None) if counts else (None, "dashboard down")


def query_metrics_remote(base_url, timeout=5):
    """Like query_metrics but falls back to SSH when direct HTTP is unreachable.

    On the first call to a given host, SSH opens a connection and keeps it
    alive for 120 s (ControlPersist). Subsequent calls reuse that socket so
    the SSH overhead is ~0 ms after the first query.

    Requires: ssh access to worker nodes and curl installed there.
    """
    return query_metrics_remote_diag(base_url, timeout=timeout)[0]


# ---------------------------------------------------------------------------
# HTCondor worker counts (matched to Dask jobs via worker log directory)
# ---------------------------------------------------------------------------

_CONDOR_STATUS = {1: 'idle', 2: 'running', 5: 'held'}


def condor_counts_for_jobs(scanned):
    """Return HTCondor worker counts per job.

    Matches Condor jobs to Dask jobs via the 'worker_log_dir' field that
    runner.py logs.  Returns::

        {job_name: {'idle': N, 'running': N, 'held': N}}

    Jobs without a worker_log_dir (old logs, non-Dask jobs) are omitted.
    """
    import json

    # Build lookup: worker_log_dir -> job_name
    dir_to_job = {}
    for name, info in scanned.items():
        wld = info.get('worker_log_dir')
        if wld:
            dir_to_job[wld.rstrip('/')] = name

    if not dir_to_job:
        return {}

    try:
        out = subprocess.check_output(
            'condor_q -json',
            shell=True,
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
            timeout=5,
        )
        all_jobs = json.loads(out) if out.strip() else []
    except Exception:
        return {}

    counts = {}  # {job_name: {'idle': 0, 'running': 0, 'held': 0}}

    for j in all_jobs:
        status = j.get('JobStatus', 0)
        key = _CONDOR_STATUS.get(status)
        if key is None:
            continue
        out_path = j.get('Out', '') or j.get('Err', '')
        # Match by checking if the job's output file lives in a known worker_log_dir
        for wld, job_name in dir_to_job.items():
            if out_path.startswith(wld):
                entry = counts.setdefault(job_name, {'idle': 0, 'running': 0, 'held': 0})
                entry[key] += 1
                break

    return counts


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

BAR_WIDTH = 30

def _progress_bar(counts):
    """
    Return a coloured progress bar string.
      green  \033[32m=\033[0m  memory    (finished, held in worker memory)
      cyan   \033[36m-\033[0m  processing (actively computing)
      red    \033[31mx\033[0m  erred
      grey   \033[90m.\033[0m  waiting
    """
    mem  = counts.get("memory",     0)
    proc = counts.get("processing", 0)
    err  = counts.get("erred",      0)
    wait = counts.get("waiting",    0)
    total = mem + proc + err + wait
    if total == 0:
        return f"|\033[90m{'.' * BAR_WIDTH}\033[0m| Mem:  -%  Mem+Run:  -%"

    def _cells(n):
        return max(0, int(round(n / total * BAR_WIDTH)))

    n_mem  = _cells(mem)
    n_proc = _cells(proc)
    n_err  = _cells(err)
    n_wait = BAR_WIDTH - n_mem - n_proc - n_err
    n_wait = max(0, n_wait)

    bar = (
        f"\033[32m{'=' * n_mem}\033[0m"
        f"\033[36m{'-' * n_proc}\033[0m"
        f"\033[31m{'x' * n_err}\033[0m"
        f"\033[90m{'.' * n_wait}\033[0m"
    )
    pct_mem     = int(round(mem / total * 100))
    pct_mem_run = int(round((mem + proc) / total * 100))
    return f"|{bar}| Mem:{pct_mem:3d}%  Mem+Run:{pct_mem_run:3d}%"


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
    worker_str = f"workers={w}"
    summary = "  ".join(parts) if parts else "idle"
    bar = _progress_bar(counts)
    return f"{bar}  {worker_str}  {summary}"


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

def snakemake_run_start_time(snakemake_log_dir=".snakemake/log"):
    """Return the mtime of the most recent Snakemake log file (= run start time)."""
    logs = sorted(glob.glob(os.path.join(snakemake_log_dir, "*.snakemake.log")))
    if not logs:
        return None
    return os.path.getmtime(logs[-1])


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
                    paths.add(os.path.abspath(m.group(1)))
    except OSError:
        pass
    return paths


# ---------------------------------------------------------------------------
# snkmt database (authoritative job status)
# ---------------------------------------------------------------------------

SNKMT_DB = os.path.expanduser("~/.local/share/snkmt/snkmt.db")


def query_snkmt(db_path=SNKMT_DB):
    """
    Query the snkmt SQLite database for job statuses in the most recent workflow.

    Returns {log_path: status} where status is e.g. 'SUCCESS' or 'RUNNING'.
    Returns an empty dict if snkmt is unavailable or the DB can't be read.
    """
    if not os.path.exists(db_path):
        return {}, None, set(), {}
    try:
        import sqlite3
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        # Prefer the actively running workflow; fall back to most recent.
        row = con.execute(
            "SELECT id, status FROM workflows WHERE status='RUNNING'"
            " ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        if not row:
            row = con.execute(
                "SELECT id, status FROM workflows ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
        if not row:
            con.close()
            return {}, None
        wf_id, wf_status = row[0], row[1]
        # Job statuses for the current workflow
        rows = con.execute(
            """SELECT f.path, j.status
               FROM jobs j JOIN files f ON f.job_id = j.id
               WHERE j.workflow_id = ? AND f.file_type = 'LOG'""",
            (wf_id,),
        ).fetchall()
        current_job_status = {path: status for path, status in rows}

        # Rule names planned in the current workflow
        current_rules = {r[0] for r in con.execute(
            "SELECT name FROM rules WHERE workflow_id = ?", (wf_id,)
        ).fetchall()}

        # For log paths NOT yet in the current workflow, look up the rule name
        # from any previous workflow so we can decide if the job is pending here.
        hist_rows = con.execute(
            """SELECT DISTINCT f.path, r.name
               FROM jobs j JOIN files f ON f.job_id = j.id
               JOIN rules r ON j.rule_id = r.id
               WHERE j.workflow_id != ? AND f.file_type = 'LOG'""",
            (wf_id,),
        ).fetchall()
        historical_rule = {path: rule for path, rule in hist_rows}

        con.close()
        return current_job_status, wf_status, current_rules, historical_rule
    except Exception:
        return {}, None, set(), {}


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

            # Authoritative job status from snkmt DB; fall back to mtime heuristic.
            snkmt_jobs, wf_status, current_rules, historical_rule = query_snkmt()
            planned_logs = expected_log_files()
            run_start    = snakemake_run_start_time()

            # Add pending jobs (in Snakemake plan but no log file yet)
            for log_path in planned_logs:
                name = os.path.basename(log_path).replace(".log", "")
                if name not in jobs and not os.path.exists(log_path):
                    jobs[name] = {'pending': True, 'log_path': log_path}

            for name, info in jobs.items():
                if info.get('pending'):
                    job_state[name] = (None, False, None, True)
                    continue

                log_path     = info.get('log_path', '')
                snkmt_status = snkmt_jobs.get(log_path)  # None if not in current workflow
                done         = info.get('done', False)

                if snkmt_jobs:
                    # snkmt available — use it as ground truth
                    if snkmt_status == 'RUNNING':
                        done = False          # job restarted; log completion is stale
                    elif snkmt_status == 'SUCCESS':
                        done = True           # definitively finished in this run
                    elif snkmt_status is None and done:
                        # Not yet submitted — check if it belongs to the current workflow.
                        # It does if: (a) Snakemake has already logged it (planned_logs), or
                        # (b) it ran under a different workflow whose rule is in current_rules.
                        hist = historical_rule.get(log_path)
                        if log_path in planned_logs or (hist and hist in current_rules):
                            job_state[name] = (None, False, None, True)  # pending
                        # else: done in a different workflow, not part of current run — omit
                        continue
                elif done and run_start and log_path in planned_logs:
                    # snkmt unavailable — fall back to mtime heuristic
                    if os.path.getmtime(log_path) < run_start:
                        done = False
                        job_state[name] = (None, False, None, True)
                        continue

                # Proxy URL (/proxy/PORT): reconstruct direct URL using the
                # logged scheduler host (cross-node), tcp:// host, or localhost.
                dashboard_url = dashboard_url_from_info(info)

                # Fallback: connect to TCP scheduler to get dashboard URL
                if not dashboard_url and not done and info.get('scheduler'):
                    dashboard_url = resolve_dashboard_via_scheduler(info['scheduler'])

                counts = query_metrics_remote(dashboard_url) if (dashboard_url and not done) else None
                job_state[name] = (dashboard_url, done, counts, False)

            os.system("clear")
            n_done    = sum(1 for _, done, _, _ in job_state.values() if done)
            n_pending = sum(1 for _, _, _, pending in job_state.values() if pending)
            n_running = len(job_state) - n_done - n_pending
            print(f"Dask job monitor  —  {time.strftime('%H:%M:%S')}  "
                  f"({len(job_state)} jobs: {n_done} done, {n_running} running, "
                  f"{n_pending} pending — refresh every 1s, Ctrl-C to quit)")
            print()

            if job_state:
                display(job_state, cols)
            else:
                print("No jobs found yet — waiting for Snakemake to start...")

            time.sleep(1)

    except KeyboardInterrupt:
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "output/"
    run(root)
