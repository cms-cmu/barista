#!/usr/bin/env python3
"""Barista workflow console — snkmt style, with live Dask job progress.

Replaces the "Workflow Info" panel with a live Dask task-progress panel
showing per-job progress bars for all running jobs in the selected workflow.

Usage (from barista root, inside pixi snakemake env):
    pixi run python src/tools/barista_console.py
    pixi run python src/tools/barista_console.py output/Run3_MvD/logs/
"""

import os
import sqlite3
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional, Union
from uuid import UUID

from rich.markup import escape
from textual import on, work
from textual.app import App, ComposeResult
from textual.command import CommandPalette
from textual.containers import Container, Horizontal, VerticalScroll
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Footer, Input, Label, ListView, Select, Static

# snkmt components we reuse as-is
from snkmt.console.command import DatabaseSourceProvider, SelectDatabaseCommand
from snkmt.console.widgets import (
    RuleTable,
    StyledStatus,
    WorkflowErrors,
    WorkflowTable,
)
from snkmt.core.db.session import AsyncDatabase
from snkmt.core.repository import WorkflowRepository
from snkmt.types.dto import WorkflowDTO
from snkmt.types.enums import DateFilter, Status
from snkmt.version import VERSION

# Monitoring logic from runner_monitor
sys.path.insert(0, str(Path(__file__).parent))
from runner_monitor import (  # noqa: E402
    SNKMT_DB,
    condor_counts_for_jobs,
    dashboard_url_from_info,
    query_metrics_remote,
    scan_logs,
)

# ---------------------------------------------------------------------------
# Progress bar (Rich markup version)
# ---------------------------------------------------------------------------

BAR_WIDTH = 30


def _rich_bar(counts: dict) -> str:
    """Return a Rich-markup coloured progress bar."""
    mem   = counts.get("memory",     0)
    proc  = counts.get("processing", 0)
    err   = counts.get("erred",      0)
    wait  = counts.get("waiting",    0)
    total = mem + proc + err + wait
    if total == 0:
        return f"[dim]{'·' * BAR_WIDTH}[/dim]  [dim]Mem:  -%  Mem+Run:  -%[/dim]"

    def cells(n: int) -> int:
        return max(0, int(round(n / total * BAR_WIDTH)))

    n_mem  = cells(mem)
    n_proc = cells(proc)
    n_err  = cells(err)
    n_wait = max(0, BAR_WIDTH - n_mem - n_proc - n_err)

    bar = (
        f"[green]{'=' * n_mem}[/green]"
        f"[cyan]{'-' * n_proc}[/cyan]"
        f"[red]{'x' * n_err}[/red]"
        f"[dim]{'.' * n_wait}[/dim]"
    )
    pct_mem = int(round(mem / total * 100))
    pct_run = int(round((mem + proc) / total * 100))
    workers = counts.get("workers", "?")
    return (
        f"|{bar}| "
        f"Mem:{pct_mem:3d}%  Mem+Run:{pct_run:3d}%  "
        f"[dim]workers={workers}  tasks={total}[/dim]"
    )


# ---------------------------------------------------------------------------
# Throughput / ETA helpers
# ---------------------------------------------------------------------------

_HISTORY_MAXLEN = 60  # keep up to 60 samples (~60 s at 1 s poll rate)

# Version marker — bump this whenever the bar-caching behaviour changes so
# users can verify which copy of the file is actually running on each host.
# Surfaced in the startup banner and logged once per session.
DASK_PANEL_REV = "2026-05-20-f"  # DRY stale-mark helper; dropped _stale_carry


def _throughput_eta(history: deque, counts: dict):
    """Return (rate_tasks_per_sec, eta_seconds) from a history deque.

    history entries are (timestamp, memory_count).
    Returns (None, None) when there is not enough data.

    Uses the oldest history entry where memory was lower than the current
    value, so temporary plateaus don't reset the rate to zero.
    """
    if len(history) < 2 or not counts:
        return None, None
    t_new, m_new = history[-1]
    # Find the oldest entry where memory was strictly lower (best long-run rate)
    t_base = m_base = None
    for t_old, m_old in history:
        if m_old < m_new:
            t_base, m_base = t_old, m_old
            break
    if t_base is None:
        return None, None  # memory hasn't grown at all in the history window
    dt = t_new - t_base
    if dt < 1:
        return None, None
    rate = (m_new - m_base) / dt
    mem  = counts.get("memory",     0)
    proc = counts.get("processing", 0)
    wait = counts.get("waiting",    0)
    remaining = proc + wait
    eta = remaining / rate if remaining > 0 else 0.0
    return rate, eta


def _fmt_eta(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


# ---------------------------------------------------------------------------
# snkmt DB helpers (synchronous, for use in worker threads)
# ---------------------------------------------------------------------------

def _running_job_logs(wf_id: str, db_path: str = SNKMT_DB) -> dict[str, str]:
    """Return {log_path: status} for all jobs in the given workflow.
    Raises on DB/FS errors so callers can distinguish a real "no jobs"
    result ({}) from a transient outage; previously this swallowed
    exceptions and returned {}, which made the console's progress
    display flicker any time snkmt/NFS hiccuped."""
    # snkmt stores UUIDs without hyphens; row keys from WorkflowTable use str(uuid) with hyphens
    wf_id_bare = wf_id.replace("-", "")
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            """SELECT f.path, j.status
               FROM jobs j JOIN files f ON f.job_id = j.id
               WHERE j.workflow_id = ? AND f.file_type = 'LOG'""",
            (wf_id_bare,),
        ).fetchall()
    finally:
        con.close()
    return {path: status for path, status in rows}


# ---------------------------------------------------------------------------
# Dask job progress panel (replaces WorkflowDetailOverview)
# ---------------------------------------------------------------------------

class DaskJobPanel(VerticalScroll):
    """Live Dask task-progress for running jobs in the selected workflow."""

    workflow_id: reactive[str | None] = reactive(None)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._metrics: dict = {}   # {job_name: {'counts': dict|None, 'status': str}}
        self._history: dict[str, deque] = {}   # {job_name: deque[(ts, memory)]}
        self._mounted = False

    def compose(self) -> ComposeResult:
        yield Static(
            "[dim]Select a workflow to view running jobs.[/dim]",
            id="dask-content",
            markup=True,
        )

    def on_mount(self) -> None:
        self._mounted = True
        self.set_interval(1, self._poll)

    def watch_workflow_id(self) -> None:
        self._metrics = {}
        self._history = {}
        self._refresh_display()

    def _mark_stale_all(self, now: float) -> None:
        """Mark every cached entry with counts as stale-since-NOW so the
        renderer keeps the bar on screen with an `unreachable Xs`
        annotation.  Used when a wider failure (snkmt DB exception, empty
        job_logs) prevents us from refreshing any single job — we never
        wipe cached data, just annotate it as not-currently-confirmable."""
        for info in self._metrics.values():
            if info.get("counts") is not None and not info.get("stale_since"):
                info["stale_since"] = now

    @work(exclusive=True, thread=True, exit_on_error=False)
    def _poll(self) -> None:
        if self.workflow_id is None:
            return

        try:
            try:
                job_logs = _running_job_logs(self.workflow_id)
            except Exception:
                # snkmt DB read failed transiently (NFS blip, lock contention,
                # etc.).  Don't wipe _metrics — mark existing entries stale
                # and let the renderer keep cached bars on screen.
                self._mark_stale_all(time.monotonic())
                self.app.call_from_thread(self._refresh_display)
                return

            if not job_logs:
                # snkmt returned 0 rows — could be a legitimate finish or
                # a transient gap between published states.  Same treatment
                # as the DB-exception case: mark stale, keep bars visible.
                self._mark_stale_all(time.monotonic())
                self.app.call_from_thread(self._refresh_display)
                return

            # A single workflow can write job logs to more than one logs/
            # directory (e.g. a mixeddata-prep stage in one output dir and the
            # hist stage in another).  Scan EVERY distinct parent dir, not just
            # the parent of the first row — otherwise running jobs that live in
            # a different dir than job_logs' first entry are silently dropped.
            log_dirs = {str(Path(p).parent) for p in job_logs}
            scanned: dict = {}
            for log_dir in log_dirs:
                scanned.update(scan_logs(log_dir))

            # Start `new_metrics` as a copy of the previous tick — anything
            # we DON'T successfully re-poll this tick is kept (and marked
            # stale later).  This protects against transient failures in
            # `scan_logs` (filesystem race / NFS blip) where a job briefly
            # disappears from the scan, AND against snkmt momentarily
            # reporting a job in a non-RUNNING transitional state.  Entries
            # are only positively *dropped* below when snkmt explicitly says
            # the job is in a terminal state.
            new_metrics: dict = dict(self._metrics)
            seen_names: set = set()

            for name, info in scanned.items():
                log_path = info.get("log_path", "")
                status   = job_logs.get(log_path)
                # Per user preference: NEVER drop a cached entry. Once a bar
                # has been shown, it stays.  If snkmt reports the job as no
                # longer RUNNING (transitional or terminal), or the log says
                # done, just skip refreshing this tick — the prior cached
                # counts will be marked stale by the post-loop pass below
                # and rendered with the `unreachable Xs` annotation.  This
                # protects against snkmt's HTCondor state transitions
                # (RUNNING → SUBMITTED → RUNNING during evictions/retries)
                # that previously caused the bar to disappear.
                if status != "RUNNING" or info.get("done"):
                    # Job isn't actively running this tick.  Distinguish a
                    # *terminal* finish (snkmt SUCCESS/ERROR or the log's
                    # completion marker) from a transient non-RUNNING blip
                    # (status None during an HTCondor state transition).  A
                    # terminal job is marked `done` so the renderer shows it as
                    # ✓ completed / ✗ failed instead of leaving the last bar to
                    # be stale-marked as "unreachable" (which looked like it was
                    # still running).  Transient blips fall through to the
                    # post-loop stale pass, preserving the old behaviour.
                    terminal = info.get("done") or status in ("SUCCESS", "ERROR")
                    if terminal and name in new_metrics:
                        new_metrics[name] = {
                            **new_metrics[name],
                            "done": True,
                            "errored": status == "ERROR",
                            "stale_since": None,
                        }
                    continue

                seen_names.add(name)

                # Resolve the dashboard URL.  Condor logs emit only a proxy
                # path (/proxy/PORT/status) with no host, so this uses the
                # `Dask scheduler host:` line runner.py logs to build a URL
                # reachable from a different node than the scheduler.  Falls
                # back to a tcp:// scheduler host, then localhost.
                dashboard_url = dashboard_url_from_info(info)

                counts = query_metrics_remote(dashboard_url) if dashboard_url else None
                # All-zero counts (Dask scheduler responding but currently
                # has no tasks counted — e.g. between waves of tasks, after
                # a scheduler restart, mid-startup) render as dots via
                # `_rich_bar`'s total==0 branch and look identical to a
                # wipe.  Treat them as "no fresh data" so the carry-over
                # below preserves the previously-shown bar.
                if counts is not None:
                    _total = (counts.get("memory", 0)
                              + counts.get("processing", 0)
                              + counts.get("erred", 0)
                              + counts.get("waiting", 0))
                    if _total == 0:
                        counts = None
                if counts is None:
                    # Probe failed (or zero-count response) — carry over
                    # previous counts so the bar stays on screen.
                    prev = self._metrics.get(name)
                    if prev and prev.get("counts") is not None:
                        new_metrics[name] = {
                            **prev,
                            "status": status,
                            "has_dashboard": bool(dashboard_url),
                            "log_path": log_path,
                            "stale_since": prev.get("stale_since") or time.monotonic(),
                            "done": False,  # snkmt says RUNNING again (restart)
                        }
                        continue
                    # First-ever observation failed — show the placeholder.
                    new_metrics[name] = {
                        "counts": None,
                        "status": status,
                        "has_dashboard": bool(dashboard_url),
                        "log_path": log_path,
                        "stale_since": None,
                    }
                    continue

                # Fresh good data — replace any prior entry (including stale).
                new_metrics[name] = {
                    "counts": counts,
                    "status": status,
                    "has_dashboard": bool(dashboard_url),
                    "log_path": log_path,
                    "stale_since": None,
                }

            # Any entry not refreshed this tick (snkmt status != RUNNING, log
            # says done, scan_logs missed it, etc.) gets marked stale so the
            # renderer adds an `unreachable Xs` annotation.  We never drop —
            # the bar stays until `watch_workflow_id` clears it (user
            # selects a different workflow) or the user restarts.
            now_loop = time.monotonic()
            for name, info in new_metrics.items():
                if name in seen_names:
                    continue
                if info.get("done"):
                    continue  # finished — render as completed, never "unreachable"
                if info.get("counts") is not None and not info.get("stale_since"):
                    info["stale_since"] = now_loop

            # HTCondor worker counts (non-blocking; returns {} if condor_q unavailable)
            condor = condor_counts_for_jobs(scanned)
            for name, cc in condor.items():
                if name in new_metrics:
                    new_metrics[name]["condor"] = cc

            # Update per-job history for throughput/ETA.  Skip entries with
            # `stale_since` set: their counts are a carried-over snapshot
            # from an earlier successful tick, not fresh measurements.
            # Appending them would dilute the rate calc with fake-zero
            # progress; pausing keeps the last real rate visible.
            now = time.monotonic()
            for name, info in new_metrics.items():
                if info.get("stale_since"):
                    continue
                mem = (info.get("counts") or {}).get("memory", 0)
                if name not in self._history:
                    self._history[name] = deque(maxlen=_HISTORY_MAXLEN)
                self._history[name].append((now, mem))

            # Alert on newly erred tasks or newly paused workers
            for name, info in new_metrics.items():
                c = info.get("counts") or {}
                prev = (self._metrics.get(name) or {}).get("counts") or {}
                n_erred = c.get("erred", 0)
                if n_erred > prev.get("erred", 0):
                    self.app.call_from_thread(
                        self.app.notify,
                        f"{name}: {n_erred} task{'s' if n_erred != 1 else ''} erred",
                        severity="error",
                        timeout=10,
                    )
                n_paused = c.get("workers_paused", 0)
                if n_paused > 0 and prev.get("workers_paused", 0) == 0:
                    self.app.call_from_thread(
                        self.app.notify,
                        f"{name}: {n_paused} worker{'s' if n_paused != 1 else ''} paused (memory pressure)",
                        severity="warning",
                        timeout=15,
                    )

            self._metrics = new_metrics
            self.app.call_from_thread(self._refresh_display)
        except Exception as exc:
            import traceback
            self.app.call_from_thread(self._show_error, repr(exc), traceback.format_exc())

    def _refresh_display(self) -> None:
        if not self._mounted:
            return
        try:
            content = self.query_one("#dask-content", Static)
        except NoMatches:
            return

        if self.workflow_id is None:
            content.update("[dim]Select a workflow to view running jobs.[/dim]")
            return

        if not self._metrics:
            content.update("[dim]No running Dask jobs.[/dim]")
            return

        lines = []
        for name, info in sorted(self._metrics.items()):
            counts = info.get("counts")
            stale_since = info.get("stale_since")
            stale_age = (time.monotonic() - stale_since) if stale_since else None

            n_erred = counts.get("erred", 0) if counts else 0
            done = info.get("done")

            if done:
                # Terminal job — show its final bar (if we ever got counts)
                # tagged as completed/failed, so it's clearly distinct from a
                # still-running or unreachable job.
                tag = ("[bold red]✗ failed[/bold red]" if info.get("errored")
                       else "[bold green]✓ completed[/bold green]")
                bar = f"{_rich_bar(counts)}  {tag}" if counts else tag
            elif counts is None:
                # No cached data has ever arrived for this job.  Per user
                # preference: NEVER render dots — they look like a wipe of
                # cached progress.  Show a text-only placeholder instead.
                if info.get("has_dashboard"):
                    bar = "[yellow]unreachable (SSH/network) — waiting for first response[/yellow]"
                else:
                    bar = "[dim]waiting for dashboard…[/dim]"
            elif stale_age is not None:
                # Cached data, current probe failing — keep the bar exactly as
                # it was, just annotate as unreachable.  No hard limit; the
                # cache persists until snkmt reports the job is no longer
                # running (which clears the entry naturally).
                bar = (f"{_rich_bar(counts)}  "
                       f"[yellow]unreachable {int(stale_age)}s[/yellow]")
            else:
                bar = _rich_bar(counts)

            n_paused = counts.get("workers_paused", 0) if counts else 0
            name_str = (
                f"[bold red]⚠ {escape(name)}[/bold red]"
                if n_erred else
                f"[bold]{escape(name)}[/bold]"
            )
            erred_str  = f"\n  [bold red]⚠ {n_erred} task{'s' if n_erred != 1 else ''} erred[/bold red]" if n_erred else ""
            paused_str = f"\n  [bold yellow]⚠ {n_paused} worker{'s' if n_paused != 1 else ''} paused (memory pressure)[/bold yellow]" if n_paused else ""

            rate, eta = _throughput_eta(self._history.get(name, deque()), counts or {})
            if done:
                throughput_str = ""  # finished — no rate/ETA
            elif rate is not None:
                eta_str = f"  [dim]~{_fmt_eta(eta)} remaining[/dim]" if eta else "  [dim]nearly done[/dim]"
                throughput_str = f"\n  [dim]{rate:.1f} tasks/s{eta_str}[/dim]"
            elif counts is not None:
                throughput_str = "\n  [dim]measuring...[/dim]"
            else:
                throughput_str = ""

            condor = None if done else info.get("condor")
            if condor:
                idle, running, held = condor['idle'], condor['running'], condor['held']
                held_str = f"[bold red]{held}H[/bold red]" if held else f"[dim]{held}H[/dim]"
                idle_str = f"[yellow]{idle}I[/yellow]" if idle else f"[dim]{idle}I[/dim]"
                condor_str = f"\n  [dim]Condor:[/dim] {idle_str} [green]{running}R[/green] {held_str}"
            else:
                condor_str = ""

            lines.append(f"{name_str}\n  {bar}{erred_str}{paused_str}{throughput_str}{condor_str}")

        content.update("\n\n".join(lines))

    def _show_error(self, short: str, detail: str) -> None:
        try:
            content = self.query_one("#dask-content", Static)
        except NoMatches:
            return
        content.update(f"[red]Poll error:[/red] {escape(short)}\n\n[dim]{escape(detail)}[/dim]")


# ---------------------------------------------------------------------------
# Overview container (same layout as snkmt, right panel swapped)
# ---------------------------------------------------------------------------

class BaristaOverviewContainer(Horizontal):
    BINDINGS = [
        ("tab",       "focus_next",     "Next"),
        ("shift+tab", "focus_previous", "Previous"),
    ]
    repo: reactive[WorkflowRepository | None] = reactive(None, recompose=True)

    def __init__(self, repo: WorkflowRepository) -> None:
        super().__init__()
        self.set_reactive(BaristaOverviewContainer.repo, repo)
        self._total_workflows   = 0
        self._filtered_workflows = 0
        self._hidden_workflows  = 0
        self.selected_workflow: str | None = None

    # --- filter handlers (identical to snkmt) ---

    @on(Input.Changed, "#name-filter")
    async def filter_by_name(self, message: Input.Changed) -> None:
        self.query_one(WorkflowTable).name_filter = message.value

    @on(Select.Changed, "#date-filter")
    async def filter_by_date(self, message: Select.Changed) -> None:
        if message.value is not Select.BLANK:
            self.query_one(WorkflowTable).date_filter = message.value  # type: ignore

    @on(Select.Changed, "#status-filter")
    async def filter_by_status(self, message: Select.Changed) -> None:
        if message.value is not Select.BLANK:
            self.query_one(WorkflowTable).status_filter = message.value  # type: ignore

    @on(WorkflowTable.TableRefreshed)
    async def handle_table_refreshed(self, message: WorkflowTable.TableRefreshed) -> None:
        try:
            label = self.query_one("#workflow-counts", Label)
            filtered_out = message.total_count - message.filtered_count
            label.update(
                f"Viewing {message.visible_count}/{message.total_count} workflows "
                f"({filtered_out} filtered, {message.hidden_count} hidden)"
            )
        except NoMatches:
            pass

    @on(WorkflowTable.UpdatedWorkflows)
    async def handle_updated_workflows(self, message: WorkflowTable.UpdatedWorkflows) -> None:
        if self.selected_workflow:
            wf = next(
                (w for w in message.workflows if str(w.id) == self.selected_workflow),
                None,
            )
            if wf:
                pass  # DaskJobPanel polls independently; nothing extra needed

    @work(exclusive=True, exit_on_error=False)
    @on(WorkflowTable.RowSelected, "#workflow-table")
    async def handle_workflow_selected(self, event: WorkflowTable.RowSelected) -> None:
        workflow_id = event.row_key.value
        self.selected_workflow = workflow_id

        # Update Dask panel
        try:
            self.query_one(DaskJobPanel).workflow_id = workflow_id
        except NoMatches:
            pass

        # Update rules and errors (same as snkmt)
        if self.repo:
            try:
                rule_table = self.query_one(RuleTable)
                rule_table.display = True
                rule_table.workflow_id = UUID(workflow_id)

                self.query_one("#rules-placeholder").display = False

                self.query_one(WorkflowErrors).workflow_id = UUID(workflow_id)
            except NoMatches:
                pass

    def compose(self) -> ComposeResult:
        if not self.repo:
            return

        # ---- Left panel: workflow list (identical to snkmt) ----
        with Container(classes="section", id="workflows") as wf_container:
            wf_container.border_title = "Workflows"
            with Container(classes="subsection", id="workflows-filters") as filters:
                filters.border_title = "Filters"
                with Horizontal(id="filter-layout"):
                    yield Input(placeholder="Filter by name...", id="name-filter", compact=True)
                    yield Select(
                        [
                            ("Any time",      DateFilter.ANY),
                            ("Today",         DateFilter.TODAY),
                            ("Yesterday",     DateFilter.YESTERDAY),
                            ("Last 7 days",   DateFilter.LAST_7_DAYS),
                            ("Last 30 days",  DateFilter.LAST_30_DAYS),
                            ("Last 90 days",  DateFilter.LAST_90_DAYS),
                            ("This year",     DateFilter.THIS_YEAR),
                        ],
                        value=DateFilter.ANY,
                        id="date-filter",
                        compact=True,
                    )
                    yield Select(
                        [
                            ("All statuses", "all"),
                            ("Running",      Status.RUNNING),
                            ("Success",      Status.SUCCESS),
                            ("Error",        Status.ERROR),
                            ("Unknown",      Status.UNKNOWN),
                        ],
                        value="all",
                        id="status-filter",
                        compact=True,
                    )
                yield Label(
                    "Viewing 0 workflows / 0 (0 filtered, 0 hidden)",
                    id="workflow-counts",
                )
            yield WorkflowTable(self.repo, id="workflow-table")

        # ---- Right panel: Dask progress + rules + errors ----
        with Container(classes="section", id="selected-workflow-detail") as detail:
            detail.border_title = "Workflow Details"

            dask_panel = DaskJobPanel(classes="subsection", id="dask-panel")
            dask_panel.border_title = "Running Jobs (Dask)"
            yield dask_panel

            with Container(classes="subsection", id="workflow-rules") as rules_container:
                rules_container.border_title = "Rules"
                yield Label(
                    "Please select a workflow to view rules.",
                    id="rules-placeholder",
                )
                rule_table = RuleTable(self.repo)
                rule_table.display = False
                yield rule_table

            errors_container = WorkflowErrors(
                repo=self.repo, classes="subsection", id="workflow-errors"
            )
            errors_container.border_title = "Errors"
            yield errors_container


# ---------------------------------------------------------------------------
# App header (same as snkmt)
# ---------------------------------------------------------------------------

class AppHeader(Horizontal):
    def __init__(self, *args, datasource: str | None, **kwargs) -> None:
        self.datasource = datasource
        super().__init__(*args, **kwargs)

    def compose(self) -> ComposeResult:
        yield Label(f"[b]barista[/] [dim]{VERSION}[/]", id="app-title")
        if self.datasource:
            yield Label(f"Connected to: {self.datasource}", id="app-db-path")
        else:
            yield Label("No database selected", id="app-db-path")


# ---------------------------------------------------------------------------
# Dashboard screen
# ---------------------------------------------------------------------------

class DashboardScreen(Screen):
    COMMANDS = {SelectDatabaseCommand}
    BINDINGS = [
        ("tab",       "focus_next",     "Next"),
        ("shift+tab", "focus_previous", "Previous"),
    ]

    def action_select_database_source(self) -> None:
        self.app.push_screen(
            CommandPalette(
                providers=[DatabaseSourceProvider],
                placeholder="Search for database sources…",
            )
        )

    def __init__(self, datasource_url: str | None = None) -> None:
        super().__init__()
        self.datasource = datasource_url

    def compose(self) -> ComposeResult:
        try:
            db   = AsyncDatabase(self.datasource, create_db=False)
            repo = db.get_workflow_repository()
            yield AppHeader(datasource=db.db_path, id="header")
            yield BaristaOverviewContainer(repo)
        except Exception as e:
            from snkmt.core.db.session import DatabaseNotFoundError
            err = Container(classes="section", id="error-container")
            err.border_title = "Database Connection Error"
            with err:
                yield Label(
                    f"{'Database Not Found' if isinstance(e, DatabaseNotFoundError) else type(e).__name__}"
                    f"\n\nDetails:\n{e}",
                    id="error-message",
                    classes="error-text",
                )
        yield Footer(id="footer")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

class BaristaConsoleApp(App):
    CSS_PATH = "barista_console.tcss"
    BINDINGS = [
        ("q",         "quit",           "Quit"),
        ("ctrl+c",    "quit",           "Quit"),
        ("tab",       "focus_next",     "Next"),
        ("shift+tab", "focus_previous", "Previous"),
    ]

    def __init__(self, databases: list[str] | None = None,
                 refresh_interval: float = 5.0) -> None:
        super().__init__()
        # snkmt 0.4+ widgets read self.app.refresh_interval in on_mount to set
        # up their own periodic update timers (workflow table, rule table).
        # Older snkmt didn't have this attribute; the installed version on
        # cmslpc does require it.  5 s is a sensible default; set to 0 to
        # disable auto-refresh.
        self.refresh_interval = refresh_interval
        self.current_source = databases[0] if databases else None

    def set_database_source(self, source: str) -> None:
        self.current_source = source
        self.notify(f"Connected to: {source}", severity="information")
        self.switch_screen(DashboardScreen(source))

    def on_ready(self) -> None:
        self.title  = "barista console"
        self.theme  = "gruvbox"
        self.push_screen(DashboardScreen(self.current_source))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        from snkmt.console.widgets import LogFileModal
        if event.item.name:
            self.push_screen(LogFileModal(Path(event.item.name)))

    def _handle_exception(self, error: Exception) -> None:
        import traceback
        log_path = "/tmp/barista_console_crash.log"
        with open(log_path, "a") as f:
            f.write(f"\n{'='*60}\n")
            traceback.print_exception(type(error), error, error.__traceback__, file=f)
        super()._handle_exception(error)

    def action_focus_next(self) -> None:
        self.screen.focus_next()

    def action_focus_previous(self) -> None:
        self.screen.focus_previous()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app = BaristaConsoleApp()
    app.run()


if __name__ == "__main__":
    main()
