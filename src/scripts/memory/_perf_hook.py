"""
Profiling hook for processor_HH4b.py.

This module is imported by perf_profile.py's wrapper BEFORE runner.py
starts. It injects a PerfTracker into the processor via its native
``tracker=`` parameter, so no duplicated process/process_shift logic
is needed.

The hook is activated when PERF_PROFILE_OUTPUT is set in the environment.
Results are written at exit via atexit.
"""

from __future__ import annotations

import atexit
import csv
import importlib
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import psutil


# ---------------------------------------------------------------------------
# Stage tracker
# ---------------------------------------------------------------------------
@dataclass
class StageRecord:
    name: str
    rss_before_mb: float = 0.0
    rss_after_mb: float = 0.0
    delta_mb: float = 0.0
    elapsed_sec: float = 0.0


@dataclass
class PerfTracker:
    """Collects per-stage timing and memory snapshots."""

    records: list[StageRecord] = field(default_factory=list)
    _process: psutil.Process = field(
        default_factory=lambda: psutil.Process(os.getpid())
    )

    def rss_mb(self) -> float:
        return self._process.memory_info().rss / (1024 * 1024)

    @contextmanager
    def stage(self, name: str):
        rss_before = self.rss_mb()
        t0 = time.perf_counter()
        yield
        elapsed = time.perf_counter() - t0
        rss_after = self.rss_mb()
        rec = StageRecord(
            name=name,
            rss_before_mb=rss_before,
            rss_after_mb=rss_after,
            delta_mb=rss_after - rss_before,
            elapsed_sec=elapsed,
        )
        self.records.append(rec)
        print(
            f"  [PERF] {name:<50s} "
            f"{elapsed:>7.2f}s | "
            f"RSS: {rss_before:>7.0f} -> {rss_after:>7.0f} MB "
            f"(delta {rec.delta_mb:+.0f} MB)"
        )

    def write_report(self, path: str):
        """Write human-readable text report."""
        with open(path, "w") as f:
            f.write(
                f"{'Stage':<55} {'Time (s)':>10} "
                f"{'RSS before':>12} {'RSS after':>12} {'Delta':>10}\n"
            )
            f.write("-" * 101 + "\n")
            for r in self.records:
                f.write(
                    f"{r.name:<55} {r.elapsed_sec:>10.2f} "
                    f"{r.rss_before_mb:>10.0f} MB "
                    f"{r.rss_after_mb:>10.0f} MB "
                    f"{r.delta_mb:>+9.0f} MB\n"
                )
            f.write("-" * 101 + "\n")
            total_time = sum(r.elapsed_sec for r in self.records)
            peak_rss = (
                max(r.rss_after_mb for r in self.records) if self.records else 0
            )
            f.write(
                f"{'TOTAL':<55} {total_time:>10.2f} "
                f"{'':>12} {'peak':>5} {peak_rss:>5.0f} MB\n"
            )
        print(f"\n[PERF] Report written to {path}")

    def write_csv(self, path: str):
        """Write CSV for easy diff."""
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["stage", "elapsed_sec", "rss_before_mb", "rss_after_mb", "delta_mb"]
            )
            for r in self.records:
                writer.writerow([
                    r.name,
                    f"{r.elapsed_sec:.3f}",
                    f"{r.rss_before_mb:.1f}",
                    f"{r.rss_after_mb:.1f}",
                    f"{r.delta_mb:.1f}",
                ])
        print(f"[PERF] CSV written to {path}")


# ---------------------------------------------------------------------------
# Global tracker
# ---------------------------------------------------------------------------
_tracker: PerfTracker | None = None


def get_tracker() -> PerfTracker:
    global _tracker
    if _tracker is None:
        _tracker = PerfTracker()
    return _tracker


# ---------------------------------------------------------------------------
# Processor injection (replaces the old monkey-patch)
# ---------------------------------------------------------------------------
def _inject_tracker(module):
    """Wrap the processor class __init__ to inject tracker= automatically."""
    cls = module.HH4bBaseProcessor
    tracker = get_tracker()
    _orig_init = cls.__init__

    def _init_with_tracker(self, *args, **kwargs):
        kwargs.setdefault("tracker", tracker)
        _orig_init(self, *args, **kwargs)

    cls.__init__ = _init_with_tracker
    print("[PERF] Tracker injected into HH4bBaseProcessor")


# ---------------------------------------------------------------------------
# Install hook
# ---------------------------------------------------------------------------
_installed = False


def install_hook():
    """Install profiling hook via importlib.import_module patch.

    Wraps importlib.import_module so that when runner.py imports the
    processor module, we inject the tracker into the processor class.
    """
    global _installed
    if _installed:
        return
    _installed = True

    output_path = os.environ.get("PERF_PROFILE_OUTPUT", "")
    if not output_path:
        return

    tracker = get_tracker()

    # Register atexit to write results
    def _write_results():
        if tracker.records:
            tracker.write_report(f"{output_path}.txt")
            tracker.write_csv(f"{output_path}.csv")
        else:
            print("[PERF] No stages recorded.")

    atexit.register(_write_results)

    # Wrap importlib.import_module to catch when the processor is loaded
    _original_import_module = importlib.import_module
    _target = "coffea4bees.analysis.processors.processor_HH4b"
    _injected = {"done": False}

    def _hooked_import_module(name, package=None):
        result = _original_import_module(name, package)
        if not _injected["done"] and name == _target:
            _injected["done"] = True
            _inject_tracker(result)
        return result

    importlib.import_module = _hooked_import_module

    print(f"[PERF] Hook installed. Output: {output_path}.{{txt,csv}}")
