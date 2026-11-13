#!/usr/bin/env python3
"""Run several classifier tasks at once on one GPU, inside the classifier image (train.sbatch batch mode).

    batch_runner.py <batch.json> --local <node-local dir> [--config-paths coffea4bees]

<batch.json> is a list of tasks, each {"label", "run_dir", "wf", "common", "template"}. Every task is
`python -m src.classifier.task.main template '{<template>}' <wf> -from <common> -flag debug` with its
own run directory (train.log, checkpoints/, yields_loaded.json, train.done), exactly as a single-task
job would have, so remote_train.py fetches and reattaches per task. A task whose train.done exists is
skipped, so a requeued batch resumes only the unfinished ones (trainings from their last epoch).

Run under an MPS daemon (train.sbatch starts one): the k-fold trainer processes of all tasks then
share the GPU concurrently. Exit 0 only if every task finished.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("batch")
    ap.add_argument("--local", required=True)
    ap.add_argument("--config-paths", default="coffea4bees")
    ap.add_argument("--extra-args", default="")
    a = ap.parse_args()
    tasks = json.loads(Path(a.batch).read_text())
    procs = []

    def stop(signum, _):
        for _, p, _ in procs:
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        for _, p, _ in procs:
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill()
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    for i, t in enumerate(tasks):
        rd = Path(t["run_dir"])
        if (rd / "train.done").exists():
            print(f"[batch] {t['label']}: already done, skipped", flush=True)
            continue
        (rd / "checkpoints").mkdir(parents=True, exist_ok=True)
        (rd / "logs").mkdir(parents=True, exist_ok=True)
        scratch = Path(a.local) / f"eval_scratch_{i}"
        scratch.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, HCR_YIELDS_FILE=str(rd / "yields_loaded.json"),
                   CLASSIFIER_CONFIG_PATHS=a.config_paths, CLASSIFIER_CHECKPOINT_DIR=str(rd / "checkpoints"),
                   EVAL_SCRATCH=str(scratch), PYTHONUNBUFFERED="1")
        cmd = [sys.executable, "-m", "src.classifier.task.main", "template", "{" + (t.get("template") or "") + "}",
               t["wf"], "-from", t["common"], "-flag", "debug", *a.extra_args.split()]
        logf = open(rd / "train.log", "a")
        logf.write(f"[batch {time.strftime('%F %T')}] task {i} of {len(tasks)} in job {os.environ.get('SLURM_JOB_ID', '?')}: "
                   f"{' '.join(cmd)}\n")
        logf.flush()
        procs.append((t, subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT), time.time()))
        print(f"[batch] {t['label']}: started", flush=True)

    failed = 0
    for t, p, t0 in procs:
        rc = p.wait()
        rd = Path(t["run_dir"])
        with open(rd / "logs" / f"timing_batch_{os.environ.get('SLURM_JOB_ID', 'x')}.txt", "a") as f:
            f.write(f"start {int(t0)}\ntrained {int(time.time())}\n")
        if rc == 0:
            (rd / "train.done").touch()
        else:
            failed += 1
        print(f"[batch] {t['label']}: exit {rc} after {(time.time() - t0) / 3600:.2f} h", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
