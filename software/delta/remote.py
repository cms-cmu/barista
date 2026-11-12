#!/usr/bin/env python3
"""Run one classifier training or evaluation on NCSA Delta from a Snakemake driver elsewhere (cmslpc).

The body of the `train_delta` / `evaluate_delta` rules of src/classifier/workflow/Snakefile, used when
the workflow config sets `train_host: delta` / `eval_host: delta` (`roast new --train-on delta
--eval-on delta`). Delta needs NCSA Duo on every login, so nothing here logs in to Delta: a pull agent
(software/delta/agent.py, in tmux on a Delta login node) watches an EOS spool (software/delta/spool.py)
and does the Slurm work. This script

  1. makes sure the agent has this checkout's code: an incremental git bundle per repository against
     the commits the agent publishes, plus `git diff HEAD` for hot patches (spool code/<key>/);
  2. writes a request (spool req/<rid>.json + the untracked workflow files it names, as a tarball);
  3. waits for the agent's status (spool status/<rid>.json), copying train.log back as it grows;
  4. copies back train.log and yields_loaded.json and exits 0 only if the job finished.

The template's `model:` is the roast's EOS path, so the models, friend trees and result.json land
where the downstream rules read them. The request id is kept in <model_dir>/delta[_<task>].json: a
restarted driver (`roast resume`) reattaches instead of submitting again; a failed job is requested
once more and resumes from its last epoch checkpoint (the run directory on Delta is per label).

Needs on the driver: xrdfs/xrdcp and a grid proxy that can write the spool. On Delta: the agent.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from spool import Spool, default_spool, safe  # noqa: E402

# code_key / tracked / referenced_files / make_tar / rel_to_root are shared with the ORCHARD driver
_spec = importlib.util.spec_from_file_location("orchard_remote_train", HERE.parent / "orchard" / "remote_train.py")
rt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rt)
REPOS = rt.REPOS
TERMINAL_OK = ("DONE",)
TERMINAL_FAIL = ("FAILED", "ERROR", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE")


def log(*a):
    print(f"[delta {time.strftime('%F %T')}]", *a, flush=True)


def worktree_diff(root: Path, sub: str) -> bytes:
    f = getattr(rt, "worktree_diff", None)
    if f:
        return f(root, sub)
    return subprocess.run(["git", "diff", "HEAD", "--binary"], cwd=root / sub, capture_output=True).stdout


def ship_code(sp: Spool, root: Path, wait: int) -> str:
    """Make the agent build this checkout's code; return its key once the agent reports it ready."""
    key = rt.code_key(root, [d for d, _ in REPOS])
    cdir = f"code/{key}"
    if not sp.exists(f"{cdir}/manifest.json"):
        man = {"key": key, "repos": {}, "created": time.strftime("%F %T"), "by": os.environ.get("USER", "")}
        with tempfile.TemporaryDirectory() as tmp:
            for sub, name in REPOS:
                d = root / sub
                sha = rt.git(["rev-parse", "HEAD"], d).strip()
                entry = {"sha": sha, "bundle": False, "diff": False}
                refs = (sp.get_text(f"agent/refs_{name}.txt") or "").split()
                cands = sorted({h for h in refs if re.fullmatch(r"[0-9a-f]{40}", h)})
                if sha not in cands:
                    ok = subprocess.run(["git", "cat-file", "--batch-check"], cwd=d, input="\n".join(cands),
                                        capture_output=True, text=True).stdout.split("\n")
                    basis = [w[0] for w in (line.split() for line in ok) if len(w) > 1 and w[1] == "commit"]
                    if not basis:
                        sys.exit(f"[delta] the Delta agent has no copy of {name} yet: run once "
                                 f"`software/delta/remote.py --bootstrap` (see docs/delta.md)")
                    bundle = f"{tmp}/{name}.bundle"
                    r = subprocess.run(["git", "bundle", "create", "-q", bundle, "HEAD", *[f"^{b}" for b in basis[:50]]],
                                       cwd=d, capture_output=True, text=True)
                    if r.returncode == 0:
                        log(f"{name}: sending {os.path.getsize(bundle) / 1e6:.2f} MB of commits")
                        sp.put(bundle, f"{cdir}/{name}.bundle")
                        entry["bundle"] = True
                    elif "empty bundle" not in r.stderr:
                        sys.exit(f"[delta] git bundle for {name} failed: {r.stderr.strip()}")
                diff = worktree_diff(root, sub)
                if diff.strip():
                    Path(f"{tmp}/{name}.diff").write_bytes(diff)
                    sp.put(f"{tmp}/{name}.diff", f"{cdir}/{name}.diff")
                    entry["diff"] = True
                    log(f"{name}: uncommitted changes shipped ({len(diff)} bytes)")
                man["repos"][name] = entry
        sp.put_json(f"{cdir}/manifest.json", man)    # last: the agent builds once it sees this
        log(f"code {key} requested from the agent")
    t0 = time.time()
    while True:
        if sp.exists(f"{cdir}/ready"):
            return key
        err = sp.get_text(f"{cdir}/error")
        if err:
            sys.exit(f"[delta] the agent could not build code {key}: {err.strip()[-1500:]}\n"
                     f"(fix it and remove {sp.url}/{cdir} to retry)")
        if time.time() - t0 > wait:
            sys.exit(f"[delta] code {key} not built after {wait} s: is the agent running? ({agent_age(sp)})")
        time.sleep(30)


def agent_age(sp: Spool) -> str:
    a = sp.get_json("agent/alive.json") or {}
    if not a.get("time"):
        return "no heartbeat from the agent"
    age = time.time() - a["time"]
    return (f"agent on {a.get('host', '?')} last seen {age / 60:.0f} min ago, "
            f"proxy {a.get('proxy_hours', '?')} h left")


def bootstrap(sp: Spool, root: Path):
    """Full bundles of every branch to spool bootstrap/: the agent clones from them once."""
    with tempfile.TemporaryDirectory() as tmp:
        for sub, name in REPOS:
            b = f"{tmp}/{name}.bundle"
            subprocess.run(["git", "bundle", "create", "-q", b, "--all"], cwd=root / sub, check=True)
            log(f"{name}: uploading {os.path.getsize(b) / 1e6:.0f} MB")
            sp.put(b, f"bootstrap/{name}.bundle")
    log(f"bundles in {sp.url}/bootstrap: the agent picks them up on its next pass")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bootstrap", action="store_true", help="upload full bundles of both repositories for a new agent")
    ap.add_argument("--spool", default="", help="EOS spool URL (default $DELTA_SPOOL or root://cmseos.fnal.gov//store/user/$USER/barista_delta)")
    ap.add_argument("--model-dir"); ap.add_argument("--wf"); ap.add_argument("--common")
    ap.add_argument("--template", help="full template string, e.g. 'model: root://...'")
    ap.add_argument("--log", help="where the job log is copied (the rule's log)")
    ap.add_argument("--yields", help="where yields_loaded.json is copied (training)")
    ap.add_argument("--task", default="train", help="train, or evaluate[_<split>]")
    ap.add_argument("--label", default=""); ap.add_argument("--roast-id", default="")
    ap.add_argument("--config-paths", default="coffea4bees", help="CLASSIFIER_CONFIG_PATHS")
    ap.add_argument("--time", default="", help="Slurm wall time per job (agent default 4:00:00; the fit requeues)")
    ap.add_argument("--partition", default=""); ap.add_argument("--cpus", default=""); ap.add_argument("--mem", default="")
    ap.add_argument("--max-restarts", default="")
    ap.add_argument("--poll", type=int, default=120, help="seconds between status checks")
    ap.add_argument("--code-wait", type=int, default=3 * 3600, help="seconds to wait for the agent to build the code")
    a = ap.parse_args()
    root = Path.cwd()
    sp = Spool(a.spool or default_spool())
    if a.bootstrap:
        bootstrap(sp, root)
        return
    for k in ("model_dir", "wf", "common", "template", "log"):
        if not getattr(a, k):
            ap.error(f"--{k.replace('_', '-')} is required")

    model_dir = Path(a.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    state_file = model_dir / ("delta.json" if a.task == "train" else f"delta_{a.task}.json")
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    task_tag = "" if a.task == "train" else f"__{a.task}"
    label = a.label or safe(f"{a.roast_id or 'run'}__{a.model_dir.strip('/')}{task_tag}")[:150]
    wf, common = rt.rel_to_root(a.wf, root), rt.rel_to_root(a.common, root)
    signal.signal(signal.SIGTERM, lambda *_: (log(f"stopped; the Delta job keeps running and a rerun "
                                                  f"reattaches to it ({state_file})"), sys.exit(143)))
    log(f"spool {sp.url}; run {label}; {agent_age(sp)}")

    def fetch_back(final: bool) -> bool:
        t = sp.get_text(f"runs/{label}/train.log")
        if t:
            Path(a.log).write_text(t)
        if not final:
            return False
        if a.yields:
            y = sp.get_text(f"runs/{label}/yields_loaded.json")
            if y and y.strip():
                Path(a.yields).write_text(y)
        s = sp.get_text(f"runs/{label}/summary.txt")
        if s:
            print(s, flush=True)
        return True

    def submit():
        key = ship_code(sp, root, a.code_wait)
        known = set(rt.tracked(root, ".") + rt.tracked(root, "coffea4bees"))
        texts = [Path(wf).read_text(), Path(common).read_text(), a.template]
        extra = sorted(set(rt.referenced_files(texts, root, known)) | {wf, common} - known)
        attempt = state.get("attempt", 0) + 1
        rid = safe(f"{label}.a{attempt}")
        with tempfile.TemporaryDirectory() as tmp:
            log(f"shipping {len(extra)} workflow file(s): {', '.join(extra[:6])}{' ...' if len(extra) > 6 else ''}")
            rt.make_tar(root, extra, f"{tmp}/wf.tgz")
            sp.put(f"{tmp}/wf.tgz", f"req/{rid}.wf.tgz")
        req = {"rid": rid, "label": label, "task": a.task, "key": key, "wf": wf, "common": common,
               "template": a.template, "config_paths": a.config_paths, "attempt": attempt,
               "time": a.time, "partition": a.partition, "cpus": a.cpus, "mem": a.mem,
               "max_restarts": a.max_restarts or ("0" if a.task.startswith("evaluate") else ""),
               "roast_id": a.roast_id, "model_dir": a.model_dir, "requested": time.strftime("%F %T"),
               "by": os.environ.get("USER", "")}
        sp.put_json(f"req/{rid}.json", req)          # last: the agent acts once it sees this
        st = {"rid": rid, "label": label, "key": key, "attempt": attempt, "requested": req["requested"]}
        state_file.write_text(json.dumps(st, indent=1))
        log(f"requested {rid} (attempt {attempt})")
        return st

    def status() -> dict:
        return sp.get_json(f"status/{state['rid']}.json") or {}

    if state.get("rid"):
        s = status()
        log(f"found request {state['rid']} from {state_file}: {s.get('state', 'no status yet')}")
        if any(s.get("state", "").startswith(x) for x in TERMINAL_FAIL):
            state = submit()
    else:
        state = submit()

    # ── wait ──────────────────────────────────────────────────────────────────────────────────────
    n, retried, last, warned = 0, False, None, 0.0
    while True:
        time.sleep(a.poll)
        n += 1
        s = status()
        st = s.get("state", "")
        if st != last:
            log(f"{state['rid']}: {st or 'not picked up yet'}" + (f" (job {s['job']})" if s.get("job") else "")
                + (f" {s['msg']}" if s.get("msg") else ""))
            last = st
        if not st and time.time() - warned > 3600:
            log(f"waiting for the agent: {agent_age(sp)}")
            warned = time.time()
        if n % 3 == 0:
            fetch_back(False)
        if st in TERMINAL_OK:
            fetch_back(True)
            log(f"{a.task} finished on Delta (job {s.get('job')})")
            for ext in (".json", ".wf.tgz"):
                sp.rm(f"req/{state['rid']}{ext}")
            sys.exit(0)
        if any(st.startswith(x) for x in TERMINAL_FAIL):
            fetch_back(False)
            log(f"Delta job {s.get('job')} ended {st}:\n{s.get('tail', '')}")
            for ext in (".json", ".wf.tgz"):
                sp.rm(f"req/{state['rid']}{ext}")
            if not retried and "OUT_OF_MEMORY" not in st and st != "ERROR":
                retried = True
                log("requesting once more (a training resumes from its last epoch checkpoint)")
                state = submit()
                continue
            sys.exit(1)


if __name__ == "__main__":
    main()
