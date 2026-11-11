#!/usr/bin/env python3
"""Run one classifier training on CMU ORCHARD from a Snakemake driver elsewhere (falcon, cmslpc, ...).

This is the body of the `train_orchard` rule of src/classifier/workflow/Snakefile, used when the
workflow config sets `train_host: orchard`. Everything else in the workflow stays where it is; only
the GPU training moves:

  1. ship this checkout (barista + coffea4bees: tracked files and uncommitted changes, plus the
     untracked files the workflow files name, such as the roast's output/<model>/wfs/train.yml) to
     $ORCHARD_PROJECT/code/<key> on ORCHARD, once per code state;
  2. submit software/orchard/submit.sh there (inputs read over XRootD; the template's `model:` is the
     roast's EOS path, so the models and result.json land where the downstream rules read them);
  3. wait for it, through requeues at the time limit or preemptions, copying train.log back;
  4. copy back train.log and yields_loaded.json, and exit 0 only if the training finished.

The job id is kept in <model_dir>/orchard.json: if the driver is stopped and restarted (`roast
resume`), the rule reattaches to the running job instead of submitting a second one, and a failed
job is resubmitted to resume from its last epoch checkpoint.

Needs on the driver host: the Google Cloud CLI logged in as the ORCHARD account (Andrew ID), found
on PATH or in ~/google-cloud-sdk/bin. On ORCHARD: the classifier image (software/orchard/setup_image.sh)
and a grid proxy in ~/.roast/x509_proxy (`bin/roast proxy` copies it there).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

TERMINAL_FAIL = ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE")
# a relative path inside a workflow file (optionally followed by @@key)
TOKEN = re.compile(r"(?<![\w./\-:])([A-Za-z0-9_][\w.\-]*(?:/[\w.\-]+)+/?)")


def log(*a):
    print(f"[orchard {time.strftime('%F %T')}]", *a, flush=True)


class Remote:
    """Commands on the ORCHARD login node through `gcloud compute ssh` (IAP tunnel)."""

    def __init__(self, a):
        g = a.gcloud or shutil.which("gcloud") or os.path.expanduser("~/google-cloud-sdk/bin/gcloud")
        if not (shutil.which(g) or os.path.exists(g)):
            sys.exit("[orchard] no gcloud on this host: install the Google Cloud CLI (see docs/software/orchard.md) "
                     "and `gcloud auth login <andrew-id>@andrew.cmu.edu`")
        self.gcloud = g
        self.instance = a.instance
        self.bucket = (a.bucket or "").rstrip("/")
        self.flags = ["--tunnel-through-iap", "--quiet", "--zone", a.zone, "--project", a.project]

    def run(self, script: str, check=True, timeout=600, retries=3) -> subprocess.CompletedProcess:
        cmd = [self.gcloud, "compute", "ssh", self.instance, *self.flags, "--command", "bash -s"]
        r = None
        for i in range(retries):
            try:
                r = subprocess.run(cmd, input=script, capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired as e:
                r = subprocess.CompletedProcess(cmd, 255, "", f"timeout after {timeout} s: {e}")
            # 255: ssh/IAP could not connect; anything else is the script's own exit code
            if r.returncode != 255:
                break
            time.sleep(20 * (i + 1))
        if check and r.returncode != 0:
            raise RuntimeError(f"remote command failed ({r.returncode}): {r.stderr.strip()[-1500:]}")
        return r

    def put(self, local: str, remote: str):
        """Copy a file to ORCHARD:~/<remote>. Through the bucket when there is one: plain HTTPS from
        here and an in-cloud copy there. `gcloud compute scp` pushes it through the IAP tunnel
        instead, which is slow and was killed on falcon's 15 GB login node with a 140 MB tarball."""
        if self.bucket:
            obj = f"{self.bucket}/transfer/{os.path.basename(remote)}"
            r = subprocess.run([self.gcloud, "storage", "cp", "--quiet", local, obj], capture_output=True, text=True)
            if r.returncode == 0:
                self.run(f"set -e; gcloud storage cp --quiet {obj} ~/{remote} && gcloud storage rm --quiet {obj}",
                         timeout=1800)
                return
            log(f"bucket upload failed ({r.stderr.strip()[-300:]}); trying scp")
        cmd = [self.gcloud, "compute", "scp", *self.flags, local, f"{self.instance}:{remote}"]
        for i in range(3):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                return
            time.sleep(20 * (i + 1))
        raise RuntimeError(f"upload of {local} failed: {r.stderr.strip()[-1500:]}")


def git(args, cwd) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout


def code_key(root: Path, repos) -> str:
    h = hashlib.sha256()
    shas = []
    for sub in repos:
        d = root / sub
        sha = git(["rev-parse", "HEAD"], d).strip()
        shas.append(sha[:10])
        h.update(sha.encode())
        h.update(subprocess.run(["git", "diff", "HEAD", "--binary"], cwd=d, capture_output=True).stdout)
    return "-".join(shas) + "-" + h.hexdigest()[:8]


def tracked(root: Path, sub: str) -> list[str]:
    d = root / sub
    files = git(["ls-files", "-z"], d).split("\0")
    pre = f"{sub}/" if sub != "." else ""
    return [pre + f for f in files if f and (d / f).is_file()]


def referenced_files(texts: list[str], root: Path, known: set[str]) -> list[str]:
    """Untracked files under `root` named in the workflow files (e.g. output/<model>/common.yml)."""
    out = set()
    for text in texts:
        for tok in TOKEN.findall(text):
            p = (root / tok.split("@@")[0]).resolve()
            try:
                rel = str(p.relative_to(root.resolve()))
            except ValueError:
                continue
            if p.is_file() and rel not in known:
                out.add(rel)
            elif p.is_dir():
                out.update(str(f.relative_to(root.resolve())) for f in p.rglob("*")
                           if f.is_file() and "__pycache__" not in f.parts and f.stat().st_size < 50e6
                           and str(f.relative_to(root.resolve())) not in known)
    return sorted(out)


def make_tar(root: Path, files: list[str], path: str):
    with tarfile.open(path, "w:gz") as t:
        for f in files:
            t.add(str(root / f), arcname=f, recursive=False)


def rel_to_root(p: str, root: Path) -> str:
    q = Path(p)
    if q.is_absolute():
        try:
            return str(q.resolve().relative_to(root.resolve()))
        except ValueError:
            sys.exit(f"[orchard] {p} is outside the checkout {root}: cannot ship it")
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, help="the rule's {model_dir} (local, for orchard.json)")
    ap.add_argument("--wf", required=True); ap.add_argument("--common", required=True)
    ap.add_argument("--template", required=True, help="full template string, e.g. 'model: root://...'")
    ap.add_argument("--log", required=True, help="where the training log is copied (the rule's log)")
    ap.add_argument("--yields", required=True, help="where yields_loaded.json is copied")
    ap.add_argument("--label", default="", help="run label on ORCHARD (default: <roast_id>__<model_dir>)")
    ap.add_argument("--roast-id", default="")
    ap.add_argument("--config-paths", default="coffea4bees", help="CLASSIFIER_CONFIG_PATHS")
    ap.add_argument("--instance", default="orchard-login-001"); ap.add_argument("--zone", default="us-central1-c")
    ap.add_argument("--project", default="cmu-gpu-cloud"); ap.add_argument("--gcloud", default="")
    ap.add_argument("--orchard-project", default="", help="remote ORCHARD_PROJECT (default /project/community/$USER/barista)")
    ap.add_argument("--bucket", default="auto", help="gs:// prefix for transfers (default gs://cmu-gpucloud-<ORCHARD user>/barista; "
                                                      "'none': gcloud compute scp)")
    ap.add_argument("--time", default="4:00:00"); ap.add_argument("--cpus", default="16"); ap.add_argument("--mem", default="160G")
    ap.add_argument("--preempt", action="store_true"); ap.add_argument("--max-restarts", default="8")
    ap.add_argument("--submit-args", default="", help="more software/orchard/submit.sh options, quoted")
    ap.add_argument("--poll", type=int, default=300, help="seconds between status checks")
    ap.add_argument("--max-unreachable", type=int, default=7200, help="seconds without contact before giving up")
    a = ap.parse_args()

    root = Path.cwd()
    model_dir = Path(a.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    state_file = model_dir / "orchard.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    label = a.label or re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{a.roast_id or 'run'}__{a.model_dir.strip('/')}")[:150]
    wf, common = rel_to_root(a.wf, root), rel_to_root(a.common, root)
    rem = Remote(a)
    proj = a.orchard_project or "/project/community/$USER/barista"
    run_dir = f"{proj}/runs/{label}"
    signal.signal(signal.SIGTERM, lambda *_: (log(f"stopped; the ORCHARD job keeps running and a rerun "
                                                  f"reattaches to it ({state_file})"), sys.exit(143)))

    # ── preflight: account, proxy, image ──────────────────────────────────────────────────────────
    pre = rem.run(f"""
        P={proj}; echo "user $USER"
        openssl x509 -checkend 14400 -noout -in ~/.roast/x509_proxy >/dev/null 2>&1 && echo proxy ok || echo proxy BAD
        [ -s "$P/images/barista_classifier_latest.sqsh" ] && echo image ok || echo image MISSING
        [ -f "{run_dir}/train.done" ] && echo already done || true
    """)
    out = pre.stdout
    ruser = out.split()[1] if len(out.split()) > 1 else ""
    if a.bucket == "auto":
        rem.bucket = f"gs://cmu-gpucloud-{ruser}/barista" if ruser else ""
    elif a.bucket == "none":
        rem.bucket = ""
    log(f"ORCHARD reachable ({ruser or '?'}); run {label}")
    if "proxy BAD" in out:
        sys.exit("[orchard] no grid proxy with 4 h left in ORCHARD:~/.roast/x509_proxy: run `bin/roast proxy` "
                 "(or `bin/roast proxy --push --to orchard`)")
    if "image MISSING" in out:
        sys.exit(f"[orchard] no classifier image under {proj}/images (purged?): run software/orchard/setup_image.sh on ORCHARD")

    def fetch_back(final: bool) -> bool:
        r = rem.run(f"cat {run_dir}/train.log 2>/dev/null", check=False, timeout=900)
        if r.returncode == 0 and r.stdout:
            Path(a.log).write_text(r.stdout)
        if not final:
            return False
        y = rem.run(f"cat {run_dir}/yields_loaded.json 2>/dev/null", check=False)
        if y.returncode == 0 and y.stdout.strip():
            Path(a.yields).write_text(y.stdout)
        s = rem.run(f"bash {code_dir_of(state)}/software/orchard/summarize.sh {run_dir} 2>/dev/null; "
                    f"test -f {run_dir}/train.done", check=False)
        print(s.stdout, flush=True)
        return s.returncode == 0

    def code_dir_of(st):
        return st.get("code_dir") or f"{proj}/code/{state.get('key', 'none')}"

    if "already done" in out and not state.get("job"):
        log(f"{run_dir}/train.done exists on ORCHARD: copying the results back")
        state.setdefault("code_dir", f"{proj}/code")
        sys.exit(0 if fetch_back(True) else 1)

    # ── reattach or submit ────────────────────────────────────────────────────────────────────────
    def job_state(job) -> str:
        r = rem.run(f"sacct -j {job} -X -n -P -o State | tail -1; test -f {run_dir}/train.done && echo TRAIN_DONE",
                    check=False, timeout=300)
        if r.returncode == 255:
            return "UNREACHABLE"
        return ("TRAIN_DONE " if "TRAIN_DONE" in r.stdout else "") + (r.stdout.split()[0] if r.stdout.split() else "UNKNOWN")

    def submit():
        key = code_key(root, [".", "coffea4bees"])
        code_dir = f"{proj}/code/{key}"
        has = rem.run(f"test -f {code_dir}/.orchard_code_ok && echo yes || true").stdout.strip() == "yes"
        files = tracked(root, ".") + tracked(root, "coffea4bees")
        known = set(files)
        texts = [Path(wf).read_text(), Path(common).read_text(), a.template]
        extra = sorted(set(referenced_files(texts, root, known)) | {wf, common} - known)
        with tempfile.TemporaryDirectory() as tmp:
            if not has:
                log(f"shipping the code ({len(files)} files, key {key}) to ORCHARD:{code_dir}")
                make_tar(root, files, f"{tmp}/code.tgz")
                rem.put(f"{tmp}/code.tgz", f"orchard_code_{key}.tgz")
                rem.run(f"set -e; P={proj}; mkdir -p {code_dir}.part && tar xzf ~/orchard_code_{key}.tgz -C {code_dir}.part "
                        f"&& rm -f ~/orchard_code_{key}.tgz && rm -rf {code_dir} && mv {code_dir}.part {code_dir} "
                        f"&& touch {code_dir}/.orchard_code_ok", timeout=1800)
            log(f"shipping {len(extra)} workflow file(s): {', '.join(extra[:6])}{' ...' if len(extra) > 6 else ''}")
            make_tar(root, extra, f"{tmp}/wf.tgz")
            rem.put(f"{tmp}/wf.tgz", f"orchard_wf_{label}.tgz")
            rem.run(f"set -e; tar xzf ~/orchard_wf_{label}.tgz -C {code_dir} && rm -f ~/orchard_wf_{label}.tgz")
        opts = ["--source", "none", "--time", a.time, "--cpus", a.cpus, "--mem", a.mem,
                "--template-full", a.template, "--common", common]
        if a.preempt:
            opts.insert(0, "--preempt")
        cmd = (f"cd {code_dir} && ORCHARD_MAX_RESTARTS={shlex.quote(a.max_restarts)} "
               f"ORCHARD_CLASSIFIER_CONFIG_PATHS={shlex.quote(a.config_paths)} "
               + (f"ORCHARD_PROJECT={proj} " if a.orchard_project else "")
               + "software/orchard/submit.sh " + " ".join(shlex.quote(o) for o in opts)
               + (f" {a.submit_args}" if a.submit_args else "") + f" {shlex.quote(label)} {shlex.quote(wf)}")
        r = rem.run(cmd)
        m = re.search(r"Submitted batch job (\d+)", r.stdout)
        if not m:
            sys.exit(f"[orchard] submission failed:\n{r.stdout}\n{r.stderr}")
        st = {"job": m.group(1), "label": label, "key": key, "code_dir": code_dir, "run_dir": run_dir,
              "submitted": time.strftime("%F %T"), "attempt": state.get("attempt", 0) + 1}
        state_file.write_text(json.dumps(st, indent=1))
        log(f"submitted ORCHARD job {st['job']} (attempt {st['attempt']}): {r.stdout.strip().splitlines()[0]}")
        return st

    if state.get("job"):
        s = job_state(state["job"])
        log(f"found ORCHARD job {state['job']} from {state_file}: {s}")
        if s.startswith("TRAIN_DONE") or any(s.startswith(x) for x in ("PENDING", "RUNNING", "REQUEUED", "COMPLETING",
                                                                         "CONFIGURING", "RESIZING", "SUSPENDED", "PREEMPTED")):
            pass
        else:
            state = submit()
    else:
        state = submit()

    # ── wait ──────────────────────────────────────────────────────────────────────────────────────
    last_ok, n, retried = time.time(), 0, False
    while True:
        time.sleep(a.poll)
        n += 1
        s = job_state(state["job"])
        if s == "UNREACHABLE":
            if time.time() - last_ok > a.max_unreachable:
                sys.exit(f"[orchard] ORCHARD unreachable for {a.max_unreachable} s; job {state['job']} may still be "
                         f"running: rerun the step to reattach")
            continue
        last_ok = time.time()
        if n % 3 == 0:
            fetch_back(False)
        if s.startswith("TRAIN_DONE") or s.startswith("COMPLETED"):
            ok = fetch_back(True)
            log(f"ORCHARD job {state['job']} {s.split()[-1]}; training {'finished' if ok else 'did NOT finish'}")
            sys.exit(0 if ok else 1)
        if any(s.startswith(x) for x in TERMINAL_FAIL):
            fetch_back(False)
            tail = rem.run(f"tail -30 {run_dir}/logs/slurm-{state['job']}.out", check=False).stdout
            log(f"ORCHARD job {state['job']} ended {s}:\n{tail}")
            if not retried and "OUT_OF_MEMORY" not in s:
                retried = True
                log("resubmitting once (the training resumes from its last epoch checkpoint)")
                state = submit()
                continue
            sys.exit(1)


if __name__ == "__main__":
    main()
