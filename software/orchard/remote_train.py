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
            # 255: ssh/IAP could not connect; < 0: gcloud killed here (falcon's login-node watchdog
            # kills anything above 60 % CPU). Anything else is the script's own exit code.
            if r.returncode != 255 and r.returncode >= 0:
                break
            time.sleep(20 * (i + 1))
        if r.returncode < 0:
            r = subprocess.CompletedProcess(cmd, 255, r.stdout, r.stderr)
        if check and r.returncode != 0:
            raise RuntimeError(f"remote command failed ({r.returncode}): {r.stderr.strip()[-1500:]}")
        return r

    def put(self, local: str, remote: str):
        """Copy a file to ORCHARD:~/<remote>. Through the bucket when there is one: plain HTTPS from
        here and an in-cloud copy there. `gcloud compute scp` pushes it through the IAP tunnel
        instead, which is slow and was killed on falcon's 15 GB login node with a 140 MB tarball."""
        if self.bucket:
            # unique per process: several rule instances upload at once
            obj = f"{self.bucket}/transfer/{os.getpid()}_{int(time.time())}_{os.path.basename(remote)}"
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


def run_label(roast_id: str, model_dir: str, task_tag: str) -> str:
    """Unique, readable ORCHARD run name: <roast>__<model_dir>[__task]. A roast's model_dir usually
    contains the roast id again; drop that repeat, and if it is still long keep the start and the end
    (mix_<n>, the task) around a hash, so two models or tasks never share a run directory."""
    rid = roast_id or "run"
    md = model_dir.strip("/").replace(rid, "").replace("//", "/")
    full = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{rid}__{md}{task_tag}")
    full = re.sub(r"_{3,}", "__", full)
    if len(full) <= 150:
        return full
    h = hashlib.sha1(full.encode()).hexdigest()[:8]
    return f"{full[:70]}_{h}_{full[-70:]}"


def git(args, cwd) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout


# untracked files here count as code: a hot patch applied with `git apply` leaves its new files
# untracked, and `git diff HEAD` alone would not ship them (a batch job then found no batch_runner.py)
SOURCE_DIRS = {".": ("bin", "software", "src"), "coffea4bees": ("analysis", "classifier", "workflows")}
SOURCE_SUFFIXES = (".py", ".sh", ".smk", ".sbatch", ".yml", ".yaml")


def worktree_diff(root: Path, sub: str) -> bytes:
    """`git diff HEAD` of a repository plus its untracked source files (as new-file diffs)."""
    d = root / sub
    out = subprocess.run(["git", "diff", "HEAD", "--binary"], cwd=d, capture_output=True).stdout
    dirs = [x for x in SOURCE_DIRS.get(sub, ()) if (d / x).is_dir()]
    if dirs:
        new = git(["ls-files", "--others", "--exclude-standard", "-z", "--", *dirs], d).split("\0")
        for f in sorted(f for f in new if f.endswith(SOURCE_SUFFIXES) and "__pycache__" not in f):
            out += subprocess.run(["git", "diff", "--no-index", "--binary", "/dev/null", f], cwd=d,
                                  capture_output=True).stdout
    return out


def code_key(root: Path, repos) -> str:
    h = hashlib.sha256()
    shas = []
    for sub in repos:
        d = root / sub
        sha = git(["rev-parse", "HEAD"], d).strip()
        shas.append(sha[:10])
        h.update(sha.encode())
        h.update(worktree_diff(root, sub))
    return "-".join(shas) + "-" + h.hexdigest()[:8]


GIT_STORE = "$HOME/.orchard_git"     # bare copies of the repositories on ORCHARD (survive the purge)
REPOS = ((".", "barista"), ("coffea4bees", "coffea4bees"))


def ship_code(rem: "Remote", root: Path, proj: str, bootstrap: bool = False) -> str:
    """_ship_code under a lock on the driver host: the rule instances of one workflow start together
    and would otherwise all ship the same code at once (and collide on the remote bundle names)."""
    import fcntl
    lock_dir = root / ".orchard_pack"
    lock_dir.mkdir(exist_ok=True)
    with open(lock_dir / "ship.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        return _ship_code(rem, root, proj, bootstrap)


def _ship_code(rem: "Remote", root: Path, proj: str, bootstrap: bool = False) -> str:
    """Make ORCHARD:<proj>/code/<key> hold this checkout's code; return that path.

    ORCHARD cannot clone from gitlab.cern.ch without credentials, and falcon's login node kills any
    process above 60 % CPU, which rules out compressing the whole tree there. So ORCHARD keeps bare
    copies of the two repositories in ~/.orchard_git, and only what they lack travels: a git bundle
    of the missing commits (usually a few KB), and `git diff HEAD` for uncommitted hot patches.
    The very first copy is a full bundle (~0.7 GB): make it from a laptop with --bootstrap.
    """
    key = code_key(root, [d for d, _ in REPOS])
    code_dir = f"{proj}/code/{key}"
    if not bootstrap and rem.run(f"test -f {code_dir}/.orchard_code_ok && echo yes || true").stdout.strip() == "yes":
        return code_dir
    with tempfile.TemporaryDirectory() as tmp:
        build = [f"set -e; G={GIT_STORE}; C={code_dir}; rm -rf $C.part; mkdir -p $C.part"]
        for sub, name in REPOS:
            d = root / sub
            sha = git(["rev-parse", "HEAD"], d).strip()
            have = rem.run(f"G={GIT_STORE}; [ -d $G/{name}.git ] || git init -q --bare $G/{name}.git; "
                           f"git --git-dir=$G/{name}.git cat-file -e {sha}^{{commit}} 2>/dev/null && echo HAVE || "
                           f"git --git-dir=$G/{name}.git for-each-ref --format='%(objectname)'").stdout.split()
            if "HAVE" not in have or bootstrap:
                cands = sorted({h for h in have if re.fullmatch(r"[0-9a-f]{40}", h)})
                ok = subprocess.run(["git", "cat-file", "--batch-check"], cwd=d, input="\n".join(cands),
                                    capture_output=True, text=True).stdout.split("\n")
                basis = [w[0] for w in (line.split() for line in ok) if len(w) > 1 and w[1] == "commit"]
                if not basis and not bootstrap:
                    sys.exit(f"[orchard] ORCHARD has no copy of {name} yet: run once, from a laptop (not a login node "
                             f"with a CPU watchdog), `software/orchard/remote_train.py --bootstrap`")
                bundle = f"{tmp}/{name}.bundle"
                # the bootstrap sends every branch, origin/* included, so later checkouts (roasts pin
                # pushed commits) always share history with ORCHARD and send only a small bundle
                revs = ["--all"] if bootstrap else ["HEAD", *[f"^{b}" for b in basis[:50]]]
                subprocess.run(["git", "bundle", "create", "-q", bundle, *revs], cwd=d, check=True, capture_output=True)
                log(f"{name}: sending {os.path.getsize(bundle) / 1e6:.1f} MB of commits ({len(basis)} known refs on ORCHARD)")
                rem.put(bundle, f"orchard_{name}.bundle")
                spec = "'+refs/*:refs/boot/*'" if bootstrap else f"HEAD:refs/orchard/{sha}"
                rem.run(f"set -e; G={GIT_STORE}; git --git-dir=$G/{name}.git fetch -q ~/orchard_{name}.bundle "
                        f"{spec} && rm -f ~/orchard_{name}.bundle", timeout=3600)
            if bootstrap:
                continue
            dst = "$C.part" + ("" if sub == "." else f"/{sub}")
            build.append(f"mkdir -p {dst} && git --git-dir=$G/{name}.git archive {sha} | tar -x -C {dst}")
            diff = worktree_diff(root, sub)
            if diff.strip():
                Path(f"{tmp}/{name}.diff").write_bytes(diff)
                rem.put(f"{tmp}/{name}.diff", f"orchard_{name}_{key}.diff")
                build.append(f"(cd {dst} && git apply --whitespace=nowarn ~/orchard_{name}_{key}.diff) && rm -f ~/orchard_{name}_{key}.diff")
                log(f"{name}: uncommitted changes applied ({len(diff)} bytes)")
        if bootstrap:
            log("repositories copied to ORCHARD:~/.orchard_git")
            return code_dir
        build.append("rm -rf $C && mv $C.part $C && touch $C/.orchard_code_ok")
        rem.run("\n".join(build), timeout=1800)
    log(f"code {key} ready on ORCHARD:{code_dir}")
    return code_dir


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
    ap.add_argument("--bootstrap", action="store_true", help="only copy this checkout's repositories to ORCHARD "
                                                              "(once, from a laptop) and exit")
    ap.add_argument("--model-dir", help="the rule's {model_dir} (local, for orchard.json)")
    ap.add_argument("--wf"); ap.add_argument("--common")
    ap.add_argument("--template", help="full template string, e.g. 'model: root://...'")
    ap.add_argument("--log", help="where the training log is copied (the rule's log)")
    ap.add_argument("--yields", help="where yields_loaded.json is copied (training)")
    ap.add_argument("--task", default="train", help="train, or evaluate[_<split>]: names the ORCHARD run "
                                                    "(<roast>__<model_dir>__<task>); an evaluation is not checkpointed, "
                                                    "so it is not requeued at the time limit")
    ap.add_argument("--label", default="", help="run label on ORCHARD (default: <roast_id>__<model_dir>)")
    ap.add_argument("--roast-id", default="")
    ap.add_argument("--config-paths", default="coffea4bees", help="CLASSIFIER_CONFIG_PATHS")
    ap.add_argument("--instance", default="orchard-login-001"); ap.add_argument("--zone", default="us-central1-c")
    ap.add_argument("--project", default="cmu-gpu-cloud"); ap.add_argument("--gcloud", default="")
    ap.add_argument("--orchard-project", default="", help="remote ORCHARD_PROJECT (default /project/community/$USER/barista)")
    ap.add_argument("--bucket", default="auto", help="gs:// prefix for transfers (default gs://cmu-gpucloud-<ORCHARD user>/barista; "
                                                      "'none': gcloud compute scp)")
    ap.add_argument("--time", default="4:00:00"); ap.add_argument("--cpus", default="16"); ap.add_argument("--mem", default="160G")
    ap.add_argument("--preempt", action="store_true"); ap.add_argument("--max-restarts", default="")
    ap.add_argument("--submit-args", default="", help="more software/orchard/submit.sh options, quoted")
    ap.add_argument("--pack", type=int, default=1, help="run up to N tasks of the same kind (training / evaluation) "
                                                        "in one ORCHARD job, sharing the GPU under MPS")
    ap.add_argument("--pack-wait", type=int, default=120, help="seconds a task waits for others to fill its batch")
    ap.add_argument("--pack-cpus", type=int, default=8, help="CPUs per task in a batch")
    ap.add_argument("--pack-mem", type=int, default=64, help="GB of memory per task in a batch")
    ap.add_argument("--poll", type=int, default=300, help="seconds between status checks")
    ap.add_argument("--max-unreachable", type=int, default=7200, help="seconds without contact before giving up")
    a = ap.parse_args()
    root = Path.cwd()
    if a.bootstrap:
        rem = Remote(a)
        u = rem.run("echo $USER").stdout.strip()
        rem.bucket = "" if a.bucket == "none" else (a.bucket if a.bucket != "auto" else f"gs://cmu-gpucloud-{u}/barista")
        ship_code(rem, root, a.orchard_project or "/project/community/$USER/barista", bootstrap=True)
        return
    for k in ("model_dir", "wf", "common", "template", "log"):
        if getattr(a, k) is None:   # an empty --template is fine: a staged workflow may need none
            ap.error(f"--{k.replace('_', '-')} is required")

    model_dir = Path(a.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    state_file = model_dir / ("orchard.json" if a.task == "train" else f"orchard_{a.task}.json")
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    evaluating = a.task.startswith("evaluate")
    if not a.max_restarts:
        a.max_restarts = "0" if evaluating else "8"
    task_tag = "" if a.task == "train" else f"__{a.task}"
    label = a.label or run_label(a.roast_id, a.model_dir, task_tag)
    if state.get("label") and not a.label:
        label = state["label"]          # keep reattaching to the run this model started with
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
        y = rem.run(f"cat {run_dir}/yields_loaded.json 2>/dev/null", check=False) if a.yields else None
        if y is not None and y.returncode == 0 and y.stdout.strip():
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

    def sbatch(cmd: str) -> str:
        """Run a submit.sh command on ORCHARD; wait while the per-user submit limit is reached."""
        while True:
            r = rem.run(cmd, check=False)
            # ORCHARD's QOS caps a user at a few submitted jobs (general: 5). A workflow with more models
            # than that (bkg_syst subsamples) waits here for a slot instead of failing the rule.
            if re.search(r"QOSMaxSubmitJob|AssocMaxSubmitJob|MaxSubmitJobs", r.stdout + r.stderr):
                log(f"ORCHARD queue full for this user (submit limit); retrying in {a.poll} s")
                time.sleep(a.poll)
                continue
            m = re.search(r"Submitted batch job (\d+)", r.stdout)
            if not m:
                sys.exit(f"[orchard] submission failed:\n{r.stdout}\n{r.stderr}")
            return m.group(1)

    key_now, code_dir_now = None, None

    def submit_packed(key: str, code_dir: str) -> dict:
        """Join a batch of up to --pack tasks of the same kind and code, run as one ORCHARD job.

        The rule instances of one driver coordinate through <checkout>/.orchard_pack: each writes its
        task to pending/, and whichever holds the lock when a batch is full (or its oldest task has
        waited --pack-wait seconds) submits it and writes assigned/<label>.json for every member.
        Each instance then waits for its own task as for a single job: the batch job id with its
        own run directory and train.done.
        """
        import fcntl
        nonlocal key_now, code_dir_now
        key_now, code_dir_now = key, code_dir
        kind = "evaluate" if evaluating else "train"
        lit = lambda p: p.replace("$USER", ruser)   # paths inside the batch file are used verbatim
        pack_dir = root / ".orchard_pack"
        for sub in ("pending", "assigned"):
            (pack_dir / sub).mkdir(parents=True, exist_ok=True)
        me = {"label": label, "run_dir": lit(run_dir), "wf": f"{lit(code_dir)}/{wf}",
              "common": f"{lit(code_dir)}/{common}", "wf_rel": wf, "common_rel": common,
              "template": a.template, "key": key, "kind": kind, "t": time.time()}
        assigned = pack_dir / "assigned" / f"{label}.json"
        if assigned.exists():
            # A batch was submitted for this task but the instance that waited for it was stopped
            # before it wrote its state file (driver interrupted and resumed): adopt that batch while
            # its job is alive, instead of packing the task again into a second job (same run_dir).
            try:
                prev = json.loads(assigned.read_text())
            except (OSError, ValueError):
                prev = {}
            alive = ("PENDING", "RUNNING", "REQUEUED", "COMPLETING", "CONFIGURING", "RESIZING", "SUSPENDED")
            s = job_state(prev["job"]) if prev.get("job") else ""
            if s.startswith(alive) or s.startswith("TRAIN_DONE"):
                assigned.unlink(missing_ok=True)
                (pack_dir / "pending" / f"{label}.json").unlink(missing_ok=True)
                log(f"adopting ORCHARD job {prev['job']} (batch {prev.get('batch')}) assigned before a restart: {s}")
                return _packed_state(prev)
        assigned.unlink(missing_ok=True)
        tmp = pack_dir / "pending" / f".{label}.tmp"
        tmp.write_text(json.dumps(me))
        tmp.rename(pack_dir / "pending" / f"{label}.json")
        log(f"waiting to be packed (up to {a.pack} {kind} tasks per ORCHARD job)")
        while True:
            with open(pack_dir / "lock", "w") as lk:
                fcntl.flock(lk, fcntl.LOCK_EX)
                if assigned.exists():
                    info = json.loads(assigned.read_text())
                    assigned.unlink()
                    break
                group = []
                for p in (pack_dir / "pending").glob("*.json"):
                    try:
                        t = json.loads(p.read_text())
                    except (OSError, ValueError):
                        continue
                    if t.get("key") == key and t.get("kind") == kind:
                        group.append(t)
                group.sort(key=lambda t: t["t"])
                if group and (len(group) >= a.pack or time.time() - group[0]["t"] >= a.pack_wait):
                    batch = group[:a.pack]
                    names = "".join(t["label"] for t in batch)
                    blabel = run_label(a.roast_id, f"batch_{kind}_{hashlib.sha1(names.encode()).hexdigest()[:8]}", "")
                    bfile = f"{code_dir}/.orchard_batches/{blabel}.json"
                    rem.run(f"mkdir -p {code_dir}/.orchard_batches && cat > {bfile} <<'ORCHARD_EOF'\n"
                            f"{json.dumps(batch, indent=1)}\nORCHARD_EOF")
                    n = len(batch)
                    opts = ["--source", "none", "--time", a.time, "--cpus", str(n * a.pack_cpus),
                            "--mem", f"{n * a.pack_mem}G", "--template-full", "", "--common", batch[0]["common_rel"]]
                    if a.preempt:
                        opts.insert(0, "--preempt")
                    job = sbatch(f"cd {code_dir} && ORCHARD_BATCH={bfile} ORCHARD_MAX_RESTARTS={shlex.quote(a.max_restarts)} "
                                 f"ORCHARD_CLASSIFIER_CONFIG_PATHS={shlex.quote(a.config_paths)} "
                                 + (f"ORCHARD_PROJECT={proj} " if a.orchard_project else "")
                                 + "software/orchard/submit.sh " + " ".join(shlex.quote(o) for o in opts)
                                 + (f" {a.submit_args}" if a.submit_args else "")
                                 + f" {shlex.quote(blabel)} {shlex.quote(batch[0]['wf_rel'])}")
                    brun = f"{proj}/runs/{blabel}"
                    for t in batch:
                        (pack_dir / "assigned" / f"{t['label']}.json").write_text(
                            json.dumps({"job": job, "batch": blabel, "batch_run_dir": brun, "size": n}))
                        (pack_dir / "pending" / f"{t['label']}.json").unlink(missing_ok=True)
                    log(f"submitted ORCHARD job {job}: batch {blabel} of {n} {kind} task(s)")
                    continue
            time.sleep(10)
        return _packed_state(info)

    def _packed_state(info: dict) -> dict:
        """Record the batch job this task runs in (state file), as for a single job."""
        st = {"job": info["job"], "label": label, "key": key_now, "code_dir": code_dir_now, "run_dir": run_dir,
              "batch": info["batch"], "batch_run_dir": info["batch_run_dir"],
              "submitted": time.strftime("%F %T"), "attempt": state.get("attempt", 0) + 1}
        state_file.write_text(json.dumps(st, indent=1))
        log(f"in ORCHARD job {st['job']} (batch {info['batch']}, {info.get('size', '?')} tasks; attempt {st['attempt']})")
        return st

    def submit():
        key = code_key(root, [d for d, _ in REPOS])
        code_dir = ship_code(rem, root, proj)
        known = set(tracked(root, ".") + tracked(root, "coffea4bees"))
        texts = [Path(wf).read_text(), Path(common).read_text(), a.template]
        extra = sorted(set(referenced_files(texts, root, known)) | {wf, common} - known)
        with tempfile.TemporaryDirectory() as tmp:
            log(f"shipping {len(extra)} workflow file(s): {', '.join(extra[:6])}{' ...' if len(extra) > 6 else ''}")
            make_tar(root, extra, f"{tmp}/wf.tgz")
            rem.put(f"{tmp}/wf.tgz", f"orchard_wf_{label}.tgz")
            # concurrent tasks extract overlapping files (shared metadata) into the same code_dir; GNU tar's
            # unlink+O_EXCL create then races ("Cannot open: File exists"), so retry a few times
            rem.run(f"f=~/orchard_wf_{label}.tgz; ok=0; for i in 1 2 3 4 5; do "
                    f"if tar xzf $f -C {code_dir}; then ok=1; break; fi; sleep $((RANDOM % 5 + 1)); done; "
                    f"rm -f $f; [ $ok = 1 ]")
        if a.pack > 1:
            return submit_packed(key, code_dir)
        opts = ["--source", "none", "--time", a.time, "--cpus", a.cpus, "--mem", a.mem,
                "--template-full", a.template, "--common", common]
        if a.preempt:
            opts.insert(0, "--preempt")
        cmd = (f"cd {code_dir} && ORCHARD_MAX_RESTARTS={shlex.quote(a.max_restarts)} "
               f"ORCHARD_CLASSIFIER_CONFIG_PATHS={shlex.quote(a.config_paths)} "
               + (f"ORCHARD_PROJECT={proj} " if a.orchard_project else "")
               + "software/orchard/submit.sh " + " ".join(shlex.quote(o) for o in opts)
               + (f" {a.submit_args}" if a.submit_args else "") + f" {shlex.quote(label)} {shlex.quote(wf)}")
        job = sbatch(cmd)
        st = {"job": job, "label": label, "key": key, "code_dir": code_dir, "run_dir": run_dir,
              "submitted": time.strftime("%F %T"), "attempt": state.get("attempt", 0) + 1}
        state_file.write_text(json.dumps(st, indent=1))
        log(f"submitted ORCHARD job {st['job']} (attempt {st['attempt']})")
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
            log(f"ORCHARD job {state['job']} {s.split()[-1]}; {a.task} {'finished' if ok else 'did NOT finish'}")
            sys.exit(0 if ok else 1)
        if any(s.startswith(x) for x in TERMINAL_FAIL):
            fetch_back(False)
            tail = rem.run(f"tail -30 {state.get('batch_run_dir', run_dir)}/logs/slurm-{state['job']}.out; "
                           f"tail -15 {run_dir}/train.log", check=False).stdout
            log(f"ORCHARD job {state['job']} ended {s}:\n{tail}")
            if not retried and "OUT_OF_MEMORY" not in s:
                retried = True
                log("resubmitting once (the training resumes from its last epoch checkpoint)")
                state = submit()
                continue
            sys.exit(1)


if __name__ == "__main__":
    main()
