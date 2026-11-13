#!/usr/bin/env python3
"""Pull agent for classifier trainings/evaluations on NCSA Delta (see docs/software/delta.md).

Runs in tmux on a Delta login node, started once by hand (Delta needs Duo for every login, so no
driver can log in by itself). It only makes outbound calls: XRootD to the EOS spool
(software/delta/spool.py) and Slurm locally. Every --poll seconds it

  1. writes a heartbeat (agent/alive.json) and the commits its bare repositories hold
     (agent/refs_<repo>.txt), which the drivers bundle against;
  2. builds every code state a driver asked for (code/<key>/manifest.json): fetches the bundles into
     $DELTA_HOME/git/<repo>.git, `git archive`s the pinned commits into $DELTA_HOME/code/<key>, applies
     the diffs, and marks code/<key>/ready;
  3. submits every new request (req/<rid>.json) with software/delta/train.sbatch on the GPU account;
  4. follows its jobs with sacct, publishing status/<rid>.json and the run's train.log, and at the end
     yields_loaded.json and summary.txt (runs/<label>/).

Its own state is in $DELTA_HOME/agent_state.json, so a restarted agent picks its jobs up again.
Run with the barista pixi environment (xrdfs/xrdcp), e.g. `software/delta/agent.sh`.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from spool import Spool, default_spool, safe  # noqa: E402

REPOS = ("barista", "coffea4bees")      # coffea4bees is extracted into <code>/coffea4bees
FINAL = ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE")


def log(*a):
    print(f"[delta-agent {time.strftime('%F %T')}]", *a, flush=True)


def sh(cmd, **kw) -> subprocess.CompletedProcess:
    kw.setdefault("capture_output", True)
    kw.setdefault("text", True)
    return subprocess.run(cmd, **kw)


class Agent:
    def __init__(self, a):
        self.a = a
        self.sp = Spool(a.spool)
        self.home = Path(a.home)
        self.git = self.home / "git"
        self.code = self.home / "code"
        self.runs = self.home / "runs"
        for d in (self.git, self.code, self.runs):
            d.mkdir(parents=True, exist_ok=True)
        self.state_file = self.home / "agent_state.json"
        self.state = json.loads(self.state_file.read_text()) if self.state_file.exists() else {"reqs": {}}
        self.bad_code: set[str] = set()

    def save(self):
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1))
        tmp.replace(self.state_file)

    # ── heartbeat and repositories ───────────────────────────────────────────────────────────────
    def proxy_hours(self) -> float:
        p = os.environ.get("X509_USER_PROXY", "")
        r = sh(["openssl", "x509", "-enddate", "-noout", "-in", p]) if p else None
        if not r or r.returncode != 0:
            return 0.0
        end = r.stdout.strip().split("=", 1)[-1]
        r2 = sh(["date", "-d", end, "+%s"])
        return round((int(r2.stdout.strip()) - time.time()) / 3600, 1) if r2.returncode == 0 else 0.0

    def heartbeat(self):
        active = [k for k, v in self.state["reqs"].items() if not v.get("final")]
        ph = self.proxy_hours()
        if ph < 24:
            log(f"WARNING: grid proxy has {ph} h left: renew it (`bin/roast proxy --push --to delta`)")
        self.sp.put_json("agent/alive.json", {"time": time.time(), "host": socket.gethostname(),
                                              "user": os.environ.get("USER"), "proxy_hours": ph,
                                              "active": active, "account": self.a.account})

    def publish_refs(self):
        for name in REPOS:
            g = self.git / f"{name}.git"
            if g.exists():
                refs = sh(["git", "--git-dir", str(g), "for-each-ref", "--format=%(objectname)"]).stdout.split()
                self.sp.put_text(f"agent/refs_{name}.txt", "\n".join(sorted(set(refs))[-2000:]) + "\n")

    def ensure_repos(self) -> bool:
        missing = [n for n in REPOS if not (self.git / f"{n}.git").exists()]
        for name in missing:
            with tempfile.TemporaryDirectory(dir=self.home) as tmp:
                b = f"{tmp}/{name}.bundle"
                if not self.sp.get(f"bootstrap/{name}.bundle", b):
                    log(f"no repository {name} yet and no bootstrap bundle: run `software/delta/remote.py --bootstrap` "
                        f"on the driver, or `git clone --bare` a bundle into {self.git}/{name}.git")
                    return False
                r = sh(["git", "clone", "-q", "--bare", b, str(self.git / f"{name}.git")])
                if r.returncode != 0:
                    log(f"clone of the {name} bundle failed: {r.stderr.strip()}")
                    return False
                log(f"{name}: bare repository created from the bootstrap bundle")
        if missing:
            self.publish_refs()
        return True

    # ── code ─────────────────────────────────────────────────────────────────────────────────────
    def build_code(self, key: str) -> bool:
        dst = self.code / key
        if (dst / ".delta_code_ok").exists():
            if not self.sp.exists(f"code/{key}/ready"):
                self.sp.put_text(f"code/{key}/ready", "ok\n")
            return True
        if key in self.bad_code:
            return False
        man = self.sp.get_json(f"code/{key}/manifest.json")
        if not man:
            return False
        part = self.code / f"{key}.part"
        shutil.rmtree(part, ignore_errors=True)
        try:
            with tempfile.TemporaryDirectory(dir=self.home) as tmp:
                for name in REPOS:
                    e = man["repos"][name]
                    g = str(self.git / f"{name}.git")
                    if e.get("bundle"):
                        b = f"{tmp}/{name}.bundle"
                        if not self.sp.get(f"code/{key}/{name}.bundle", b):
                            raise RuntimeError(f"cannot download code/{key}/{name}.bundle")
                        r = sh(["git", "--git-dir", g, "fetch", "-q", b, f"HEAD:refs/delta/{e['sha']}"])
                        if r.returncode != 0:
                            raise RuntimeError(f"{name}: git fetch of the bundle failed: {r.stderr.strip()}")
                    else:
                        sh(["git", "--git-dir", g, "update-ref", f"refs/delta/{e['sha']}", e["sha"]])
                    sub = part if name == "barista" else part / name
                    sub.mkdir(parents=True, exist_ok=True)
                    r = sh(f"git --git-dir={shlex.quote(g)} archive {e['sha']} | tar -x -C {shlex.quote(str(sub))}",
                           shell=True)
                    if r.returncode != 0:
                        raise RuntimeError(f"{name}: git archive {e['sha']} failed: {r.stderr.strip()}")
                    if e.get("diff"):
                        d = f"{tmp}/{name}.diff"
                        if not self.sp.get(f"code/{key}/{name}.diff", d):
                            raise RuntimeError(f"cannot download code/{key}/{name}.diff")
                        r = sh(["git", "apply", "--whitespace=nowarn", d], cwd=sub)
                        if r.returncode != 0:
                            raise RuntimeError(f"{name}: git apply failed: {r.stderr.strip()}")
            (part / ".delta_code_ok").touch()
            shutil.rmtree(dst, ignore_errors=True)
            part.rename(dst)
        except Exception as e:  # noqa: BLE001 -- reported to the driver
            log(f"code {key}: {e}")
            self.sp.put_text(f"code/{key}/error", f"{e}\n")
            self.bad_code.add(key)
            shutil.rmtree(part, ignore_errors=True)
            return False
        self.sp.put_text(f"code/{key}/ready", "ok\n")
        self.publish_refs()
        log(f"code {key} built in {dst}")
        return True

    def pending_code(self):
        for key in self.sp.ls("code"):
            if not (self.code / key / ".delta_code_ok").exists() and key not in self.bad_code \
                    and self.sp.exists(f"code/{key}/manifest.json"):
                self.build_code(key)

    # ── requests ─────────────────────────────────────────────────────────────────────────────────
    def status(self, rid: str, state: str, **kw):
        rec = self.state["reqs"].setdefault(rid, {})
        rec.update(kw)
        rec["state"] = state
        self.sp.put_json(f"status/{rid}.json", {"rid": rid, "state": state, "updated": time.strftime("%F %T"),
                                                **{k: v for k, v in rec.items() if k in ("job", "msg", "tail", "label")}})
        self.save()

    def submit(self, req: dict) -> None:
        rid, label, key = req["rid"], safe(req["label"]), req["key"]
        if not self.build_code(key):
            if key in self.bad_code:
                self.status(rid, "ERROR", msg=f"code {key} could not be built", final=True, label=label)
            return                                    # manifest not there yet: try again next pass
        code = self.code / key
        with tempfile.TemporaryDirectory(dir=self.home) as tmp:
            if self.sp.get(f"req/{rid}.wf.tgz", f"{tmp}/wf.tgz"):
                with tarfile.open(f"{tmp}/wf.tgz") as t:
                    t.extractall(code, filter="data") if sys.version_info >= (3, 12) else t.extractall(code)
        run_dir = self.runs / label
        if (run_dir / "train.done").exists():
            log(f"{rid}: {run_dir}/train.done exists, reporting it done")
            self.status(rid, "DONE", label=label, run_dir=str(run_dir), final=True)
            self.upload(rid, final=True)
            return
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        env = dict(os.environ,
                   DELTA_LABEL=label, DELTA_TASK=req.get("task", "train"), DELTA_BARISTA=str(code),
                   DELTA_TRAIN=str(code / req["wf"]), DELTA_COMMON=str(code / req["common"]),
                   DELTA_RUN_DIR=str(run_dir), DELTA_TEMPLATE_FULL=req["template"],
                   DELTA_CLASSIFIER_CONFIG_PATHS=req.get("config_paths") or "coffea4bees",
                   DELTA_MAX_RESTARTS=str(req.get("max_restarts") or self.a.max_restarts))
        cmd = ["sbatch", "--parsable", f"--job-name=barista-{label}"[:120],
               f"--account={self.a.account}", f"--partition={req.get('partition') or self.a.partition}",
               f"--time={req.get('time') or self.a.time}", f"--cpus-per-task={req.get('cpus') or self.a.cpus}",
               f"--mem={req.get('mem') or self.a.mem}", "--gpus-per-node=1", "--nodes=1", "--ntasks=1",
               f"--output={run_dir}/logs/slurm-%j.out", "--export=ALL",
               *shlex.split(self.a.sbatch_args), str(code / "software" / "delta" / "train.sbatch")]
        r = sh(cmd, env=env, cwd=str(code))
        if r.returncode != 0:
            log(f"{rid}: sbatch failed: {r.stderr.strip()}")
            self.status(rid, "ERROR", msg=f"sbatch failed: {r.stderr.strip()[-800:]}", final=True, label=label)
            return
        job = r.stdout.strip().split(";")[0]
        log(f"{rid}: submitted job {job} ({' '.join(cmd[3:8])})")
        self.status(rid, "PENDING", job=job, label=label, run_dir=str(run_dir), submitted=time.time(), final=False)

    def job_state(self, job: str) -> str:
        r = sh(["sacct", "-j", job, "-X", "-n", "-P", "-o", "State"])
        s = r.stdout.split()
        return s[-1] if s else ("UNKNOWN" if r.returncode == 0 else "SACCT_ERROR")

    def upload(self, rid: str, final: bool):
        rec = self.state["reqs"][rid]
        run_dir = Path(rec["run_dir"])
        label = rec["label"]
        names = ["train.log"] + (["yields_loaded.json", "summary.txt"] if final else [])
        for n in names:
            p = run_dir / n
            if p.exists() and (final or p.stat().st_mtime > rec.get("uploaded", 0)):
                try:
                    self.sp.put(str(p), f"runs/{label}/{n}")
                except RuntimeError as e:
                    log(f"{rid}: {e}")
        rec["uploaded"] = time.time()

    def follow(self, rid: str, rec: dict, n: int):
        s = self.job_state(rec["job"])
        if s == "SACCT_ERROR":
            return
        run_dir = Path(rec["run_dir"])
        if s != rec.get("state"):
            log(f"{rid}: job {rec['job']} {s}")
        if s.split()[0] in FINAL or s.startswith("CANCELLED"):
            done = (run_dir / "train.done").exists()
            tail = ""
            if not done:
                logs = sorted((run_dir / "logs").glob(f"slurm-{rec['job']}*.out"))
                if logs:
                    tail = "\n".join(logs[-1].read_text(errors="replace").splitlines()[-30:])
            self.upload(rid, final=True)
            state = "DONE" if done else (s if s != "COMPLETED" else "FAILED")
            # a worker killed by the cgroup leaves the batch step FAILED, not OUT_OF_MEMORY
            if not done and "oom" in (tail + sh(["sacct", "-j", rec["job"], "-n", "-P", "-o", "State"]).stdout).lower():
                state = "OUT_OF_MEMORY"
            self.status(rid, state, tail=tail, final=True)
            return
        if s != rec.get("state"):
            self.status(rid, s)
        if n % self.a.log_every == 0:
            self.upload(rid, final=False)

    def cycle(self, n: int):
        self.heartbeat()
        if not self.ensure_repos():
            return
        if n == 0:
            self.publish_refs()
        self.pending_code()
        for f in self.sp.ls("req"):
            if not f.endswith(".json"):
                continue
            rid = f[:-5]
            if rid in self.state["reqs"] and (self.state["reqs"][rid].get("job") or self.state["reqs"][rid].get("final")):
                continue
            req = self.sp.get_json(f"req/{f}")
            if req and req.get("rid") == rid:
                self.submit(req)
        for rid, rec in list(self.state["reqs"].items()):
            if rec.get("job") and not rec.get("final"):
                self.follow(rid, rec, n)


def main():
    proj = os.environ.get("DELTA_PROJECT", "bjsv")
    user = os.environ.get("USER", "")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spool", default=default_spool(), help="EOS spool URL (as the drivers use; $DELTA_SPOOL)")
    ap.add_argument("--home", default=os.environ.get("DELTA_HOME", f"/projects/{proj}/{user}/barista_delta"),
                    help="code, bare repositories and run directories ($DELTA_HOME)")
    ap.add_argument("--account", default=os.environ.get("DELTA_GPU_ACCOUNT", f"{proj}-delta-gpu"))
    ap.add_argument("--partition", default="gpuA100x4", help="default partition (a request may override)")
    # one long job per fit: every requeue re-enters the queue, and on Delta a job start can take many
    # hours once the account has some usage (docs/software/delta.md: 4 chunks of 1:30 = 16.6 h queued)
    ap.add_argument("--time", default="8:00:00", help="wall time per job; trainings requeue themselves "
                                                      "15 min before it and resume from the last epoch")
    ap.add_argument("--cpus", default="16"); ap.add_argument("--mem", default="62g")
    ap.add_argument("--max-restarts", default="12")
    ap.add_argument("--sbatch-args", default="", help="more sbatch options for every job, quoted")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--log-every", type=int, default=5, help="upload train.log every N polls")
    ap.add_argument("--once", action="store_true", help="one pass, then exit (testing)")
    a = ap.parse_args()
    if not shutil.which("xrdfs"):
        sys.exit("[delta-agent] no xrdfs on PATH: start it with software/delta/agent.sh (pixi environment)")
    ag = Agent(a)
    log(f"spool {a.spool}; home {a.home}; account {a.account}; partition {a.partition}; time {a.time}")
    n = 0
    while True:
        try:
            ag.cycle(n)
        except Exception as e:  # noqa: BLE001 -- an EOS hiccup must not kill the agent
            log(f"ERROR in pass {n}: {type(e).__name__}: {e}")
        if a.once:
            return
        n += 1
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
