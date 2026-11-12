"""EOS spool shared by the Delta driver (software/delta/remote.py, on cmslpc/falcon) and the Delta
agent (software/delta/agent.py, on a Delta login node). Plain XRootD command-line tools only
(xrdfs, xrdcp: on cmslpc natively, elsewhere in the barista pixi environment).

Layout under the spool URL (default root://cmseos.fnal.gov//store/user/$USER/barista_delta):

    agent/alive.json             agent heartbeat (time, host, proxy hours left, active jobs)
    agent/refs_<repo>.txt        commits the agent's bare repositories have (basis for bundles)
    bootstrap/<repo>.bundle      full bundles for an agent that has no repositories yet
    code/<key>/                  <repo>.bundle, <repo>.diff, manifest.json (written last), ready | error
    req/<rid>.json, .wf.tgz      one request per attempt (json written last)
    status/<rid>.json            the agent's view of that request
    runs/<label>/                train.log, yields_loaded.json, summary.txt, gpu.csv, timing.txt

Nothing secret goes into the spool: the grid proxy stays on each host (~/.roast/x509_proxy).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time

DEFAULT_SPOOL = "root://cmseos.fnal.gov//store/user/{user}/barista_delta"


def default_spool() -> str:
    return os.environ.get("DELTA_SPOOL") or DEFAULT_SPOOL.format(user=os.environ.get("USER", "unknown"))


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:180]


class Spool:
    def __init__(self, url: str):
        m = re.match(r"(root://[^/]+)/(/.*)$", url.rstrip("/"))
        if not m:
            raise ValueError(f"spool must be root://host//path, got {url!r}")
        self.host, self.path, self.url = m.group(1), m.group(2), url.rstrip("/")

    # ── low level ────────────────────────────────────────────────────────────────────────────────
    def _fs(self, *args, timeout=300) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(["xrdfs", self.host, *args], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            return subprocess.CompletedProcess(e.cmd, 124, "", f"timeout after {timeout} s")

    def full(self, rel: str) -> str:
        return f"{self.path}/{rel}"

    def mkdir(self, rel: str):
        self._fs("mkdir", "-p", self.full(rel))

    def exists(self, rel: str) -> bool:
        return self._fs("stat", self.full(rel), timeout=120).returncode == 0

    def ls(self, rel: str) -> list[str]:
        r = self._fs("ls", self.full(rel), timeout=120)
        if r.returncode != 0:
            return []
        return [os.path.basename(x.rstrip("/")) for x in r.stdout.split() if x.strip()]

    def rm(self, rel: str):
        self._fs("rm", self.full(rel), timeout=120)

    def put(self, local: str, rel: str, retries=3):
        """Upload a file (overwriting); the parent directory is created."""
        self.mkdir(os.path.dirname(rel))
        dst = f"{self.host}/{self.full(rel)}"
        err = ""
        for i in range(retries):
            r = subprocess.run(["xrdcp", "-f", "-s", local, dst], capture_output=True, text=True, timeout=3600)
            if r.returncode == 0:
                return
            err = r.stderr.strip()[-500:]
            time.sleep(10 * (i + 1))
        raise RuntimeError(f"xrdcp {local} -> {dst} failed: {err}")

    def get(self, rel: str, local: str) -> bool:
        r = subprocess.run(["xrdcp", "-f", "-s", f"{self.host}/{self.full(rel)}", local],
                           capture_output=True, text=True, timeout=3600)
        return r.returncode == 0

    # ── small files ──────────────────────────────────────────────────────────────────────────────
    def put_text(self, rel: str, text: str):
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".spool") as f:
            f.write(text)
        try:
            self.put(f.name, rel)
        finally:
            os.unlink(f.name)

    def get_text(self, rel: str) -> str | None:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "f")
            if not self.get(rel, p):
                return None
            with open(p, errors="replace") as f:
                return f.read()

    def put_json(self, rel: str, obj):
        self.put_text(rel, json.dumps(obj, indent=1, sort_keys=True))

    def get_json(self, rel: str):
        t = self.get_text(rel)
        try:
            return json.loads(t) if t else None
        except json.JSONDecodeError:
            return None
