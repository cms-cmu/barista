#!/usr/bin/env python3
"""roast -- reproducible production runs of barista Snakemake workflows.

A *roast* is one production batch: a pinned barista sha, a pinned coffea4bees
sha, a captured workflow config, and an ordered list of steps (Snakemake
invocations) each bound to a host.  Every roast gets its own isolated git
checkout on each host, outside the mutagen-synced development trees, so the
code that produced a result is exactly the code recorded in its manifest.

Results are published to the owner's CERNBox www area and catalogued in the
barista GitLab Pages site ("cupping notes", docs/prod/).

Typical session (from the barista root on your laptop):

    roast init                                   # once: ~/.config/roast/config.json
    roast proxy                                  # voms-proxy-init on cmslpc (weekly); --check to see time left
    roast new --config coffea4bees/workflows/config/nominal_run2.yml --phases B,C,D,F
    roast checkout <id>                          # isolated trees on cmslpc + falcon
    roast submit   <id> --step B --dry-run       # plan only
    roast submit   <id> --step B --test          # small local slice
    roast submit   <id> --step B                 # the real thing, tmux window on cmslpc
    roast status   <id>  |  roast attach <id>
    roast submit   <id> --step C                 # after B is done, on falcon
    roast resume   <id> --step C                 # after a dead driver: --unlock + --rerun-incomplete
    roast publish  <id>                          # CERNBox + docs/prod/<id>.md + index
    roast ls

Ids look like <label>_<YYYYMMDD>_<barista7>-<coffea4bees7>; any unique prefix works.

Stdlib only.  Manifests live in roasts/<id>/roast.json (commit them).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import getpass
import hashlib
import html
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOOL = "roast"
CONFIG_PATH = Path(os.environ.get("ROAST_CONFIG", "~/.config/roast/config.json")).expanduser()
TMUX_SESSION = "roast"
# Reuse one authenticated connection per host for the life of a command, and a minute beyond.
# roast makes several calls per roast -- step state, then EOS, then the deletion -- and a fresh
# handshake to cmslpc costs 2 to 4 seconds against 0.25 for a multiplexed one, so on a `clean
# --all` the handshakes were most of the wall clock.  ControlPath must stay under 104 bytes, so
# it uses %C, a fixed-length hash of the connection parameters.
SSH_CONTROL_DIR = Path("~/.ssh").expanduser()


def ssh_opts(target: str = "") -> list[str]:
    """SSH command line options.

    Hosts like Bridges-2 use persistent 2FA master sockets managed by ssh-guardian
    or ~/.ssh/config. Overriding ControlPath breaks their existing authentication.
    For standard hosts (cmslpc, falcon), multiplexing into ~/.ssh/roast-cm-%C is enabled.
    """
    opts = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "LogLevel=ERROR",
            "-o", "HostName=%h"]
    if target and ("bridges2" in target or "psc.edu" in target):
        return opts
    return [*opts, "-o", "ControlMaster=auto", "-o", f"ControlPath={SSH_CONTROL_DIR}/roast-cm-%C",
            "-o", "ControlPersist=60"]


SSH_OPTS = ssh_opts()

# Phase -> (host, snakefile).  Mirrors coffea4bees/workflows/README.md.
PHASES = {
    "A": ("cmslpc", "coffea4bees/workflows/Snakefile_PhaseA.smk"),
    "B": ("cmslpc", "coffea4bees/workflows/Snakefile_PhaseB.smk"),
    "C": ("falcon", "coffea4bees/workflows/Snakefile_PhaseC.smk"),
    "C4": ("cmslpc", "coffea4bees/workflows/Snakefile_PhaseC_4_FvT_closure.smk"),   # FvT closure: processor + plots + cutflow with the new FvT
    "D": ("falcon", "coffea4bees/workflows/Snakefile_PhaseD.smk"),
    "F": ("cmslpc", "coffea4bees/workflows/Snakefile_PhaseF.smk"),
    "BKG_AB": ("cmslpc", "coffea4bees/workflows/Snakefile_bkg_syst_AB.smk"),
    "BKG_C":  ("falcon", "coffea4bees/workflows/Snakefile_bkg_syst_C.smk"),
    "BKG_F":  ("cmslpc", "coffea4bees/workflows/Snakefile_bkg_syst_F.smk"),
}

# What `publish` ships to CERNBox: small, human-readable artefacts.  Override any key under
# "publish" in ~/.config/roast/config.json, or per roast under "publish" in roast.json.
# include: filename globs (find -name); exclude: path globs (find -path, relative to the checkout).
PUBLISH_DEFAULTS = {
    "include": ["*.pdf", "*.png", "*.svg", "*.html", "*.yml", "*.yaml", "*.json", "*.txt", "*.log", "*.md", "*.csv", "*.tex"],
    # per-job logs under output/ stay on the host; the step log logs/<step>.log always ships
    "exclude": ["*_test", "*_test/*", "*dask-report*", "*/performance", "*/performance/*",
                "*/classifier_inputs/classifier_inputs_dataset_*",
                "output/*/logs", "output/*/logs/*", "output/*/*/logs", "output/*/*/logs/*"],
    "max_mb": 50,
}
# What `archive` ships to FNAL EOS (heavy, machine-readable products).  Same override mechanism ("archive" key).
ARCHIVE_DEFAULTS = {
    "include": ["*.coffea", "*.root", "*.yml", "*.yaml", "*.json", "*.pkl"],
    # merged products only: no per-dataset singlefiles, no per-dataset classifier-input manifests
    # (classifier_inputs_friends.json contains every path they do)
    "exclude": ["*_test", "*_test/*", "*dask-report*", "*/performance", "*/performance/*",
                "*/singlefiles", "*/singlefiles/*", "*/classifier_inputs/classifier_inputs_dataset_*",
                "output/*/logs", "output/*/logs/*", "output/*/*/logs", "output/*/*/logs/*"],
    "max_mb": 0,   # 0 = no limit
}

DEFAULT_CONFIG = {
    "hosts": {
        "cmslpc": {
            "ssh": "<user>@cmslpc-el9.fnal.gov",
            "prod_root": "~/nobackup/HH4b/prod",
            "reference": "~/nobackup/HH4b/Run3/barista",
            "cores": 8,
        },
        "falcon": {
            "ssh": "<user>@falcon.phys.cmu.edu",
            "prod_root": "~/work/prod",
            "reference": "~/work/barista",
            "cores": 4,
        },
        "bridges2": {
            "ssh": "<user>@bridges2.psc.edu",
            "prod_root": "/ocean/projects/phy260026p/<user>/HH4b/prod",
            "reference": "/ocean/projects/phy260026p/<user>/HH4b/barista",
            "cores": 16,
        },
        "nautilus": {
            "namespace": "cms-cmu",
            "user": "<user>",
            "prod_root": "/workspace/users/<user>/prod",
            "reference": "/workspace/users/<user>/barista",
            "cores": 16,
        },
    },
    "cernbox": {
        "eos_path": "/eos/user/<x>/<user>/www/ttHbb/prod",
        "url": "https://<user>.web.cern.ch/<user>/ttHbb/prod",
        "analyses": {
            "ttHbb": {
                "eos_path": "/eos/user/<x>/<user>/www/ttHbb/prod",
                "url": "https://<user>.web.cern.ch/<user>/ttHbb/prod",
            },
            "HH4b": {
                "eos_path": "/eos/user/<x>/<user>/www/HH4b/prod",
                "url": "https://<user>.web.cern.ch/<user>/HH4b/prod",
            },
        },
    },
    "owner": "<your name>",
    "eos": {"url": "root://cmseos.fnal.gov", "path": "/store/user/<lpc_user>/HH4b_prod"},
    "publish": PUBLISH_DEFAULTS,
    "archive": ARCHIVE_DEFAULTS,
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def die(msg: str, code: int = 1) -> None:
    print(f"{TOOL}: {msg}", file=sys.stderr)
    sys.exit(code)


def info(msg: str) -> None:
    print(f"\033[1;34m[{TOOL}]\033[0m {msg}", file=sys.stderr)


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def sh(cmd, cwd=None, check=True, capture=True, input=None) -> subprocess.CompletedProcess:
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    # errors="replace": remote logs carry progress bars and box drawing, and any tool on the
    # far side may hand us a partial character.  Status must survive that.
    return subprocess.run(cmd, cwd=cwd, check=check, text=True, errors="replace", input=input,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.PIPE if capture else None)


def repo_root() -> Path:
    """Barista root: walk up from cwd looking for run_container."""
    p = Path.cwd()
    for cand in [p, *p.parents]:
        if (cand / "run_container").exists() and (cand / "coffea4bees").exists():
            return cand
    here = Path(__file__).resolve().parent.parent.parent
    if (here / "run_container").exists():
        return here
    die("not inside a barista checkout (no run_container found)")


ROOT = repo_root()
ROASTS = ROOT / "roasts"
DOCS_PROD = ROOT / "docs" / "prod"

# Pages under docs/prod that are written by hand and committed, so the stale-page sweep in
# write_index() must leave them alone.  "index" is generated but always wanted; the rest are
# prose that lives next to the catalogue because colleagues reach it from the same nav.
HANDWRITTEN_PAGES = {"index", "using_roast"}

# A grid proxy on shared home.  /tmp is node-local on LPC and the login gateway is
# round-robin, so the proxy you just created is usually not on the node a given roast
# was pinned to; this copy is readable from all of them.
SHARED_PROXY = "~/.roast/x509_proxy"

DOCS_SITE_FALLBACK = "https://barista.docs.cern.ch"


def docs_site() -> str:
    """Where the cupping notes are served, read from mkdocs.yml so there is one source of truth."""
    try:
        for line in (ROOT / "docs" / "mkdocs.yml").read_text().splitlines():
            if line.startswith("site_url:"):
                return line.split(":", 1)[1].strip().rstrip("/")
    except OSError:
        pass
    return DOCS_SITE_FALLBACK


def docs_page_url(rid: str) -> str:
    """The published cupping note for a roast.  Derived, not stored: mkdocs serves every page
    under docs/ at its own path, so the id is the address."""
    return f"{docs_site()}/prod/{rid}/"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        die(f"no config at {CONFIG_PATH}; run `{TOOL} init` and edit it")
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    for k in ("hosts", "cernbox"):
        if k not in cfg:
            die(f"config missing '{k}'")
    return cfg


def host_cfg(cfg: dict, host: str) -> dict:
    if host == "nautilus" and "nautilus" not in cfg.get("hosts", {}):
        user = getpass.getuser()
        return {
            "namespace": "cms-cmu",
            "user": user,
            "prod_root": f"/workspace/users/{user}/prod",
            "reference": f"/workspace/users/{user}/barista",
            "cores": 16,
        }
    if host == "bridges2" and "bridges2" not in cfg.get("hosts", {}):
        user = getpass.getuser()
        return {
            "ssh": f"{user}@bridges2.psc.edu",
            "prod_root": f"/ocean/projects/phy260026p/{user}/HH4b/prod",
            "reference": f"/ocean/projects/phy260026p/{user}/HH4b/barista",
            "cores": 16,
        }
    try:
        return cfg["hosts"][host]
    except KeyError:
        die(f"host '{host}' not in {CONFIG_PATH}")


def k8s_cmd(*args) -> list[str]:
    """Return kubectl command vector, using pixi if kubectl is not on host PATH."""
    if "KUBECONFIG" not in os.environ and (Path.home() / ".kube" / "config").exists():
        os.environ["KUBECONFIG"] = str(Path.home() / ".kube" / "config")
    base = ["kubectl"] if shutil.which("kubectl") else ["pixi", "run", "kubectl"]
    return [*base, *args]


def _k8s_job_name(r: dict, step: dict) -> str:
    """RFC 1123 compliant Kubernetes Job name."""
    h = hashlib.sha256(r["id"].encode()).hexdigest()[:6]
    raw = f"roast-{r['label'][:35]}-{h}-{step['name'][:10]}".lower()
    return re.sub(r"[^a-z0-9-]", "-", raw).strip("-")[:63]


def resolve_ssh(hc: dict) -> str:
    """Live ssh target for a host.  Optional 'host_file' (e.g. ~/.cmslpc-claude-host, maintained by
    cmslpc-failover) overrides the hostname so a dead pinned node can be swapped without editing config."""
    target = hc["ssh"]
    hf = hc.get("host_file")
    if hf and Path(hf).expanduser().exists():
        node = Path(hf).expanduser().read_text().strip()
        if node:
            user, _, hostname = target.rpartition("@")
            if "." not in node and "." in hostname:
                node = node + hostname[hostname.index("."):]
            target = f"{user}@{node}" if user else node
    return target


def roast_ssh(cfg: dict, r: dict, host: str) -> str:
    """The node to talk to for this roast.

    Only the driver process and its tmux window are node-bound; the filesystem is shared
    across LPC interactive nodes and HTCondor runs central schedds.  So the configured
    target may be the round-robin gateway, and `checkout` records whichever node it landed
    on.  Everything afterwards follows that record, because that is where the run lives.
    """
    node = (r.get("hosts", {}).get(host) or {}).get("ssh")
    return node or resolve_ssh(host_cfg(cfg, host))


def ssh_run(target: str, script: str, check=True, capture=True) -> subprocess.CompletedProcess:
    """Run a bash script on a remote host via stdin (no quoting games)."""
    return sh(["ssh", *ssh_opts(target), target, "bash -s"], check=check, capture=capture, input=script)


def scp_to(target: str, srcs: list[Path], dst: str) -> None:
    # -p: keep mtimes. The captured config.yml is an input of the workflows' create_*_config rules;
    # re-shipping it with a fresh mtime on every submit/resume made snakemake regenerate those
    # configs and rerun every processor job downstream of them.
    # On clusters without scp (e.g. PSC Bridges-2), stream a tar archive over ssh.
    if "bridges2" in target or "psc.edu" in target:
        import io
        import tarfile
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            for s in srcs:
                tar.add(str(s), arcname=s.name)
        cmd = ["ssh", *ssh_opts(target), target, f"mkdir -p {rq(dst)} && tar -xf - -C {rq(dst)}"]
        subprocess.run(cmd, input=buf.getvalue(), check=True)
        return
    sh(["scp", "-q", "-p", *ssh_opts(target), *map(str, srcs), f"{target}:{dst}"])


def rq(path: str) -> str:
    """Quote a remote path for bash, letting a leading ~/ expand to $HOME."""
    if path.startswith("~/"):
        rest = path[2:].replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")
        return f'"$HOME/{rest}"'
    return shlex.quote(path)


def git(args: str, cwd: Path) -> str:
    return sh(f"git {args}", cwd=cwd).stdout.strip()


def gitlab_web(origin: str) -> str:
    """ssh://git@gitlab.cern.ch:7999/cms-cmu/barista.git -> https://gitlab.cern.ch/cms-cmu/barista"""
    m = re.match(r"(?:ssh://)?(?:git@)?([^:/]+)(?::\d+)?[:/](.+?)(?:\.git)?$", origin)
    if not m:
        return origin
    return f"https://{m.group(1)}/{m.group(2)}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _merge_config(base: dict, over: dict) -> dict:
    """Recursive dict merge, `over` winning; lists and scalars are replaced.
    Same semantics as snakemake's own merge of repeated --configfile, so a layered config
    run by hand as `--configfile <base> --configfile <file>` sees what the roast captured."""
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge_config(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _load_layered_config(path: Path, seen: tuple = ()) -> tuple[dict, list[dict]]:
    """Load a workflow config and the chain of `base:` configs under it.
    Returns the merged dict and the bases, innermost first, as [{source, sha256}]."""
    import yaml     # only layered configs need it; roast is otherwise stdlib (see CI job code_roast_barista)
    if path.resolve() in seen:
        die(f"config base cycle: {' -> '.join(str(p) for p in (*seen, path))}")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base is None:
        return cfg, []
    base_path = ROOT / base      # relative to the barista root, like every other path in these configs
    if not base_path.exists():
        die(f"{path}: base config {base} not found")
    merged, chain = _load_layered_config(base_path, (*seen, path.resolve()))
    return _merge_config(merged, cfg), [*chain, {"source": str(base), "sha256": sha256_file(base_path)}]


def _norm_url(url: str) -> str:
    """Compare EOS URLs by path: `root://host//store/x` and `root://host/store/x/` are one place."""
    return re.sub(r"(?<!:)/{2,}", "/", url.split("@@", 1)[0]).rstrip("/")


def _roast_eos(cfg: dict, up: dict) -> str:
    """Where a roast's products live on EOS: its archive once archived, else the per-roast
    namespace (eos.path/<id>) that its jobs write into while it runs."""
    if up.get("archive", {}).get("eos"):
        return up["archive"]["eos"]
    eos = cfg.get("eos") or {}
    if not eos.get("path") or "<" in eos["path"]:
        die(f"upstream roast {up['id']} is not archived and eos.path is unset in {CONFIG_PATH}")
    return f"{eos.get('url', 'root://cmseos.fnal.gov')}/{eos['path'].rstrip('/')}/{up['id']}"


# {roast:<id>} -> that roast's EOS area; {roast:<id>:<key>} -> a value from its captured config
# (key `id`: the roast id itself). <id> may also be an alias from a mapping-form upstream_roasts.
# Same syntax as coffea4bees/workflows/helpers/common.smk, which resolves it at run time.
UPSTREAM_REF = re.compile(r"\{roast:([^}:]+)(?::([^}]+))?\}")


def upstream_aliases(upstream) -> dict:
    """`upstream_roasts` as {name: roast id}: a mapping gives short aliases, a str / list names
    the roasts by their own ids."""
    if isinstance(upstream, dict):
        return {str(k): str(v) for k, v in upstream.items()}
    ids = [upstream] if isinstance(upstream, str) else list(upstream or [])
    return {str(i): str(i) for i in ids}


def upstream_config_value(rid: str, key: str) -> str:
    """Top-level `key` of roast `rid`'s captured config, with its {roast_id} filled in."""
    if key == "id":
        return rid
    import yaml     # as in _load_layered_config: roast is otherwise stdlib
    captured = roast_dir(rid) / "config.yml"
    if not captured.exists():
        raise ValueError(f"{{roast:{rid}:{key}}}: no captured config {captured}")
    value = (yaml.safe_load(captured.read_text()) or {}).get(key)
    if not isinstance(value, str):
        raise ValueError(f"{{roast:{rid}:{key}}}: {captured} has no top-level string {key!r}")
    return value.replace("{roast_id}", rid)


def check_inputs(cfg: dict, wf: dict, analysis: str | None = None) -> dict | None:
    """Validate a workflow config's `inputs:` block: the products it reads from other roasts.

        inputs:
          upstream_roasts: [nominal_run3_...]      # str or list -- or {alias: id}
          FvT: root://.../HH4b_prod/nominal_run3_.../friend/FvT_nominal/result.json@@analysis.0.merged
          JCM: "{roast:<id or alias>}/{roast:<id or alias>:output_path}computeJCM/..."

    Every other value under `inputs` (nested allowed) must be a URL inside the EOS area of one of
    the named roasts, or null (an input of a `base:` config that this one drops). Otherwise a dependent roast can quietly read a hand-run product or another
    production's, and nothing downstream would notice. Returns the record kept in roast.json.
    """
    block = wf.get("inputs")
    if block is None:
        return None
    if not isinstance(block, dict):
        die("config `inputs:` must be a mapping")
    aliases = upstream_aliases(block.get("upstream_roasts"))
    if not aliases:
        die("config `inputs:` needs `upstream_roasts:` naming the roast(s) its URLs come from")
    ups = [load_roast(i) for i in aliases.values()]
    # name in the config -> the full id load_roast found (it also accepts a unique prefix)
    aliases = {name: up["id"] for name, up in zip(aliases, ups)}
    for up in ups:
        if not up.get("archive", {}).get("ok"):
            info(f"WARNING: upstream roast {up['id']} is not archived; its products may still change")
    areas = {up["id"]: _norm_url(_roast_eos(cfg, up)) for up in ups}
    # configs name the production area as {eos_prod} / {web_prod} (resolved at submit time):
    # expand them the same way before checking where an input lives
    placeholders = {f"{{{k}}}": v for k, v in user_paths(cfg, analysis=analysis).items()}

    refs, bad = {}, []
    def walk(node, key):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{key}.{k}" if key else k)
            return
        if node is None:
            # `key: null` in a layered config drops an input its base chain sets (`base:` merges
            # dicts recursively, so a key cannot be removed otherwise): nothing is read
            return
        if not isinstance(node, str):
            bad.append(f"{key}: not a URL ({node!r})")
            return
        if "{roast_id}" in node:
            bad.append(f"{key}: {{roast_id}} refers to this roast, not an upstream one")
            return
        resolved = node
        for ph, value in placeholders.items():
            resolved = resolved.replace(ph, value)
        # {roast:<id>} names an upstream roast's own area (the workflow resolves it through that
        # roast's manifest, helpers/common.smk); {roast:<id>:<key>} a top-level value of its
        # captured config, e.g. output_path. The id must be one of upstream_roasts.
        unknown = [name for name, _ in UPSTREAM_REF.findall(resolved)
                   if aliases.get(name, name) not in areas]
        if unknown:
            bad.append(f"{key}: {{roast:{unknown[0]}}} is not in upstream_roasts")
            return
        def sub(m):
            rid = aliases.get(m.group(1), m.group(1))
            return upstream_config_value(rid, m.group(2)) if m.group(2) else areas[rid]
        try:
            resolved = UPSTREAM_REF.sub(sub, resolved)
        except ValueError as e:
            bad.append(f"{key}: {e}")
            return
        url = _norm_url(resolved)
        owner = next((i for i, a in areas.items() if url == a or url.startswith(a + "/")), None)
        if owner is None:
            bad.append(f"{key}: {node} is not under any upstream roast ({', '.join(areas.values())})")
        else:
            refs[key] = {"url": node, "roast": owner}
    walk({k: v for k, v in block.items() if k != "upstream_roasts"}, "")
    if bad:
        die("config `inputs:` problems:\n  " + "\n  ".join(bad))
    return {"upstream": [{"id": up["id"], "barista": up["barista"]["sha"], "coffea4bees": up["coffea4bees"]["sha"],
                          "eos": _roast_eos(cfg, up), "archived": bool(up.get("archive", {}).get("ok"))}
                         for up in ups],
            "refs": refs}


def capture_config(src: Path, dest: Path) -> list[dict]:
    """Write the config a roast runs from.  A plain config is copied verbatim.  One with a
    top-level `base: <path>` holds only its differences from that base: it is merged over
    the base (recursively, bases may have bases) and the self-contained result is captured,
    so the roast never depends on the base file changing later.  Returns the base chain."""
    if not re.search(r"^base:", src.read_text(), re.M):
        shutil.copy2(src, dest)
        return []
    try:
        import yaml
    except ImportError:
        die(f"{src} has a `base:` key; capturing a layered config needs PyYAML")
    merged, chain = _load_layered_config(src)
    header = (f"# Captured by `roast new`: {src} merged over "
              + " <- ".join(b["source"] for b in reversed(chain))
              + ".\n# Comments are not carried over; read the sources for them.\n")
    with open(dest, "w") as f:
        f.write(header)
        yaml.safe_dump(merged, f, sort_keys=False, default_flow_style=False, width=1000)
    return chain


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def roast_dir(rid: str) -> Path:
    return ROASTS / rid


def load_roast(rid: str) -> dict:
    p = roast_dir(rid) / "roast.json"
    if not p.exists():
        # allow unique prefix match
        matches = [d for d in ROASTS.glob(f"{rid}*") if (d / "roast.json").exists()] if ROASTS.exists() else []
        if len(matches) == 1:
            p = matches[0] / "roast.json"
        elif len(matches) > 1:
            die(f"ambiguous id '{rid}': " + ", ".join(m.name for m in matches))
        else:
            die(f"no roast '{rid}' (looked in {ROASTS})")
    with open(p) as f:
        return json.load(f)


def save_roast(r: dict) -> None:
    d = roast_dir(r["id"])
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "roast.json", "w") as f:
        json.dump(r, f, indent=2)
        f.write("\n")


def log_event(r: dict, event: str, **kw) -> None:
    r.setdefault("history", []).append({"ts": now(), "event": event, **kw})


def all_roasts() -> list[dict]:
    if not ROASTS.exists():
        return []
    out = []
    for p in sorted(ROASTS.glob("*/roast.json")):
        with open(p) as f:
            out.append(json.load(f))
    return out


def find_step(r: dict, name: str) -> dict:
    for s in r["steps"]:
        if s["name"] == name:
            return s
    die(f"no step '{name}' in roast {r['id']}; steps: " + ", ".join(s["name"] for s in r["steps"]))


def checkout_path(cfg: dict, r: dict, host: str) -> str:
    hc = host_cfg(cfg, host)
    return f"{hc['prod_root'].rstrip('/')}/{r['id']}/barista"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_init(args) -> None:
    if CONFIG_PATH.exists() and not args.force:
        die(f"{CONFIG_PATH} exists (use --force to overwrite)")
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if args.cmslpc_user:
        cfg["hosts"]["cmslpc"]["ssh"] = f"{args.cmslpc_user}@{args.cmslpc_node or 'cmslpc-el9'}.fnal.gov"
        cfg["eos"]["path"] = f"/store/user/{args.cmslpc_user}/HH4b_prod"
    if args.falcon_user:
        cfg["hosts"]["falcon"]["ssh"] = f"{args.falcon_user}@falcon.phys.cmu.edu"
    if getattr(args, "bridges2_user", None):
        u = args.bridges2_user
        cfg["hosts"]["bridges2"]["ssh"] = f"{u}@bridges2.psc.edu"
        cfg["hosts"]["bridges2"]["prod_root"] = f"/ocean/projects/phy260026p/{u}/HH4b/prod"
        cfg["hosts"]["bridges2"]["reference"] = f"/ocean/projects/phy260026p/{u}/HH4b/barista"
    if args.cern_user:
        u = args.cern_user
        cfg["cernbox"]["eos_path"] = f"/eos/user/{u[0]}/{u}/www/ttHbb/prod"
        cfg["cernbox"]["url"] = f"https://{u}.web.cern.ch/{u}/ttHbb/prod"
        cfg["cernbox"]["analyses"] = {
            "ttHbb": {
                "eos_path": f"/eos/user/{u[0]}/{u}/www/ttHbb/prod",
                "url": f"https://{u}.web.cern.ch/{u}/ttHbb/prod",
            },
            "HH4b": {
                "eos_path": f"/eos/user/{u[0]}/{u}/www/HH4b/prod",
                "url": f"https://{u}.web.cern.ch/{u}/HH4b/prod",
            },
        }
    cfg["owner"] = args.owner or getpass.getuser()
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    info(f"wrote {CONFIG_PATH}")
    for key, val in (("cmslpc ssh", cfg["hosts"]["cmslpc"]["ssh"]), ("falcon ssh", cfg["hosts"]["falcon"]["ssh"]),
                     ("EOS archive", cfg["eos"]["path"]), ("CERNBox", cfg["cernbox"]["url"])):
        info(f"  {key:12s} {val}")
    if any("<" in json.dumps(cfg[k]) for k in ("hosts", "cernbox", "eos")):
        info("  placeholders in <angle brackets> still need editing")


def detect_analysis(r: dict | None = None, config_path: Path | str | None = None, label: str | None = None) -> str:
    """Detect whether a roast belongs to HH4b or ttHbb. Defaults to ttHbb."""
    if r and r.get("analysis"):
        return r["analysis"]
    if r:
        web = (r.get("user_paths") or {}).get("web_prod", "") or (r.get("publish") or {}).get("eos", "")
        if "/www/HH4b/" in web:
            return "HH4b"
        if "/www/ttHbb/" in web:
            return "ttHbb"
    text_to_check = []
    if label:
        text_to_check.append(str(label).lower())
    if r:
        if r.get("id"):
            text_to_check.append(str(r["id"]).lower())
        if r.get("label"):
            text_to_check.append(str(r["label"]).lower())
        if r.get("config", {}).get("source"):
            text_to_check.append(str(r["config"]["source"]).lower())
    if config_path:
        text_to_check.append(str(config_path).lower())

    joined = " ".join(text_to_check)
    if "tthbb" in joined or "ttbb" in joined:
        return "ttHbb"
    if any(k in joined for k in ("hh4b", "lowpt", "nominal", "mvd", "declustered", "mixeddata")):
        return "HH4b"

    cfg_file = None
    if config_path and Path(config_path).exists():
        cfg_file = Path(config_path)
    elif r and r.get("id") and (roast_dir(r["id"]) / "config.yml").exists():
        cfg_file = roast_dir(r["id"]) / "config.yml"
    if cfg_file:
        try:
            content = cfg_file.read_text()
            if re.search(r"channel:\s*['\"]?ttHbb", content, re.I):
                return "ttHbb"
            if re.search(r"(processor_HH4b|channel:\s*['\"]?HH4b|friends_HH4b)", content, re.I):
                return "HH4b"
        except Exception:
            pass

    return "ttHbb"


def cernbox_paths(cfg: dict, r: dict | None = None, analysis: str | None = None) -> tuple[str, str]:
    """Return (eos_path, url) for CERNBox.
    Defaults to ttHbb from cfg['cernbox'], or selects from cfg['cernbox']['analyses'] / replaces path for HH4b.
    """
    cb = cfg.get("cernbox") or {}
    ana = analysis or (r.get("analysis") if r else None) or detect_analysis(r)
    analyses = cb.get("analyses") or {}
    if ana and ana in analyses:
        eos_path = analyses[ana].get("eos_path", "")
        url = analyses[ana].get("url", "")
    elif cb.get("eos_path"):
        base_eos = cb["eos_path"]
        base_url = cb.get("url", "")
        if ana == "HH4b" and "/www/ttHbb" in base_eos:
            eos_path = base_eos.replace("/www/ttHbb", "/www/HH4b")
            url = base_url.replace("/ttHbb", "/HH4b")
        elif ana == "ttHbb" and "/www/HH4b" in base_eos:
            eos_path = base_eos.replace("/www/HH4b", "/www/ttHbb")
            url = base_url.replace("/HH4b", "/ttHbb")
        else:
            eos_path = base_eos
            url = base_url
    else:
        eos_path = ""
        url = ""
    return eos_path, url


def user_paths(cfg: dict, r: dict | None = None, analysis: str | None = None) -> dict:
    """The per-user production areas, passed to the workflow as {eos_prod} / {web_prod}.

    Derived from what `roast init` already wrote, so a config in git never names a person,
    and a colleague re-running your roast writes into their own area rather than yours.
    xrootd wants root://host//abs/path, hence the doubled slash before an absolute path.
    """
    out = {}
    eos = cfg.get("eos") or {}
    if eos.get("path") and "<" not in eos["path"]:
        host = eos.get("url", "root://cmseos.fnal.gov").rstrip("/")
        out["eos_prod"] = f"{host}//{eos['path'].strip('/')}"
    web, _ = cernbox_paths(cfg, r, analysis)
    if web and "<" not in web:
        out["web_prod"] = f"root://eosuser.cern.ch//{web.strip('/')}"
    return out


def cmd_new(args) -> None:
    cfg = load_config()
    config_src = Path(args.config)
    if not config_src.exists():
        die(f"config file {config_src} not found")

    barista_sha = args.barista or git("rev-parse HEAD", ROOT)
    c4b_dir = ROOT / "coffea4bees"
    c4b_sha = args.coffea4bees or git("rev-parse HEAD", c4b_dir)
    barista_origin = git("remote get-url origin", ROOT)
    c4b_origin = git("remote get-url origin", c4b_dir)

    dirty = git("status --porcelain --untracked-files=no", ROOT) or git("status --porcelain --untracked-files=no", c4b_dir)
    if dirty and not args.barista and not args.coffea4bees:
        info("WARNING: working tree has uncommitted tracked changes; the roast records HEAD, not your edits")

    steps = []
    for ph in (args.phases.replace(" ", "").split(",") if args.phases else []):
        ph = ph.upper()
        if ph not in PHASES:
            die(f"unknown phase '{ph}'; known: {','.join(PHASES)}")
        host, smk = PHASES[ph]
        if getattr(args, "bridges2", False) and host == "falcon":
            host = "bridges2"
        elif getattr(args, "nautilus", False) and host == "falcon":
            host = "nautilus"
        steps.append({"name": ph, "host": host, "snakefile": smk, "targets": "", "extra": ""})
    for spec in args.step or []:
        # host:snakefile[:targets]
        parts = spec.split(":", 2)
        if len(parts) < 2:
            die(f"--step wants host:snakefile[:targets], got '{spec}'")
        host, smk = parts[0], parts[1]
        targets = parts[2] if len(parts) > 2 else ""
        name = Path(smk).stem.replace("Snakefile_", "")
        steps.append({"name": name, "host": host, "snakefile": smk, "targets": targets, "extra": ""})
    if not steps:
        die("nothing to run: give --phases and/or --step")
    for s in steps:
        host_cfg(cfg, s["host"])
        if not (ROOT / s["snakefile"]).exists():
            info(f"WARNING: {s['snakefile']} not found in this tree (may exist at the pinned sha)")

    label = args.label or re.sub(r"[^A-Za-z0-9_.-]", "_", config_src.stem)
    analysis = getattr(args, "analysis", None) or detect_analysis(config_path=config_src, label=label)
    rid = f"{label}_{dt.datetime.now():%Y%m%d}_{barista_sha[:7]}-{c4b_sha[:7]}"
    if args.id:
        rid = args.id
    if roast_dir(rid).exists():
        die(f"roast {rid} already exists")

    # Products read from other roasts, checked before anything is written
    inputs = None
    if re.search(r"^(inputs|base):", config_src.read_text(), re.M):    # a base may carry the inputs
        try:
            import yaml  # noqa: F401  (only configs with `inputs:` or `base:` need it)
        except ImportError:
            die(f"{config_src} has an `inputs:` or `base:` block; reading it needs PyYAML")
        inputs = check_inputs(cfg, _load_layered_config(config_src)[0], analysis=analysis)

    d = roast_dir(rid)
    d.mkdir(parents=True)
    # Verbatim, or merged over its `base:` chain; {roast_id} is resolved at run time via --config roast_id
    config_bases = capture_config(config_src, d / "config.yml")

    r = {
        "id": rid,
        "label": label,
        "analysis": analysis,
        "created": now(),
        "owner": cfg.get("owner", getpass.getuser()),
        "barista": {"sha": barista_sha, "origin": barista_origin, "web": gitlab_web(barista_origin)},
        "coffea4bees": {"sha": c4b_sha, "origin": c4b_origin, "web": gitlab_web(c4b_origin)},
        "config": {"source": str(config_src), "captured": "config.yml", "sha256": sha256_file(d / "config.yml"),
                   **({"base": config_bases} if config_bases else {})},
        **({"inputs": inputs} if inputs else {}),
        "user_paths": user_paths(cfg, analysis=analysis),
        "steps": steps,
        "hosts": {},
        "publish": {},
        "history": [],
        "notes": args.notes or "",
        "category": args.category or "",
    }
    log_event(r, "new")
    save_roast(r)
    print(rid)
    info(f"manifest: {d / 'roast.json'}")
    info(f"next: {TOOL} checkout {rid}")


def _checkout_script(hc: dict, r: dict, ckpt: str) -> str:
    ref = hc["reference"]
    return textwrap.dedent(f"""\
        set -euo pipefail
        CK={rq(ckpt)}
        mkdir -p "$(dirname "$CK")"
        if [ ! -d "$CK/.git" ]; then
            git clone -q --no-checkout {rq(ref)} "$CK"
        fi
        echo "NODE|$(hostname -f)"
        cd "$CK"
        git remote set-url origin {shlex.quote(r['barista']['origin'])}
        if [ ! -d coffea4bees/.git ]; then
            git clone -q --no-checkout {rq(ref + '/coffea4bees')} coffea4bees
        fi
        (cd coffea4bees && git remote set-url origin {shlex.quote(r['coffea4bees']['origin'])})
        if [ -s {rq(ref + '/proxy/x509_proxy')} ] && [ ! -s proxy/x509_proxy ]; then
            mkdir -p proxy && cp -p {rq(ref + '/proxy/x509_proxy')} proxy/x509_proxy 2>/dev/null || true
        fi
        echo "prepared $CK"
    """)


def _finish_checkout_script(r: dict, ckpt: str) -> str:
    return textwrap.dedent(f"""\
        set -euo pipefail
        cd {rq(ckpt)}
        git checkout -q --detach {r['barista']['sha']}
        (cd coffea4bees && git checkout -q --detach {r['coffea4bees']['sha']})
        mkdir -p logs roasts/{r['id']}
        echo "barista     $(git rev-parse --short HEAD)  $(git log -1 --format=%s | cut -c1-60)"
        echo "coffea4bees $(cd coffea4bees && git rev-parse --short HEAD)  $(cd coffea4bees && git log -1 --format=%s | cut -c1-60)"
    """)


def _checkout_nautilus(cfg: dict, r: dict, ckpt: str) -> None:
    hc = host_cfg(cfg, "nautilus")
    ns = hc.get("namespace", "cms-cmu")
    user = hc.get("user", getpass.getuser())
    ref = hc.get("reference", f"/workspace/users/{user}/barista")

    info(f"[nautilus] preparing {ckpt}")

    # Verify connection to Nautilus
    check_ns = sh(k8s_cmd("get", "ns", ns), check=False)
    if check_ns.returncode != 0:
        die(f"cannot reach Nautilus namespace '{ns}': {check_ns.stderr.strip()}")

    # 1. Create directory structure on CephFS via storage-helper
    setup_cmd = f"mkdir -p {ckpt}/roasts/{r['id']} {ckpt}/logs /workspace/users/{user}/.kube"
    sh_run = sh(k8s_cmd("exec", "-i", "storage-helper", "-n", ns, "--", "sh", "-c", setup_cmd), check=False)
    if sh_run.returncode != 0:
        die(f"failed to create directories on Nautilus CephFS: {sh_run.stderr.strip()}")

    # 2. Hardlink reference tree on CephFS (instantaneous cp -al)
    copy_script = f"""
    for item in coffea4bees output proxy run_container runner.py software src roasts; do
        if [ -e "{ref}/$item" ] && [ ! -e "{ckpt}/$item" ]; then
            cp -al "{ref}/$item" "{ckpt}/$item" 2>/dev/null || cp -r "{ref}/$item" "{ckpt}/$item"
        fi
    done
    """
    sh(k8s_cmd("exec", "-i", "storage-helper", "-n", ns, "--", "sh", "-c", copy_script), check=False)

    # 3. Ship roast files (config.yml, roast.json) via kubectl cp
    rdir = roast_dir(r["id"])
    for f in sorted(p for p in rdir.iterdir() if p.is_file()):
        sh(k8s_cmd("cp", str(f), f"{ns}/storage-helper:{ckpt}/roasts/{r['id']}/{f.name}"), check=False)
    # Also ship upstream roasts if present locally
    for up in r.get("inputs", {}).get("upstream", []):
        up_id = up.get("id")
        up_dir = roast_dir(up_id)
        if up_dir.is_dir():
            sh(k8s_cmd("exec", "-i", "storage-helper", "-n", ns, "--", "mkdir", "-p", f"{ckpt}/roasts/{up_id}"), check=False)
            for f in sorted(p for p in up_dir.iterdir() if p.is_file()):
                sh(k8s_cmd("cp", str(f), f"{ns}/storage-helper:{ckpt}/roasts/{up_id}/{f.name}"), check=False)

    # 4. Stage user's local kubeconfig to CephFS if not already present
    check_kube = sh(k8s_cmd("exec", "-i", "storage-helper", "-n", ns, "--", "test", "-f", f"/workspace/users/{user}/.kube/config"), check=False)
    if check_kube.returncode != 0:
        local_kube = Path.home() / ".kube" / "config"
        if local_kube.is_file():
            sh(k8s_cmd("cp", str(local_kube), f"{ns}/storage-helper:/workspace/users/{user}/.kube/config"), check=False)
            sh(k8s_cmd("exec", "-i", "storage-helper", "-n", ns, "--", "chmod", "600", f"/workspace/users/{user}/.kube/config"), check=False)

    r.setdefault("hosts", {})["nautilus"] = {"checkout": ckpt, "checked_out": now()}
    log_event(r, "checkout", host="nautilus")
    save_roast(r)
    _sync_nautilus_code(cfg, r, ckpt)
    info(f"[nautilus] prepared {ckpt}")


def _sync_nautilus_code(cfg: dict, r: dict, ckpt: str) -> None:
    """Stream clean code from the roast's pinned commit SHAs directly to Nautilus CephFS.

    Avoids running Git on CephFS while ensuring deleted files (ghosts) are purged and
    both barista and coffea4bees match the exact committed state for this roast.
    """
    hc = host_cfg(cfg, "nautilus")
    ns = hc.get("namespace", "cms-cmu")
    info(f"[nautilus] syncing code to {ckpt}...")

    # 1. Clean code directories in the roast checkout to prevent stale or deleted ghost files
    clean_dirs = [
        f"{ckpt}/software",
        f"{ckpt}/src",
        f"{ckpt}/coffea4bees/classifier",
        f"{ckpt}/coffea4bees/workflows",
        f"{ckpt}/coffea4bees/analysis",
    ]
    sh(k8s_cmd("exec", "-i", "storage-helper", "-n", ns, "--", "rm", "-rf", *clean_dirs), check=False)

    # 2. Stream barista code at pinned sha
    barista_sha = r.get("barista", {}).get("sha", "HEAD")
    barista_paths = [
        p for p in ["src", "software", "runner.py"]
        if (ROOT / p).exists()
    ]
    if barista_paths:
        p_barista = subprocess.Popen(
            ["git", "archive", "--format=tar.gz", barista_sha, *barista_paths],
            cwd=ROOT,
            stdout=subprocess.PIPE,
        )
        res1 = subprocess.run(
            k8s_cmd("exec", "-i", "storage-helper", "-n", ns, "--", "tar", "-xzf", "-", "-C", ckpt),
            stdin=p_barista.stdout,
            capture_output=True,
            timeout=300,
        )
        p_barista.stdout.close()
        if res1.returncode != 0:
            info(f"[nautilus] warning: barista code sync returned {res1.returncode}")

    # 3. Stream coffea4bees code at pinned sha
    c4b_dir = ROOT / "coffea4bees"
    if c4b_dir.is_dir() and "coffea4bees" in r:
        c4b_sha = r["coffea4bees"]["sha"]
        c4b_paths = [
            p for p in ["classifier/", "workflows/", "analysis/"]
            if (c4b_dir / p.strip("/")).exists()
        ]
        if c4b_paths:
            c4b_args = [*c4b_paths, ":!analysis/trigger_emulator"]
            sh(k8s_cmd("exec", "-i", "storage-helper", "-n", ns, "--", "mkdir", "-p", f"{ckpt}/coffea4bees"), check=False)
            p_c4b = subprocess.Popen(
                ["git", "archive", "--format=tar.gz", c4b_sha, *c4b_args],
                cwd=c4b_dir,
                stdout=subprocess.PIPE,
            )
            res2 = subprocess.run(
                k8s_cmd("exec", "-i", "storage-helper", "-n", ns, "--", "tar", "-xzf", "-", "-C", f"{ckpt}/coffea4bees"),
                stdin=p_c4b.stdout,
                capture_output=True,
                timeout=300,
            )
            p_c4b.stdout.close()
            if res2.returncode != 0:
                info(f"[nautilus] warning: coffea4bees code sync returned {res2.returncode}")

    info(f"[nautilus] synced code to {ckpt}")


def cmd_checkout(args) -> None:
    cfg = load_config()
    r = load_roast(args.id)
    hosts = [args.host] if args.host else sorted({s["host"] for s in r["steps"]})
    for host in hosts:
        ckpt = checkout_path(cfg, r, host)
        if host == "nautilus":
            _checkout_nautilus(cfg, r, ckpt)
            continue
        hc = host_cfg(cfg, host)
        target = resolve_ssh(hc)
        info(f"[{host}] preparing {ckpt}")
        res = ssh_run(target, _checkout_script(hc, r, ckpt))
        node = next((l[5:].strip() for l in res.stdout.splitlines() if l.startswith("NODE|")), "")
        print("\n".join(l for l in res.stdout.splitlines() if not l.startswith("NODE|")).strip())
        if host == "cmslpc" and node and "@" in target:
            user = target.split("@", 1)[0]
            if f"{user}@{node}" != target:
                info(f"[{host}] the gateway placed this roast on {node}")
            target = f"{user}@{node}"
        # Ship the exact objects from here: works even if the remote's reference
        # tree never saw these commits and cannot reach GitLab non-interactively.
        info(f"[{host}] pushing pinned commits")
        sh(["git", "push", "-q", f"{target}:{ckpt}", f"{r['barista']['sha']}:refs/roasts/{r['id']}"], cwd=ROOT)
        sh(["git", "push", "-q", f"{target}:{ckpt}/coffea4bees", f"{r['coffea4bees']['sha']}:refs/roasts/{r['id']}"],
           cwd=ROOT / "coffea4bees")
        res = ssh_run(target, _finish_checkout_script(r, ckpt))
        print(res.stdout.strip())
        files = sorted(p for p in roast_dir(r["id"]).iterdir() if p.is_file())
        scp_to(target, files, f"{ckpt}/roasts/{r['id']}/")
        for up in r.get("inputs", {}).get("upstream", []):
            up_id = up.get("id")
            up_dir = roast_dir(up_id)
            if up_dir.is_dir():
                ssh_run(target, f"mkdir -p {rq(f'{ckpt}/roasts/{up_id}')}")
                up_files = sorted(p for p in up_dir.iterdir() if p.is_file())
                if up_files:
                    scp_to(target, up_files, f"{ckpt}/roasts/{up_id}/")
        r["hosts"][host] = {"checkout": ckpt, "ssh": target, "checked_out": now()}
        log_event(r, "checkout", host=host)
        save_roast(r)
    info(f"next: {TOOL} submit {r['id']} --step {r['steps'][0]['name']}")





def _window_name(r: dict, step: dict) -> str:
    """tmux window for a step.  Must identify the roast, not just its label and date:
    submit closes a window of this name when no driver of *this* roast is alive, so two
    roasts sharing a name would let one kill the other's running step."""
    # The FULL label: a [:14] prefix made mvd_run3_30x_c5 / _c5b / _c6 (same shas, same step)
    # collide, and submitting one closed the other's running window (2026-09-28).
    return f"{r['label']}_{r['id'].rsplit('_', 1)[-1]}_{step['name']}"


def _tmux_session(r: dict | None = None) -> str:
    """tmux session name for a roast.

    Each roast runs in its own isolated tmux session ('roast_<id>') so that
    multiple roasts running concurrently or resumed independently on the same
    host do not clobber each other's sessions.
    """
    if not r or "id" not in r:
        return TMUX_SESSION
    clean = re.sub(r"[^A-Za-z0-9_-]", "_", r["id"])
    return f"roast_{clean}"


def _run_script(cfg: dict, r: dict, step: dict, ckpt: str, cores: int, extra: str, resume: bool,
                targets: str = "") -> str:
    """The bash script that runs one step inside its tmux window."""
    name = step["name"]
    smk = step["snakefile"]
    configfile = f"roasts/{r['id']}/config.yml"
    # -p/--printshellcmds: every rule's resolved shell command lands in logs/<step>.log, so a job can be re-run by hand
    # --config roast_id: resolves {roast_id} placeholders in the config (helpers/common.smk), e.g. run-scoped EOS paths
    # --jobs as well as --cores: hosts whose run_container injects a remote-executor snakemake profile
    # (falcon: software/snakemake/profiles/falcon, executor slurm) refuse to run without --jobs N.
    # Targets go BEFORE the options: `--config` takes every argument after it, so a target
    # appended after `--config roast_id=...` is parsed as a malformed name=value entry and
    # snakemake exits ("Config entries have to be defined as name=value pairs").
    tgts = " ".join(t for t in (step.get("targets") or "", targets or "") if t)
    # --config carries what the captured YAML deliberately does not know: which run this is
    # (from the manifest) and whose production area to write into (from ~/.config/roast/config.json).
    # More name=value pairs are safe here; a bare target is not, hence tgts above.
    settings = "".join(f" {k}={shlex.quote(v)}" for k, v in sorted(user_paths(cfg, r).items()))
    base = (f"./run_container snakemake -s {shlex.quote(smk)} {tgts + ' ' if tgts else ''}--configfile {configfile} "
            f"--cores {cores} --jobs {cores} --printshellcmds --config roast_id={r['id']}{settings}")
    if step.get("extra"):
        base += f" {step['extra']}"
    if extra:
        base += f" {extra}"
    unlock = f"{base} --unlock || true\n" if resume else ""
    if resume:
        base += " --rerun-incomplete"
    ref = host_cfg(cfg, step["host"]).get("reference", "")
    ref_proxy = f"{ref}/proxy/x509_proxy" if ref else ""
    proxy_srcs = ['"${X509_USER_PROXY:-}"', '"$HOME/.roast/x509_proxy"']
    if ref_proxy:
        proxy_srcs.append(rq(ref_proxy))
    proxy_srcs.append('"/tmp/x509up_u$(id -u)"')
    proxy_list = " ".join(proxy_srcs)
    return textwrap.dedent(f"""\
        #!/usr/bin/env bash
        # roast {r['id']} step {name} -- generated {now()}
        # Everything lives in main() so bash parses the whole file before executing: a later
        # regeneration of this script cannot derail a run that is already in progress.
        main() {{
        cd {rq(ckpt)} || exit 97
        LOG=logs/{name}.log
        EXIT=logs/{name}.exit
        rm -f "$EXIT"
        echo "=== roast {r['id']} step {name} start $(date) on $(hostname) ===" | tee -a "$LOG"
        echo "=== barista $(git rev-parse --short HEAD) coffea4bees $(cd coffea4bees && git rev-parse --short HEAD) ===" | tee -a "$LOG"
        # Grid proxy: run_container binds ./proxy/x509_proxy into the container; seed it from the
        # user's standard proxy (voms-proxy-init writes /tmp/x509up_u<uid>) or the shared copy
        # `roast proxy` leaves in ~/.roast (LPC /tmp is node-local), whichever is newer than the
        # checkout's. run_container `touch`es a missing proxy, so an empty file counts as none.
        mkdir -p proxy
        for _p in {proxy_list}; do
            if [ -n "$_p" ] && [ -s "$_p" ] && voms-proxy-info -exists -valid 0:05 -file "$_p" >/dev/null 2>&1; then
                cp -f "$_p" proxy/x509_proxy && echo "=== proxy copied from $_p ===" | tee -a "$LOG"
                break
            fi
        done
        if [ -s proxy/kubeconfig ]; then
            export KUBECONFIG="${{PWD}}/proxy/kubeconfig"
        elif [ -s "$HOME/.kube/config" ]; then
            export KUBECONFIG="$HOME/.kube/config"
        fi
        command -v voms-proxy-info >/dev/null && voms-proxy-info --file proxy/x509_proxy --timeleft 2>/dev/null | sed 's/^/=== proxy seconds left: /' | tee -a "$LOG"
        {unlock.strip()}
        {base} 2>&1 | tee -a "$LOG"
        RC=${{PIPESTATUS[0]}}
        echo "=== roast {r['id']} step {name} exit $RC $(date) ===" | tee -a "$LOG"
        echo "$RC" > "$EXIT"
        if [ "$RC" != "0" ]; then
            echo "step {name} FAILED (rc=$RC). Shell kept open for inspection; exit to close."
            exec bash
        fi
        sleep 5
        }}
        main "$@"
    """)


def _submit_nautilus(cfg: dict, r: dict, step: dict, ckpt: str, cores: int, args, resume: bool, targets: str = "") -> None:
    hc = host_cfg(cfg, "nautilus")
    ns = hc.get("namespace", "cms-cmu")
    user = hc.get("user", getpass.getuser())
    jname = _k8s_job_name(r, step)

    # 1. Guard against double-submit
    jres = sh(k8s_cmd("get", "job", jname, "-n", ns, "-o", "json"), check=False)
    if jres.returncode == 0:
        try:
            jinfo = json.loads(jres.stdout)
            if jinfo.get("status", {}).get("active", 0) > 0:
                die(f"a Kubernetes Job '{jname}' for this roast is still running on Nautilus; refusing to double-submit")
        except Exception:
            pass
        sh(k8s_cmd("delete", "job", jname, "-n", ns, "--ignore-not-found"), check=False)
        info(f"[nautilus] removed previous job {jname}")

    name = step["name"]
    smk = step["snakefile"]
    configfile = f"roasts/{r['id']}/config.yml"
    tgts = " ".join(t for t in (step.get("targets") or "", targets or "") if t)
    settings = "".join(f" {k}={shlex.quote(v)}" for k, v in sorted(user_paths(cfg, r).items()))
    base = (f"snakemake -s {shlex.quote(smk)} {tgts + ' ' if tgts else ''}--configfile {configfile} "
            f"--profile software/snakemake/profiles/nautilus --jobs {cores} --printshellcmds --config roast_id={r['id']}{settings}")
    if step.get("extra"):
        base += f" {step['extra']}"
    if args.extra:
        base += f" {args.extra}"
    if resume:
        base += " --rerun-incomplete"

    unlock = f"snakemake -s {shlex.quote(smk)} {tgts + ' ' if tgts else ''}--configfile {configfile} --profile software/snakemake/profiles/nautilus --unlock || true\n" if resume else ""

    # 2. Build Job manifest
    manifest = textwrap.dedent(f"""\
        apiVersion: batch/v1
        kind: Job
        metadata:
          name: {jname}
          namespace: {ns}
          labels:
            roast: "{r['id']}"
            step: "{name}"
            app: "roast-runner"
        spec:
          backoffLimit: 3
          template:
            metadata:
              labels:
                roast: "{r['id']}"
                step: "{name}"
            spec:
              restartPolicy: Never
              serviceAccountName: roast-runner
              containers:
              - name: orchestrator
                image: docker.io/snakemake/snakemake:v9.27.0
                imagePullPolicy: IfNotPresent
                workingDir: {ckpt}
                command: ["bash", "-c"]
                args:
                - |
                  set -e
                  echo "=== roast {r['id']} step {name} start $(date) on Nautilus ==="
                  pip install --no-cache-dir --quiet snakemake-executor-plugin-cluster-generic fsspec fsspec-xrootd
                  mkdir -p /etc/grid-security && ln -sfn /workspace/shared/certificates /etc/grid-security/certificates
                  export X509_USER_PROXY="$PWD/proxy/x509_proxy"
                  export PATH=/workspace/shared/bin:$PATH
                  export NAUTILUS_WORKDIR={ckpt}
                  export CLASSIFIER_CONFIG_PATHS=coffea4bees
                  export ROAST_ID="{r['id']}"
                  export ROAST_STEP="{name}"
                  cd {ckpt}
                  mkdir -p logs
                  LOG=logs/{name}.log
                  EXIT=logs/{name}.exit
                  rm -f "$EXIT"
                  rm -rf .snakemake/locks || true
                  {unlock}
                  {base} 2>&1 | tee "$LOG"
                  RC=${{PIPESTATUS[0]}}
                  echo "$RC" > "$EXIT"
                  echo "=== roast {r['id']} step {name} exit $RC $(date) ===" | tee -a "$LOG"
                  exit $RC
                resources:
                  limits:
                    cpu: "2"
                    memory: "4Gi"
                  requests:
                    cpu: "2"
                    memory: "4Gi"
                volumeMounts:
                - mountPath: /workspace
                  name: ceph-storage
              volumes:
              - name: ceph-storage
                persistentVolumeClaim:
                  claimName: cms-cmu-storage
    """)

    # Sync code at pinned roast SHAs to CephFS checkout (clean archive stream, no ghost files)
    _sync_nautilus_code(cfg, r, ckpt)

    manifest_path = roast_dir(r["id"]) / f"job_{name}.yaml"
    manifest_path.write_text(manifest)

    # 3. Apply manifest
    res = sh(k8s_cmd("apply", "--validate=false", "-f", str(manifest_path)), check=False)
    if res.returncode != 0:
        die(f"failed to submit Kubernetes Job to Nautilus: {res.stderr.strip()}")
    print(res.stdout.strip())

    # 4. Sync updated config.yml to CephFS if changed
    cfg_sha = sha256_file(roast_dir(r["id"]) / "config.yml")
    if cfg_sha != r["config"]["sha256"]:
        r["config"]["sha256"] = cfg_sha
        log_event(r, "config-edited", sha256=cfg_sha[:12])
        sh(k8s_cmd("cp", str(roast_dir(r["id"]) / "config.yml"),
                   f"{ns}/storage-helper:{ckpt}/roasts/{r['id']}/config.yml"), check=False)

    step.setdefault("runs", []).append({
        "ts": now(), "cores": cores, "extra": args.extra or "",
        "targets": targets or "", "dry_run": bool(args.dry_run),
        "test": bool(args.test), "resume": resume,
        "job_name": jname, "namespace": ns, "config_sha256": cfg_sha[:12]
    })
    log_event(r, "resume" if resume else "submit", step=name, host="nautilus")
    save_roast(r)
    info(f"[nautilus] launched Kubernetes Job {jname}")
    info(f"attach: {TOOL} attach {r['id']} --step {name}    |    {TOOL} status {r['id']}")


def _submit(args, resume: bool) -> None:
    cfg = load_config()
    r = load_roast(args.id)
    step = find_step(r, args.step)
    host = step["host"]
    hc = host_cfg(cfg, host)
    if host not in r["hosts"]:
        die(f"roast not checked out on {host}; run `{TOOL} checkout {r['id']} --host {host}`")
    ckpt = r["hosts"][host]["checkout"]
    target = roast_ssh(cfg, r, host) if host != "nautilus" else ""
    cores = args.cores or hc.get("cores", 4)
    extra = args.extra
    targets = getattr(args, "targets", None)
    if resume and extra is None and not args.test and not args.dry_run and step.get("runs"):
        # resume repeats the snakemake args of the last REAL submit: skip dry runs (-n) and test
        # slices, and take the user's extra args only (not the -n / test flags roast appended).
        # (Replaying a `submit -n` here once turned a resume into a 12-second no-op.)
        for run in reversed(step["runs"]):
            if run.get("dry_run") or run.get("test") or "-n" in (run.get("extra") or "").split():
                continue
            extra = run.get("user_extra", run.get("extra", ""))
            cores = args.cores or run.get("cores", cores)
            break
    if resume and targets is None and step.get("runs"):
        # Targets are reused even when --extra is given: a resume that adds e.g. --forcerun must
        # still stop at the sub-step the last real run targeted, not run the whole workflow.
        for run in reversed(step["runs"]):
            if run.get("dry_run") or run.get("test") or "-n" in (run.get("extra") or "").split():
                continue
            targets = run.get("targets", "")
            break
    user_extra = extra or ""
    parts = [extra or ""]
    if args.test:
        parts.append("--config test=true")             # workflows' small-slice mode (local execution, few files)
        # Keep test outputs out of the real output_path, otherwise snakemake later sees the
        # test files as up-to-date targets and the full run is "Nothing to be done".
        m = re.search(r'^output_path:\s*["\']?([^"\'\s#]+)', (roast_dir(r["id"]) / "config.yml").read_text(), re.M)
        if m:
            test_path = m.group(1).rstrip("/") + "_test/"
            parts.append(f"output_path={test_path}")
            info(f"test outputs go to {test_path}")
        else:
            info("WARNING: no top-level output_path in the captured config; test outputs share the real output dir")
    if args.dry_run:
        parts.append("-n")                             # snakemake dry run: plan only, nothing produced
    args.extra = " ".join(x for x in parts if x).strip()

    if host == "nautilus":
        _submit_nautilus(cfg, r, step, ckpt, cores, args, resume, targets or "")
        return

    if host == "cmslpc":
        local_kube = Path.home() / ".kube" / "config"
        if local_kube.is_file():
            sh(["scp", "-q", "-p", *ssh_opts(target), str(local_kube), f"{target}:{ckpt}/proxy/kubeconfig"], check=False)

    session = _tmux_session(r)
    window = _window_name(r, step)
    driver_pat = f"snakemake.*roasts/{r['id']}/config.yml"
    # Guard FIRST, before touching any file on the host: a live snakemake driver for this roast means
    # "running" (refuse); a window without a driver is a finished or derailed step and is closed.
    guard = textwrap.dedent(f"""\
        tmux has-session -t {session} 2>/dev/null || tmux new-session -d -s {session} -n hub
        if pgrep -u "$USER" -f {shlex.quote(driver_pat)} >/dev/null 2>&1; then
            echo "a snakemake driver for this roast is still running on $(hostname); refusing to double-submit" >&2
            exit 3
        fi
        if tmux list-windows -t {session} -F '#W' 2>/dev/null | grep -qx {shlex.quote(window)}; then
            tmux kill-window -t {session}:{shlex.quote(window)}
            if [ -f {rq(ckpt)}/logs/{step['name']}.exit ]; then
                echo "closed finished window {window} (exit=$(cat {rq(ckpt)}/logs/{step['name']}.exit))"
            else
                echo "closed stale window {window} (no driver process, no exit file)"
            fi
        fi
        if tmux list-windows -t {TMUX_SESSION} -F '#W' 2>/dev/null | grep -qx {shlex.quote(window)}; then
            tmux kill-window -t {TMUX_SESSION}:{shlex.quote(window)} 2>/dev/null || true
        fi
    """)
    res = ssh_run(target, guard, check=False)
    if res.returncode == 255:
        # The node this roast was placed on is unreachable.  Its filesystem is shared, so
        # go back through the gateway, take whichever node it gives, and re-pin the roast.
        gateway = resolve_ssh(hc)
        probe = ssh_run(gateway, "hostname -f", check=False) if gateway != target else None
        if probe is not None and probe.returncode == 0 and probe.stdout.strip():
            target = f"{target.split('@')[0]}@{probe.stdout.strip()}"
            info(f"[{host}] {r['hosts'][host]['ssh']} is unreachable; moving this roast to {probe.stdout.strip()}")
            r["hosts"][host]["ssh"] = target
            log_event(r, "repin", host=host, node=probe.stdout.strip())
            save_roast(r)
            res = ssh_run(target, guard, check=False)
    if res.returncode != 0:
        die(res.stderr.strip() or res.stdout.strip())
    if res.stdout.strip():
        print(res.stdout.strip())
    script = _run_script(cfg, r, step, ckpt, cores, args.extra, resume, targets or "")
    local = roast_dir(r["id"]) / f"run_{step['name']}.sh"
    local.write_text(script)
    # Re-ship the whole roast dir: the captured config.yml may have been edited since checkout.
    scp_to(target, sorted(p for p in roast_dir(r["id"]).iterdir() if p.is_file()), f"{ckpt}/roasts/{r['id']}/")
    remote = textwrap.dedent(f"""\
        set -e
        SCRIPT={rq(f"{ckpt}/roasts/{r['id']}/run_{step['name']}.sh")}
        chmod +x "$SCRIPT"
        tmux new-window -d -t {session} -n {shlex.quote(window)} "bash $SCRIPT"
        echo "launched tmux {session}:{window}"
    """)
    res = ssh_run(target, remote, check=False)
    if res.returncode != 0:
        die(res.stderr.strip() or res.stdout.strip())
    print(res.stdout.strip())
    paths = user_paths(cfg, r)
    if paths and r.get("user_paths") != paths:
        r["user_paths"] = paths          # where this run writes; the captured config stays user-neutral
    cfg_sha = sha256_file(roast_dir(r["id"]) / "config.yml")
    if cfg_sha != r["config"]["sha256"]:
        info("captured config.yml changed since `new`; recording the new sha256")
        r["config"]["sha256"] = cfg_sha
        log_event(r, "config-edited", sha256=cfg_sha[:12])
    step.setdefault("runs", []).append({"ts": now(), "cores": cores, "extra": args.extra or "", "user_extra": user_extra,
                                        "targets": targets or "",
                                        "dry_run": bool(args.dry_run), "test": bool(args.test), "resume": resume,
                                        "window": window, "ssh": target, "config_sha256": cfg_sha[:12]})
    log_event(r, "resume" if resume else "submit", step=step["name"], host=host)
    save_roast(r)
    info(f"attach: ssh -t {target} tmux attach -t {session}    |    {TOOL} status {r['id']}")


# Lines worth seeing when scanning a step's log for what went wrong. `Traceback` is deliberately
# absent: dask tears its client down noisily at the end of every successful job, so matching it
# buries the real failures under a dozen benign stacks per wave of jobs.
_LOG_ERROR_RE = (r"Error in rule|WorkflowError|LockException|[A-Za-z]*Error:|"
                 r"Missing(Output|Input)Exception|"
                 r"Exiting because a job execution failed|Removing output files of failed job|"
                 r"JOB EXECUTION FAILED|exited with non-zero|unbound variable|"
                 r"Killed|Out of memory|Segmentation fault|Cutflow check MISMATCH|=== roast .* exit ")


def cmd_log(args) -> None:
    """Show, follow or summarise a step's log without taking over the terminal.

    `attach` needs the tmux window to still exist and replaces this process; `status` shows only
    the last line. This is the in-between: read the log of a finished or running step, over ssh.
    """
    cfg = load_config()
    r = load_roast(args.id)
    step = find_step(r, args.step)
    if step["host"] == "nautilus":
        hc = host_cfg(cfg, "nautilus")
        ns = hc.get("namespace", "cms-cmu")
        jname = _k8s_job_name(r, step)
        if args.follow:
            info(f"following Nautilus job {jname} in namespace {ns} (ctrl-c to stop)")
            cmd = k8s_cmd("logs", f"job/{jname}", "-n", ns, "-f")
            os.execvp(cmd[0], cmd)
        cmd = k8s_cmd("logs", f"job/{jname}", "-n", ns, f"--tail={int(args.lines)}")
        res = sh(cmd, check=False)
        out = (res.stdout or res.stderr or "").strip()
        if out:
            print(out)
        else:
            info(f"no logs found for job {jname} in namespace {ns}")
        return

    target = roast_ssh(cfg, r, step["host"])
    log = f"{checkout_path(cfg, r, step['host'])}/logs/{step['name']}.log"

    # Everything after the last start marker, i.e. the most recent run of this step rather than
    # the whole history of resubmissions appended to the same file.
    since_start = f"tac {rq(log)} | sed '/=== roast .* start /q' | tac"

    if args.follow:
        info(f"following {step['host']}:{log}  (ctrl-c to stop)")
        cmd = f"tail -n {int(args.lines)} -F {rq(log)}"
        if args.errors:
            cmd += f" | grep -E --line-buffered {shlex.quote(_LOG_ERROR_RE)}"
        os.execvp("ssh", ["ssh", "-t", *ssh_opts(target), target, cmd])

    if args.stats:
        # The last "Job stats:" block: the job counts and the reasons snakemake gives for them.
        # This is the only place a dry run's job count appears -- it prints no progress lines, so
        # `status` shows an empty progress column for one.
        script = (f"awk '/^Job stats:/{{n=NR}} {{a[NR]=$0}} "
                  f"END{{if(!n){{print \"no Job stats block in this run\"; exit}} "
                  f"for(i=n;i<=NR;i++) print a[i]}}' <({since_start})")
    elif args.errors:
        script = f"({since_start}) | grep -E {shlex.quote(_LOG_ERROR_RE)} || echo 'no failures in the last run'"
    else:
        script = f"({since_start}) | tail -n {int(args.lines)}"

    res = ssh_run(target, f"set -o pipefail\n{script}\n", check=False)
    out = (res.stdout or "").rstrip()
    if out:
        print(out)
    elif res.returncode != 0:
        die((res.stderr or "").strip() or f"could not read {log} on {step['host']}")
    else:
        info(f"{log} is empty on {step['host']}")


def cmd_attach(args) -> None:
    """Replace this process with `ssh -t <host> tmux attach -t roast`, selecting the step's window."""
    cfg = load_config()
    window, r = None, None
    if args.id:
        r = load_roast(args.id)
        if args.step:
            step = find_step(r, args.step)
            host = step["host"]
            runs = step.get("runs", [])
            window = runs[-1]["window"] if runs else None
        else:
            host = args.host or (r["steps"][0]["host"] if len({s["host"] for s in r["steps"]}) == 1 else None)
            if host is None:
                die("roast spans several hosts; give --step or --host")
            # newest submitted step on that host
            for s in reversed(r["steps"]):
                if s["host"] == host and s.get("runs"):
                    window = s["runs"][-1]["window"]
                    break
    elif args.host:
        host = args.host
    else:
        die("give a roast id, or --host")

    if host == "nautilus":
        hc = host_cfg(cfg, "nautilus")
        ns = hc.get("namespace", "cms-cmu")
        target_step = None
        if args.id and args.step:
            target_step = find_step(r, args.step)
        elif r:
            for s in reversed(r["steps"]):
                if s["host"] == "nautilus" and s.get("runs"):
                    target_step = s
                    break
            if not target_step:
                target_step = next((s for s in r["steps"] if s["host"] == "nautilus"), None)
        if not target_step:
            die("no Nautilus step found to attach")
        jname = _k8s_job_name(r, target_step)
        info(f"attaching to Nautilus job {jname} in namespace {ns}")
        cmd = k8s_cmd("logs", f"job/{jname}", "-n", ns, "-f")
        os.execvp(cmd[0], cmd)

    target = roast_ssh(cfg, r, host) if r else resolve_ssh(host_cfg(cfg, host))
    session = _tmux_session(r)
    if window:
        tmux = f"tmux attach -t {session} \\; select-window -t {shlex.quote(window)} 2>/dev/null || tmux attach -t {TMUX_SESSION} \\; select-window -t {shlex.quote(window)}"
    else:
        tmux = f"tmux attach -t {session} 2>/dev/null || tmux attach -t {TMUX_SESSION}"
    info(f"ssh -t {target} {tmux}")
    os.execvp("ssh", ["ssh", "-t", *ssh_opts(target), target, tmux])


def cmd_proxy(args) -> None:
    """Create (interactively) or check the grid proxy on a host.

    voms-proxy-init writes /tmp/x509up_u<uid>, which every roast launcher copies into its
    checkout's proxy/x509_proxy.  On LPC /tmp is node-local while the gateway is round-robin, so
    the proxy you just made is often on a different node from the roast you want to use it for.
    A copy therefore goes to ~/.roast/x509_proxy, which is on shared home and readable from
    whichever node a roast was pinned to.
    """
    cfg = load_config()
    target = resolve_ssh(host_cfg(cfg, args.host))
    if args.check:
        res = ssh_run(target, f"echo -n 'node '; hostname -s; "
                              f"echo -n 'local '; voms-proxy-info --timeleft 2>/dev/null || echo none; "
                              f"echo -n 'shared '; voms-proxy-info --timeleft -file {rq(SHARED_PROXY)} 2>/dev/null || echo none",
                      check=False)
        where = {}
        for line in res.stdout.split("\n"):
            k, _, v = line.strip().partition(" ")
            if k:
                where[k] = v.strip()
        print(f"{args.host} ({where.get('node', '?')}):")
        for k in ("local", "shared"):
            left = where.get(k, "none")
            pretty = f"{int(left)//3600} h left" if left.isdigit() and int(left) > 0 else "expired or absent"
            print(f"  {k:7s} {pretty}" + (f"   ({SHARED_PROXY}, readable from every node)" if k == "shared" else "   (/tmp, this node only)"))
        return
    # the copy is part of the same command: os.execvp hands the terminal to ssh for the passphrase,
    # so there is no second round trip to do it in
    cmd = (f"voms-proxy-init -rfc -voms cms --valid {shlex.quote(args.valid)}"
           f" && mkdir -p ~/.roast && cp -f /tmp/x509up_u$(id -u) {SHARED_PROXY}"
           f" && chmod 600 {SHARED_PROXY} && echo 'copied to {SHARED_PROXY} for the other nodes'")
    info(f"ssh -t {target} {cmd}")
    os.execvp("ssh", ["ssh", "-t", *ssh_opts(target), target, cmd])


def cmd_pull(args) -> None:
    """rsync a roast's outputs from a host into <barista>/output/roasts/<id>/<path> on this machine."""
    cfg = load_config()
    r = load_roast(args.id)
    hosts = [args.host] if args.host else list(r["hosts"])
    for host in hosts:
        ckpt = r["hosts"][host]["checkout"]
        rel = args.path.strip("/")
        dst = ROOT / "output" / "roasts" / r["id"] / rel
        dst.mkdir(parents=True, exist_ok=True)
        if host == "nautilus":
            hc = host_cfg(cfg, "nautilus")
            ns = hc.get("namespace", "cms-cmu")
            info(f"[nautilus] pulling {rel} -> {dst}/")
            if not args.dry_run:
                p1 = subprocess.Popen(k8s_cmd("exec", "-i", "storage-helper", "-n", ns, "--", "tar", "-cf", "-", "-C", ckpt, rel), stdout=subprocess.PIPE)
                p2 = subprocess.Popen(["tar", "-xf", "-", "-C", str(ROOT / "output" / "roasts" / r["id"])], stdin=p1.stdout)
                p1.stdout.close()
                p2.communicate()
                if p2.returncode != 0:
                    die(f"pull from nautilus failed ({p2.returncode})")
                log_event(r, "pull", host="nautilus", path=rel)
                save_roast(r)
            continue
        target = roast_ssh(cfg, r, host)
        src = f"{target}:{ckpt}/{rel}/"
        cmd = ["rsync", "-a", "--progress", "--exclude=*_test", "--exclude=*_test/"]
        if not args.include_test:
            pass
        else:
            cmd = ["rsync", "-a", "--progress"]
        for g in args.exclude or []:
            cmd.append(f"--exclude={g}")
        if args.only:
            # keep only matching files (directories still traversed)
            for g in args.only:
                cmd.append(f"--include={g}")
            cmd += ["--include=*/", "--exclude=*"]
        if args.dry_run:
            cmd += ["-n", "-v"]
        cmd += ["-e", "ssh " + " ".join(ssh_opts(target)), src, str(dst) + "/"]
        info(f"[{host}] {' '.join(shlex.quote(c) for c in cmd[1:-2])} ...")
        info(f"[{host}] {src} -> {dst}/")
        res = subprocess.run(cmd, text=True)
        if res.returncode != 0:
            die(f"rsync from {host} failed ({res.returncode})")
        if not args.dry_run:
            log_event(r, "pull", host=host, path=rel)
            save_roast(r)
    if not args.dry_run:
        info(f"pulled into {ROOT / 'output' / 'roasts' / r['id']}")


def proxy_candidates() -> list[str]:
    """Where a grid proxy may be, best first, as bash words.

    The roast's own copy (scripts that need it run from the checkout), then this node's /tmp,
    then the shared-home copy.  A path that does not exist costs nothing to skip.
    """
    return ['"$PWD/proxy/x509_proxy"', '"${X509_USER_PROXY:-/tmp/x509up_u$(id -u)}"', rq(SHARED_PROXY)]


def _xrd_auth(eos_url: str, *proxies: str) -> str:
    """One line of bash choosing the xrootd credential for eos_url: a kerberos ticket if present,
    else the first of `proxies` (bash words) that still has time left.

    Validity, not existence.  LPC /tmp is node-local and the login alias is round-robin, so the
    proxy sitting beside a roast is routinely weeks stale, and an expired proxy does not fail
    as a credential error -- CERN EOS answers "[3010] unauthorized identity used", which reads
    like the data being missing and cost an afternoon twice.

    For CERN EOS only a @CERN.CH ticket counts: cmslpc logins carry a @FNAL.GOV one, which eosuser
    rejects the same way. Then use the proxy and hide that ticket with KRB5CCNAME -- pinning
    XrdSecPROTOCOL=gsi instead fails on eosuser ("No protocols left to try").

    Stays on ONE line: it is interpolated into dedent()ed templates, and a multi-line insert
    whose continuation lines carry no indentation would stop dedent finding a common prefix.
    """
    cern = "cern.ch" in eos_url
    realm = "CERN\\.CH" if cern else "[A-Z.]*"
    hide = "; export KRB5CCNAME=FILE:/dev/null" if cern else ""
    cands = " ".join(proxies) or '"${X509_USER_PROXY:-/tmp/x509up_u$(id -u)}"'
    return (f'PROXY=""; for c in {cands}; do if [ -s "$c" ] && '
            f'voms-proxy-info -exists -valid 0:05 -file "$c" >/dev/null 2>&1; then PROXY="$c"; break; fi; done; '
            f'if klist -s 2>/dev/null && klist 2>/dev/null | grep -q "Default principal: .*@{realm}$"; then :; '
            f'elif [ -s "$PROXY" ]; then export X509_USER_PROXY="$PROXY"{hide}; fi')


def _eos_rm_tree_script(eos_url: str, path: str, *proxies: str) -> str:
    """xrdfs has no recursive delete: remove files in parallel, then directories deepest-first.
    Only a "no such file" stat counts as absent; any other stat failure (auth) is an error, not a
    silent success."""
    return textwrap.dedent(f"""\
        set -u
        {_xrd_auth(eos_url, *(proxies or proxy_candidates()))}
        EOS={eos_url}; P={shlex.quote(path)}
        if ! ST=$(xrdfs $EOS stat "$P" 2>&1); then
            if printf '%s' "$ST" | grep -qiE "no such file|\\[3011\\]"; then echo "  (not present) $EOS/$P"; exit 0; fi
            echo "  ERROR: cannot stat $EOS/$P: $ST" >&2; exit 1
        fi
        xrdfs $EOS ls -l -R "$P" 2>/dev/null | awk '$1 !~ /^d/ {{ print $NF }}' | tr '\\n' '\\0' | xargs -0 -r -P 16 -n 1 xrdfs $EOS rm >/dev/null 2>&1
        xrdfs $EOS ls -l -R "$P" 2>/dev/null | awk '$1 ~ /^d/ {{ print $NF }}' | awk '{{ print length($0), $0 }}' | sort -rn | cut -d" " -f2- | while read -r d; do xrdfs $EOS rmdir "$d" >/dev/null 2>&1; done
        xrdfs $EOS rmdir "$P" >/dev/null 2>&1
        xrdfs $EOS stat "$P" >/dev/null 2>&1 && echo "  WARNING: $EOS/$P still exists" || echo "  removed $EOS/$P"
    """)


def _eos_count_script(eos_url: str, path: str, *proxies: str) -> str:
    """Print `COUNT=<n>` for the files under `path`: `missing` if the tree is not there, `error`
    if it cannot be read.

    `clean` deletes the only other copy of a roast's output, so "the copy reported success once"
    is not enough -- this asks the server what is there now, before anything is removed.
    """
    return textwrap.dedent(f"""\
        set -u
        {_xrd_auth(eos_url, *(proxies or proxy_candidates()))}
        EOS={eos_url}; P={shlex.quote(path)}
        if ! ST=$(xrdfs $EOS stat "$P" 2>&1); then
            if printf '%s' "$ST" | grep -qiE "no such file|\\[3011\\]"; then echo "COUNT=missing"; exit 0; fi
            echo "  cannot read $EOS/$P: $ST" >&2; echo "COUNT=error"; exit 0
        fi
        echo "COUNT=$(xrdfs $EOS ls -l -R "$P" 2>/dev/null | awk '$1 !~ /^d/' | wc -l | tr -d ' ')"
    """)


def _eos_counts_script(items: list[tuple[str, str, str]], proxies: list[str]) -> str:
    """Count several EOS trees in one go, as `COUNT|<label>|<n>` lines.

    Each tree runs in its own subshell because the credential differs by server: CERN EOS needs
    the FNAL kerberos ticket hidden, FNAL EOS does not.  One ssh instead of one per tree, which
    on a `clean --all` saves a handshake per roast.
    """
    return "\n".join(f"( {_eos_count_script(url, path, *proxies)} ) | sed 's/^COUNT=/COUNT|{label}|/'"
                     for label, url, path in items)


def _eos_file_counts(cfg: dict, r: dict, items: list[tuple[str, str, str]]) -> dict:
    """{label: (files, why not)} for several EOS trees, in a single ssh to cmslpc."""
    ck = (r.get("hosts", {}).get("cmslpc") or {}).get("checkout")
    proxies = proxy_candidates() + ([rq(f"{ck}/proxy/x509_proxy")] if ck else [])
    res = ssh_run(roast_ssh(cfg, r, "cmslpc"), _eos_counts_script(items, proxies), check=False)
    errs = [ln.strip() for ln in res.stderr.splitlines() if "[ERROR]" in ln or "error" in ln.lower()]
    detail = (errs[-1] if errs else (res.stderr.strip().splitlines() or [""])[-1])[:200]
    out = {}
    for line in res.stdout.splitlines():
        if line.startswith("COUNT|"):
            _, label, val = line.split("|", 2)
            val = val.strip()
            out[label] = (int(val), "") if val.isdigit() else \
                         (-1, "no such directory" if val == "missing" else (detail or "unreadable"))
    for label, _, _ in items:
        out.setdefault(label, (-1, detail or f"no answer from {roast_ssh(cfg, r, 'cmslpc')}"))
    return out


def _eos_file_count(cfg: dict, r: dict, eos_url: str, path: str) -> tuple[int, str]:
    """(files under an EOS path, why not) -- the count is -1 when the tree cannot be read.

    Asked from cmslpc.  The reason is carried back because "missing" and "you have no CERN
    credential on this node" call for very different responses.
    """
    # the usual places, plus this roast's own copy by absolute path: the script runs from home,
    # not from the checkout, so "$PWD/proxy/x509_proxy" would not find it
    ck = (r.get("hosts", {}).get("cmslpc") or {}).get("checkout")
    proxies = proxy_candidates() + ([rq(f"{ck}/proxy/x509_proxy")] if ck else [])
    res = ssh_run(roast_ssh(cfg, r, "cmslpc"), _eos_count_script(eos_url, path, *proxies), check=False)
    # xrootd puts the useful sentence on an "[ERROR] ..." line; the rest is the path echoed back
    errs = [ln.strip() for ln in res.stderr.splitlines() if "[ERROR]" in ln or "error" in ln.lower()]
    detail = (errs[-1] if errs else res.stderr.strip().splitlines()[-1] if res.stderr.strip() else "")[:200]
    for line in res.stdout.splitlines():
        if line.startswith("COUNT="):
            val = line.split("=", 1)[1].strip()
            if val.isdigit():
                return int(val), ""
            return -1, "no such directory" if val == "missing" else (detail or "unreadable")
    return -1, detail or f"no answer from {roast_ssh(cfg, r, 'cmslpc')}"


def _last_copy_ok(r: dict, kind: str):
    """Did the most recent `publish` or `archive` reach every host it was asked to?

    True, False, or None when it was never attempted.  Read from history rather than the
    `publish`/`archive` summary: a `--docs-only` publish writes a cupping note without copying
    anything, and it used to overwrite that summary, which made five already-published roasts
    read as unpublished.  History only ever records copies that actually ran.
    """
    last_by_host = {}
    for h in r.get("history", []):
        if h.get("event") == kind:
            last_by_host[h.get("host")] = h
    if last_by_host:
        return all(bool(h.get("ok")) for h in last_by_host.values())
    summary = r.get(kind) or {}
    return summary.get("ok") if summary else None


def _clean_blockers(r: dict, parsed: dict) -> list[str]:
    """Why a roast's host checkouts may not be deleted yet.

    Each entry is a way the results would not survive the deletion.  Publishing and archiving are
    judged on what the manifest recorded; the steps are judged on what the hosts say now, because
    the manifest only records that a step was submitted, never how it ended.
    """
    out = []
    for kind in ("publish", "archive"):
        state = _last_copy_ok(r, kind)
        if state is None:
            out.append(f"never {'published' if kind == 'publish' else 'archived'}")
        elif not state:
            out.append(f"the last {kind} reported a failure")
    for host, st in sorted(parsed.items()):
        if st.get("error"):
            out.append(f"{host}: {st['error']}")
    for step in r["steps"]:
        st = parsed.get(step["host"])
        # `gone` is a checkout that is already removed: that is the goal, not an objection, and
        # treating it as unfinished would leave a half-cleaned roast impossible to finish
        if st is None or st.get("error") or st.get("gone"):
            continue
        text = dict(st["steps"]).get(step["name"], "not started")
        word = _state_word(text)
        if word != "exit=0":
            out.append(f"step {step['name']} on {step['host']} is {word}, not exit=0")
        elif re.search(r"tmux=[1-9]", re.sub(r"\033\[[0-9;]*m", "", text)):
            out.append(f"step {step['name']} still has a tmux window open on {step['host']} "
                       f"(`{TOOL} attach {r['id']} --step {step['name']}`)")
    return out


def _rm_tree_script(top: str) -> str:
    """Remove a roast's directory on a host.

    A plain `rm -rf` walks a pixi environment's tens of thousands of small files one unlink at a
    time, and on NFS each one is a round trip.  Unlinking in parallel first turns that wait into
    concurrency; the `rm -rf` afterwards clears the directories and anything missed.
    """
    return textwrap.dedent(f"""\
        set -u
        T={rq(top)}
        [ -d "$T" ] || {{ echo "  already gone: $T"; exit 0; }}
        find "$T" -type f -print0 2>/dev/null | xargs -0 -r -P 16 -n 512 rm -f 2>/dev/null || true
        rm -rf "$T"
        [ -d "$T" ] && {{ echo "  WARNING: $T still exists" >&2; exit 1; }} || echo "  removed $T"
    """)


def _clean_probe(cfg: dict, r: dict) -> dict:
    """Everything `clean` needs to know about one roast, gathered over the network.

    Separated from the reporting so that a `--all` run can probe many roasts at once: the work is
    almost entirely waiting on ssh and EOS.  It reads, it never deletes.
    """
    rid = r["id"]
    parsed = {}
    for host, hinfo in r["hosts"].items():
        target = roast_ssh(cfg, r, host)
        res = ssh_run(target, _status_script(r, hinfo["checkout"],
                                             [s for s in r["steps"] if s["host"] == host], host), check=False)
        if res.returncode != 0:
            parsed[host] = {"error": f"unreachable ({target})" if res.returncode == 255
                            else f"ssh failed: {res.stderr.strip()[:60]}"}
        else:
            st = _parse_status(res.stdout)
            # a checkout that is already gone is not a reason to refuse: that is the goal, and a
            # half-cleaned roast has to be able to finish
            parsed[host] = {"gone": True} if st["nocheckout"] else dict(st, error=None)

    probe = {"blockers": _clean_blockers(r, parsed), "checks": [], "counts": {}, "tops": {}, "unsafe": []}
    if probe["blockers"]:
        return probe                       # no point asking EOS about a roast that is not finished

    # What EOS should still be holding.  An empty tree means the copy that "succeeded" left
    # nothing behind, and deleting the checkout would lose the run.
    if (r.get("archive") or {}).get("eos"):
        # stored as one string, "root://cmseos.fnal.gov//store/user/.../<id>"; xrdfs wants the
        # server and the path apart again
        scheme, server, path = r["archive"]["eos"].split("//", 2)
        probe["checks"].append(("archive", f"{scheme}//{server}", "/" + path))
    if (r.get("publish") or {}).get("eos"):
        probe["checks"].append(("published", "root://eosuser.cern.ch", r["publish"]["eos"]))
    if probe["checks"]:
        probe["counts"] = _eos_file_counts(cfg, r, probe["checks"])

    # The directory removed is the checkout's parent, which roast creates per roast.  Insist that
    # it is named after the roast: a hand-edited manifest whose checkout is "~/barista" would
    # otherwise point the deletion at a home directory.
    for host, h in r["hosts"].items():
        top = h["checkout"].rstrip("/").rsplit("/", 1)[0]
        if not top.endswith("/" + rid):
            probe["unsafe"].append(f"{host} checkout {h['checkout']} is not inside a directory "
                                   f"named {rid}; refusing to remove {top}")
        probe["tops"][host] = top
    return probe


def cmd_clean(args) -> None:
    """Delete the host checkouts of roasts whose results are safely on EOS and CERNBox.

    The record survives: the manifest, the cupping note, the archive and the published area all
    stay, so the roast keeps its row in the catalogue.  Only the scratch goes -- the checkout, its
    pixi environment and its job logs -- and `roast checkout <id>` rebuilds it, because the
    manifest pins both shas.
    """
    cfg = load_config()
    if args.id:
        candidates = [load_roast(args.id)]
    elif args.all:
        candidates = [r for r in all_roasts() if r.get("hosts") and not r.get("cleaned")]
    else:
        die(f"name a roast, or pass --all to clean every eligible one (`{TOOL} ls` to choose)")
    if not candidates:
        info("nothing to clean")
        return

    # Probe every candidate at once.  Each probe is a couple of ssh calls that spend their time
    # waiting, so threads turn a queue of roasts into one round of waiting.  Nothing is printed or
    # deleted here: the report has to come out in the order the roasts were listed.
    todo = [r for r in candidates if not r.get("cleaned")]
    probes = {}
    if todo:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(todo))) as pool:
            for r, probe in zip(todo, pool.map(lambda x: _clean_probe(cfg, x), todo)):
                probes[r["id"]] = probe

    cleaned = eligible = 0
    for r in candidates:
        rid = r["id"]
        if r.get("cleaned"):
            info(f"{rid}: already cleaned on {r['cleaned']['ts'][:10]}")
            continue
        print(f"\n\033[1m{rid}\033[0m")
        probe = probes[rid]
        if probe["blockers"]:
            for b in probe["blockers"]:
                print(f"  \033[31mcannot clean\033[0m: {b}")
            continue

        short = False
        for what, url, path in probe["checks"]:
            n, why = probe["counts"].get(what, (-1, "not checked"))
            if n > 0:
                print(f"  {what:10s} {n} files at {url}{path}")
            else:
                print(f"  \033[31mcannot clean\033[0m: {what} tree at {url}{path}: "
                      + ("it is empty" if n == 0 else why))
                if re.search(r"3010|unauthorized identity|Permission denied", why):
                    print(f"  {'':10s} that is a credential, not a missing directory: "
                          f"`{TOOL} proxy` on cmslpc, then try again")
                short = True
        for bad in probe["unsafe"]:
            print(f"  \033[31mcannot clean\033[0m: {bad}")
            short = True
        if short:
            continue
        tops = probe["tops"]
        if not args.yes:
            for host, top in tops.items():
                print(f"  would remove   {host:7s} {top}")
            eligible += 1
            continue
        # Both hosts at once, and the files inside one host in parallel too: deleting a checkout
        # is tens of thousands of unlinks on NFS, where latency rather than bandwidth is the cost.
        ok = True
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(tops))) as pool:
            for (host, top), res in zip(tops.items(),
                                        pool.map(lambda ht: ssh_run(roast_ssh(cfg, r, ht[0]),
                                                                    _rm_tree_script(ht[1]), check=False),
                                                 list(tops.items()))):
                print((res.stdout + res.stderr).strip() or f"  removed {host}: {top}")
                ok &= res.returncode == 0
        if not ok:
            info(f"{rid}: some checkouts could not be removed; not marking it cleaned")
            continue
        r["cleaned"] = {"ts": now(), "hosts": sorted(tops)}
        log_event(r, "clean", hosts=sorted(tops))
        save_roast(r)
        write_docs(r)
        cleaned += 1

    sys.stdout.flush()           # info() writes to stderr; without this the note lands first
    if cleaned:
        write_index()
        info(f"cleaned {cleaned} roast{'s' if cleaned != 1 else ''}; "
             f"`{TOOL} checkout <id>` rebuilds one from its pinned shas")
    elif eligible:
        info(f"dry run: {eligible} roast{'s' if eligible != 1 else ''} ready. "
             f"Re-run with --yes to delete the checkouts listed above.")
    else:
        info("nothing is ready to clean")


def cmd_rm(args) -> None:
    """Delete a roast everywhere: local manifest + docs page, host checkouts, EOS archive, CERNBox publish."""
    cfg = load_config()
    r = load_roast(args.id)
    rid = r["id"]
    eos = cfg.get("eos") or {}
    plan = [f"local   {roast_dir(rid)}", f"local   {DOCS_PROD / (rid + '.md')} (+ regenerate index)"]
    for host, h in r.get("hosts", {}).items():
        plan.append(f"{host:7s} {h['checkout'].rsplit('/', 1)[0]}   (whole roast dir on the host)")
    if r.get("archive", {}).get("eos") or (eos.get("path") and "<" not in eos["path"]):
        plan.append(f"eos     {eos.get('url', 'root://cmseos.fnal.gov')}/{eos.get('path', '').rstrip('/')}/{rid}   (archive AND run-scoped outputs such as friend trees)")
    cb_eos, _ = cernbox_paths(cfg, r)
    plan.append(f"cernbox root://eosuser.cern.ch/{cb_eos.rstrip('/')}/{rid}")
    print(f"roast rm {rid} would remove:")
    for line in plan:
        print("  " + line)
    if not args.yes:
        info("dry run. Re-run with --yes to delete. Use --keep-eos / --keep-cernbox / --keep-hosts to spare a location.")
        return
    # refuse while a step is running
    for host, h in r.get("hosts", {}).items():
        if host == "nautilus":
            hc = host_cfg(cfg, "nautilus")
            ns = hc.get("namespace", "cms-cmu")
            chk = sh(k8s_cmd("get", "pods", "-n", ns, "-l", f"roast={rid}", "--field-selector=status.phase=Running"), check=False)
            if chk.returncode == 0 and "Running" in (chk.stdout or ""):
                die(f"a Kubernetes pod for this roast is still running on Nautilus; cancel it first")
            continue
        target = roast_ssh(cfg, r, host)
        session = _tmux_session(r)
        chk = ssh_run(target, f"(tmux list-windows -t {session} -F '#W' 2>/dev/null; tmux list-windows -t {TMUX_SESSION} -F '#W' 2>/dev/null) | grep -c {shlex.quote(r['label'][:16] + '_')} || true", check=False)
        if chk.stdout.strip() not in ("", "0"):
            die(f"a tmux window for this roast is still open on {host}; finish or kill it first (`{TOOL} attach {rid}`)")
    ok = True
    # remote locations first, while we still have the manifest -- and EOS before the host checkouts,
    # whose proxy/x509_proxy is the credential CERN EOS needs from a cmslpc node (see _xrd_auth)
    lpc = roast_ssh(cfg, r, "cmslpc")
    lpc_ck = (r.get("hosts", {}).get("cmslpc") or {}).get("checkout")
    proxy = rq(f"{lpc_ck}/proxy/x509_proxy") if lpc_ck else '"${X509_USER_PROXY:-/tmp/x509up_u$(id -u)}"'
    if not args.keep_eos and eos.get("path") and "<" not in eos["path"]:
        res = ssh_run(lpc, _eos_rm_tree_script(eos.get("url", "root://cmseos.fnal.gov"), f"{eos['path'].rstrip('/')}/{rid}", proxy), check=False)
        print((res.stdout + res.stderr).strip()); ok &= res.returncode == 0
    if not args.keep_cernbox:
        res = ssh_run(lpc, _eos_rm_tree_script("root://eosuser.cern.ch", f"{cb_eos.rstrip('/')}/{rid}", proxy), check=False)
        print((res.stdout + res.stderr).strip()); ok &= res.returncode == 0
    for host, h in r.get("hosts", {}).items():
        if not args.keep_hosts:
            if host == "nautilus":
                hc = host_cfg(cfg, "nautilus")
                ns = hc.get("namespace", "cms-cmu")
                sh(k8s_cmd("delete", "jobs,pods", "-n", ns, "-l", f"roast={rid}", "--ignore-not-found"), check=False)
                top = h["checkout"].rsplit("/", 1)[0]
                sh(k8s_cmd("exec", "-i", "storage-helper", "-n", ns, "--", "rm", "-rf", top), check=False)
                print(f"  removed {top} on Nautilus")
                continue
            top = h["checkout"].rsplit("/", 1)[0]
            res = ssh_run(roast_ssh(cfg, r, host), f"rm -rf {rq(top)} && echo '  removed {top}'", check=False)
            print((res.stdout + res.stderr).strip()); ok &= res.returncode == 0
    if not ok:
        die("some remote deletions failed; local manifest kept so you can retry")
    shutil.rmtree(roast_dir(rid), ignore_errors=True)
    (DOCS_PROD / f"{rid}.md").unlink(missing_ok=True)
    write_index()
    info(f"removed roast {rid}")


# ---------------------------------------------------------------------------
# pourover: inspect a step's merged histograms locally with the config it ran with
# ---------------------------------------------------------------------------

POUROVER_DEFAULT_PYTHON = "~/python-environments/pourover/bin/python"


def _plot_cmds(target: str, ckpt: str, step_name: str) -> list[dict]:
    """The makePlots invocations a step actually ran, in order, deduped.

    They are in the step log because the launcher runs snakemake with --printshellcmds,
    so this is the config that produced the plots, not a guess from the workflow."""
    script = textwrap.dedent(f"""\
        cd {rq(ckpt)} 2>/dev/null || {{ echo "NOCHECKOUT"; exit 0; }}
        grep -hoE "makePlots\\.py [^ ]+ -o [^ ]+ -m [^ ]+" logs/{step_name}.log 2>/dev/null | awk '!seen[$0]++'
    """)
    res = ssh_run(target, script, check=False)
    if res.returncode != 0:
        die(f"ssh failed: {(res.stderr or res.stdout).strip()[:200]}")
    out = []
    for line in res.stdout.splitlines():
        if line.strip() == "NOCHECKOUT":
            die(f"checkout missing at {ckpt}")
        t = line.split()
        if len(t) < 6 or "-m" not in t or "-o" not in t:
            continue
        out.append({"coffea": t[1], "outdir": t[t.index("-o") + 1], "metadata": t[t.index("-m") + 1], "cmd": line})
    return out


def _rsync_from(target: str, ckpt: str, rel_paths: list[str], dest_root: Path) -> None:
    """Pull specific files, preserving their paths under dest_root.  rsync is the cache:
    a file already present with the same size and mtime is not transferred again."""
    dest_root.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-a", "-v", "--files-from=-",
           "-e", "ssh " + " ".join(ssh_opts(target)), f"{target}:{ckpt}/", str(dest_root) + "/"]
    res = subprocess.run(cmd, text=True, input="\n".join(rel_paths) + "\n")
    if res.returncode != 0:
        die(f"rsync failed ({res.returncode})")


def cmd_pourover(args) -> None:
    cfg = load_config()
    r = load_roast(args.id)
    step = find_step(r, args.step)
    host = args.host or step["host"]
    if host not in r["hosts"]:
        die(f"roast not checked out on {host}")
    ckpt = r["hosts"][host]["checkout"]
    target = roast_ssh(cfg, r, host)
    dest_root = ROOT / "output" / "roasts" / r["id"]

    coffea_rel, meta_rel = args.coffea, args.config
    if not (coffea_rel and meta_rel) or args.list:
        cmds = _plot_cmds(target, ckpt, step["name"])
        if not cmds:
            die(f"no makePlots command found in logs/{step['name']}.log on {host}; "
                f"pass --coffea and --config explicitly (paths relative to the checkout)")
        if args.list:
            for i, c in enumerate(cmds, 1):
                print(f"{i}. {c['coffea']}\n   -m {c['metadata']}   (-o {c['outdir']})")
            return
        if args.match:
            cmds = [c for c in cmds if args.match in c["cmd"]] or die(f"no makePlots command matching {args.match!r}")
        if args.which:
            if not 1 <= args.which <= len(cmds):
                die(f"--which {args.which} out of range (1..{len(cmds)}); use --list")
            chosen = cmds[args.which - 1]
        elif len(cmds) > 1 and not args.match:
            info(f"{len(cmds)} makePlots commands in this step; using the last (see --list, --which, --match)")
            chosen = cmds[-1]
        else:
            chosen = cmds[-1]
        coffea_rel = coffea_rel or chosen["coffea"]
        meta_rel = meta_rel or chosen["metadata"]

    want = [coffea_rel] + ([meta_rel] if not Path(meta_rel).is_absolute() else [])
    if args.no_pull:
        info("skipping the pull (--no-pull)")
    else:
        info(f"[{host}] pulling {', '.join(want)} -> {dest_root}")
        _rsync_from(target, ckpt, want, dest_root)

    coffea_local = dest_root / coffea_rel
    if not coffea_local.exists():
        die(f"{coffea_local} not present (pull it without --no-pull)")
    # The metadata that ran is pulled alongside; a local override path is used as given.
    meta_local = dest_root / meta_rel
    if args.config and not (dest_root / args.config).exists():
        meta_local = Path(args.config)
    if not meta_local.exists():
        die(f"plot config {meta_local} not found")

    py = Path(cfg.get("pourover", {}).get("python", POUROVER_DEFAULT_PYTHON)).expanduser()
    script = ROOT / "coffea4bees" / "plots" / "pourOver.py"
    if not script.exists():
        die(f"{script} not found")
    cmd = [str(py) if py.exists() else sys.executable, str(script), str(coffea_local), "-m", str(meta_local),
           "--port", str(args.port), "--output-dir", str(ROOT / "pourover_output" / "roasts" / f"{r['id']}_{step['name']}")]
    if not py.exists():
        info(f"{py} not found; falling back to {sys.executable} (see `pourover.python` in {CONFIG_PATH})")
    cmd += shlex.split(args.extra or "")
    log_event(r, "pourover", step=step["name"], coffea=coffea_rel, metadata=meta_rel)
    save_roast(r)
    info(f"{coffea_local.name}  -m {meta_local.name}   (pourOver serves in the background and prints how to stop it;"
         f" add --extra=--foreground to keep it in this terminal)")
    os.execv(cmd[0], cmd)


def cmd_submit(args) -> None:
    _submit(args, resume=False)


def cmd_resume(args) -> None:
    _submit(args, resume=True)


def _status_script(r: dict, ckpt: str, steps: list[dict], host: str) -> str:
    # "<step>:<this roast's window name>" pairs.  The window has to be matched exactly: counting
    # names that merely end in "_<step>" reported another roast's leftover window as this one's,
    # which `clean` then refused to work around.
    step_windows = " ".join(shlex.quote(f"{s['name']}:{_window_name(r, s)}") for s in steps)
    rid = shlex.quote(r["id"])
    # Batch-system view, filtered to this roast: both schedulers record the submitting
    # directory per job (condor Iwd, slurm WorkDir), and that is the roast checkout.
    if host == "cmslpc":
        batch = "\n".join([
            # Attribute the queue to its step.  Spooling overwrites Iwd, so the submitting
            # directory is gone; what survives is the dask scheduler each worker was told to
            # call, and the step log records the scheduler it started.  Jobs no local log
            # claims are reported too: they belong to a roast this checkout cannot see.
            f'''condor_q -af:t JobStatus Arguments 2>/dev/null | awk -F\'\\t\' \'{{ s="-"; if (match($2, /tcp:\\/\\/[0-9.]+:[0-9]+/)) s=substr($2, RSTART, RLENGTH); n[s "\\t" $1]++ }} END {{ for (k in n) {{ split(k, a, "\\t"); print a[1] "\\t" a[2] "\\t" n[k] }} }}\' | while IFS="$(printf \'\\t\')" read -r sched st cnt; do owner=$(grep -l -- "$sched" logs/*.log 2>/dev/null | head -1); owner=${{owner##*/}}; owner=${{owner%%.log}}; echo "CONDOR|${{owner:-?}}|$sched|$st|$cnt"; done || true''',
        ])
    elif host in ("falcon", "bridges2"):
        batch = "\n".join([
            # live jobs, each followed by the last line of its own slurm log (training progress, loss, ...)
            f'''squeue -u "$USER" -h -o \'%i|%k|%T|%M|%l|%R|%C|%m|%b|%Z\' 2>/dev/null | awk -F\'|\' -v rid={rid} \'$10 ~ rid {{ print $1 "|" $2 "|" $3 "|" $4 "|" $5 "|" $6 "|" $7 "|" $8 "|" $9 }}\' | while IFS="|" read -r jid rest; do hit=$(grep -H -m1 "SLURM jobid $jid " logs/*.log 2>/dev/null | head -1); step=${{hit%%:*}}; step=${{step##*/}}; step=${{step%%.log}}; [ -n "$hit" ] || step=""; log=$(ls -t slurm_logs/*/$jid.log slurm_logs/$jid.log 2>/dev/null | head -1); rule=$(echo "$rest" | cut -d"|" -f1); [ -n "$rule" ] || rule=$(echo "$hit" | grep -oE "slurm_logs/[^/]+/" | head -1 | cut -d/ -f2); echo "SLURM|$jid|$step|$rule|$(echo "$rest" | cut -d"|" -f2-)|$(tail -n 1 "$log" 2>/dev/null | tr -d "\\r" | tr "|" " ")"; done''',
            # recently finished jobs of this roast (failures are what you want to see)
            f'''sacct -u "$USER" -S now-2days -P -n -o JobID,JobName,State,Elapsed,MaxRSS,WorkDir 2>/dev/null | awk -F\'|\' -v rid={rid} \'$1 !~ /\\./ && $6 ~ rid && $3 !~ /RUNNING|PENDING/ {{ print $1 "|" $2 "|" $3 "|" $4 "|" $5 }}\' | tail -40 | while IFS="|" read -r jid rest; do hit=$(grep -H -m1 "SLURM jobid $jid " logs/*.log 2>/dev/null | head -1); step=${{hit%%:*}}; step=${{step##*/}}; step=${{step%%.log}}; [ -n "$hit" ] || step=""; rule=$(echo "$hit" | grep -oE "slurm_logs/[^/]+/" | head -1 | cut -d/ -f2); if [ -z "$rule" ]; then rule=$(echo "$rest" | cut -d"|" -f1); case "$rule" in ????????-????-????-????-????????????) rule=${{rule%%-*}};; esac; fi; echo "SLURMDONE|$jid|$step|$rule|$(echo "$rest" | cut -d"|" -f2-)"; done''',
            f'''sinfo -h -o "SINFO|%P|%D|%T|%G" 2>/dev/null | head -3''',
        ])
    else:
        batch = "true"
    session = _tmux_session(r)
    return textwrap.dedent(f"""\
        cd {rq(ckpt)} 2>/dev/null || {{ echo "NOCHECKOUT"; exit 0; }}
        if pgrep -u "$USER" -f {shlex.quote(f"snakemake.*roasts/{r['id']}/config.yml")} >/dev/null 2>&1; then live=1; else live=0; fi
        WINDOWS=$( (tmux list-windows -t {session} -F '#W' 2>/dev/null; tmux list-windows -t {TMUX_SESSION} -F '#W' 2>/dev/null) || true )
        for SW in {step_windows}; do
            S=${{SW%%:*}}; W=${{SW#*:}}
            if [ -f logs/$S.exit ]; then st="exit=$(cat logs/$S.exit)";
            elif [ -f logs/$S.log ]; then
                if [ "$live" != "0" ]; then st="running";
                elif tac logs/$S.log | sed '/=== roast .* start /q' | grep -Eq "WorkflowError|Exiting because a job execution failed|Error in rule"; then st="error";
                else st="stalled"; fi
            else st="not started"; fi
            win=$(printf '%s\\n' "$WINDOWS" | grep -cx -- "$W" || true)
            prog=$(tac logs/$S.log 2>/dev/null | sed '/=== roast .* start /q' | grep -a -m1 -oE '[0-9]+ of [0-9]+ steps \\([0-9]+%\\) done')
            last=$(tail -n 1 logs/$S.log 2>/dev/null | tr -d "\\r" | tr "|" " ")
            echo "STEP|$S|$st|tmux=$win|$prog|$last"
        done
        {batch}
    """)


def _jobnum(j: dict) -> int:
    head = j["jid"].split("_")[0].split(".")[0]
    return int(head) if head.isdigit() else 0


CONDOR_STATES = {"1": "idle", "2": "running", "3": "removing", "4": "done",
                 "5": "HELD", "6": "ERROR", "7": "suspended"}


def _condor_row(groups: list[dict]) -> str:
    """One line per dask cluster: how its workers are split across the queue."""
    by_sched = {}
    for g in groups:
        by_sched.setdefault(g["sched"], {})[CONDOR_STATES.get(g["state"], g["state"])] = int(g["n"] or 0)
    out = []
    for sched, states in sorted(by_sched.items()):
        total = sum(states.values())
        detail = ", ".join(f"{v} {k}" for k, v in sorted(states.items(), key=lambda kv: -kv[1]))
        word = "worker " if total == 1 else "workers"
        body = f"{total:>5d} {word}"
        if states.get("HELD"):
            body = f"\033[31m{body}\033[0m"
        out.append(f"condor  {body}  {detail}" + (f"   {sched}" if sched != "-" else ""))
    return "\n".join(out)


def _slurm_rows(live: list[dict], done: list[dict]) -> list[str]:
    """One line per rule, showing its latest attempt.

    Snakemake resubmits a failed rule (and people retry jobs by hand), so the scheduler
    keeps several job ids for the same rule.  Listing them all put stale failures next to
    the attempt that replaced them, which reads as if the rule were still broken.  Older
    attempts are folded into a note on the line that superseded them.
    """
    def strip(rule: str) -> str:
        return rule[5:] if rule.startswith("rule_") else rule

    for j in live + done:
        j["rule"] = strip(j["rule"]) or "-"
    latest_live = {j["rule"]: j for j in sorted(live, key=_jobnum)}
    attempts: dict[str, list[dict]] = {}
    for j in sorted(done, key=_jobnum):
        attempts.setdefault(j["rule"], []).append(j)

    def note(rule: str, superseded: list[dict]) -> str:
        failed = [a for a in superseded if a["state"] != "COMPLETED"]
        if not superseded:
            return ""
        if failed:
            return f"   (after {len(failed)} failed attempt{'s' if len(failed) > 1 else ''})"
        return f"   (attempt {len(superseded) + 1})"

    rows = []
    for rule, j in latest_live.items():
        colour = "\033[33m" if j["state"] == "RUNNING" else "\033[31m" if j["state"] not in ("PENDING", "COMPLETED") else ""
        rows.append((_jobnum(j),
                     f"slurm   {j['jid']:>7s} {rule:20s} {colour}{j['state']:9s}\033[0m "
                     f"{j['el']:>8s}/{j['lim']:<8s} {j['node']:12s} cpu={j['cpus']} mem={j['mem']} {j['tres']}"
                     f"{note(rule, attempts.get(rule, []))}".rstrip()))
        if j["tail"].strip():
            rows.append((_jobnum(j) + 0.5, f"                {j['tail'].strip()}"))
    finished = []
    for rule, tries in attempts.items():
        if rule in latest_live:      # a live attempt supersedes every finished one
            continue
        j = tries[-1]
        colour = "\033[32m" if j["state"] == "COMPLETED" else "\033[31m"
        finished.append((_jobnum(j),
                         f"slurm   {j['jid']:>7s} {rule[:20]:20s} {colour}{j['state']:9s}\033[0m {j['el']:>8s}"
                         f"{('  maxrss=' + j['rss']) if j['rss'] else ''}{note(rule, tries[:-1])}".rstrip()))
    rows += sorted(finished)[-6:]
    return [text for _, text in sorted(rows)]


def _parse_status(stdout: str) -> dict:
    """Turn the host script's output into {steps, live, done, extra, nocheckout}.

    `live` and `done` are keyed by step name, so each step's jobs are rendered under it and
    a rule that several steps happen to share (two phases both run `train`) is never merged.
    Jobs the step logs do not account for land under the empty key.
    """
    out = {"steps": [], "live": {}, "done": {}, "condor": {}, "extra": [], "nocheckout": False}
    for line in stdout.splitlines():
        if line.startswith("STEP|"):
            _, name, st, win, prog, last = (line.split("|", 5) + [""] * 6)[:6]
            colour = ("\033[32m" if st == "exit=0" else "\033[31m" if st.startswith("exit=") or st in ("error", "stalled")
                      else "\033[33m" if st == "running" else "")
            out["steps"].append((name, f"{colour}{st:12s}\033[0m {win:7s} {prog:28s} {last[:90]}".rstrip()))
        elif line.startswith("CONDOR|"):
            _, step, sched, code, n = (line.split("|", 4) + [""] * 5)[:5]
            out["condor"].setdefault(step, []).append({"sched": sched, "state": code, "n": n})
        elif line.startswith("SLURM|"):
            f = (line.split("|") + [""] * 12)[:12]
            out["live"].setdefault(f[2], []).append(
                dict(zip(("jid", "rule", "state", "el", "lim", "node", "cpus", "mem", "tres", "tail"),
                         [f[1]] + f[3:11] + [f[11][:100]])))
        elif line.startswith("SLURMDONE|"):
            f = (line.split("|") + [""] * 7)[:7]
            out["done"].setdefault(f[2], []).append(
                dict(zip(("jid", "rule", "state", "el", "rss"), [f[1]] + f[3:])))
        elif line.startswith("SINFO|"):
            _, part, nodes, state, gres = (line.split("|") + [""] * 5)[:5]
            out["extra"].append(f"cluster {part} {nodes} node(s) {state} {gres}")
        elif line == "NOCHECKOUT":
            out["nocheckout"] = True
    return out


STATE_GLYPH = {"exit=0": ("\u2713", "\033[32m"), "running": ("\u25cf", "\033[33m"),
               "error": ("\u2717", "\033[31m"), "stalled": ("?", "\033[31m"),
               "not started": ("\u00b7", ""), "unreachable": ("!", "\033[31m")}


def _state_word(text: str) -> str:
    """The bare state word out of a rendered step line."""
    plain = re.sub(r"\033\[[0-9;]*m", "", text).strip()
    word = plain.split("  ")[0].strip()
    return word if word in STATE_GLYPH else ("exit=n" if word.startswith("exit=") else word)


def _summary_line(r: dict, parsed: dict, width: int) -> str:
    """One line per roast: what each step is, and what is happening right now."""
    glyphs, running = [], []
    for step in r["steps"]:
        host, name = step["host"], step["name"]
        st = parsed.get(host)
        if st is None:
            state = "not started"
        elif st.get("error"):
            state = "unreachable"
        else:
            state = _state_word(dict(st["steps"]).get(name, "not started"))
        mark, colour = STATE_GLYPH.get(state, ("\u2717", "\033[31m"))
        glyphs.append(f"{colour}{name[:8]}{mark}\033[0m")
        if state == "running":
            tail = ""
            live = (st["live"].get(name) or [{}])[0] if st else {}
            if live.get("tail"):
                tail = live["tail"]
            else:
                tail = re.sub(r"\033\[[0-9;]*m", "", dict(st["steps"])[name]).split("  ")[-1]
            # progress bars pad themselves with runs of spaces; squeeze so the budget buys signal
            tail = " ".join(tail.split())
            running.append(f"{name} on {host}: {tail[:46]}" if tail else f"{name} on {host}")
    # While something is running, what it is doing is the news.  Once nothing is, the useful thing
    # is where to read the results, so a finished roast carries its cupping note like a cleaned one.
    note = "  ".join(running) or (f"published  {docs_page_url(r['id'])}"
                                  if r.get("publish", {}).get("url") else "")
    marks = " ".join(glyphs)
    visible = len(re.sub(r"\033\[[0-9;]*m", "", marks))
    return f"  {r['id']:<{width}s}  {marks}{' ' * max(1, 24 - visible)} {note}".rstrip()


def _by_family(roasts: list[dict]) -> list[tuple[str, dict]]:
    """(family, roast) in catalogue order: families alphabetically, newest first within one.

    The same split the cupping notes use for their sub-pages, so a production reads the same way
    in the terminal as on the website.
    """
    families = {}
    for r in roasts:
        families.setdefault(roast_category(r), []).append(r)
    out = []
    for fam in sorted(families):
        out += [(fam, r) for r in sorted(families[fam], key=lambda x: x["created"], reverse=True)]
    return out


def _links(r: dict) -> list[str]:
    """Where a roast can be read once it is published: the cupping note, the results, the archive."""
    out = [f"docs:    {docs_page_url(r['id'])}"]
    if (r.get("publish") or {}).get("url"):
        out.append(f"results: {r['publish']['url']}")
    if (r.get("archive") or {}).get("eos"):
        out.append(f"archive: {r['archive']['eos']}/")
    return out


def _cleaned_summary(r: dict, width: int) -> str:
    """One line for a roast whose checkouts have been removed.

    No host is contacted: `clean` refuses unless every step ended exit=0 and both EOS copies were
    verified, so being cleaned at all is what the glyphs record.
    """
    mark, colour = STATE_GLYPH["exit=0"]
    glyphs = " ".join(f"{colour}{s['name'][:8]}{mark}\033[0m" for s in r["steps"])
    visible = len(re.sub(r"\033\[[0-9;]*m", "", glyphs))
    note = f"cleaned {r['cleaned']['ts'][:10]}  {docs_page_url(r['id'])}"
    return f"  {r['id']:<{width}s}  {glyphs}{' ' * max(1, 24 - visible)} {note}".rstrip()


def _is_step_worker(pod: dict, r: dict, step_name: str, jname: str) -> bool:
    """Is `pod` a worker of this roast's step? By label when it has one (older pods may lack
    `step`), else by the roast hash nautilus_submit.py puts in the pod name."""
    md = pod.get("metadata", {})
    pname = md.get("name", "")
    if jname in pname:                                   # the step's own orchestrator
        return False
    labels = md.get("labels") or {}
    if labels.get("roast"):
        return labels["roast"] == r["id"] and labels.get("step", step_name) == step_name
    h = hashlib.sha256(r["id"].encode()).hexdigest()[:6]
    return pname.startswith(f"smk-{h}-")


def _parse_nautilus_status(cfg: dict, r: dict, steps: list[dict]) -> dict:
    hc = host_cfg(cfg, "nautilus")
    ns = hc.get("namespace", "cms-cmu")
    out = {"steps": [], "live": {}, "done": {}, "condor": {}, "k8s_pods": {}, "extra": [], "nocheckout": False, "error": None}

    # Query all pods in namespace to catch both labeled orchestrator and worker jobs
    res = sh(k8s_cmd("get", "pods", "-n", ns, "-o", "json"), check=False)
    all_pods = []
    if res.returncode == 0 and res.stdout.strip():
        try:
            all_pods = json.loads(res.stdout).get("items", [])
        except Exception:
            pass

    for s in steps:
        name = s["name"]
        jname = _k8s_job_name(r, s)
        jres = sh(k8s_cmd("get", "job", jname, "-n", ns, "-o", "json"), check=False)
        job_info = {}
        if jres.returncode == 0 and jres.stdout.strip():
            try:
                job_info = json.loads(jres.stdout)
            except Exception:
                pass

        if not job_info:
            st = "not started"
            colour = ""
            prog = ""
        else:
            status_f = job_info.get("status", {})
            if status_f.get("succeeded", 0) > 0:
                st = "exit=0"
                colour = "\033[32m"
            elif status_f.get("failed", 0) > 0:
                st = "error"
                colour = "\033[31m"
            elif status_f.get("active", 0) > 0:
                st = "running"
                colour = "\033[33m"
            else:
                st = "pending"
                colour = ""

            # Child worker pods (nautilus_submit.py): labelled roast=<id>, step=<step>, and named
            # smk-<sha256(id)[:6]>-<rule>-... Other roasts' pods in the shared namespace must not count.
            child_pods = [p for p in all_pods if _is_step_worker(p, r, name, jname)]

            n_done, n_run, n_pend, n_fail = 0, 0, 0, 0
            rows = []
            for p in sorted(child_pods, key=lambda x: x.get("metadata", {}).get("name", "")):
                pname = p.get("metadata", {}).get("name", "")
                phase = p.get("status", {}).get("phase", "Unknown")
                node = p.get("spec", {}).get("nodeName", "-")
                c_status = p.get("status", {}).get("containerStatuses", [{}])[0] if p.get("status", {}).get("containerStatuses") else {}
                state_desc = phase
                if c_status:
                    state = c_status.get("state", {})
                    if "waiting" in state:
                        state_desc = state["waiting"].get("reason", "Waiting")
                    elif "terminated" in state:
                        reason = state["terminated"].get("reason", "Terminated")
                        ec = state["terminated"].get("exitCode", 0)
                        state_desc = f"{reason}({ec})"
                    elif "running" in state:
                        state_desc = "Running"

                is_fail = phase == "Failed" or (any(k in state_desc for k in ["CrashLoop", "Error", "ImagePull", "Terminated", "OOMKilled"]) and not state_desc.startswith("Completed(0)"))
                is_done = phase == "Succeeded" or state_desc == "Completed(0)"
                is_pend = phase == "Pending" or state_desc in ("ContainerCreating", "PodInitializing", "Waiting")
                is_run = phase == "Running" and state_desc == "Running"

                if is_fail:
                    n_fail += 1
                    pcol = "\033[31m"
                elif is_done:
                    n_done += 1
                    pcol = "\033[32m"
                elif is_pend:
                    n_pend += 1
                    pcol = "\033[33m"
                else:
                    n_run += 1
                    pcol = "\033[36m"

                rows.append(f"nautilus {pname[:28]:28s} {pcol}{state_desc:14s}\033[0m node={node}")
            out["k8s_pods"][name] = rows

            # Extract any Snakemake step completion from orchestrator log
            smk_prog = ""
            log_res = sh(k8s_cmd("logs", "-n", ns, f"job/{jname}", "--tail=60"), check=False)
            if log_res.returncode == 0 and log_res.stdout:
                m = re.findall(r"(\d+ of \d+ steps \(\d+%\) done)", log_res.stdout)
                if m:
                    smk_prog = m[-1]

            summary_parts = []
            if n_run > 0: summary_parts.append(f"{n_run} running")
            if n_pend > 0: summary_parts.append(f"{n_pend} pending")
            if n_fail > 0: summary_parts.append(f"\033[31m{n_fail} failed\033[0m")
            if n_done > 0: summary_parts.append(f"\033[32m{n_done} done\033[0m")

            k8s_summary = f"k8s: {', '.join(summary_parts)}" if summary_parts else "k8s-job"
            if smk_prog:
                prog = f"{smk_prog} ({k8s_summary})"
            else:
                prog = k8s_summary

        out["steps"].append((name, f"{colour}{st:12s}\033[0m {prog}".rstrip()))

    if all_pods:
        out["extra"].append(f"nautilus {len(all_pods)} pod(s) in namespace {ns}")

    return out


def _gather_status(cfg: dict, r: dict) -> dict:
    """{host: parsed status} for every host a roast has a checkout on (one ssh/kubectl round per host)."""
    parsed = {}
    for host, hinfo in r["hosts"].items():
        if host == "nautilus":
            parsed[host] = _parse_nautilus_status(cfg, r, [s for s in r["steps"] if s["host"] == host])
            continue
        target = roast_ssh(cfg, r, host)
        res = ssh_run(target, _status_script(r, hinfo["checkout"], [s for s in r["steps"] if s["host"] == host], host), check=False)
        if res.returncode == 255:
            node = target.split("@")[-1]
            parsed[host] = {"error": f"{node} unreachable — the driver is gone; "
                                     f"`{TOOL} resume <id> --step <step>` moves this roast to a live node"}
        elif res.returncode != 0:
            parsed[host] = {"error": f"ssh failed ({target}): {res.stderr.strip()[:80]}"}
        else:
            st = _parse_status(res.stdout)
            st["error"] = f"checkout missing at {hinfo['checkout']}" if st["nocheckout"] else None
            parsed[host] = st
    return parsed


def cmd_status(args) -> None:
    cfg = load_config()
    roasts = [load_roast(args.id)] if args.id else [r for r in all_roasts() if r.get("hosts")]
    if args.active:
        roasts = [r for r in roasts if not r.get("cleaned")]
    # Naming a roast means you want its detail; asking for all of them means you want the shape.
    detail = args.detail or (bool(args.id) and not args.summary)
    width = max((len(r["id"]) for r in roasts), default=10)
    # Grouped by the same families as the catalogue's sub-pages, newest first within a family,
    # so the terminal and the website order a production the same way.
    last_family = None
    for family, r in _by_family(roasts):
        if not args.id and family != last_family:
            print(f"\n\033[1m{family}\033[0m")
            last_family = family
        # A cleaned roast has no checkout left to ask about, so it costs no ssh at all.  This is
        # most of what makes `status` fast once a production is finished.
        if r.get("cleaned"):
            if not detail:
                print(_cleaned_summary(r, width))
                continue
            print(f"\n\033[1m{r['id']}\033[0m  barista {r['barista']['sha'][:7]}  coffea4bees {r['coffea4bees']['sha'][:7]}  {r['config']['source']}")
            w = max(4, min(30, max((len(s["name"]) for s in r["steps"]), default=4)))
            for step in r["steps"]:
                print(f"  {step['name']:<{w}s} {step['host']:7s} exit=0 (checkout removed)")
            print(f"  {'':<{w}s} cleaned {r['cleaned']['ts'][:10]} from " + ", ".join(r["cleaned"]["hosts"])
                  + f"; `{TOOL} checkout {r['id']}` rebuilds it")
            for line in _links(r):
                print(f"  {'':<{w}s} {line}")
            continue
        if detail:
            print(f"\n\033[1m{r['id']}\033[0m  barista {r['barista']['sha'][:7]}  coffea4bees {r['coffea4bees']['sha'][:7]}  {r['config']['source']}")
        # One ssh per host, but the report is ordered by step: a roast is a pipeline, and
        # which machine a phase happens to run on matters less than where the pipeline is.
        parsed = _gather_status(cfg, r)
        if not detail:
            print(_summary_line(r, parsed, width))
            continue
        # Width follows the longest step name of THIS roast: a `--step host:Snakefile` name
        # can be long, and truncating it would hide the argument you need to type back.
        w = max(4, min(30, max((len(s["name"]) for s in r["steps"]), default=4)))
        pad = " " * (w + 8)
        for step in r["steps"]:
            name, host = step["name"], step["host"]
            st = parsed.get(host)
            if st is None:
                print(f"  {name:<{w}s} {host:7s} not checked out")
                continue
            if st.get("error"):
                print(f"  {name:<{w}s} {host:7s} {st['error']}")
                continue
            text = dict(st["steps"]).get(name)
            print(f"  {name:<{w}s} {host:7s} {text if text is not None else 'not reported'}")
            if st.get("condor", {}).get(name):
                for row in _condor_row(st["condor"][name]).splitlines():
                    print(f"{pad}{row}")
            for row in _slurm_rows(st["live"].get(name, []), st["done"].get(name, [])):
                print(f"{pad}{row}")
            if st.get("k8s_pods", {}).get(name):
                for row in st["k8s_pods"][name]:
                    print(f"{pad}{row}")
        for host, st in parsed.items():
            if st.get("error"):
                continue
            for text in st["extra"]:
                print(f"  {'':<{w}s} {host:7s} {text}")
        # Jobs submitted from the checkout that no step log accounts for: things run by hand.
        # A live one is worth seeing (it is holding resources); finished ones are just history,
        # so they are counted rather than listed unless asked for.
        for host, st in parsed.items():
            if st.get("error"):
                continue
            for sched_step, groups in sorted(st.get("condor", {}).items()):
                if sched_step in {n for n, _ in st["steps"]}:
                    continue                      # already shown under its step
                print(f"  {'':<{w}s} {host:7s} condor jobs from a roast this checkout does not know:")
                for row in _condor_row(groups).splitlines():
                    print(f"{pad}{row}")
            stray_live, stray_done = st["live"].get("", []), st["done"].get("", [])
            if stray_live or (stray_done and args.all):
                print(f"  {'':<{w}s} {host:7s} not part of any step (submitted by hand from the checkout):")
                for row in _slurm_rows(stray_live, stray_done if args.all else []):
                    print(f"{pad}{row}")
            elif stray_done:
                n = len(stray_done)
                failed = sum(1 for j in stray_done if j["state"] != "COMPLETED")
                # not `detail`: that is the flag controlling this whole block, and rebinding it
                # here turned the rest of a multi-roast `--detail` run back into summary lines
                failed_note = f", {failed} failed" if failed else ""
                print(f"  {'':<{w}s} {host:7s} {n} finished job{'s' if n > 1 else ''} not part of any step{failed_note} (--all to list)")
        if r.get("publish", {}).get("url"):
            for line in _links(r):
                print(f"  {'':<{w}s} {line}")


ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _roast_snapshot(cfg: dict, r: dict) -> dict:
    """JSON-ready state of one roast for the HTML monitor: the data `status --detail` prints, as fields."""
    snap = {"id": r["id"], "label": r.get("label", r["id"]), "family": roast_category(r),
            "created": r["created"], "config": r["config"]["source"], "steps": [], "errors": [],
            "cleaned": bool(r.get("cleaned")), "docs": (r.get("publish") or {}).get("url", "")}
    if r.get("cleaned"):
        snap["steps"] = [{"name": s["name"], "host": s["host"], "state": "exit=0", "pct": 100, "detail": "checkout removed", "jobs": []}
                         for s in r["steps"]]
        return snap
    parsed = _gather_status(cfg, r)
    for host, st in parsed.items():
        if st.get("error"):
            snap["errors"].append(f"{host}: {st['error']}")
    for step in r["steps"]:
        name, host = step["name"], step["host"]
        st = parsed.get(host)
        if st is None or st.get("error"):
            snap["steps"].append({"name": name, "host": host, "state": "unreachable" if st else "not started",
                                  "pct": None, "detail": "", "jobs": []})
            continue
        text = ANSI_RE.sub("", dict(st["steps"]).get(name, "not started"))
        state = _state_word(text)
        m = re.search(r"(\d+) of (\d+) steps \((\d+)%\) done", text)
        pct = int(m.group(3)) if m else (100 if state == "exit=0" else None)
        detail = " ".join(text.replace(m.group(0), "").split()[1:]) if m else " ".join(text.split()[1:])
        jobs = []
        if st.get("condor", {}).get(name):
            jobs += _condor_row(st["condor"][name]).splitlines()
        jobs += _slurm_rows(st["live"].get(name, []), st["done"].get(name, []))
        jobs += st.get("k8s_pods", {}).get(name, [])
        snap["steps"].append({"name": name, "host": host, "state": state, "pct": pct,
                              "done": int(m.group(1)) if m else None, "total": int(m.group(2)) if m else None,
                              "detail": detail[:160], "jobs": [" ".join(ANSI_RE.sub("", j).split()) for j in jobs]})
    return snap


def _roast_finished(snap: dict) -> bool:
    return bool(snap["steps"]) and all(s["state"] == "exit=0" for s in snap["steps"])


def cmd_monitor(args) -> None:
    """Serve a self-refreshing HTML dashboard of roast progress (stdlib http.server, bound to localhost)."""
    import http.server
    import threading
    import time

    cfg = load_config()
    page = Path(__file__).with_name("roast_monitor.html")
    if not page.exists():
        die(f"{page} is missing")
    snaps: dict[str, dict] = {}                  # id -> snapshot (+ "polled": epoch seconds)
    lock = threading.Lock()
    force = threading.Event()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)

    def wanted() -> list[dict]:
        if args.ids:
            return [load_roast(i) for i in args.ids]
        return [r for r in all_roasts() if r.get("hosts") and not r.get("cleaned")]

    def poll(r: dict) -> None:
        try:
            snap = _roast_snapshot(cfg, r)
        except Exception as e:                   # one bad roast must not blank the page
            snap = {"id": r["id"], "label": r.get("label", r["id"]), "family": roast_category(r), "created": r["created"],
                    "config": r["config"]["source"], "steps": [], "errors": [f"monitor: {e}"], "cleaned": False, "docs": ""}
        snap["polled"] = time.time()
        with lock:
            snaps[r["id"]] = snap

    def loop() -> None:
        while True:
            try:
                todo = []
                for r in wanted():
                    old = snaps.get(r["id"])
                    # a roast whose every step is exit=0 cannot change; the refresh button re-polls it anyway
                    if old and _roast_finished(old) and not force.is_set():
                        continue
                    todo.append(r)
                force.clear()
                list(pool.map(poll, todo))
            except Exception as e:
                print(f"monitor loop: {e}", file=sys.stderr)
            force.wait(args.interval)

    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.split("?")[0] == "/api/status":
                with lock:
                    data = {"now": time.time(), "interval": args.interval,
                            "roasts": sorted(snaps.values(), key=lambda x: x["created"], reverse=True)}
                self._send(200, json.dumps(data).encode(), "application/json")
            elif self.path.split("?")[0] in ("/", "/index.html"):
                self._send(200, page.read_bytes(), "text/html; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path == "/api/refresh":
                force.set()
                self._send(200, b"{}", "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def log_message(self, *a):
            pass

    threading.Thread(target=loop, daemon=True).start()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    info(f"roast monitor on {url}  (polls every {args.interval}s; Ctrl-C to stop)")
    if not args.no_open:
        import webbrowser
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def copy_settings(kind: str, cfg: dict, r: dict, include_test: bool = False, with_pdfs: bool = False) -> dict:
    """kind = 'publish' | 'archive'.  Defaults <- user config[kind] <- roast.json[kind + '_rules']."""
    base = PUBLISH_DEFAULTS if kind == "publish" else ARCHIVE_DEFAULTS
    ps = dict(base)
    for src in (cfg.get(kind) or {}, r.get(kind + "_rules") or {}):
        ps.update({k: v for k, v in src.items() if k in base})
    if include_test:
        ps["exclude"] = [e for e in ps["exclude"] if "_test" not in e]
    if kind == "publish" and not with_pdfs:
        ps["include"] = [e for e in ps["include"] if e not in ("*.pdf", "*.svg")]
    return ps


def _copy_script(r: dict, ckpt: str, eos_url: str, dst_dir: str, ps: dict, jobs: int = 32,
                 dry_run: bool = False, with_logs: bool = True, htaccess: bool = False) -> str:
    """Bash for the host: select files under output/ (+ logs/, roasts/<id>/), skip ones already at the
    destination with the same size, xrdcp the rest in parallel.  dst_dir is an absolute EOS path."""
    eos_root = dst_dir.rsplit("/", 1)[0]
    exts = " -o ".join(f"-name {shlex.quote(g)}" for g in ps["include"]) or "-false"
    prune = " ".join(f"-path {shlex.quote(g)} -prune -o" for g in ps["exclude"])
    size = f"-size -{int(ps['max_mb'])}M" if int(ps.get("max_mb") or 0) > 0 else ""
    extra_dirs = f"find logs roasts/{r['id']} -type f;" if with_logs else ""
    dry = "cat \"$LIST\"; echo; echo \"$(wc -l < \"$LIST\") files would be copied (dry run)\"; exit 0" if dry_run else ""
    ht = textwrap.dedent(f"""\
        # CERN EOS websites return 403 on directories without an index; enable Apache listings once at the root.
        HT=$(mktemp); printf 'Options +Indexes\\n' > "$HT"
        if ! xrdfs $EOS ls -l {eos_root}/.htaccess >/dev/null 2>&1; then
            xrdcp -f -s "$HT" "$EOS/{eos_root}/.htaccess" && echo "directory listings enabled via {eos_root}/.htaccess"
        fi
        rm -f "$HT"
        """) if htaccess else ""
    return textwrap.dedent(f"""\
        set -uo pipefail
        cd {rq(ckpt)} 2>/dev/null || {{ echo "no checkout at {ckpt} on $(hostname): nothing to copy. \\`{TOOL} checkout\\` rebuilds the code, but a cleaned or deleted checkout's outputs are gone" >&2; exit 4; }}
        command -v xrdcp >/dev/null || {{ echo "xrdcp not found on $(hostname)" >&2; exit 2; }}
        {_xrd_auth(eos_url, *proxy_candidates())}
        [ -n "$PROXY" ] || klist -s 2>/dev/null || {{ echo "no grid proxy with time left on $(hostname) and no kerberos ticket: run \\`{TOOL} proxy\\`" >&2; exit 3; }}
        EOS={eos_url}
        DST_BASE=$EOS/{dst_dir}
        LIST=$(mktemp); HAVE=$(mktemp); RAW_HAVE=$(mktemp); TODO=$(mktemp)
        trap 'rm -f "$LIST" "$HAVE" "$RAW_HAVE" "$TODO"' EXIT
        {{ [ -d output ] && find -L output {prune} -type f \\( {exts} \\) {size} -print; \\
           {extra_dirs} \\
           [ -f index.html ] && echo "index.html"; }} | sort -u > "$LIST"
        N=$(wc -l < "$LIST")
        {dry}
        # Discover publishable HTML pages for cupping notes and index summary
        awk '/\\.html$/ && !/dask-report/ && !/_test/ && !/\\/logs\\// && !/\\/singlefiles\\// {{ print "PAGE:" $0 }}' "$LIST"
        # What is already there (path<TAB>size), so reruns only copy new or changed files.
        xrdfs $EOS ls -l -R {dst_dir} 2>/dev/null > "$RAW_HAVE" || true
        awk -v b="{dst_dir}/" '$1 !~ /^d/ {{ p=$NF; sub(b,"",p); print p "\\t" $4 }}' "$RAW_HAVE" | sort > "$HAVE"
        rm -f "$RAW_HAVE"
        python3 -c '
import os, sys
have = dict()
with open(sys.argv[2], "r", errors="replace") as f:
    for line in f:
        parts = line.strip().split("\t", 1)
        if len(parts) == 2 and parts[1].isdigit():
            have[parts[0]] = int(parts[1])
with open(sys.argv[1], "r", errors="replace") as f_in, open(sys.argv[3], "w") as f_out:
    for line in f_in:
        p = line.strip()
        if not p:
            continue
        try:
            if have.get(p) != os.path.getsize(p):
                print(p, file=f_out)
        except OSError:
            pass
' "$LIST" "$HAVE" "$TODO" || {{ echo "python diff failed" >&2; exit 1; }}
        T=$(wc -l < "$TODO")
        {ht}
        echo "copying from $(hostname):$PWD -> $DST_BASE"
        echo "$N files selected, $((N-T)) already up to date, $T to copy ({jobs} in parallel)"
        [ "$T" = 0 ] && exit 0
        xrdfs $EOS mkdir -p {dst_dir} 2>/dev/null || true
        export DST_BASE
        tr '\\n' '\\0' < "$TODO" | xargs -0 -P {jobs} -n 25 bash -c '
            for f in "$@"; do
                xrdcp -f -s "$f" "$DST_BASE/$f" && echo "ok $f" || echo "FAILED $f"
            done
        ' _ \\
            | awk '/^FAILED/ {{ print; fail++ }} /^ok/ {{ ok++; if (ok % 100 == 0) print "  ... " ok " copied" }} END {{ printf "done: %d copied, %d failed\\n", ok, fail; exit fail>0 }}'
    """)


def _copy_failure_hint(rc: int, credential_hint: str) -> str:
    """Why a copy failed, when the exit code already says.  Guessing "check your proxy" at a
    missing checkout sends people to look at the wrong thing."""
    return {2: " (xrdcp is not on that host)",
            3: " (no grid proxy with time left there)",
            4: " (no checkout on that host: its outputs are gone)"}.get(rc, f" ({credential_hint})")


def _refuse_if_cleaned(r: dict, what: str, allow: bool = False) -> None:
    """A cleaned roast has no checkout, so anything reading from one fails deep inside a remote
    script with a `cd` error.  Say what happened instead, and how to get the checkout back."""
    if r.get("cleaned") and not allow:
        die(f"{r['id']} was cleaned on {r['cleaned']['ts'][:10]}: there is no checkout to {what} from. "
            f"`{TOOL} checkout {r['id']}` rebuilds it from the pinned shas.")


def _sync_nautilus_to_host(cfg: dict, r: dict, target_host: str = "cmslpc", publish_mode: bool = False) -> None:
    if "nautilus" not in r.get("hosts", {}) or target_host not in r.get("hosts", {}):
        return
    hc = host_cfg(cfg, "nautilus")
    ns = hc.get("namespace", "cms-cmu")
    nautilus_ck = r["hosts"]["nautilus"]["checkout"]
    target_ck = r["hosts"][target_host]["checkout"]
    target_ssh = roast_ssh(cfg, r, target_host)
    info(f"[nautilus] syncing outputs to [{target_host}]...")
    extra_tar_args = []
    if publish_mode:
        for pat in ("*.pt", "*.root", "*.parquet", "*.coffea", "*.tar.gz", "*.bin", "*.pkl", "*/performance/*", "*/classifier_inputs/*"):
            extra_tar_args.extend(["--exclude", pat])
    p1 = subprocess.Popen(k8s_cmd("exec", "-i", "storage-helper", "-n", ns, "--", "tar", "-czf", "-", *extra_tar_args, "-C", nautilus_ck, "output"), stdout=subprocess.PIPE)
    p2 = subprocess.Popen(["ssh", *ssh_opts(target_ssh), target_ssh, f"tar -xzf - -C {rq(target_ck)}"], stdin=p1.stdout)
    p1.stdout.close()
    p2.communicate()
    if p2.returncode == 0:
        info(f"[nautilus] synced outputs to [{target_host}]")
    else:
        info(f"[nautilus] sync to [{target_host}] returned code {p2.returncode}")


def cmd_publish(args) -> None:
    cfg = load_config()
    r = load_roast(args.id)
    _refuse_if_cleaned(r, "publish", allow=args.docs_only)
    analysis = getattr(args, "analysis", None) or r.get("analysis") or detect_analysis(r)
    eos_base, url_base = cernbox_paths(cfg, r, analysis)
    eos_dir = f"{eos_base.rstrip('/')}/{r['id']}"
    url = f"{url_base.rstrip('/')}/{r['id']}/"
    hosts = [args.host] if args.host else list(r["hosts"])
    ok = True
    ps = copy_settings("publish", cfg, r, args.include_test, with_pdfs=getattr(args, "with_pdfs", False))
    if args.dry_run:
        info("publish rules: include=" + " ".join(ps["include"]) + "  exclude=" + " ".join(ps["exclude"]) + f"  max_mb={ps['max_mb']}")
    pages = set(r.get("publish", {}).get("pages", []))
    if not args.docs_only:
        if not args.host:
            pages = set()   # full publish: rebuild the list from what the hosts have now
        if not args.dry_run and "nautilus" in hosts and "cmslpc" in r.get("hosts", {}):
            _sync_nautilus_to_host(cfg, r, "cmslpc", publish_mode=True)
        for host in hosts:
            if host == "nautilus":
                continue
            hinfo = r["hosts"][host]
            info(f"[{host}] publishing to {eos_dir}")
            res = ssh_run(roast_ssh(cfg, r, host),
                          _copy_script(r, hinfo["checkout"], "root://eosuser.cern.ch", eos_dir, ps, jobs=args.jobs,
                                       dry_run=args.dry_run, with_logs=True, htaccess=True), check=False)
            combined_out = res.stdout + res.stderr
            for line in combined_out.splitlines():
                if line.startswith("PAGE:"):
                    pages.add(line.split("PAGE:", 1)[1].strip())
            filtered_out = "\n".join(l for l in combined_out.splitlines() if not l.startswith("PAGE:"))
            if filtered_out.strip():
                print(filtered_out.strip())
            if args.dry_run:
                continue
            if res.returncode != 0:
                ok = False
                info(f"[{host}] publish FAILED" + _copy_failure_hint(res.returncode,
                     "CERN auth on that host? try `kinit <user>@CERN.CH` or a voms proxy there"))
            log_event(r, "publish", host=host, ok=res.returncode == 0)
    if args.dry_run:
        return
    prev = r.get("publish") or {}
    sorted_pages = sorted(pages)
    r["publish"] = {"eos": eos_dir, "url": url,
                    "ts": prev.get("ts", now()) if args.docs_only else now(),
                    "ok": prev.get("ok") if args.docs_only else ok,
                    "pages": sorted_pages}
    if args.docs_only:
        r["publish"]["docs_ts"] = now()
    save_roast(r)
    write_docs(r)
    write_index()

    summary_html = generate_roast_index_html(r, sorted_pages)
    rdir = roast_dir(r["id"])
    if rdir.exists():
        (rdir / "index.html").write_text(summary_html, encoding="utf-8")
    if not args.dry_run and not args.docs_only and ok:
        pub_host = None
        for h in hosts:
            if h != "nautilus" and h in r.get("hosts", {}):
                pub_host = h
                break
        if pub_host:
            hinfo = r["hosts"][pub_host]
            ck = hinfo["checkout"]
            info(f"[{pub_host}] publishing summary roast index.html -> {url}index.html")
            upload_script = textwrap.dedent(f"""\
                set -e
                cd {rq(ck)}
                cat > index.html
                {_xrd_auth('root://eosuser.cern.ch', *proxy_candidates())}
                xrdcp -f -s index.html "root://eosuser.cern.ch/{eos_dir}/index.html"
            """)
            target_ssh = roast_ssh(cfg, r, pub_host)
            up_res = subprocess.run(
                ["ssh", *ssh_opts(target_ssh), target_ssh, f"bash -l -c {rq(upload_script)}"],
                input=summary_html.encode("utf-8"),
                capture_output=True,
            )
            if up_res.returncode != 0:
                err_msg = up_res.stderr.decode("utf-8", errors="replace").strip()
                info(f"[{pub_host}] warning: failed to upload summary index.html: {err_msg}")
            else:
                info(f"summary roast index: {url}index.html")

    info(f"results: {url}")
    info(f"docs: {DOCS_PROD / (r['id'] + '.md')} and index.md regenerated; run `{TOOL} commit {r['id']}` to commit the roast manifest")


def cmd_archive(args) -> None:
    """Copy heavy products (.coffea/.root/...) to FNAL EOS under <eos.path>/<id>/, mirroring the checkout layout."""
    cfg = load_config()
    r = load_roast(args.id)
    _refuse_if_cleaned(r, "archive")
    eos = cfg.get("eos") or {}
    if not eos.get("path") or "<" in eos.get("path", ""):
        die("set \"eos\": {\"url\": \"root://cmseos.fnal.gov\", \"path\": \"/store/user/<you>/HH4b_prod\"} in " + str(CONFIG_PATH))
    eos_url = eos.get("url", "root://cmseos.fnal.gov")
    dst_dir = f"{eos['path'].rstrip('/')}/{r['id']}"
    ps = copy_settings("archive", cfg, r, args.include_test)
    if args.dry_run:
        info("archive rules: include=" + " ".join(ps["include"]) + "  exclude=" + " ".join(ps["exclude"]) + f"  max_mb={ps['max_mb'] or 'none'}")
    hosts = [args.host] if args.host else list(r["hosts"])
    ok = True
    if not args.dry_run and "nautilus" in hosts and "cmslpc" in r.get("hosts", {}):
        _sync_nautilus_to_host(cfg, r, "cmslpc")
    for host in hosts:
        if host == "nautilus":
            continue
        hinfo = r["hosts"][host]
        info(f"[{host}] archiving to {eos_url}/{dst_dir}")
        res = ssh_run(roast_ssh(cfg, r, host),
                      _copy_script(r, hinfo["checkout"], eos_url, dst_dir, ps, jobs=args.jobs,
                                   dry_run=args.dry_run, with_logs=True, htaccess=False), check=False)
        print((res.stdout + res.stderr).strip())
        if args.dry_run:
            continue
        if res.returncode != 0:
            ok = False
            info(f"[{host}] archive FAILED" + _copy_failure_hint(res.returncode,
                 f"grid proxy on that host? `{TOOL} proxy --check`"))
        log_event(r, "archive", host=host, ok=res.returncode == 0)
    if args.dry_run:
        return
    r["archive"] = {"eos": f"{eos_url}/{dst_dir}", "ts": now(), "ok": ok}
    save_roast(r)
    write_docs(r)
    write_index()
    info(f"archive: {eos_url}/{dst_dir}   (list: xrdfs {eos_url} ls -R {dst_dir})")


def _default_commit_message(r: dict) -> str:
    rid = r["id"]
    completed_steps = []
    for h in r.get("history", []):
        if isinstance(h, dict) and h.get("event") == "exit" and h.get("code") == 0:
            step = h.get("step")
            if step and step not in completed_steps:
                completed_steps.append(step)

    details = []
    if completed_steps:
        s_text = f"step{'s' if len(completed_steps) > 1 else ''} {','.join(completed_steps)} done"
        details.append(s_text)
    if r.get("archive", {}).get("ok"):
        details.append("archived")
    if r.get("publish", {}).get("ok"):
        details.append("published")

    summary = "; ".join(details) if details else "update manifest and configuration"
    return f"roast {rid}: {summary}"


def cmd_commit(args) -> None:
    """Commit the roast manifest and associated files under roasts/<id>/ to git, with optional push."""
    r = load_roast(args.id)
    rid = r["id"]
    rdir = roast_dir(rid)
    if not rdir.exists():
        die(f"roast directory {rdir} does not exist")

    rel_rdir = rdir.relative_to(ROOT)

    res_status = sh(["git", "status", "--porcelain", "--", str(rel_rdir)], cwd=ROOT, check=True)
    res_staged = sh(["git", "diff", "--name-only", "--cached", "--", str(rel_rdir)], cwd=ROOT, check=True)

    status_out = res_status.stdout.strip()
    staged_out = res_staged.stdout.strip()

    if not status_out and not staged_out:
        info(f"Nothing to commit in {rel_rdir}")
        return

    if args.message:
        msg = args.message.strip()
        if not msg.startswith(f"roast {rid}"):
            msg = f"roast {rid}: {msg}"
    else:
        msg = _default_commit_message(r)

    if args.dry_run:
        info(f"[dry-run] would stage: git add {rel_rdir}")
        info(f"[dry-run] would commit with message: {msg}")
        if args.push:
            res_branch = sh(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, check=True)
            branch = res_branch.stdout.strip()
            remote = args.remote or "origin"
            info(f"[dry-run] would push to {remote} {branch}")
        return

    info(f"staging {rel_rdir}")
    sh(["git", "add", str(rel_rdir)], cwd=ROOT, check=True)

    res_diff = sh(["git", "diff", "--name-only", "--cached", "--", str(rel_rdir)], cwd=ROOT, check=True)
    if not res_diff.stdout.strip():
        info(f"Nothing staged to commit in {rel_rdir}")
        return

    info(f"committing: {msg.splitlines()[0]}")
    sh(["git", "commit", "-m", msg, "--", str(rel_rdir)], cwd=ROOT, check=True)

    if args.push:
        res_branch = sh(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, check=True)
        branch = res_branch.stdout.strip()
        if not branch or branch == "HEAD":
            die("cannot push from detached HEAD")
        remote = args.remote or "origin"
        info(f"pushing to {remote} {branch}")
        sh(["git", "push", "-u", remote, branch], cwd=ROOT, check=True)



# ---------------------------------------------------------------------------
# Cupping notes (docs/prod)
# ---------------------------------------------------------------------------

def _step_state(r: dict, step: dict) -> str:
    runs = step.get("runs", [])
    if not runs:
        return "not submitted"
    return f"submitted {runs[-1]['ts'][:10]}" + (" (resumed)" if runs[-1].get("resume") else "")


def write_docs(r: dict) -> None:
    DOCS_PROD.mkdir(parents=True, exist_ok=True)
    b, c = r["barista"], r["coffea4bees"]
    url = r.get("publish", {}).get("url", "")
    lines = [
        f"# {r['id']}",
        "",
        f"*{r.get('label','')} — roasted by {r.get('owner','?')} on {r['created'][:10]}*",
        "",
        "| | |",
        "|---|---|",
        f"| barista | [`{b['sha'][:12]}`]({b['web']}/-/commit/{b['sha']}) |",
        f"| coffea4bees | [`{c['sha'][:12]}`]({c['web']}/-/commit/{c['sha']}) |",
        f"| config | `{r['config']['source']}` (sha256 `{r['config']['sha256'][:12]}`) — [captured copy]({url}roasts/{r['id']}/config.yml) |" if url
        else f"| config | `{r['config']['source']}` (sha256 `{r['config']['sha256'][:12]}`) |",
        f"| results | [{url}]({url}) |" if url else "| results | not published |",
        f"| manifest | [roast.json]({url}roasts/{r['id']}/roast.json) |" if url else "",
        (f"| production area | `{r['user_paths']['eos_prod']}/{r['id']}/` |" if r.get("user_paths", {}).get("eos_prod") else ""),
        (f"| archive (EOS) | `{r['archive']['eos']}/` (list: `xrdfs root://{r['archive']['eos'].split('//')[1]} ls -R /{r['archive']['eos'].split('//')[2]}`) |"
         if r.get("archive", {}).get("eos") else ""),
        (f"| checkouts | removed {r['cleaned']['ts'][:10]}; `{TOOL} checkout {r['id']}` rebuilds them from the pinned shas |"
         if r.get("cleaned") else ""),
        "",
        "## Steps",
        "",
        "| step | host | snakefile | state | log |",
        "|---|---|---|---|---|",
    ]
    for s in r["steps"]:
        log = f"[{s['name']}.log]({url}logs/{s['name']}.log)" if url else ""
        lines.append(f"| {s['name']} | {s['host']} | `{s['snakefile']}` {s.get('targets','')} | {_step_state(r, s)} | {log} |")
    pages = r.get("publish", {}).get("pages", [])
    if url and pages:
        # galleries (…/index.html) and cutflow closure tables (…/cutflow_*.html), grouped by workflow dir
        lines += ["", "## Pages", "", "| workflow | page |", "|---|---|"]
        for p in pages:
            parts = p.split("/")
            wf = parts[2] if len(parts) > 3 else parts[-2]          # output/<label>/<workflow>/...
            name = parts[-2] + " gallery" if parts[-1] == "index.html" else parts[-1].removesuffix(".html")
            lines.append(f"| {wf} | [{name}]({url}{p}) |")
    if r.get("notes"):
        lines += ["", "## Notes", "", r["notes"]]
    lines += ["", "## History", ""]
    for h in r.get("history", []):
        extra = " ".join(f"{k}={v}" for k, v in h.items() if k not in ("ts", "event"))
        lines.append(f"- {h['ts']} {h['event']} {extra}")
    (DOCS_PROD / f"{r['id']}.md").write_text("\n".join(l for l in lines if l is not None) + "\n")


def generate_roast_index_html(r: dict, pages: list[str] | None = None) -> str:
    """Generate a standalone, rich HTML summary page (index.html) for the root of the published roast folder on CERNBox."""
    rid = html.escape(str(r.get("id", "")))
    label = html.escape(str(r.get("label", "")))
    owner = html.escape(str(r.get("owner", "unknown")))
    created = html.escape(str(r.get("created", ""))[:19])
    analysis = html.escape(str(r.get("analysis") or detect_analysis(r) or "HH4b"))

    pub_info = r.get("publish", {})
    pub_ok = pub_info.get("ok")
    pub_ts = html.escape(str(pub_info.get("ts", ""))[:19])

    if pub_ok is True:
        status_badge = '<span class="badge badge-success">● Published</span>'
    elif pub_ok is False:
        status_badge = '<span class="badge badge-danger">● Failed</span>'
    else:
        status_badge = '<span class="badge badge-warning">● Unpublished</span>'

    b = r.get("barista", {})
    c = r.get("coffea4bees", {})
    b_sha = str(b.get("sha", ""))
    c_sha = str(c.get("sha", ""))
    b_web = b.get("web", "https://gitlab.cern.ch/cms-cmu/barista")
    c_web = c.get("web", "https://gitlab.cern.ch/cms-cmu/coffea4bees")

    b_link = f'<a href="{b_web}/-/commit/{b_sha}" target="_blank" class="code-link">{b_sha[:8]}</a>' if b_sha else "—"
    c_link = f'<a href="{c_web}/-/commit/{c_sha}" target="_blank" class="code-link">{c_sha[:8]}</a>' if c_sha else "—"

    cfg_src = html.escape(str(r.get("config", {}).get("source", "—")))
    cfg_sha = html.escape(str(r.get("config", {}).get("sha256", ""))[:12])

    hosts = list(r.get("hosts", {}).keys())
    hosts_html = " ".join(f'<span class="badge badge-host">{html.escape(h)}</span>' for h in hosts) or "—"

    history = r.get("history", [])
    step_rows = []
    for s in r.get("steps", []):
        sname = html.escape(str(s.get("name", "")))
        shost = html.escape(str(s.get("host", "")))
        snakefile = html.escape(str(s.get("snakefile", "")))
        targets = html.escape(str(s.get("targets", "")))

        last_exit = None
        for h in history:
            if isinstance(h, dict) and h.get("event") == "exit" and h.get("step") == s.get("name"):
                last_exit = h.get("code")

        if last_exit == 0:
            st_badge = '<span class="badge badge-success">Done</span>'
        elif last_exit is not None:
            st_badge = f'<span class="badge badge-danger">Exit {last_exit}</span>'
        else:
            state_text = _step_state(r, s)
            st_badge = f'<span class="badge badge-secondary">{html.escape(state_text)}</span>'

        log_path = f"logs/{s.get('name')}.log"
        log_cell = f'<a href="{log_path}" class="file-link">{sname}.log</a>'

        target_str = f" <span class='muted'>({targets})</span>" if targets else ""
        step_rows.append(f"""
        <tr>
            <td><strong>{sname}</strong></td>
            <td><span class="badge badge-host">{shost}</span></td>
            <td><code>{snakefile}</code>{target_str}</td>
            <td>{st_badge}</td>
            <td>{log_cell}</td>
        </tr>""")

    steps_table_html = "\n".join(step_rows) if step_rows else "<tr><td colspan='5' class='muted'>No steps registered</td></tr>"

    if pages is None:
        pages = r.get("publish", {}).get("pages", [])

    galleries = []
    cutflows = []
    others = []

    for p in pages:
        p_esc = html.escape(p)
        parts = p.split("/")
        if p.endswith("index.html"):
            wf = parts[2] if len(parts) > 3 else parts[-2] if len(parts) > 1 else "root"
            sub = parts[-2] if len(parts) > 2 and parts[-2] != wf else ""
            title = f"{wf} ({sub}) gallery" if sub else f"{wf} gallery"
            galleries.append((title, p_esc, p))
        elif "cutflow" in p.lower():
            name = parts[-1].removesuffix(".html")
            cutflows.append((name, p_esc, p))
        else:
            name = parts[-1].removesuffix(".html")
            others.append((name, p_esc, p))

    def make_page_items(items):
        if not items:
            return '<div class="empty-state">None found</div>'
        html_items = []
        for title, p_esc, p_raw in items:
            html_items.append(f"""
            <div class="page-item" data-search="{html.escape(p_raw.lower())}">
                <a href="{p_esc}" class="page-link" target="_blank">
                    <span class="page-icon">↗</span>
                    <span class="page-title">{html.escape(title)}</span>
                </a>
                <span class="page-path">{p_esc}</span>
            </div>""")
        return "\n".join(html_items)

    galleries_html = make_page_items(galleries)
    cutflows_html = make_page_items(cutflows)
    others_html = make_page_items(others)
    total_pages = len(pages)

    history_items = []
    for h in reversed(history):
        if isinstance(h, dict):
            hts = html.escape(str(h.get("ts", "")))
            hevt = html.escape(str(h.get("event", "")))
            extra = html.escape(" ".join(f"{k}={v}" for k, v in h.items() if k not in ("ts", "event")))
            history_items.append(f"<li><span class='muted'>{hts}</span> <strong>{hevt}</strong> {extra}</li>")
    history_html = "\n".join(history_items) if history_items else "<li>No recorded history</li>"

    notes_html = ""
    if r.get("notes"):
        notes_esc = html.escape(str(r["notes"]))
        notes_html = f"""
        <div class="card">
            <div class="card-header">
                <h2>Notes</h2>
            </div>
            <div class="card-body">
                <pre class="notes-block">{notes_esc}</pre>
            </div>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Roast {rid} — Summary</title>
    <style>
        :root {{
            --bg: #0f172a;
            --card-bg: #1e293b;
            --card-border: #334155;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --accent: #38bdf8;
            --accent-hover: #7dd3fc;
            --border: #334155;
            --success: #22c55e;
            --success-bg: rgba(34, 197, 94, 0.15);
            --danger: #ef4444;
            --danger-bg: rgba(239, 68, 68, 0.15);
            --warning: #f59e0b;
            --warning-bg: rgba(245, 158, 11, 0.15);
            --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            font-size: 14px;
            line-height: 1.5;
            padding: 2rem 1.5rem;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; display: flex; flex-direction: column; gap: 1.5rem; }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            flex-wrap: wrap;
            gap: 1rem;
            padding-bottom: 1.25rem;
            border-bottom: 1px solid var(--border);
        }}
        .header-main h1 {{
            font-size: 1.75rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-family: var(--font-mono);
        }}
        .header-sub {{ color: var(--text-muted); margin-top: 0.25rem; font-size: 0.9rem; }}
        .header-actions {{ display: flex; gap: 0.5rem; flex-wrap: wrap; }}
        .btn {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.45rem 0.85rem;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 500;
            text-decoration: none;
            color: var(--text);
            background: var(--card-bg);
            border: 1px solid var(--border);
            transition: all 0.15s ease;
        }}
        .btn:hover {{
            background: #273549;
            border-color: var(--accent);
            color: var(--accent);
        }}
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 8px;
            overflow: hidden;
        }}
        .card-header {{
            padding: 0.85rem 1.25rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(255, 255, 255, 0.02);
        }}
        .card-header h2 {{ font-size: 1.05rem; font-weight: 600; }}
        .card-body {{ padding: 1.25rem; }}
        .grid-2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1rem; }}
        .meta-table {{ width: 100%; border-collapse: collapse; }}
        .meta-table td {{ padding: 0.45rem 0.6rem; vertical-align: top; }}
        .meta-label {{ color: var(--text-muted); width: 120px; font-weight: 500; }}
        .badge {{
            display: inline-block;
            padding: 0.15rem 0.55rem;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }}
        .badge-success {{ background: var(--success-bg); color: var(--success); border: 1px solid var(--success); }}
        .badge-danger {{ background: var(--danger-bg); color: var(--danger); border: 1px solid var(--danger); }}
        .badge-warning {{ background: var(--warning-bg); color: var(--warning); border: 1px solid var(--warning); }}
        .badge-host {{ background: rgba(56, 189, 248, 0.15); color: var(--accent); border: 1px solid rgba(56, 189, 248, 0.3); font-family: var(--font-mono); text-transform: none; }}
        .badge-analysis {{ background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); text-transform: none; font-family: var(--font-mono); }}
        .badge-secondary {{ background: rgba(148, 163, 184, 0.15); color: var(--text-muted); border: 1px solid rgba(148, 163, 184, 0.3); text-transform: none; }}
        code {{ font-family: var(--font-mono); font-size: 0.85em; background: rgba(0,0,0,0.3); padding: 0.15rem 0.35rem; border-radius: 4px; }}
        a.code-link {{ font-family: var(--font-mono); color: var(--accent); text-decoration: none; }}
        a.code-link:hover {{ text-decoration: underline; }}
        a.file-link {{ color: var(--accent); text-decoration: none; }}
        a.file-link:hover {{ text-decoration: underline; }}
        table.data-table {{ width: 100%; border-collapse: collapse; text-align: left; }}
        table.data-table th, table.data-table td {{ padding: 0.65rem 0.85rem; border-bottom: 1px solid var(--border); }}
        table.data-table th {{ color: var(--text-muted); font-size: 0.8rem; text-transform: uppercase; font-weight: 600; }}
        table.data-table tr:hover td {{ background: rgba(255, 255, 255, 0.02); }}
        .search-box {{
            width: 100%;
            padding: 0.55rem 0.85rem;
            background: #0b1120;
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text);
            font-size: 0.9rem;
            outline: none;
            margin-bottom: 1rem;
        }}
        .search-box:focus {{ border-color: var(--accent); }}
        .section-title {{ font-size: 0.9rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; margin: 1rem 0 0.5rem 0; letter-spacing: 0.05em; }}
        .page-list {{ display: flex; flex-direction: column; gap: 0.35rem; }}
        .page-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 0.75rem;
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid var(--border);
            border-radius: 6px;
            transition: background 0.15s ease;
        }}
        .page-item:hover {{ background: rgba(56, 189, 248, 0.06); border-color: rgba(56, 189, 248, 0.3); }}
        .page-link {{ color: var(--accent); text-decoration: none; font-weight: 500; display: flex; align-items: center; gap: 0.4rem; }}
        .page-icon {{ font-size: 0.85rem; opacity: 0.7; }}
        .page-path {{ font-family: var(--font-mono); font-size: 0.78rem; color: var(--text-muted); }}
        .empty-state {{ color: var(--text-muted); font-style: italic; padding: 0.5rem 0; font-size: 0.85rem; }}
        .muted {{ color: var(--text-muted); }}
        .notes-block {{ white-space: pre-wrap; font-family: inherit; font-size: 0.9rem; }}
        details summary {{ cursor: pointer; color: var(--text-muted); font-weight: 500; padding: 0.5rem 0; user-select: none; }}
        details summary:hover {{ color: var(--text); }}
        .history-list {{ list-style: none; padding-left: 0.5rem; margin-top: 0.5rem; display: flex; flex-direction: column; gap: 0.35rem; font-size: 0.85rem; }}
        footer {{ text-align: center; color: var(--text-muted); font-size: 0.8rem; padding: 1.5rem 0; border-top: 1px solid var(--border); }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="header-main">
                <span class="badge badge-analysis">{analysis}</span>
                <h1>{rid}</h1>
                {status_badge}
                <div class="header-sub">
                    <strong>{label}</strong> • Roasted by {owner} on {created}
                    {f" • Published {pub_ts}" if pub_ts else ""}
                </div>
            </div>
            <div class="header-actions">
                <a href="output/" class="btn">📁 Browse output/</a>
                <a href="roasts/{rid}/roast.json" class="btn">⚙️ roast.json</a>
                <a href="roasts/{rid}/config.yml" class="btn">📝 config.yml</a>
            </div>
        </header>

        <div class="grid-2">
            <div class="card">
                <div class="card-header">
                    <h2>Configuration & Provenance</h2>
                </div>
                <div class="card-body">
                    <table class="meta-table">
                        <tr>
                            <td class="meta-label">barista</td>
                            <td>{b_link}</td>
                        </tr>
                        <tr>
                            <td class="meta-label">coffea4bees</td>
                            <td>{c_link}</td>
                        </tr>
                        <tr>
                            <td class="meta-label">config</td>
                            <td><code>{cfg_src}</code> <span class="muted">(sha256 {cfg_sha})</span></td>
                        </tr>
                        <tr>
                            <td class="meta-label">hosts</td>
                            <td>{hosts_html}</td>
                        </tr>
                    </table>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <h2>Pipeline Steps</h2>
                </div>
                <div class="card-body" style="padding: 0;">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Step</th>
                                <th>Host</th>
                                <th>Snakefile</th>
                                <th>State</th>
                                <th>Log</th>
                            </tr>
                        </thead>
                        <tbody>
                            {steps_table_html}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                <h2>Interactive Results & Galleries</h2>
                <span class="muted">{total_pages} pages available</span>
            </div>
            <div class="card-body">
                <input type="text" id="pageSearch" class="search-box" placeholder="Filter pages by name, workflow, or dataset (e.g. cutflow, signal, 4b)...">

                <div class="section-title">📊 Plot Galleries</div>
                <div class="page-list" id="galleriesList">
                    {galleries_html}
                </div>

                <div class="section-title">📈 Cutflow & Closure Tables</div>
                <div class="page-list" id="cutflowsList">
                    {cutflows_html}
                </div>

                <div class="section-title">📑 Other Pages & Reports</div>
                <div class="page-list" id="othersList">
                    {others_html}
                </div>
            </div>
        </div>

        {notes_html}

        <details>
            <summary>Execution History & Timeline ({len(history)} events)</summary>
            <div class="card" style="margin-top: 0.5rem;">
                <div class="card-body">
                    <ul class="history-list">
                        {history_html}
                    </ul>
                </div>
            </div>
        </details>

        <footer>
            Roast published with barista • CERNBox Web Distribution
        </footer>
    </div>

    <script>
        const searchInput = document.getElementById('pageSearch');
        if (searchInput) {{
            searchInput.addEventListener('input', function() {{
                const q = this.value.toLowerCase().trim();
                document.querySelectorAll('.page-item').forEach(item => {{
                    const s = item.getAttribute('data-search') || '';
                    item.style.display = s.includes(q) ? '' : 'none';
                }});
            }});
        }}
    </script>
</body>
</html>
"""


def roast_category(r: dict) -> str:
    """Which production family a roast belongs to, for the catalogue's sub-pages.

    Taken from the label: everything before the run period, so nominal_run3_30x and
    nominal_run2 share a page.  `roast new --category` overrides it when the label does not
    say what the roast really is (libA_run3 is a DeClustered library build, not its own kind).
    """
    explicit = (r.get("category") or "").strip()
    if explicit:
        return explicit
    label = r.get("label") or r["id"]
    head = re.split(r"_run[23]\b", label)[0] or label
    return head.split("_")[0] or "other"          # display case kept; the file name is slugged


def _table(rs: list[dict]) -> list[str]:
    lines = ["| roast | date | owner | barista | coffea4bees | config | steps | results |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rs:
        b, c = r["barista"], r["coffea4bees"]
        url = r.get("publish", {}).get("url", "")
        steps = " ".join(s["name"] for s in r["steps"])
        res = f"[browse]({url})" if url else "\u2014"
        lines.append(
            f"| [{r['id']}]({r['id']}.md) | {r['created'][:10]} | {r.get('owner','')} "
            f"| [`{b['sha'][:7]}`]({b['web']}/-/commit/{b['sha']}) | [`{c['sha'][:7]}`]({c['web']}/-/commit/{c['sha']}) "
            f"| `{Path(r['config']['source']).name}` | {steps} | {res} |")
    return lines


BLURB = ("Production roasts of the barista workflows, one row per reproducible run. "
         "Each roast pins a barista and a coffea4bees commit plus the exact workflow config; "
         "results live in the owner's CERNBox area.")


def write_index() -> None:
    """The catalogue: a landing page per family, and one page per family.

    Everything here is derived from roasts/*/roast.json and regenerated by the docs build,
    so none of it is committed -- which is what stops two people publishing roasts from
    conflicting on the same table.
    """
    DOCS_PROD.mkdir(parents=True, exist_ok=True)
    rs = sorted(all_roasts(), key=lambda r: r["created"], reverse=True)
    families = {}
    for r in rs:
        families.setdefault(roast_category(r), []).append(r)

    lines = ["# Cupping notes", "", BLURB, "",
             "Generated from the roast manifests by `src/tools/roast.py`; do not edit by hand.", "",
             "| family | roasts | most recent |", "|---|---|---|"]
    for fam, members in sorted(families.items()):
        newest = members[0]
        lines.append(f"| [{fam}]({fam.lower()}.md) | {len(members)} | [{newest['id']}]({newest['id']}.md) "
                     f"({newest['created'][:10]}) |")
    (DOCS_PROD / "index.md").write_text("\n".join(lines) + "\n")

    for fam, members in families.items():
        page = [f"# {fam}", "", f"{len(members)} roast{'s' if len(members) != 1 else ''}. "
                f"[All families](index.md).", ""] + _table(members)
        (DOCS_PROD / f"{fam.lower()}.md").write_text("\n".join(page) + "\n")

    # a family page whose roasts have all been removed would otherwise linger
    for stale in DOCS_PROD.glob("*.md"):
        if stale.stem in HANDWRITTEN_PAGES or stale.stem in {f.lower() for f in families}:
            continue
        if not any(stale.stem == r["id"] for r in rs):
            stale.unlink()


def cmd_index(args) -> None:
    # No load_config(): the pages are a pure function of roasts/*/roast.json, and the docs
    # jobs regenerate them in a clone that has no ~/.config/roast/config.json.
    for r in all_roasts():
        write_docs(r)
    write_index()
    info(f"regenerated {DOCS_PROD}")


def cmd_ls(args) -> None:
    rs = sorted(all_roasts(), key=lambda r: r["created"], reverse=True)
    if not rs:
        print("no roasts yet")
        return
    print(f"{'id':48s} {'created':10s} {'hosts':14s} {'steps':16s} published")
    for r in rs:
        hosts = ",".join(r.get("hosts", {}))
        steps = " ".join(s["name"] + ("*" if s.get("runs") else "") for s in r["steps"])
        pub = "yes" if r.get("publish", {}).get("url") else "-"
        print(f"{r['id']:48s} {r['created'][:10]:10s} {hosts:14s} {steps:16s} {pub}")
    print("(* = submitted at least once)")


def cmd_show(args) -> None:
    print(json.dumps(load_roast(args.id), indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    # ssh refuses to multiplex into a directory that is not there, and silently falls back to a
    # fresh handshake every call -- which is the slow path this is here to avoid
    SSH_CONTROL_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    p = argparse.ArgumentParser(prog=TOOL, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="write ~/.config/roast/config.json")
    s.add_argument("--cmslpc-user", help="LPC username: sets the ssh target and the EOS archive path")
    s.add_argument("--cmslpc-node", help="pin every roast to one LPC node; by default the "
                   "cmslpc-el9 gateway assigns one per roast and the roast remembers it")
    s.add_argument("--falcon-user"); s.add_argument("--bridges2-user"); s.add_argument("--cern-user", help="CERN username: sets the CERNBox web area")
    s.add_argument("--owner", help="name shown in the cupping notes")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("new", help="create a roast manifest from the current HEADs and a config file")
    s.add_argument("--config", required=True, help="workflow --configfile YAML to capture")
    s.add_argument("--phases", help="comma list from A,B,C,D,E,F (host chosen per phase)")
    s.add_argument("--step", action="append", help="host:snakefile[:targets] for non-phase workflows (repeatable)")
    s.add_argument("--barista", help="barista sha (default: HEAD)")
    s.add_argument("--coffea4bees", help="coffea4bees sha (default: HEAD)")
    s.add_argument("--label", help="short label (default: config file stem)")
    s.add_argument("--id", help="override the generated roast id")
    s.add_argument("--notes", help="free text for the cupping notes")
    s.add_argument("--category", help="catalogue family (default: the label up to _run2/_run3)")
    s.add_argument("--bridges2", action="store_true", help="run GPU phases (C, D, BKG_C) on PSC Bridges-2 instead of Falcon")
    s.add_argument("--nautilus", action="store_true", help="run GPU phases (C, D, BKG_C) on NRP Nautilus instead of Falcon")
    s.add_argument("--analysis", choices=["HH4b", "ttHbb"], help="analysis channel for CERNBox web prod (default: auto-detected, fallback ttHbb)")
    s.set_defaults(func=cmd_new)

    s = sub.add_parser("checkout", help="create isolated checkouts at the pinned shas on each host")
    s.add_argument("id"); s.add_argument("--host")
    s.set_defaults(func=cmd_checkout)

    for name, fn, hlp in (("submit", cmd_submit, "launch one step in a tmux window on its host"),
                          ("resume", cmd_resume, "relaunch a step with --unlock and --rerun-incomplete")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("id"); s.add_argument("--step", required=True)
        s.add_argument("--cores", type=int); s.add_argument("--extra", help="extra snakemake args, quoted (e.g. --extra=\"--resources gres=mps:25\")")
        s.add_argument("--targets", help="snakemake targets (rules or files) for this run, quoted if several, e.g. "
                                         "--targets all_M1; placed before the options (a target in --extra would "
                                         "be swallowed by --config). resume reuses the last real run's targets")
        s.add_argument("-n", "--dry-run", action="store_true", help="snakemake -n: show the plan, run nothing")
        s.add_argument("-t", "--test", action="store_true", help="--config test=true: the workflow's small local test slice")
        s.set_defaults(func=fn)

    s = sub.add_parser("proxy", help="voms-proxy-init on a host (interactive), or --check its time left")
    s.add_argument("--host", default="cmslpc"); s.add_argument("--valid", default="168:00", help="hours:minutes (default 168:00)")
    s.add_argument("--check", action="store_true")
    s.set_defaults(func=cmd_proxy)

    s = sub.add_parser("log", help="show, follow or summarise a step's log (no tmux needed)")
    s.add_argument("id"); s.add_argument("--step", required=True)
    s.add_argument("-n", "--lines", type=int, default=50, help="how many lines (default 50)")
    s.add_argument("-f", "--follow", action="store_true", help="tail -F the live log")
    s.add_argument("--stats", action="store_true",
                   help="the last 'Job stats:' block — the job counts and snakemake's reasons, "
                        "which for a dry run is the only place the count appears")
    s.add_argument("--errors", action="store_true", help="only failure lines from the last run")
    s.set_defaults(func=cmd_log)

    s = sub.add_parser("attach", help="open the host's roast tmux session, on the step's window")
    s.add_argument("id", nargs="?"); s.add_argument("--step"); s.add_argument("--host")
    s.set_defaults(func=cmd_attach)

    s = sub.add_parser("status", help="per-step state on each host")
    s.add_argument("id", nargs="?")
    s.add_argument("--all", action="store_true", help="also list finished jobs that belong to no step")
    s.add_argument("-d", "--detail", action="store_true", help="per-step lines and batch jobs (the default when you name a roast)")
    s.add_argument("-s", "--summary", action="store_true", help="one line per roast, even for a single one")
    s.add_argument("--active", action="store_true", help="hide cleaned roasts (whose checkouts are gone)")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("monitor", help="serve an auto-refreshing HTML dashboard of roast progress on localhost")
    s.add_argument("ids", nargs="*", help="roasts to watch (default: every roast with a live checkout)")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--interval", type=int, default=60, help="seconds between polls of each host")
    s.add_argument("--workers", type=int, default=4, help="roasts polled in parallel")
    s.add_argument("--no-open", action="store_true", help="do not open a browser")
    s.set_defaults(func=cmd_monitor)

    s = sub.add_parser("publish", help="copy results to CERNBox and write cupping notes")
    s.add_argument("id"); s.add_argument("--host"); s.add_argument("--docs-only", action="store_true")
    s.add_argument("--jobs", type=int, default=32, help="parallel xrdcp streams (default 32)")
    s.add_argument("--with-pdfs", action="store_true", help="include heavy .pdf and .svg files (excluded by default for fast web publishing)")
    s.add_argument("--include-test", action="store_true", help="also publish output/*_test/ directories")
    s.add_argument("-n", "--dry-run", action="store_true", help="list the files that would be published, copy nothing")
    s.add_argument("--analysis", choices=["HH4b", "ttHbb"], help="override analysis destination (default: roast analysis or auto-detected)")
    s.set_defaults(func=cmd_publish)

    s = sub.add_parser("archive", help="copy heavy products (.coffea/.root/yml/json) to FNAL EOS under eos.path/<id>/")
    s.add_argument("id"); s.add_argument("--host")
    s.add_argument("--jobs", type=int, default=16); s.add_argument("--include-test", action="store_true")
    s.add_argument("-n", "--dry-run", action="store_true", help="list the files that would be archived, copy nothing")
    s.set_defaults(func=cmd_archive)

    s = sub.add_parser("pourover", help="serve a step's merged histograms locally with the plot config that step ran")
    s.add_argument("id"); s.add_argument("--step", required=True); s.add_argument("--host")
    s.add_argument("--list", action="store_true", help="show the makePlots commands this step ran, then exit")
    s.add_argument("--which", type=int, help="pick the Nth command from --list (default: the last)")
    s.add_argument("--match", help="pick the command containing this substring")
    s.add_argument("--coffea", help="override the histogram file (path relative to the checkout)")
    s.add_argument("--config", help="override the plot config (checkout-relative, or a local path)")
    s.add_argument("--port", type=int, default=5000)
    s.add_argument("--no-pull", action="store_true", help="use what is already under output/roasts/<id>/")
    s.add_argument("--extra", help="further pourOver.py args, quoted (e.g. --extra=\"--pregallery -j 8\")")
    s.set_defaults(func=cmd_pourover)

    s = sub.add_parser("pull", help="rsync a roast's output/ (or a sub-path) from its host to output/roasts/<id>/ here")
    s.add_argument("id"); s.add_argument("--host")
    s.add_argument("--path", default="output", help="path inside the checkout (default: output)")
    s.add_argument("--only", action="append", help="only files matching this glob, e.g. --only '*.coffea' (repeatable)")
    s.add_argument("--exclude", action="append", help="extra rsync exclude glob (repeatable)")
    s.add_argument("--include-test", action="store_true", help="also pull *_test directories")
    s.add_argument("-n", "--dry-run", action="store_true")
    s.set_defaults(func=cmd_pull)

    s = sub.add_parser("clean", help="remove the host checkouts of a finished, archived and published roast (dry run unless --yes)")
    s.add_argument("id", nargs="?")
    s.add_argument("--all", action="store_true", help="every roast that qualifies")
    s.add_argument("--yes", action="store_true", help="actually delete the checkouts")
    s.set_defaults(func=cmd_clean)

    s = sub.add_parser("rm", help="delete a roast everywhere (dry run unless --yes): local, host checkouts, EOS archive, CERNBox")
    s.add_argument("id"); s.add_argument("--yes", action="store_true")
    s.add_argument("--keep-hosts", action="store_true"); s.add_argument("--keep-eos", action="store_true"); s.add_argument("--keep-cernbox", action="store_true")
    s.set_defaults(func=cmd_rm)

    s = sub.add_parser("index", help="regenerate docs/prod from all manifests")
    s.set_defaults(func=cmd_index)

    s = sub.add_parser("ls", help="list roasts")
    s.set_defaults(func=cmd_ls)

    s = sub.add_parser("show", help="print a manifest")
    s.add_argument("id")
    s.set_defaults(func=cmd_show)

    s = sub.add_parser("commit", help="stage and commit roasts/<id>/ (and optionally push)")
    s.add_argument("id")
    s.add_argument("-m", "--message", help="commit message (default: generated from roast status)")
    s.add_argument("--push", action="store_true", help="push the commit to the current branch on the remote")
    s.add_argument("--remote", default="origin", help="git remote for --push (default: origin)")
    s.add_argument("-n", "--dry-run", action="store_true", help="show what would be committed without committing")
    s.set_defaults(func=cmd_commit)

    args = p.parse_args(argv)
    try:
        args.func(args)
    except subprocess.CalledProcessError as e:
        die(f"command failed ({e.returncode}): {' '.join(map(str, e.cmd))}\n{(e.stderr or '').strip()}")
    except KeyboardInterrupt:
        die("interrupted", 130)


if __name__ == "__main__":
    main()
