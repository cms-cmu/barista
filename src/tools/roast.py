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
import datetime as dt
import getpass
import hashlib
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
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=25", "-o", "LogLevel=ERROR"]

# Phase -> (host, snakefile).  Mirrors coffea4bees/workflows/README.md.
PHASES = {
    "A": ("cmslpc", "coffea4bees/workflows/Snakefile_PhaseA.smk"),
    "B": ("cmslpc", "coffea4bees/workflows/Snakefile_PhaseB.smk"),
    "C": ("falcon", "coffea4bees/workflows/Snakefile_PhaseC.smk"),
    "C4": ("cmslpc", "coffea4bees/workflows/Snakefile_PhaseC_4_FvT_closure.smk"),   # FvT closure: processor + plots + cutflow with the new FvT
    "D": ("falcon", "coffea4bees/workflows/Snakefile_PhaseD.smk"),
    "E": ("cmslpc", "coffea4bees/workflows/Snakefile_PhaseE.smk"),
    "F": ("cmslpc", "coffea4bees/workflows/Snakefile_PhaseF.smk"),
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
            "ssh": "<user>@cmslpc307.fnal.gov",
            "prod_root": "~/nobackup/HH4b/prod",
            "reference": "~/nobackup/HH4b/Run3/barista",
            "cores": 8,
            "host_file": "~/.cmslpc-claude-host",
        },
        "falcon": {
            "ssh": "<user>@falcon.phys.cmu.edu",
            "prod_root": "~/work/prod",
            "reference": "~/work/barista",
            "cores": 4,
        },
    },
    "cernbox": {
        "eos_path": "/eos/user/<x>/<user>/www/HH4b/prod",
        "url": "https://<user>.web.cern.ch/<user>/HH4b/prod",
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
    return subprocess.run(cmd, cwd=cwd, check=check, text=True, input=input,
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
    try:
        return cfg["hosts"][host]
    except KeyError:
        die(f"host '{host}' not in {CONFIG_PATH}")


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


def ssh_run(target: str, script: str, check=True, capture=True) -> subprocess.CompletedProcess:
    """Run a bash script on a remote host via stdin (no quoting games)."""
    return sh(["ssh", *SSH_OPTS, target, "bash -s"], check=check, capture=capture, input=script)


def scp_to(target: str, srcs: list[Path], dst: str) -> None:
    # -p: keep mtimes. The captured config.yml is an input of the workflows' create_*_config rules;
    # re-shipping it with a fresh mtime on every submit/resume made snakemake regenerate those
    # configs and rerun every processor job downstream of them.
    sh(["scp", "-q", "-p", *SSH_OPTS, *map(str, srcs), f"{target}:{dst}"])


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
        cfg["hosts"]["cmslpc"]["ssh"] = f"{args.cmslpc_user}@{args.cmslpc_node}.fnal.gov"
        cfg["eos"]["path"] = f"/store/user/{args.cmslpc_user}/HH4b_prod"
    if args.falcon_user:
        cfg["hosts"]["falcon"]["ssh"] = f"{args.falcon_user}@falcon.phys.cmu.edu"
    if args.cern_user:
        u = args.cern_user
        cfg["cernbox"]["eos_path"] = f"/eos/user/{u[0]}/{u}/www/HH4b/prod"
        cfg["cernbox"]["url"] = f"https://{u}.web.cern.ch/{u}/HH4b/prod"
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
    rid = f"{label}_{dt.datetime.now():%Y%m%d}_{barista_sha[:7]}-{c4b_sha[:7]}"
    if args.id:
        rid = args.id
    if roast_dir(rid).exists():
        die(f"roast {rid} already exists")

    d = roast_dir(rid)
    d.mkdir(parents=True)
    shutil.copy2(config_src, d / "config.yml")   # verbatim; {roast_id} is resolved at run time via --config roast_id

    r = {
        "id": rid,
        "label": label,
        "created": now(),
        "owner": cfg.get("owner", getpass.getuser()),
        "barista": {"sha": barista_sha, "origin": barista_origin, "web": gitlab_web(barista_origin)},
        "coffea4bees": {"sha": c4b_sha, "origin": c4b_origin, "web": gitlab_web(c4b_origin)},
        "config": {"source": str(config_src), "captured": "config.yml", "sha256": sha256_file(d / "config.yml")},
        "steps": steps,
        "hosts": {},
        "publish": {},
        "history": [],
        "notes": args.notes or "",
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
        cd "$CK"
        git remote set-url origin {shlex.quote(r['barista']['origin'])}
        if [ ! -d coffea4bees/.git ]; then
            git clone -q --no-checkout {rq(ref + '/coffea4bees')} coffea4bees
        fi
        (cd coffea4bees && git remote set-url origin {shlex.quote(r['coffea4bees']['origin'])})
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


def cmd_checkout(args) -> None:
    cfg = load_config()
    r = load_roast(args.id)
    hosts = [args.host] if args.host else sorted({s["host"] for s in r["steps"]})
    for host in hosts:
        hc = host_cfg(cfg, host)
        target = resolve_ssh(hc)
        ckpt = checkout_path(cfg, r, host)
        info(f"[{host}] preparing {ckpt}")
        res = ssh_run(target, _checkout_script(hc, r, ckpt))
        print(res.stdout.strip())
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
        r["hosts"][host] = {"checkout": ckpt, "ssh": target, "checked_out": now()}  # ssh = node used at checkout (record only)
        log_event(r, "checkout", host=host)
        save_roast(r)
    info(f"next: {TOOL} submit {r['id']} --step {r['steps'][0]['name']}")


def _run_script(cfg: dict, r: dict, step: dict, ckpt: str, cores: int, extra: str, resume: bool) -> str:
    """The bash script that runs one step inside its tmux window."""
    name = step["name"]
    smk = step["snakefile"]
    configfile = f"roasts/{r['id']}/config.yml"
    # -p/--printshellcmds: every rule's resolved shell command lands in logs/<step>.log, so a job can be re-run by hand
    # --config roast_id: resolves {roast_id} placeholders in the config (helpers/common.smk), e.g. run-scoped EOS paths
    # --jobs as well as --cores: hosts whose run_container injects a remote-executor snakemake profile
    # (falcon: software/snakemake/profiles/falcon, executor slurm) refuse to run without --jobs N.
    base = f"./run_container snakemake -s {shlex.quote(smk)} --configfile {configfile} --cores {cores} --jobs {cores} --printshellcmds --config roast_id={r['id']}"
    if step.get("targets"):
        base += f" {step['targets']}"
    if step.get("extra"):
        base += f" {step['extra']}"
    if extra:
        base += f" {extra}"
    unlock = f"{base} --unlock || true\n" if resume else ""
    if resume:
        base += " --rerun-incomplete"
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
        # user's standard proxy (voms-proxy-init writes /tmp/x509up_u<uid>) when the checkout has none.
        mkdir -p proxy
        if [ ! -s proxy/x509_proxy ] || [ "${{X509_USER_PROXY:-/tmp/x509up_u$(id -u)}}" -nt proxy/x509_proxy ]; then
            cp -f "${{X509_USER_PROXY:-/tmp/x509up_u$(id -u)}}" proxy/x509_proxy 2>/dev/null && echo "=== proxy copied from ${{X509_USER_PROXY:-/tmp/x509up_u$(id -u)}} ===" | tee -a "$LOG"
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


def _submit(args, resume: bool) -> None:
    cfg = load_config()
    r = load_roast(args.id)
    step = find_step(r, args.step)
    host = step["host"]
    hc = host_cfg(cfg, host)
    if host not in r["hosts"]:
        die(f"roast not checked out on {host}; run `{TOOL} checkout {r['id']} --host {host}`")
    ckpt = r["hosts"][host]["checkout"]
    target = resolve_ssh(hc)
    cores = args.cores or hc.get("cores", 4)
    extra = args.extra
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
    window = f"{r['label'][:16]}_{r['created'][:10].replace('-', '')}_{step['name']}"
    driver_pat = f"snakemake.*roasts/{r['id']}/config.yml"
    # Guard FIRST, before touching any file on the host: a live snakemake driver for this roast means
    # "running" (refuse); a window without a driver is a finished or derailed step and is closed.
    guard = textwrap.dedent(f"""\
        tmux has-session -t {TMUX_SESSION} 2>/dev/null || tmux new-session -d -s {TMUX_SESSION} -n hub
        if pgrep -u "$USER" -f {shlex.quote(driver_pat)} >/dev/null 2>&1; then
            echo "a snakemake driver for this roast is still running on $(hostname); refusing to double-submit" >&2
            exit 3
        fi
        if tmux list-windows -t {TMUX_SESSION} -F '#W' | grep -qx {shlex.quote(window)}; then
            tmux kill-window -t {TMUX_SESSION}:{shlex.quote(window)}
            if [ -f {rq(ckpt)}/logs/{step['name']}.exit ]; then
                echo "closed finished window {window} (exit=$(cat {rq(ckpt)}/logs/{step['name']}.exit))"
            else
                echo "closed stale window {window} (no driver process, no exit file)"
            fi
        fi
    """)
    res = ssh_run(target, guard, check=False)
    if res.returncode != 0:
        die(res.stderr.strip() or res.stdout.strip())
    if res.stdout.strip():
        print(res.stdout.strip())
    script = _run_script(cfg, r, step, ckpt, cores, args.extra, resume)
    local = roast_dir(r["id"]) / f"run_{step['name']}.sh"
    local.write_text(script)
    # Re-ship the whole roast dir: the captured config.yml may have been edited since checkout.
    scp_to(target, sorted(p for p in roast_dir(r["id"]).iterdir() if p.is_file()), f"{ckpt}/roasts/{r['id']}/")
    remote = textwrap.dedent(f"""\
        set -e
        SCRIPT={rq(f"{ckpt}/roasts/{r['id']}/run_{step['name']}.sh")}
        chmod +x "$SCRIPT"
        tmux new-window -d -t {TMUX_SESSION} -n {shlex.quote(window)} "bash $SCRIPT"
        echo "launched tmux {TMUX_SESSION}:{window}"
    """)
    res = ssh_run(target, remote, check=False)
    if res.returncode != 0:
        die(res.stderr.strip() or res.stdout.strip())
    print(res.stdout.strip())
    cfg_sha = sha256_file(roast_dir(r["id"]) / "config.yml")
    if cfg_sha != r["config"]["sha256"]:
        info("captured config.yml changed since `new`; recording the new sha256")
        r["config"]["sha256"] = cfg_sha
        log_event(r, "config-edited", sha256=cfg_sha[:12])
    step.setdefault("runs", []).append({"ts": now(), "cores": cores, "extra": args.extra or "", "user_extra": user_extra,
                                        "dry_run": bool(args.dry_run), "test": bool(args.test), "resume": resume,
                                        "window": window, "ssh": target, "config_sha256": cfg_sha[:12]})
    log_event(r, "resume" if resume else "submit", step=step["name"], host=host)
    save_roast(r)
    info(f"attach: ssh -t {target} tmux attach -t {TMUX_SESSION}    |    {TOOL} status {r['id']}")


# Lines worth seeing when scanning a step's log for what went wrong. `Traceback` is deliberately
# absent: dask tears its client down noisily at the end of every successful job, so matching it
# buries the real failures under a dozen benign stacks per wave of jobs.
_LOG_ERROR_RE = (r"Error in rule|WorkflowError|LockException|[A-Za-z]*Error:|"
                 r"Missing(Output|Input)Exception|"
                 r"Exiting because a job execution failed|Removing output files of failed job|"
                 r"JOB EXECUTION FAILED|exited with non-zero|unbound variable|"
                 r"Killed|Out of memory|Segmentation fault|=== roast .* exit ")


def cmd_log(args) -> None:
    """Show, follow or summarise a step's log without taking over the terminal.

    `attach` needs the tmux window to still exist and replaces this process; `status` shows only
    the last line. This is the in-between: read the log of a finished or running step, over ssh.
    """
    cfg = load_config()
    r = load_roast(args.id)
    step = find_step(r, args.step)
    target = resolve_ssh(host_cfg(cfg, step["host"]))
    log = f"{checkout_path(cfg, r, step['host'])}/logs/{step['name']}.log"

    # Everything after the last start marker, i.e. the most recent run of this step rather than
    # the whole history of resubmissions appended to the same file.
    since_start = f"tac {rq(log)} | sed '/=== roast .* start /q' | tac"

    if args.follow:
        info(f"following {step['host']}:{log}  (ctrl-c to stop)")
        cmd = f"tail -n {int(args.lines)} -F {rq(log)}"
        if args.errors:
            cmd += f" | grep -E --line-buffered {shlex.quote(_LOG_ERROR_RE)}"
        os.execvp("ssh", ["ssh", "-t", *SSH_OPTS, target, cmd])

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
    window = None
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
    target = resolve_ssh(host_cfg(cfg, host))
    tmux = f"tmux attach -t {TMUX_SESSION}"
    if window:
        tmux += f" \\; select-window -t {shlex.quote(window)}"
    info(f"ssh -t {target} {tmux}")
    os.execvp("ssh", ["ssh", "-t", *SSH_OPTS, target, tmux])


def cmd_proxy(args) -> None:
    """Create (interactively) or check the grid proxy on a host.  Writes the standard /tmp/x509up_u<uid>,
    which every roast launcher copies into its checkout's proxy/x509_proxy."""
    cfg = load_config()
    target = resolve_ssh(host_cfg(cfg, args.host))
    if args.check:
        res = ssh_run(target, "voms-proxy-info --timeleft 2>&1 | head -1", check=False)
        left = (res.stdout or res.stderr).strip()
        print(f"{args.host}: {int(left)//3600} h left" if left.isdigit() else f"{args.host}: {left or 'no proxy'}")
        return
    cmd = f"voms-proxy-init -rfc -voms cms --valid {shlex.quote(args.valid)}"
    info(f"ssh -t {target} {cmd}")
    os.execvp("ssh", ["ssh", "-t", *SSH_OPTS, target, cmd])


def cmd_pull(args) -> None:
    """rsync a roast's outputs from a host into <barista>/output/roasts/<id>/<path> on this machine."""
    cfg = load_config()
    r = load_roast(args.id)
    hosts = [args.host] if args.host else list(r["hosts"])
    for host in hosts:
        target = resolve_ssh(host_cfg(cfg, host))
        ckpt = r["hosts"][host]["checkout"]
        rel = args.path.strip("/")
        src = f"{target}:{ckpt}/{rel}/"
        dst = ROOT / "output" / "roasts" / r["id"] / rel
        dst.mkdir(parents=True, exist_ok=True)
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
        cmd += ["-e", "ssh " + " ".join(SSH_OPTS), src, str(dst) + "/"]
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


def _eos_rm_tree_script(eos_url: str, path: str) -> str:
    """xrdfs has no recursive delete: remove files in parallel, then directories deepest-first."""
    return textwrap.dedent(f"""\
        set -u
        EOS={eos_url}; P={shlex.quote(path)}
        xrdfs $EOS stat "$P" >/dev/null 2>&1 || {{ echo "  (not present) $EOS/$P"; exit 0; }}
        xrdfs $EOS ls -l -R "$P" 2>/dev/null | awk '$1 !~ /^d/ {{ print $NF }}' | tr '\\n' '\\0' | xargs -0 -r -P 16 -n 1 xrdfs $EOS rm >/dev/null 2>&1
        xrdfs $EOS ls -l -R "$P" 2>/dev/null | awk '$1 ~ /^d/ {{ print $NF }}' | awk '{{ print length($0), $0 }}' | sort -rn | cut -d" " -f2- | while read -r d; do xrdfs $EOS rmdir "$d" >/dev/null 2>&1; done
        xrdfs $EOS rmdir "$P" >/dev/null 2>&1
        xrdfs $EOS stat "$P" >/dev/null 2>&1 && echo "  WARNING: $EOS/$P still exists" || echo "  removed $EOS/$P"
    """)


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
    plan.append(f"cernbox root://eosuser.cern.ch/{cfg['cernbox']['eos_path'].rstrip('/')}/{rid}")
    print(f"roast rm {rid} would remove:")
    for line in plan:
        print("  " + line)
    if not args.yes:
        info("dry run. Re-run with --yes to delete. Use --keep-eos / --keep-cernbox / --keep-hosts to spare a location.")
        return
    # refuse while a step is running
    for host, h in r.get("hosts", {}).items():
        target = resolve_ssh(host_cfg(cfg, host))
        chk = ssh_run(target, f"tmux list-windows -t {TMUX_SESSION} -F '#W' 2>/dev/null | grep -c {shlex.quote(r['label'][:16] + '_')} || true", check=False)
        if chk.stdout.strip() not in ("", "0"):
            die(f"a tmux window for this roast is still open on {host}; finish or kill it first (`{TOOL} attach {rid}`)")
    ok = True
    # remote locations first, while we still have the manifest
    lpc = None
    for host, h in r.get("hosts", {}).items():
        target = resolve_ssh(host_cfg(cfg, host))
        if host == "cmslpc":
            lpc = target
        if not args.keep_hosts:
            top = h["checkout"].rsplit("/", 1)[0]
            res = ssh_run(target, f"rm -rf {rq(top)} && echo '  removed {top}'", check=False)
            print((res.stdout + res.stderr).strip()); ok &= res.returncode == 0
    lpc = lpc or resolve_ssh(host_cfg(cfg, "cmslpc"))
    if not args.keep_eos and eos.get("path") and "<" not in eos["path"]:
        res = ssh_run(lpc, _eos_rm_tree_script(eos.get("url", "root://cmseos.fnal.gov"), f"{eos['path'].rstrip('/')}/{rid}"), check=False)
        print((res.stdout + res.stderr).strip()); ok &= res.returncode == 0
    if not args.keep_cernbox:
        res = ssh_run(lpc, _eos_rm_tree_script("root://eosuser.cern.ch", f"{cfg['cernbox']['eos_path'].rstrip('/')}/{rid}"), check=False)
        print((res.stdout + res.stderr).strip()); ok &= res.returncode == 0
    if not ok:
        die("some remote deletions failed; local manifest kept so you can retry")
    shutil.rmtree(roast_dir(rid), ignore_errors=True)
    (DOCS_PROD / f"{rid}.md").unlink(missing_ok=True)
    write_index(cfg)
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
           "-e", "ssh " + " ".join(SSH_OPTS), f"{target}:{ckpt}/", str(dest_root) + "/"]
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
    target = resolve_ssh(host_cfg(cfg, host))
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
    step_names = " ".join(s["name"] for s in steps)
    rid = shlex.quote(r["id"])
    # Batch-system view, filtered to this roast: both schedulers record the submitting
    # directory per job (condor Iwd, slurm WorkDir), and that is the roast checkout.
    if host == "cmslpc":
        batch = "\n".join([
            f'''echo "TOTALS|$(condor_q -totals 2>/dev/null | grep -m1 "Total for query" || true)"''',
            f'''condor_q -af:t JobBatchName JobStatus Iwd 2>/dev/null | awk -F\'\\t\' -v rid={rid} \'$3 ~ rid {{ n[$1 "\\t" $2]++ }} END {{ for (k in n) {{ split(k, a, "\\t"); print "CONDOR|" a[1] "|" a[2] "|" n[k] }} }}\' || true''',
        ])
    elif host == "falcon":
        batch = "\n".join([
            # live jobs, each followed by the last line of its own slurm log (training progress, loss, ...)
            f'''squeue -u "$USER" -h -o \'%i|%k|%T|%M|%l|%R|%C|%m|%b|%Z\' 2>/dev/null | awk -F\'|\' -v rid={rid} \'$10 ~ rid {{ print $1 "|" $2 "|" $3 "|" $4 "|" $5 "|" $6 "|" $7 "|" $8 "|" $9 }}\' | while IFS="|" read -r jid rest; do hit=$(grep -H -m1 "SLURM jobid $jid " logs/*.log 2>/dev/null | head -1); step=${{hit%%:*}}; step=${{step##*/}}; step=${{step%%.log}}; [ -n "$hit" ] || step=""; log=$(ls -t slurm_logs/*/$jid.log slurm_logs/$jid.log 2>/dev/null | head -1); rule=$(echo "$rest" | cut -d"|" -f1); [ -n "$rule" ] || rule=$(echo "$hit" | grep -oE "slurm_logs/[^/]+/" | head -1 | cut -d/ -f2); echo "SLURM|$jid|$step|$rule|$(echo "$rest" | cut -d"|" -f2-)|$(tail -n 1 "$log" 2>/dev/null | tr -d "\\r" | tr "|" " " | cut -c1-100)"; done''',
            # recently finished jobs of this roast (failures are what you want to see)
            f'''sacct -u "$USER" -S now-2days -P -n -o JobID,JobName,State,Elapsed,MaxRSS,WorkDir 2>/dev/null | awk -F\'|\' -v rid={rid} \'$1 !~ /\\./ && $6 ~ rid && $3 !~ /RUNNING|PENDING/ {{ print $1 "|" $2 "|" $3 "|" $4 "|" $5 }}\' | tail -40 | while IFS="|" read -r jid rest; do hit=$(grep -H -m1 "SLURM jobid $jid " logs/*.log 2>/dev/null | head -1); step=${{hit%%:*}}; step=${{step##*/}}; step=${{step%%.log}}; [ -n "$hit" ] || step=""; rule=$(echo "$hit" | grep -oE "slurm_logs/[^/]+/" | head -1 | cut -d/ -f2); if [ -z "$rule" ]; then rule=$(echo "$rest" | cut -d"|" -f1); case "$rule" in ????????-????-????-????-????????????) rule=${{rule%%-*}};; esac; fi; echo "SLURMDONE|$jid|$step|$rule|$(echo "$rest" | cut -d"|" -f2-)"; done''',
            f'''sinfo -h -o "SINFO|%P|%D|%T|%G" 2>/dev/null | head -3''',
        ])
    else:
        batch = "true"
    return textwrap.dedent(f"""\
        cd {rq(ckpt)} 2>/dev/null || {{ echo "NOCHECKOUT"; exit 0; }}
        if pgrep -u "$USER" -f {shlex.quote(f"snakemake.*roasts/{r['id']}/config.yml")} >/dev/null 2>&1; then live=1; else live=0; fi
        for S in {step_names}; do
            if [ -f logs/$S.exit ]; then st="exit=$(cat logs/$S.exit)";
            elif [ -f logs/$S.log ]; then
                if [ "$live" != "0" ]; then st="running";
                elif tac logs/$S.log | sed '/=== roast .* start /q' | grep -Eq "WorkflowError|Exiting because a job execution failed|Error in rule"; then st="error";
                else st="stalled"; fi
            else st="not started"; fi
            win=$(tmux list-windows -t {TMUX_SESSION} -F '#W' 2>/dev/null | grep -c "_$S\\$" || true)
            prog=$(tac logs/$S.log 2>/dev/null | sed '/=== roast .* start /q' | grep -m1 -oE '[0-9]+ of [0-9]+ steps \\([0-9]+%\\) done')
            last=$(tail -n 1 logs/$S.log 2>/dev/null | cut -c1-90)
            echo "STEP|$S|$st|tmux=$win|$prog|$last"
        done
        {batch}
    """)


def _jobnum(j: dict) -> int:
    head = j["jid"].split("_")[0].split(".")[0]
    return int(head) if head.isdigit() else 0


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
    out = {"steps": [], "live": {}, "done": {}, "extra": [], "nocheckout": False}
    for line in stdout.splitlines():
        if line.startswith("STEP|"):
            _, name, st, win, prog, last = (line.split("|", 5) + [""] * 6)[:6]
            colour = ("\033[32m" if st == "exit=0" else "\033[31m" if st.startswith("exit=") or st in ("error", "stalled")
                      else "\033[33m" if st == "running" else "")
            out["steps"].append((name, f"{colour}{st:12s}\033[0m {win:7s} {prog:28s} {last}".rstrip()))
        elif line.startswith("TOTALS|") and line[7:].strip():
            out["extra"].append(f"condor  {line[7:].strip()}")
        elif line.startswith("CONDOR|"):
            _, name, code, n = (line.split("|", 3) + [""] * 4)[:4]
            states = {"1": "idle", "2": "running", "3": "removing", "4": "done", "5": "HELD", "6": "ERROR", "7": "suspended"}
            out["extra"].append(f"condor  {n:>4s} x {states.get(code, code):9s} {name}")
        elif line.startswith("SLURM|"):
            f = (line.split("|") + [""] * 12)[:12]
            out["live"].setdefault(f[2], []).append(
                dict(zip(("jid", "rule", "state", "el", "lim", "node", "cpus", "mem", "tres", "tail"), [f[1]] + f[3:])))
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


def cmd_status(args) -> None:
    cfg = load_config()
    roasts = [load_roast(args.id)] if args.id else [r for r in all_roasts() if r.get("hosts")]
    for r in roasts:
        print(f"\n\033[1m{r['id']}\033[0m  barista {r['barista']['sha'][:7]}  coffea4bees {r['coffea4bees']['sha'][:7]}  {r['config']['source']}")
        # One ssh per host, but the report is ordered by step: a roast is a pipeline, and
        # which machine a phase happens to run on matters less than where the pipeline is.
        parsed = {}
        for host, hinfo in r["hosts"].items():
            target = resolve_ssh(host_cfg(cfg, host))
            res = ssh_run(target, _status_script(r, hinfo["checkout"], [s for s in r["steps"] if s["host"] == host], host), check=False)
            if res.returncode != 0:
                parsed[host] = {"error": f"ssh failed ({target}): {res.stderr.strip()[:80]}"}
            else:
                st = _parse_status(res.stdout)
                st["error"] = f"checkout missing at {hinfo['checkout']}" if st["nocheckout"] else None
                parsed[host] = st
        w = max(4, min(18, max((len(s["name"]) for s in r["steps"]), default=4)))
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
            for row in _slurm_rows(st["live"].get(name, []), st["done"].get(name, [])):
                print(f"{pad}{row}")
        for host, st in parsed.items():
            if st.get("error"):
                continue
            loose = _slurm_rows(st["live"].get("", []), st["done"].get("", []))
            if loose:
                print(f"  {'':<{w}s} {host:7s} (jobs not traced to a step)")
                for row in loose:
                    print(f"{pad}{row}")
            for text in st["extra"]:
                print(f"  {'':<{w}s} {host:7s} {text}")
        if r.get("publish", {}).get("url"):
            print(f"  published: {r['publish']['url']}")


def copy_settings(kind: str, cfg: dict, r: dict, include_test: bool = False) -> dict:
    """kind = 'publish' | 'archive'.  Defaults <- user config[kind] <- roast.json[kind + '_rules']."""
    base = PUBLISH_DEFAULTS if kind == "publish" else ARCHIVE_DEFAULTS
    ps = dict(base)
    for src in (cfg.get(kind) or {}, r.get(kind + "_rules") or {}):
        ps.update({k: v for k, v in src.items() if k in base})
    if include_test:
        ps["exclude"] = [e for e in ps["exclude"] if "_test" not in e]
    return ps


def _copy_script(r: dict, ckpt: str, eos_url: str, dst_dir: str, ps: dict, jobs: int = 16,
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
        cd {rq(ckpt)}
        command -v xrdcp >/dev/null || {{ echo "xrdcp not found on $(hostname)" >&2; exit 2; }}
        # Auth: a kerberos ticket if present, else the grid proxy run_container keeps in ./proxy
        if ! klist -s 2>/dev/null && [ -s proxy/x509_proxy ]; then export X509_USER_PROXY="$PWD/proxy/x509_proxy"; fi
        EOS={eos_url}
        DST_BASE=$EOS/{dst_dir}
        LIST=$(mktemp); HAVE=$(mktemp); TODO=$(mktemp); DIRS=$(mktemp)
        trap 'rm -f "$LIST" "$HAVE" "$TODO" "$DIRS"' EXIT
        {{ [ -d output ] && find output {prune} -type f \\( {exts} \\) {size} -print; \\
           {extra_dirs} }} | sort -u > "$LIST"
        N=$(wc -l < "$LIST")
        {dry}
        # What is already there (path<TAB>size), so reruns only copy new or changed files.
        xrdfs $EOS ls -l -R {dst_dir} 2>/dev/null | awk -v b="{dst_dir}/" '$1 !~ /^d/ {{ p=$NF; sub(b,"",p); print p "\\t" $4 }}' | sort > "$HAVE" || true
        while IFS= read -r f; do
            sz=$(stat -c %s "$f")
            if ! grep -qxF "$f	$sz" "$HAVE"; then echo "$f"; fi
        done < "$LIST" > "$TODO"
        T=$(wc -l < "$TODO")
        {ht}
        echo "copying from $(hostname):$PWD -> $DST_BASE"
        echo "$N files selected, $((N-T)) already up to date, $T to copy ({jobs} in parallel)"
        [ "$T" = 0 ] && exit 0
        sed 's#/[^/]*$##' "$TODO" | sort -u > "$DIRS"
        while IFS= read -r d; do xrdfs $EOS mkdir -p "{dst_dir}/$d" 2>/dev/null || true; done < "$DIRS"
        export DST_BASE
        tr '\\n' '\\0' < "$TODO" | xargs -0 -P {jobs} -I{{}} bash -c 'xrdcp -f -s "$1" "$DST_BASE/$1" && echo "ok $1" || echo "FAILED $1"' _ {{}} \\
            | awk '/^FAILED/ {{ print; fail++ }} /^ok/ {{ ok++; if (ok % 100 == 0) print "  ... " ok " copied" }} END {{ printf "done: %d copied, %d failed\\n", ok, fail; exit fail>0 }}'
    """)


def cmd_publish(args) -> None:
    cfg = load_config()
    r = load_roast(args.id)
    eos_dir = f"{cfg['cernbox']['eos_path'].rstrip('/')}/{r['id']}"
    url = f"{cfg['cernbox']['url'].rstrip('/')}/{r['id']}/"
    hosts = [args.host] if args.host else list(r["hosts"])
    ok = True
    ps = copy_settings("publish", cfg, r, args.include_test)
    if args.dry_run:
        info("publish rules: include=" + " ".join(ps["include"]) + "  exclude=" + " ".join(ps["exclude"]) + f"  max_mb={ps['max_mb']}")
    if not args.docs_only:
        for host in hosts:
            hinfo = r["hosts"][host]
            info(f"[{host}] publishing to {eos_dir}")
            res = ssh_run(resolve_ssh(host_cfg(cfg, host)),
                          _copy_script(r, hinfo["checkout"], "root://eosuser.cern.ch", eos_dir, ps, jobs=args.jobs,
                                       dry_run=args.dry_run, with_logs=True, htaccess=True), check=False)
            print((res.stdout + res.stderr).strip())
            if args.dry_run:
                continue
            if res.returncode != 0:
                ok = False
                info(f"[{host}] publish FAILED (CERN auth on that host? try `kinit <user>@CERN.CH` or a voms proxy there)")
            log_event(r, "publish", host=host, ok=res.returncode == 0)
    if args.dry_run:
        return
    # Remember the HTML pages that were shipped (plot galleries, cutflow closure tables, ...) so the
    # docs page can link them; they are otherwise only reachable by knowing the output path.
    pages = set(r.get("publish", {}).get("pages", []))
    if not args.docs_only:
        if not args.host:
            pages = set()   # full publish: rebuild the list from what the hosts have now
        for host in hosts:
            ck = r["hosts"][host]["checkout"]
            res = ssh_run(resolve_ssh(host_cfg(cfg, host)),
                          f"cd {rq(ck)} && find output -name '*.html' -not -name '*dask-report*' -not -path '*_test*' -not -path '*/logs/*' -not -path '*/singlefiles/*' | sort",
                          check=False)
            pages.update(p.strip() for p in res.stdout.splitlines() if p.strip())
    r["publish"] = {"eos": eos_dir, "url": url, "ts": now(), "ok": None if args.docs_only else ok, "pages": sorted(pages)}
    save_roast(r)
    write_docs(cfg, r)
    write_index(cfg)
    info(f"results: {url}")
    info(f"docs: {DOCS_PROD / (r['id'] + '.md')} and index.md regenerated; commit roasts/ and docs/prod/ to update the Pages site")


def cmd_archive(args) -> None:
    """Copy heavy products (.coffea/.root/...) to FNAL EOS under <eos.path>/<id>/, mirroring the checkout layout."""
    cfg = load_config()
    r = load_roast(args.id)
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
    for host in hosts:
        hinfo = r["hosts"][host]
        info(f"[{host}] archiving to {eos_url}/{dst_dir}")
        res = ssh_run(resolve_ssh(host_cfg(cfg, host)),
                      _copy_script(r, hinfo["checkout"], eos_url, dst_dir, ps, jobs=args.jobs,
                                   dry_run=args.dry_run, with_logs=True, htaccess=False), check=False)
        print((res.stdout + res.stderr).strip())
        if args.dry_run:
            continue
        if res.returncode != 0:
            ok = False
            info(f"[{host}] archive FAILED (grid proxy on that host? `{TOOL} proxy --check`)")
        log_event(r, "archive", host=host, ok=res.returncode == 0)
    if args.dry_run:
        return
    r["archive"] = {"eos": f"{eos_url}/{dst_dir}", "ts": now(), "ok": ok}
    save_roast(r)
    write_docs(cfg, r)
    write_index(cfg)
    info(f"archive: {eos_url}/{dst_dir}   (list: xrdfs {eos_url} ls -R {dst_dir})")


# ---------------------------------------------------------------------------
# Cupping notes (docs/prod)
# ---------------------------------------------------------------------------

def _step_state(r: dict, step: dict) -> str:
    runs = step.get("runs", [])
    if not runs:
        return "not submitted"
    return f"submitted {runs[-1]['ts'][:10]}" + (" (resumed)" if runs[-1].get("resume") else "")


def write_docs(cfg: dict, r: dict) -> None:
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
        (f"| archive (EOS) | `{r['archive']['eos']}/` (list: `xrdfs root://{r['archive']['eos'].split('//')[1]} ls -R /{r['archive']['eos'].split('//')[2]}`) |"
         if r.get("archive", {}).get("eos") else ""),
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


def write_index(cfg: dict) -> None:
    DOCS_PROD.mkdir(parents=True, exist_ok=True)
    rs = sorted(all_roasts(), key=lambda r: r["created"], reverse=True)
    lines = [
        "# Cupping notes",
        "",
        "Production roasts of the barista workflows, one row per reproducible run. "
        "Each roast pins a barista and a coffea4bees commit plus the exact workflow config; "
        "results live in the owner's CERNBox area. Generated by `src/tools/roast.py`.",
        "",
        "| roast | date | owner | barista | coffea4bees | config | steps | results |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rs:
        b, c = r["barista"], r["coffea4bees"]
        url = r.get("publish", {}).get("url", "")
        steps = " ".join(s["name"] for s in r["steps"])
        res = f"[browse]({url})" if url else "—"
        lines.append(
            f"| [{r['id']}]({r['id']}.md) | {r['created'][:10]} | {r.get('owner','')} "
            f"| [`{b['sha'][:7]}`]({b['web']}/-/commit/{b['sha']}) | [`{c['sha'][:7]}`]({c['web']}/-/commit/{c['sha']}) "
            f"| `{Path(r['config']['source']).name}` | {steps} | {res} |")
    (DOCS_PROD / "index.md").write_text("\n".join(lines) + "\n")


def cmd_index(args) -> None:
    cfg = load_config()
    for r in all_roasts():
        write_docs(cfg, r)
    write_index(cfg)
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
    p = argparse.ArgumentParser(prog=TOOL, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="write ~/.config/roast/config.json")
    s.add_argument("--cmslpc-user", help="LPC username: sets the ssh target and the EOS archive path")
    s.add_argument("--cmslpc-node", default="cmslpc307", help="LPC interactive node to pin (default: cmslpc307)")
    s.add_argument("--falcon-user"); s.add_argument("--cern-user", help="CERN username: sets the CERNBox web area")
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
    s.set_defaults(func=cmd_new)

    s = sub.add_parser("checkout", help="create isolated checkouts at the pinned shas on each host")
    s.add_argument("id"); s.add_argument("--host")
    s.set_defaults(func=cmd_checkout)

    for name, fn, hlp in (("submit", cmd_submit, "launch one step in a tmux window on its host"),
                          ("resume", cmd_resume, "relaunch a step with --unlock and --rerun-incomplete")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("id"); s.add_argument("--step", required=True)
        s.add_argument("--cores", type=int); s.add_argument("--extra", help="extra snakemake args, quoted (e.g. --extra=\"--resources gres=mps:25\")")
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
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("publish", help="copy results to CERNBox and write cupping notes")
    s.add_argument("id"); s.add_argument("--host"); s.add_argument("--docs-only", action="store_true")
    s.add_argument("--jobs", type=int, default=16, help="parallel xrdcp streams (default 16)")
    s.add_argument("--include-test", action="store_true", help="also publish output/*_test/ directories")
    s.add_argument("-n", "--dry-run", action="store_true", help="list the files that would be published, copy nothing")
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

    args = p.parse_args(argv)
    try:
        args.func(args)
    except subprocess.CalledProcessError as e:
        die(f"command failed ({e.returncode}): {' '.join(map(str, e.cmd))}\n{(e.stderr or '').strip()}")
    except KeyboardInterrupt:
        die("interrupted", 130)


if __name__ == "__main__":
    main()
