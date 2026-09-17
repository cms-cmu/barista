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
    roast new --config coffea4bees/workflows/config/nominal_run2.yml --phases B,C,D,F
    roast checkout <id>                          # isolated trees on cmslpc + falcon
    roast submit   <id> --step B                 # tmux window on cmslpc
    roast status   <id>
    roast submit   <id> --step C                 # after B is done, on falcon
    ...
    roast publish  <id>                          # CERNBox + docs/prod/<id>.md + index
    roast ls

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
    "D": ("falcon", "coffea4bees/workflows/Snakefile_PhaseD.smk"),
    "E": ("cmslpc", "coffea4bees/workflows/Snakefile_PhaseE.smk"),
    "F": ("cmslpc", "coffea4bees/workflows/Snakefile_PhaseF.smk"),
}

# Files worth publishing to CERNBox (small, human-readable artefacts).
PUBLISH_EXT = ("pdf", "png", "svg", "yml", "yaml", "json", "txt", "log", "md", "html", "csv", "tex")
PUBLISH_MAX_MB = 50

DEFAULT_CONFIG = {
    "hosts": {
        "cmslpc": {
            "ssh": "<user>@cmslpc307.fnal.gov",
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
    },
    "cernbox": {
        "eos_path": "/eos/user/<x>/<user>/www/HH4b/prod",
        "url": "https://<user>.web.cern.ch/<user>/HH4b/prod",
    },
    "owner": "<your name>",
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


def ssh_run(target: str, script: str, check=True, capture=True) -> subprocess.CompletedProcess:
    """Run a bash script on a remote host via stdin (no quoting games)."""
    return sh(["ssh", *SSH_OPTS, target, "bash -s"], check=check, capture=capture, input=script)


def scp_to(target: str, srcs: list[Path], dst: str) -> None:
    sh(["scp", "-q", *SSH_OPTS, *map(str, srcs), f"{target}:{dst}"])


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
        cfg["hosts"]["cmslpc"]["ssh"] = f"{args.cmslpc_user}@cmslpc307.fnal.gov"
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
    info(f"wrote {CONFIG_PATH}; edit ssh targets, paths and cores to taste")


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
    rid = f"{dt.datetime.now():%Y%m%d}_{label}_{barista_sha[:7]}-{c4b_sha[:7]}"
    if args.id:
        rid = args.id
    if roast_dir(rid).exists():
        die(f"roast {rid} already exists")

    d = roast_dir(rid)
    d.mkdir(parents=True)
    shutil.copy2(config_src, d / "config.yml")

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
        target = hc["ssh"]
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
        r["hosts"][host] = {"checkout": ckpt, "ssh": target, "checked_out": now()}
        log_event(r, "checkout", host=host)
        save_roast(r)
    info(f"next: {TOOL} submit {r['id']} --step {r['steps'][0]['name']}")


def _run_script(cfg: dict, r: dict, step: dict, ckpt: str, cores: int, extra: str, resume: bool) -> str:
    """The bash script that runs one step inside its tmux window."""
    name = step["name"]
    smk = step["snakefile"]
    configfile = f"roasts/{r['id']}/config.yml"
    base = f"./run_container snakemake -s {shlex.quote(smk)} --configfile {configfile} --cores {cores}"
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
        cd {rq(ckpt)} || exit 97
        LOG=logs/{name}.log
        EXIT=logs/{name}.exit
        rm -f "$EXIT"
        echo "=== roast {r['id']} step {name} start $(date) on $(hostname) ===" | tee -a "$LOG"
        echo "=== barista $(git rev-parse --short HEAD) coffea4bees $(cd coffea4bees && git rev-parse --short HEAD) ===" | tee -a "$LOG"
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
    target = r["hosts"][host]["ssh"]
    cores = args.cores or hc.get("cores", 4)
    script = _run_script(cfg, r, step, ckpt, cores, args.extra or "", resume)
    local = roast_dir(r["id"]) / f"run_{step['name']}.sh"
    local.write_text(script)
    scp_to(target, [local], f"{ckpt}/roasts/{r['id']}/")
    window = f"{r['id'][:8]}_{r['label'][:12]}_{step['name']}"
    remote = textwrap.dedent(f"""\
        set -e
        SCRIPT={rq(f"{ckpt}/roasts/{r['id']}/run_{step['name']}.sh")}
        chmod +x "$SCRIPT"
        tmux has-session -t {TMUX_SESSION} 2>/dev/null || tmux new-session -d -s {TMUX_SESSION} -n hub
        if tmux list-windows -t {TMUX_SESSION} -F '#W' | grep -qx {shlex.quote(window)}; then
            echo "window {window} already exists in tmux session {TMUX_SESSION}; refusing to double-submit" >&2
            exit 3
        fi
        tmux new-window -d -t {TMUX_SESSION} -n {shlex.quote(window)} "bash $SCRIPT"
        echo "launched tmux {TMUX_SESSION}:{window}"
    """)
    res = ssh_run(target, remote, check=False)
    if res.returncode != 0:
        die(res.stderr.strip() or res.stdout.strip())
    print(res.stdout.strip())
    step.setdefault("runs", []).append({"ts": now(), "cores": cores, "extra": args.extra or "", "resume": resume, "window": window})
    log_event(r, "resume" if resume else "submit", step=step["name"], host=host)
    save_roast(r)
    info(f"attach: ssh -t {target} tmux attach -t {TMUX_SESSION}    |    {TOOL} status {r['id']}")


def cmd_submit(args) -> None:
    _submit(args, resume=False)


def cmd_resume(args) -> None:
    _submit(args, resume=True)


def _status_script(r: dict, ckpt: str, steps: list[dict], host: str) -> str:
    step_names = " ".join(s["name"] for s in steps)
    condor = 'condor_q -totals 2>/dev/null | grep -m1 "Total for query" || true' if host == "cmslpc" else "true"
    gpu = ('for NV in nvidia-smi /usr/bin/nvidia-smi /usr/local/cuda/bin/nvidia-smi; do command -v $NV >/dev/null 2>&1 && '
           '{ $NV --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | '
           'sed "s/^/GPU /"; break; }; done || true') if host == "falcon" else "true"
    return textwrap.dedent(f"""\
        cd {rq(ckpt)} 2>/dev/null || {{ echo "NOCHECKOUT"; exit 0; }}
        for S in {step_names}; do
            if [ -f logs/$S.exit ]; then st="exit=$(cat logs/$S.exit)";
            elif [ -f logs/$S.log ]; then st="running";
            else st="not started"; fi
            win=$(tmux list-windows -t {TMUX_SESSION} -F '#W' 2>/dev/null | grep -c "_$S\\$" || true)
            prog=$(ls -t .snakemake/log/*.snakemake.log 2>/dev/null | head -1 | xargs -r grep -h -o '[0-9]* of [0-9]* steps ([0-9]*%) done' 2>/dev/null | tail -1)
            last=$(tail -n 1 logs/$S.log 2>/dev/null | cut -c1-90)
            echo "STEP|$S|$st|tmux=$win|$prog|$last"
        done
        echo "HOST|$({condor})"
        {gpu}
    """)


def cmd_status(args) -> None:
    cfg = load_config()
    roasts = [load_roast(args.id)] if args.id else [r for r in all_roasts() if r.get("hosts")]
    for r in roasts:
        print(f"\n\033[1m{r['id']}\033[0m  barista {r['barista']['sha'][:7]}  coffea4bees {r['coffea4bees']['sha'][:7]}  {r['config']['source']}")
        for host, hinfo in r["hosts"].items():
            steps = [s for s in r["steps"] if s["host"] == host]
            res = ssh_run(hinfo["ssh"], _status_script(r, hinfo["checkout"], steps, host), check=False)
            if res.returncode != 0:
                print(f"  {host:7s} ssh failed: {res.stderr.strip()[:100]}")
                continue
            for line in res.stdout.splitlines():
                if line.startswith("STEP|"):
                    _, s, st, win, prog, last = (line.split("|", 5) + [""] * 6)[:6]
                    colour = "\033[32m" if st == "exit=0" else "\033[31m" if st.startswith("exit=") else "\033[33m" if st == "running" else ""
                    print(f"  {host:7s} {s:14s} {colour}{st:12s}\033[0m {win:7s} {prog:28s} {last}".rstrip())
                elif line.startswith("HOST|") and line[5:].strip():
                    print(f"  {host:7s} condor: {line[5:].strip()}")
                elif line.startswith("GPU "):
                    print(f"  {host:7s} {line}")
                elif line == "NOCHECKOUT":
                    print(f"  {host:7s} checkout missing at {hinfo['checkout']}")
        if r.get("publish", {}).get("url"):
            print(f"  published: {r['publish']['url']}")


def _publish_script(r: dict, ckpt: str, eos_dir: str) -> str:
    exts = " -o ".join(f"-name '*.{e}'" for e in PUBLISH_EXT)
    return textwrap.dedent(f"""\
        set -euo pipefail
        cd {rq(ckpt)}
        command -v xrdcp >/dev/null || {{ echo "xrdcp not found on $(hostname)" >&2; exit 2; }}
        # Auth for eosuser.cern.ch: a CERN kerberos ticket if present, else the grid proxy run_container keeps in ./proxy
        if ! klist -s 2>/dev/null && [ -s proxy/x509_proxy ]; then export X509_USER_PROXY="$PWD/proxy/x509_proxy"; fi
        DST_BASE=root://eosuser.cern.ch/{eos_dir}
        LIST=$(mktemp)
        {{ [ -d output ] && find output -type f \\( {exts} \\) -size -{PUBLISH_MAX_MB}M; \\
           find logs roasts/{r['id']} -type f; }} | sort -u > "$LIST"
        N=$(wc -l < "$LIST")
        echo "publishing $N files from $(hostname):$PWD -> $DST_BASE"
        sed 's#/[^/]*$##' "$LIST" | sort -u | while read -r d; do xrdfs root://eosuser.cern.ch mkdir -p "{eos_dir}/$d" 2>/dev/null || true; done
        xrdfs root://eosuser.cern.ch mkdir -p "{eos_dir}" 2>/dev/null || true
        FAIL=0
        while read -r f; do
            xrdcp -f -s "$f" "$DST_BASE/$f" || {{ echo "FAILED $f"; FAIL=$((FAIL+1)); }}
        done < "$LIST"
        echo "done: $((N-FAIL)) ok, $FAIL failed"
        [ "$FAIL" = 0 ]
    """)


def cmd_publish(args) -> None:
    cfg = load_config()
    r = load_roast(args.id)
    eos_dir = f"{cfg['cernbox']['eos_path'].rstrip('/')}/{r['id']}"
    url = f"{cfg['cernbox']['url'].rstrip('/')}/{r['id']}/"
    hosts = [args.host] if args.host else list(r["hosts"])
    ok = True
    if not args.docs_only:
        for host in hosts:
            hinfo = r["hosts"][host]
            info(f"[{host}] publishing to {eos_dir}")
            res = ssh_run(hinfo["ssh"], _publish_script(r, hinfo["checkout"], eos_dir), check=False)
            print((res.stdout + res.stderr).strip())
            if res.returncode != 0:
                ok = False
                info(f"[{host}] publish FAILED (CERN auth on that host? try `kinit <user>@CERN.CH` or a voms proxy there)")
            log_event(r, "publish", host=host, ok=res.returncode == 0)
    r["publish"] = {"eos": eos_dir, "url": url, "ts": now(), "ok": None if args.docs_only else ok}
    save_roast(r)
    write_docs(cfg, r)
    write_index(cfg)
    info(f"results: {url}")
    info(f"docs: {DOCS_PROD / (r['id'] + '.md')} and index.md regenerated; commit roasts/ and docs/prod/ to update the Pages site")


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
        "",
        "## Steps",
        "",
        "| step | host | snakefile | state | log |",
        "|---|---|---|---|---|",
    ]
    for s in r["steps"]:
        log = f"[{s['name']}.log]({url}logs/{s['name']}.log)" if url else ""
        lines.append(f"| {s['name']} | {s['host']} | `{s['snakefile']}` {s.get('targets','')} | {_step_state(r, s)} | {log} |")
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
    s.add_argument("--cmslpc-user"); s.add_argument("--falcon-user"); s.add_argument("--cern-user"); s.add_argument("--owner")
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
        s.add_argument("--cores", type=int); s.add_argument("--extra", help="extra snakemake args, quoted")
        s.set_defaults(func=fn)

    s = sub.add_parser("status", help="per-step state on each host")
    s.add_argument("id", nargs="?")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("publish", help="copy results to CERNBox and write cupping notes")
    s.add_argument("id"); s.add_argument("--host"); s.add_argument("--docs-only", action="store_true")
    s.set_defaults(func=cmd_publish)

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
