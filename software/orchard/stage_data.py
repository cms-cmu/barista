#!/usr/bin/env python3
"""Stage classifier training inputs onto node-local disk and point the configs at the copies.

A training workflow (train.yml + common.yml) names metadata YAML and friend-tree JSON files that list
their ROOT files as ``root://<host>//<path>`` URLs. This tool

  1. finds every repository file the workflow files reference (metadata YAML, friend JSON, JCM, ...),
  2. collects the ``root://`` URLs in them (optionally filtered with --include / --exclude),
  3. writes copies of those files, and of the workflow files, in which each selected URL is replaced
     by its node-local path ``<data>/<host>/<path>``,
  4. copies the ROOT files themselves, from XRootD or from a GCS mirror of the same layout.

Config files that are themselves remote (a friend-tree JSON or JCM YAML given as ``root://...json``,
as the roast workflows do) are downloaded during ``plan`` and treated like repository files.

Only the standard library is used and the files are edited as text, so it runs on the host as well
as inside the classifier image. Nothing in the repository is modified.

    stage_data.py plan  --stage DIR --data DIR [--source ..] [--include RE] [--exclude RE] train.yml common.yml
    stage_data.py fetch --stage DIR --data DIR --source xrootd|gcs [--gcs-prefix gs://bucket/dir]
    stage_data.py push  --stage DIR --data DIR --gcs-prefix gs://bucket/dir     (fill the GCS mirror)

``plan`` prints the staged path of each workflow file, in the order given.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import re
import subprocess
import sys
import time
from pathlib import Path

URL = re.compile(r"root://([A-Za-z0-9.\-]+(?::\d+)?)/(/[^\s\"',\]\}@]+)")
# a repository path inside an option string, optionally followed by @@key
TOKEN = re.compile(r"(?<![\w./\-:])([A-Za-z0-9_][\w.\-]*(?:/[\w.\-]+)+/?)")
CONFIG_SUFFIX = (".json", ".yml", ".yaml")
MANIFEST = "manifest.txt"


def log(*a):
    print("[stage_data]", *a, file=sys.stderr, flush=True)


def local_path(data: Path, host: str, path: str) -> Path:
    return data / host.split(":")[0] / path.lstrip("/")


def referenced(text: str, root: Path) -> dict[str, list[Path]]:
    """token -> the repository files it stands for (file, <file>.yml, or every YAML/JSON in a dir)."""
    out: dict[str, list[Path]] = {}
    for tok in set(TOKEN.findall(text)):
        p = root / tok
        if p.is_dir():
            files = sorted(f for f in p.iterdir() if f.suffix in (".yml", ".yaml", ".json"))
        elif p.is_file():
            files = [p]
        elif p.with_name(p.name + ".yml").is_file():  # --metadata takes the name without .yml
            files = [p.with_name(p.name + ".yml")]
        else:
            continue
        if files:
            out[tok] = files
    return out


def run(cmd, retries=3):
    for i in range(retries):
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if r.returncode == 0:
            return None
        time.sleep(5 * (i + 1))
    return f"{' '.join(cmd)}\n{r.stdout[-2000:]}"


def download(url: str, dst: Path, source: str, gcs_prefix: str | None):
    """Copy one remote file (a config) to dst, from XRootD or from the GCS mirror."""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    m = URL.fullmatch(url)
    if source == "gcs":
        cmd = ["gcloud", "storage", "cp", f"{gcs_prefix.rstrip('/')}/{m.group(1).split(':')[0]}/{m.group(2).lstrip('/')}", str(dst)]
    else:
        cmd = ["xrdcp", "--silent", "--force", url, str(dst)]
    err = run(cmd)
    if err:
        sys.exit(f"[stage_data] cannot fetch config {url}:\n{err}")


def cmd_plan(a):
    root = Path(a.root).resolve()
    stage, data = Path(a.stage).resolve(), Path(a.data).resolve()
    cfg = stage / "config"
    inc = re.compile(a.include) if a.include else None
    exc = re.compile(a.exclude) if a.exclude else None
    urls: dict[str, Path] = {}

    remote_cfg: dict[str, Path] = {}

    def remote_config(url: str) -> Path:
        """Download a remote config file once, return the path of its rewritten copy."""
        if url not in remote_cfg:
            m = URL.fullmatch(url)
            raw = local_path(data, m.group(1), m.group(2))  # kept beside the ROOT files: `push` mirrors it
            download(url, raw, a.source, a.gcs_prefix)
            out = cfg / "remote" / m.group(1).split(":")[0] / m.group(2).lstrip("/")
            out.parent.mkdir(parents=True, exist_ok=True)
            remote_cfg[url] = out
            out.write_text(rewrite(raw.read_text()))
            log(f"remote config staged: {url}")
        return remote_cfg[url]

    def rewrite(text: str, known_only: bool = False) -> str:
        def sub(m):
            url = m.group(0)
            if url.endswith(CONFIG_SUFFIX):
                return str(remote_config(url)) if a.source != "none" and not known_only else url
            if known_only:  # a metadata directory lists every dataset: stage only what is used
                return str(urls[url]) if url in urls else url
            if (inc and not inc.search(url)) or (exc and exc.search(url)):
                return url
            dst = local_path(data, m.group(1), m.group(2))
            urls[url] = dst
            return str(dst)

        return URL.sub(sub, text)

    # Files named explicitly (friend-tree JSON, a metadata file, remote configs) select the URLs to
    # stage. Files that are only reached through a directory (--metadata some/dir/) are rewritten
    # afterwards, and only for URLs already selected: the friend JSON names every picoAOD it extends.
    wfs = [(Path(w).resolve(), Path(w).resolve().read_text()) for w in a.workflow]
    refs = [referenced(text, root) for _, text in wfs]
    is_dir = lambda tok: (root / tok).is_dir()
    texts = [rewrite(text) for _, text in wfs]
    for known_only in (False, True):
        for r in refs:
            for tok, files in r.items():
                if is_dir(tok) != known_only:
                    continue
                for src in files:
                    dst = cfg / src.relative_to(root)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_text(rewrite(src.read_text(), known_only))
    staged_wfs = []
    for (wf, _), text, r in zip(wfs, texts, refs):
        # longest tokens first so a directory token never clobbers a longer file token
        for tok in sorted(r, key=len, reverse=True):
            new = str(cfg / tok) + ("/" if tok.endswith("/") else "")
            text = re.sub(r"(?<![\w./\-:])" + re.escape(tok) + r"(?![\w\-])", lambda _: new, text)
        out = cfg / "workflow" / wf.parent.name / wf.name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        staged_wfs.append(out)
        log(f"{wf.name}: {len(r)} referenced config path(s) staged")
    if not urls:
        log("WARNING: no root:// URLs selected, nothing will be staged (check --include/--exclude)")
    (stage / MANIFEST).write_text("".join(f"{u}\t{p}\n" for u, p in sorted(urls.items())))
    log(f"{len(urls)} ROOT files in {stage / MANIFEST}")
    for w in staged_wfs:
        print(w)


def read_manifest(stage: Path, data: Path):
    rows = []
    for line in (stage / MANIFEST).read_text().splitlines():
        url, _ = line.split("\t")
        m = URL.fullmatch(url)
        rows.append((url, local_path(data, m.group(1), m.group(2)), m.group(1).split(":")[0], m.group(2).lstrip("/")))
    return rows


def cmd_fetch(a):
    stage, data = Path(a.stage).resolve(), Path(a.data).resolve()
    rows = read_manifest(stage, data)
    todo = [r for r in rows if not r[1].exists()]
    log(f"{len(rows)} files, {len(todo)} to copy from {a.source}")
    t0 = time.time()
    if a.source == "gcs":
        prefix = a.gcs_prefix.rstrip("/")
        # one rsync per dataset directory: few transactions, parallel composite downloads
        dirs = sorted({(h, str(Path(p).parent)) for _, _, h, p in todo})
        fails = []
        for h, d in dirs:
            (data / h / d).mkdir(parents=True, exist_ok=True)
            err = run(["gcloud", "storage", "rsync", "--recursive", f"{prefix}/{h}/{d}", str(data / h / d)])
            if err:
                fails.append(err)
    else:
        def one(row):
            url, dst, _, _ = row
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".part")
            err = run(["xrdcp", "--silent", "--force", "--retry", "2", url, str(tmp)])
            if err is None:
                tmp.rename(dst)
            return err

        with cf.ThreadPoolExecutor(a.jobs) as ex:
            fails = [e for e in ex.map(one, todo) if e]
    missing = [str(r[1]) for r in rows if not r[1].exists()]
    size = sum(r[1].stat().st_size for r in rows if r[1].exists())
    dt = max(time.time() - t0, 1e-9)
    log(f"staged {size / 1e9:.1f} GB total in {dt:.0f} s; {len(missing)} missing")
    if fails or missing:
        for e in fails[:5]:
            log("FAILED:", e)
        for m in missing[:10]:
            log("MISSING:", m)
        sys.exit(1)


def cmd_push(a):
    """Mirror the staged files into the bucket so later jobs can use --source gcs."""
    stage, data = Path(a.stage).resolve(), Path(a.data).resolve()
    prefix = a.gcs_prefix.rstrip("/")
    hosts = sorted({h for _, _, h, _ in read_manifest(stage, data)})
    for h in hosts:
        err = run(["gcloud", "storage", "rsync", "--recursive", str(data / h), f"{prefix}/{h}"], retries=2)
        if err:
            sys.exit(f"[stage_data] push failed: {err}")
    log(f"mirrored {', '.join(hosts)} to {prefix}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "fetch", "push"):
        s = sub.add_parser(name)
        s.add_argument("--stage", required=True, help="directory for the manifest and rewritten configs")
        s.add_argument("--data", required=True, help="node-local directory for the ROOT files")
        if name == "plan":
            s.add_argument("--root", default=os.getcwd(), help="barista directory (default: cwd)")
            s.add_argument("--include", help="only stage URLs matching this regex")
            s.add_argument("--exclude", help="do not stage URLs matching this regex")
            s.add_argument("workflow", nargs="+", help="train.yml, common.yml, ...")
        if name in ("plan", "fetch"):
            s.add_argument("--source", choices=("xrootd", "gcs"), default="xrootd")
        if name == "fetch":
            s.add_argument("--jobs", type=int, default=16, help="parallel xrdcp streams")
        s.add_argument("--gcs-prefix", required=(name == "push"), help="gs://bucket/dir holding <host>/<path>")
    a = p.parse_args()
    if a.cmd != "push" and a.source == "gcs" and not a.gcs_prefix:
        p.error("--source gcs needs --gcs-prefix")
    {"plan": cmd_plan, "fetch": cmd_fetch, "push": cmd_push}[a.cmd](a)


if __name__ == "__main__":
    main()
