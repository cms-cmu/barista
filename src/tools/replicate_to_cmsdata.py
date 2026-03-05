#!/usr/bin/env python3
"""
Replicate HH4b picoAOD and classifier (FvT/SvB) files from FNAL EOS to CMU EOS.

Two sources of files are combined:
  1. picoAOD / friend-tree files explicitly listed in the dataset YML files
  2. FvT*.root / SvB*.root companion files discovered by scanning the same
     source directories on FNAL EOS

All paths are flattened: user-specific prefixes are stripped and the files
are placed under the era-specific group area on CMU EOS.

Usage:
  # Generate commands file only
  python src/scripts/replicate_to_cmsdata.py --era Run2

  # Generate and submit SLURM array job
  python src/scripts/replicate_to_cmsdata.py --era Run2 --submit --proxy proxy/x509_proxy

  # Dry run (no files written, no jobs submitted)
  python src/scripts/replicate_to_cmsdata.py --era Run3 --dry-run

  # Force-overwrite existing files at destination (for reruns)
  python src/scripts/replicate_to_cmsdata.py --era Run2 --submit --force --proxy proxy/x509_proxy

  # Skip companion file scan (only copy files listed in YMLs)
  python src/scripts/replicate_to_cmsdata.py --era Run2 --no-companions
"""

import argparse
import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BARISTA_ROOT = Path(__file__).resolve().parent.parent.parent
META_BASE = BARISTA_ROOT / "coffea4bees/metadata"

# ---------------------------------------------------------------------------
# Era configurations
# ---------------------------------------------------------------------------

ERA_CONFIG = {
    "Run2": {
        "yml_base": META_BASE / "datasets_HH4b_Run2",
        "ymls": [
            "data.yml",
            "GluGluToHHTo4B.yml",
            "TT.yml",
            "ZH4b.yml",
            "ZZ4b.yml",
            "VBFHHTo4B.yml",
            "others.yml",
            "mixeddata_4b.yml",
            "mixeddata_all.yml",
            "synthetic_datasets.yml",
            "2024_v1/datasets_HH4b_v1.yml",
            "2024_v1/datasets_HH4b_v1p1.yml",
            "2024_v1/datasets_HH4b_v1p2.yml",
            "2024_v1/datasets_HH4b_2024_v1.yml",
            "2024_v2/datasets_HH4b_2024_v2.yml",
        ],
        "dest_base": "root://cmsdata.phys.cmu.edu//store/group/HH4b/Run2",
        # Longest/most-specific prefix first to avoid partial matches
        "source_prefixes": [
            "root://cmseos.fnal.gov//store/user/jda102/condor/ZH4b/ULTrig/",
            "root://cmseos.fnal.gov//store/user/algomez/XX4b/",
            "root://cmseos.fnal.gov//store/user/jda102/XX4b/",
            "root://cmseos.fnal.gov//store/user/smurthy/XX4b/",
        ],
        "xrdfs_host": "cmseos.fnal.gov",
    },
    "Run3": {
        "yml_base": META_BASE / "datasets_HH4b_Run3",
        "ymls": [
            "data.yml",
            "GluGlutoHHto4B.yml",
            "TT.yml",
            "NMSSM_XtoYHto4B.yml",
            "mixeddata_4b.yml",
            "mixeddata_all.yml",
            "synthetic_data.yml",
        ],
        "dest_base": "root://cmsdata.phys.cmu.edu//store/group/HH4b/Run3",
        "source_prefixes": [
            "root://cmseos.fnal.gov//store/user/jda102/XX4b/",
            "root://cmseos.fnal.gov//store/user/smurthy/XX4b/",
        ],
        "xrdfs_host": "cmseos.fnal.gov",
    },
}

COMPANION_PREFIXES = ("FvT", "SvB")

# ---------------------------------------------------------------------------
# YAML key classification
# ---------------------------------------------------------------------------

NON_FILE_KEYS = {
    "nanoAOD", "xs", "count", "sumw", "sumw2", "saved_events", "total_events",
    "sumw_raw", "sumw2_raw", "sumw_diff", "sumw2_diff", "outliers", "missing",
    "bad_files", "nSamples", "lumi", "trigger", "top_reconstruction",
    "use_kfold",
    "FvT_name_template", "FvT_name_ZZandZHinSB_template",
    "FvT_name_ZZinSB_template", "FvT_name_kfold_template",
    "JCM_load_template",
}

FVT_FILE_KEYS = {
    "FvT_file_template",
    "FvT_file_ZZandZHinSB_template",
    "FvT_file_ZZinSB_template",
    "FvT_file_kfold_template",
}

# ---------------------------------------------------------------------------
# File collection from YMLs
# ---------------------------------------------------------------------------

def is_fnal(path):
    return isinstance(path, str) and path.startswith("root://cmseos.fnal.gov")


def expand(template, n_samples):
    if "vXXX" in template:
        return [template.replace("vXXX", f"v{i}") for i in range(n_samples)]
    if "seedXXX" in template:
        return [template.replace("seedXXX", f"seed{i}") for i in range(n_samples)]
    return [template]


def collect_from_node(node, n_samples, result):
    if not isinstance(node, dict):
        return
    for key, val in node.items():
        if key in NON_FILE_KEYS:
            continue
        if key == "files":
            for f in (val or []):
                if is_fnal(f):
                    result.add(f)
        elif key == "files_template":
            for tmpl in (val or []):
                if is_fnal(tmpl):
                    for f in expand(tmpl, n_samples):
                        result.add(f)
        elif key in FVT_FILE_KEYS:
            if is_fnal(val):
                for f in expand(val, n_samples):
                    result.add(f)
        else:
            collect_from_node(val, n_samples, result)


def load_yml(path):
    with open(path) as fh:
        data = yaml.safe_load(fh)
    if isinstance(data, dict) and "datasets" in data:
        data = data["datasets"]
    return data or {}


def collect_yml_files(yml_base, ymls):
    result = set()
    for rel in ymls:
        path = yml_base / rel
        data = load_yml(path)
        for dataset_val in data.values():
            if not isinstance(dataset_val, dict):
                continue
            n = dataset_val.get("nSamples", 1)
            collect_from_node(dataset_val, n, result)
    return result


# ---------------------------------------------------------------------------
# Companion file discovery via xrdfs ls
# ---------------------------------------------------------------------------

def xrdfs_ls(host, remote_url):
    """List files in a remote xrootd directory. Returns basenames only."""
    path = "/" + remote_url.split(host)[-1].lstrip("/")
    r = subprocess.run(
        ["xrdfs", host, "ls", path],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0:
        return []
    return [line.strip().rsplit("/", 1)[-1] for line in r.stdout.splitlines() if line.strip()]


def scan_dir_for_companions(src_dir, host):
    """Return full xrootd URLs of FvT/SvB companion files in src_dir."""
    names = xrdfs_ls(host, src_dir)
    return [
        f"{src_dir}/{name}"
        for name in names
        if name.endswith(".root") and any(name.startswith(p) for p in COMPANION_PREFIXES)
    ]


def collect_companion_files(yml_files, host, workers):
    """Scan all source directories referenced by yml_files for companion files."""
    src_dirs = sorted({url.rsplit("/", 1)[0] for url in yml_files})
    print(f"  Scanning {len(src_dirs)} source directories for FvT/SvB companions ...")

    result = set()
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(scan_dir_for_companions, d, host): d for d in src_dirs}
        done = 0
        for future in concurrent.futures.as_completed(futures):
            done += 1
            d = futures[future]
            try:
                companions = future.result()
                result.update(companions)
                if companions:
                    print(f"    [{done:3d}/{len(src_dirs)}] {len(companions)} found: {d.rsplit('/', 1)[-1]}")
            except Exception as e:
                errors.append((d, str(e)))
    if errors:
        print(f"  WARNING: {len(errors)} directories could not be scanned.")
    return result


# ---------------------------------------------------------------------------
# Path flattening
# ---------------------------------------------------------------------------

def flatten(src_url, source_prefixes, dest_base):
    for prefix in source_prefixes:
        if src_url.startswith(prefix):
            relative = src_url[len(prefix):].lstrip("/")
            return f"{dest_base}/{relative}"
    return None


# ---------------------------------------------------------------------------
# SLURM script generation and submission
# ---------------------------------------------------------------------------

SLURM_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/copy_%A_%a.out
#SBATCH --error={log_dir}/copy_%A_%a.err
#SBATCH --partition=work
#SBATCH --time=04:00:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --array=1-{n_tasks}

export X509_USER_PROXY="$SLURM_SUBMIT_DIR/{proxy}"
if [ ! -f "$X509_USER_PROXY" ]; then
    echo "ERROR: proxy file not found at $X509_USER_PROXY"
    exit 1
fi

COMMANDS_FILE="$SLURM_SUBMIT_DIR/{commands_file}"
BATCH_SIZE={batch_size}
TOTAL=$(wc -l < "$COMMANDS_FILE")

START=$(( (SLURM_ARRAY_TASK_ID - 1) * BATCH_SIZE + 1 ))
END=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
if [ "$END" -gt "$TOTAL" ]; then END=$TOTAL; fi

echo "Task $SLURM_ARRAY_TASK_ID: copying lines $START-$END of $TOTAL"
echo "Node: $SLURMD_NODENAME  Date: $(date)"
echo "Proxy: $(voms-proxy-info --timeleft 2>/dev/null || echo 'unknown') seconds remaining"

FAIL=0
COUNT=0

while IFS=$'\\t' read -r SRC DEST; do
    COUNT=$((COUNT + 1))
    echo "[$COUNT] xrdcp $SRC"
    if xrdcp --silent {force_flag}"$SRC" "$DEST"; then
        echo "  OK"
    else
        echo "  FAIL: $SRC"
        echo "$SRC\\t$DEST" >> "$SLURM_SUBMIT_DIR/{log_dir}/failures_${{SLURM_ARRAY_TASK_ID}}.log"
        FAIL=$((FAIL + 1))
    fi
done < <(sed -n "${{START}},${{END}}p" "$COMMANDS_FILE")

echo ""
echo "Task $SLURM_ARRAY_TASK_ID done: $((COUNT - FAIL)) OK, $FAIL failed."
"""


def generate_slurm_script(commands_file, log_dir, job_name, proxy, n_tasks,
                           batch_size, force, output_path):
    script = SLURM_TEMPLATE.format(
        job_name=job_name,
        log_dir=log_dir,
        n_tasks=n_tasks,
        proxy=proxy,
        commands_file=commands_file,
        batch_size=batch_size,
        force_flag="--force " if force else "",
    )
    with open(output_path, "w") as fh:
        fh.write(script)
    os.chmod(output_path, 0o755)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--era", choices=list(ERA_CONFIG), required=True,
                        help="Data-taking era to replicate")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for commands file, SLURM script, and logs "
                             "(default: slurm_replication/<era>/)")
    parser.add_argument("--no-companions", action="store_true",
                        help="Skip scanning for FvT/SvB companion files")
    parser.add_argument("--submit", action="store_true",
                        help="Submit a SLURM array job after generating commands")
    parser.add_argument("--proxy", default="proxy/x509_proxy",
                        help="Path to VOMS proxy file relative to submit dir "
                             "(default: proxy/x509_proxy)")
    parser.add_argument("--force", action="store_true",
                        help="Pass --force to xrdcp to overwrite existing files")
    parser.add_argument("--workers", type=int, default=16,
                        help="Parallel workers for companion file scanning (default: 16)")
    parser.add_argument("--batch-size", type=int, default=49,
                        help="Files per SLURM array task (default: 49)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print summary without writing files or submitting")
    args = parser.parse_args()

    cfg = ERA_CONFIG[args.era]

    out_dir = Path(args.output_dir) if args.output_dir else BARISTA_ROOT / "slurm_replication" / args.era
    log_dir = out_dir / "logs"

    # --- Collect files from YMLs ---
    print(f"Era: {args.era}")
    print(f"Parsing YML files ...")
    yml_files = collect_yml_files(cfg["yml_base"], cfg["ymls"])
    print(f"  {len(yml_files)} unique files from YMLs")

    # --- Collect companion files ---
    companion_files = set()
    if not args.no_companions:
        companion_files = collect_companion_files(yml_files, cfg["xrdfs_host"], args.workers)
        print(f"  {len(companion_files)} companion files (FvT/SvB) found")

    all_sources = yml_files | companion_files
    print(f"  {len(all_sources)} total unique source files")

    # --- Flatten paths ---
    pairs, unmatched = [], []
    for src in sorted(all_sources):
        dest = flatten(src, cfg["source_prefixes"], cfg["dest_base"])
        if dest:
            pairs.append((src, dest))
        else:
            unmatched.append(src)

    if unmatched:
        print(f"\nWARNING: {len(unmatched)} files with unrecognized prefix (skipped):")
        for f in unmatched[:10]:
            print(f"  {f}")

    import math
    n_tasks = math.ceil(len(pairs) / args.batch_size)
    print(f"\n{len(pairs)} files → {n_tasks} SLURM tasks ({args.batch_size} files/task)")

    if args.dry_run:
        print("\nDRY RUN — no files written.")
        print(f"Would write commands to: {out_dir}/commands.txt")
        if args.submit:
            print(f"Would submit SLURM job: {out_dir}/submit.sh")
        return

    # --- Write commands file ---
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    commands_file = out_dir / "commands.txt"
    with open(commands_file, "w") as fh:
        for src, dest in pairs:
            fh.write(f"{src}\t{dest}\n")
    print(f"Commands written to: {commands_file}")

    # --- Generate SLURM script ---
    slurm_script = out_dir / "submit.sh"
    generate_slurm_script(
        commands_file=str(commands_file.relative_to(BARISTA_ROOT)),
        log_dir=str(log_dir.relative_to(BARISTA_ROOT)),
        job_name=f"replicate_{args.era.lower()}",
        proxy=args.proxy,
        n_tasks=n_tasks,
        batch_size=args.batch_size,
        force=args.force,
        output_path=slurm_script,
    )
    print(f"SLURM script written to: {slurm_script}")

    if args.submit:
        print("Submitting SLURM job ...")
        r = subprocess.run(["sbatch", str(slurm_script)], capture_output=True, text=True)
        print(r.stdout.strip() or r.stderr.strip())
        if r.returncode != 0:
            sys.exit(r.returncode)
    else:
        print(f"\nTo submit: sbatch {slurm_script}")


if __name__ == "__main__":
    main()
