"""Check that a skim processed every event of its input datasets.

For each dataset in the skimmer registries (picoaod_*.yml), look up the NanoAOD dataset name(s)
in the datasets metadata, ask DBS for the number of valid events and lumi sections, and compare
with the registry's total_events and lumis_processed. Writes a YAML report and a markdown table
next to it, and exits non-zero if any event count differs or a dataset cannot be checked.

DBS is queried over its REST API with the grid proxy (X509_USER_PROXY, else proxy/x509_proxy),
so the check runs the same way in the analysis container at any site, without dasgoclient.

    python src/tools/check_event_counts.py \
        -f output/skims/skimmer/picoaod_*.yml -m coffea4bees/metadata/datasets/ \
        -o output/skims/checks/event_counts.yml
"""
import argparse
import os
import sys

import requests
import yaml

DBS_URL = "https://cmsweb.cern.ch/dbs/prod/global/DBSReader"


def proxy_path():
    for p in (os.environ.get("X509_USER_PROXY"), "proxy/x509_proxy", f"/tmp/x509up_u{os.getuid()}"):
        if p and os.path.exists(p):
            return p
    raise RuntimeError("no grid proxy found (X509_USER_PROXY, proxy/x509_proxy, /tmp/x509up_u<uid>)")


def dbs_summary(das_name, proxy):
    """Valid-file event, lumi and file counts of one dataset, from DBS."""
    ca = os.environ.get("X509_CERT_DIR", "/etc/grid-security/certificates")
    r = requests.get(f"{DBS_URL}/filesummaries",
                     params={"dataset": das_name, "validFileOnly": 1},
                     cert=(proxy, proxy), verify=ca if os.path.isdir(ca) else True, timeout=120)
    r.raise_for_status()
    rows = r.json()
    if not rows or rows[0].get("num_event") is None:
        raise RuntimeError(f"DBS knows no valid files for {das_name}")
    return {k: int(rows[0][f"num_{k}"]) for k in ("event", "lumi", "file")}


def load_metadata(paths):
    """Merge datasets metadata files and directories (as runner.py -m) into {dataset: {year: ...}}."""
    files = []
    for p in paths:
        files += sorted(os.path.join(p, f) for f in os.listdir(p) if f.endswith((".yml", ".yaml"))) \
            if os.path.isdir(p) else [p]
    merged = {}
    for fname in files:
        with open(fname) as f:
            content = yaml.safe_load(f) or {}
        for dataset, years in content.get("datasets", content).items():
            if isinstance(years, dict):
                merged.setdefault(dataset, {}).update(years)
    return merged


def das_names_by_key(metadata, data_tier):
    """Map registry keys to NanoAOD dataset names, following the runner's naming.

    Data:  {dataset}_{year}{era}  (data_tier is a dict of era -> name)
    MC:    {dataset}_{year}       (data_tier is a name or a list of names)
    """
    names = {}
    for dataset, years in metadata.items():
        if not isinstance(years, dict):
            continue
        for year, tiers in years.items():
            entry = tiers.get(data_tier) if isinstance(tiers, dict) else None
            if isinstance(entry, dict):
                for era, name in entry.items():
                    names[f"{dataset}_{year}{era}"] = [name] if isinstance(name, str) else list(name)
            elif isinstance(entry, str):
                names[f"{dataset}_{year}"] = [entry]
            elif isinstance(entry, list):
                names[f"{dataset}_{year}"] = list(entry)
    return names


def n_lumis(lumis_processed):
    return sum(len(set(ls)) for ls in (lumis_processed or {}).values())


def check(registries, metadata, data_tier="nanoAOD"):
    names = das_names_by_key(metadata, data_tier)
    proxy = proxy_path()
    report = {}
    for reg in registries:
        with open(reg) as f:
            content = yaml.load(f, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader)) or {}
        for key, entry in content.items():
            row = {"registry": os.path.basename(reg),
                   "skim_events": int(entry["total_events"]),
                   "skim_lumis": n_lumis(entry.get("lumis_processed"))}
            if key not in names:
                row.update(status="NOT_FOUND", error=f"no {data_tier} dataset for '{key}' in the metadata")
                report[key] = row
                continue
            try:
                summaries = [dbs_summary(n, proxy) for n in names[key]]
            except Exception as e:
                row.update(das_names=names[key], status="ERROR", error=str(e))
                report[key] = row
                continue
            row.update(das_names=names[key],
                       das_events=sum(s["event"] for s in summaries),
                       das_lumis=sum(s["lumi"] for s in summaries),
                       das_files=sum(s["file"] for s in summaries))
            row["events_diff"] = row["skim_events"] - row["das_events"]
            row["lumis_diff"] = row["skim_lumis"] - row["das_lumis"]
            row["status"] = "OK" if row["events_diff"] == 0 else "MISMATCH"
            report[key] = row
    return report


def markdown(report):
    lines = ["| dataset | DAS name | DAS events | skim events | diff | DAS lumis | skim lumis | status |",
             "|---|---|--:|--:|--:|--:|--:|---|"]
    for key, r in sorted(report.items()):
        fmt = lambda k: f"{r[k]:,}" if isinstance(r.get(k), int) else "-"
        lines.append(f"| {key} | {'<br>'.join(r.get('das_names', ['-']))} | {fmt('das_events')} | "
                     f"{fmt('skim_events')} | {fmt('events_diff')} | {fmt('das_lumis')} | "
                     f"{fmt('skim_lumis')} | {r['status']} |")
    ok = sum(r["status"] == "OK" for r in report.values())
    lines += ["", f"{ok} of {len(report)} datasets have every DBS event in the skim."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-f", "-y", "--registries", nargs="+", required=True,
                        help="skimmer output registries (picoaod_*.yml)")
    parser.add_argument("-m", "--metadata", nargs="+", default=["coffea4bees/metadata/datasets/"],
                        help="datasets metadata (directory or files), as passed to runner.py")
    parser.add_argument("--data-tier", default="nanoAOD", help="metadata key holding the input dataset names")
    parser.add_argument("-o", "--output", help="YAML report; a .md table is written next to it")
    parser.add_argument("--allow-errors", action="store_true",
                        help="exit 0 when DAS lookup fails (for private productions not in DBS);"
                             " MISMATCH (counts disagree) still exits 1")
    args = parser.parse_args()

    report = check(args.registries, load_metadata(args.metadata), args.data_tier)
    table = markdown(report)
    print(table)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            yaml.safe_dump(report, f, sort_keys=True)
        with open(os.path.splitext(args.output)[0] + ".md", "w") as f:
            f.write(table)

    bad = {k: r["status"] for k, r in report.items() if r["status"] != "OK"}
    if bad:
        print(f"FAILED: {bad}", file=sys.stderr)
        if not args.allow_errors or any(r["status"] == "MISMATCH" for r in report.values()):
            sys.exit(1)


if __name__ == "__main__":
    main()
