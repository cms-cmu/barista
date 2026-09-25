#!/usr/bin/env python3
"""Snakemake `cluster-generic` submit command for CERN lxplus HTCondor.

Used by software/snakemake/profiles/lxplus_gpu. Rules that need a GPU (a `gres`, `gpus` or
`request_gpus` resource, or a name in the GPU rule list) are submitted as HTCondor jobs that run
the Snakemake jobscript on a GPU node with the user's Kerberos credential; Snakemake then starts
the rule's container itself (`apptainer exec --nv ...`). Every other rule runs locally on the
login node, synchronously, and reports `local_job_<rule>_<jobid>` (understood by status_wrapper.py).

The GPU rule list can be extended with `--config lxplus_condor_rules=rule1,rule2` or the
LXPLUS_CONDOR_RULES environment variable; `--config lxplus_condor_all=True` sends every rule.
"""
import json
import os
import re
import subprocess
import sys

try:
    from snakemake.utils import read_job_properties
except ImportError:  # keep the module importable (and testable) without snakemake
    def read_job_properties(jobscript, prefix="# properties", pattern=re.compile(r"# properties = (.*)")):
        with open(jobscript) as f:
            for line in f:
                if line.startswith(prefix):
                    return json.loads(pattern.match(line).group(1))
        return {}

# CERN HTCondor job flavours and their maximum wall time in minutes.
FLAVOURS = [
    ("espresso", 20),
    ("microcentury", 60),
    ("longlunch", 120),
    ("workday", 480),
    ("tomorrow", 1440),
    ("testmatch", 4320),
    ("nextweek", 10080),
]
DEFAULT_CONDOR_RULES = {
    "train",
    "evaluate",
    "evaluate_all",
    "evaluate_single_dataset",
    "evaluate_all_svb",
    "evaluate_single_dataset_svb",
}
LOG_DIR = "condor_logs"  # relative to the Snakemake working directory; status_wrapper.py globs it


def job_flavour(runtime_minutes):
    """Smallest CERN JobFlavour whose limit covers `runtime_minutes` (default: workday)."""
    if runtime_minutes is None:
        return "workday"
    try:
        runtime_minutes = float(runtime_minutes)
    except (TypeError, ValueError):
        return "workday"
    for name, limit in FLAVOURS:
        if runtime_minutes <= limit:
            return name
    return FLAVOURS[-1][0]


def n_gpus(resources):
    """Number of GPUs implied by the rule resources (`request_gpus`, `gpus`, or a SLURM-style `gres`)."""
    for key in ("request_gpus", "gpus"):
        if key in resources:
            try:
                return int(resources[key])
            except (TypeError, ValueError):
                return 1
    gres = str(resources.get("gres", ""))
    if gres:
        m = re.match(r"^(gpu|mps)(?::[^:]+)?:(\d+)$", gres)
        if m and m.group(1) == "gpu":
            return int(m.group(2))
        return 1  # e.g. "mps:50" -> one GPU share
    return 0


def condor_rules(config):
    rules = set(DEFAULT_CONDOR_RULES)
    extra = config.get("lxplus_condor_rules") or os.environ.get("LXPLUS_CONDOR_RULES", "")
    if isinstance(extra, str):
        extra = [r for r in extra.split(",") if r]
    rules.update(extra or [])
    return rules


def should_submit(rule, resources, config):
    if str(config.get("lxplus_condor_all", "")).lower() in ("true", "1", "yes"):
        return True
    return n_gpus(resources) > 0 or rule in condor_rules(config)


def build_jdl(jobscript, rule, jobid, threads, resources, gpus, log_dir=LOG_DIR, workdir=None):
    workdir = workdir or os.getcwd()
    log_dir = os.path.join(workdir, log_dir)
    mem_mb = resources.get("mem_mb", 4000)
    lines = [
        "universe = vanilla",
        f"executable = {os.path.abspath(jobscript)}",
        f"initialdir = {workdir}",
        f"output = {log_dir}/job_{rule}_{jobid}_$(Cluster).out",
        f"error = {log_dir}/job_{rule}_{jobid}_$(Cluster).err",
        f"log = {log_dir}/job_{rule}_{jobid}_$(Cluster).log",
        "MY.SendCredential = True",
        f'+JobFlavour = "{job_flavour(resources.get("runtime"))}"',
        f"request_cpus = {threads}",
        f"request_memory = {mem_mb}MB",
        "should_transfer_files = YES",
        "when_to_transfer_output = ON_EXIT",
        'transfer_output_files = ""',
    ]
    if gpus > 0:
        lines.append(f"request_gpus = {gpus}")
        lines.append('requirements = !regexp("MIG", TARGET.GPUs_DeviceName)')
    lines.append("queue")
    return "\n".join(lines) + "\n"


def main():
    jobscript = sys.argv[-1]
    props = read_job_properties(jobscript)
    rule = props.get("rule", "unknown")
    jobid = props.get("jobid", 0)
    threads = props.get("threads", 1)
    resources = props.get("resources", {}) or {}
    config = props.get("config", {}) or {}

    if not should_submit(rule, resources, config):
        print(f"[lxplus_condor_submit] running rule '{rule}' locally on the login node", file=sys.stderr)
        result = subprocess.run(["/bin/bash", jobscript], stdout=sys.stderr)
        print(f"local_job_{rule}_{jobid}")
        sys.exit(result.returncode)

    gpus = n_gpus(resources) or 1
    os.makedirs(LOG_DIR, exist_ok=True)
    jdl_path = os.path.join(LOG_DIR, f"job_{rule}_{jobid}.sub")
    with open(jdl_path, "w") as f:
        f.write(build_jdl(jobscript, rule, jobid, threads, resources, gpus))

    try:
        result = subprocess.run(["condor_submit", jdl_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"[lxplus_condor_submit] condor_submit failed ({e.returncode})\n{e.stdout.decode()}\n{e.stderr.decode()}", file=sys.stderr)
        sys.exit(1)
    stdout = result.stdout.decode()
    m = re.search(r"submitted to cluster (\d+)\.", stdout)
    if not m:
        print(f"[lxplus_condor_submit] could not parse cluster id from: {stdout}", file=sys.stderr)
        sys.exit(1)
    cluster_id = m.group(1)
    print(f"[lxplus_condor_submit] rule '{rule}' job {jobid} -> HTCondor cluster {cluster_id} ({gpus} GPU, {jdl_path})", file=sys.stderr)
    print(cluster_id)


if __name__ == "__main__":
    main()
