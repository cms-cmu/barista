#!/usr/bin/env python3
"""Snakemake `cluster-generic` status command for NRP Nautilus Kubernetes."""
import json
import os
import subprocess
import sys

NAMESPACE = os.environ.get("NAUTILUS_NAMESPACE", "cms-cmu")


def main():
    if len(sys.argv) < 2:
        print("Usage: nautilus_status.py <jobid>", file=sys.stderr)
        sys.exit(1)

    jobid = sys.argv[-1].strip()

    if jobid.startswith("local_job_"):
        print("success")
        sys.exit(0)

    cmd = [
        "kubectl",
        "get",
        "job",
        jobid,
        "-n",
        NAMESPACE,
        "-o",
        "json",
    ]

    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        job_data = json.loads(res.stdout.decode())
    except subprocess.CalledProcessError:
        # Job not found in Kubernetes
        print("failed")
        sys.exit(0)
    except Exception as e:
        print(f"Error querying job {jobid}: {e}", file=sys.stderr)
        print("running")
        sys.exit(0)

    status = job_data.get("status", {})
    succeeded = status.get("succeeded", 0)
    failed = status.get("failed", 0)
    active = status.get("active", 0)

    if succeeded > 0:
        print("success")
    elif failed > 0:
        print("failed")
    elif active > 0:
        print("running")
    else:
        # Pod may still be preparing/pending
        print("running")


if __name__ == "__main__":
    main()
