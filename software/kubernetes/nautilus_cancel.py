#!/usr/bin/env python3
"""Snakemake `cluster-generic` cancel command for NRP Nautilus Kubernetes."""
import os
import subprocess
import sys

NAMESPACE = os.environ.get("NAUTILUS_NAMESPACE", "cms-cmu")


def main():
    if len(sys.argv) < 2:
        sys.exit(0)

    jobid = sys.argv[-1].strip()

    if jobid.startswith("local_job_"):
        sys.exit(0)

    subprocess.run(
        ["kubectl", "delete", "job", jobid, "-n", NAMESPACE, "--ignore-not-found"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


if __name__ == "__main__":
    main()
