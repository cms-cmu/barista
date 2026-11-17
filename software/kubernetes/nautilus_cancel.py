#!/usr/bin/env python3
"""Snakemake `cluster-generic` cancel command for NRP Nautilus Kubernetes."""
import os
import subprocess
import sys

NAMESPACE = os.environ.get("NAUTILUS_NAMESPACE", "cms-cmu")
CONTEXT = os.environ.get("NAUTILUS_CONTEXT", "").strip()


def find_kubectl():
    import shutil
    if not CONTEXT and os.path.exists("/workspace/shared/bin/kubectl"):
        return "/workspace/shared/bin/kubectl"
    return shutil.which("kubectl") or "kubectl"


def main():
    if len(sys.argv) < 2:
        sys.exit(0)

    jobid = sys.argv[-1].strip()

    if jobid.startswith("local_job_"):
        sys.exit(0)

    cmd = [find_kubectl()]
    if CONTEXT:
        cmd.extend(["--context", CONTEXT])
    cmd.extend(["delete", "job", jobid, "-n", NAMESPACE, "--ignore-not-found"])
    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


if __name__ == "__main__":
    main()
