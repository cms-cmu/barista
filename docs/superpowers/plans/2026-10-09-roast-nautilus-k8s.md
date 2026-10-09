# Implementation Plan: First-Class Nautilus Kubernetes Backend in `roast`

**Date:** 2026-10-09  
**Branch:** `feat-nautilus-k8s-profile` (in worktree `../barista-nautilus-k8s`)  
**Objective:** Add native NRP Nautilus support to `roast`, enabling single-command submission, status tracking, log streaming, and resumption of GPU training steps via in-cluster Kubernetes runner jobs on NRP Nautilus.

---

## 1. Architecture Overview

`roast` currently supports two SSH-and-tmux-based hosts: `cmslpc` (HTCondor) and `falcon` (Slurm).
We will add `nautilus` as a first-class host target:

```
[Laptop / CLI]
      │
      ├── roast new --phases BKG_AB,BKG_C,BKG_F --nautilus
      │     └── BKG_AB -> cmslpc (Condor)
      │     └── BKG_C  -> nautilus (Kubernetes Job)
      │     └── BKG_F  -> cmslpc (Condor)
      │
      ├── roast checkout <id> --host nautilus
      │     └── Prepares roast dir on CephFS (/workspace/users/<user>/prod/<id>)
      │     └── Stages pinned config.yml and kubeconfig
      │
      ├── roast submit <id> --step BKG_C
      │     └── Renders Job manifest: roast-<id>-<step>
      │     └── Submits to Nautilus via `pixi run kubectl apply`
      │     └── Job runs in-cluster Snakemake runner pod on CephFS
      │     └── Snakemake dispatches 16 GPU fold pods via profile nautilus
      │
      ├── roast status <id>
      │     └── Combines Condor (LPC), Slurm (Falcon), and Nautilus (K8s pods) in one table
      │
      └── roast log <id> --step BKG_C -f  /  roast attach <id>
            └── Streams live logs directly from Nautilus runner pod
```

---

## 2. Key Changes & Components

### A. Configuration (`DEFAULT_CONFIG` in `src/tools/roast.py`)
Add default configuration for host `nautilus`:
```python
"nautilus": {
    "namespace": "cms-cmu",
    "user": "<user>",
    "prod_root": "/workspace/users/<user>/prod",
    "reference": "/workspace/users/<user>/barista",
    "cores": 16,
}
```

### B. `roast new` Flag: `--nautilus`
Add `--nautilus` flag to `roast new`:
When `--nautilus` is passed, GPU training phases (`C`, `D`, `BKG_C`) default to host `nautilus` instead of `falcon`.

### C. `roast checkout` for Nautilus
When checking out on `nautilus`:
- Uses `kubectl exec -i storage-helper -n <namespace>` to prepare `/workspace/users/<user>/prod/<id>`.
- Copies `roasts/<id>/config.yml` to CephFS via `kubectl cp`.
- Ensures user's `~/.kube/config` is staged to `/workspace/users/<user>/.kube/config` with `chmod 600`.

### D. `roast submit` / `roast resume` for Nautilus
- Generates a standard Job manifest `roast-<label>-<step>.yaml` under `roasts/<id>/`.
- The Job:
  - Image: `gitlab-registry.cern.ch/cms-cmu/barista:classifier_latest`
  - Mounts: CephFS `/workspace`, CVMFS `/cvmfs`, `/dev/shm`
  - Resources: 2 CPUs, 8 GB RAM, 0 GPUs
  - Command:
    ```bash
    export KUBECONFIG=/workspace/users/<user>/.kube/config
    cd /workspace/users/<user>/prod/<id>
    snakemake -s <snakefile> \
              --configfile roasts/<id>/config.yml \
              --profile software/snakemake/profiles/nautilus \
              --jobs <cores> \
              --config roast_id=<id> ...
    ```
- Submits with `kubectl apply -f ...`.
- On `resume`: deletes existing failed/completed job (`kubectl delete job ... --ignore-not-found`) and appends `--rerun-incomplete`.

### E. `roast status` for Nautilus
- Queries `kubectl get pods -n <namespace> -l roast=<id>` and `kubectl get job roast-<label>-<step>`.
- Parses state:
  - `RUNNING`: active runner pod + number of active/completed fold pods.
  - `COMPLETED`: runner job Succeeded.
  - `FAILED`: runner job Failed.
- Renders seamlessly under the roast steps view.

### F. `roast log` / `roast attach` for Nautilus
- `roast log <id> --step <step> -f`: runs `kubectl logs -f job/roast-<label>-<step> -n <namespace>`.
- `roast attach <id>`: detects if the active step is on Nautilus and streams logs directly.

---

## 3. Verification & Validation Steps
1. Unit check: `pixi run bin/roast init --help` and verify `nautilus` configuration parsing.
2. Dry run: `roast new --config ... --phases BKG_C --nautilus` -> verify manifest records `host: "nautilus"`.
3. Submit dry-run: `roast submit <id> --step BKG_C -n` -> verify Job manifest generation and dry-run execution on Nautilus.
4. Status check: `roast status <id>` -> verify it parses Kubernetes status without error.
5. Worktree sync: Ensure all changes are isolated in `../barista-nautilus-k8s` and left uncommitted for user review.
