#!/usr/bin/env python3
"""Snakemake `cluster-generic` submit command for NRP Nautilus Kubernetes.

Used by software/snakemake/profiles/nautilus.
Rules that require GPU resources (e.g. train, evaluate, or rules with `gpus`/`gres`)
are automatically submitted as Kubernetes Jobs to NRP Nautilus using the shared
CephFS volume. All lightweight/local rules run locally on the submit machine.
"""
import getpass
import json
import os
import re
import subprocess
import sys
import uuid

try:
    from snakemake.utils import read_job_properties
except ImportError:
    def read_job_properties(jobscript, prefix="# properties", pattern=re.compile(r"# properties = (.*)")):
        with open(jobscript) as f:
            for line in f:
                if line.startswith(prefix):
                    return json.loads(pattern.match(line).group(1))
        return {}

DEFAULT_GPU_RULES = {
    "train",
    "evaluate",
    "evaluate_all",
    "evaluate_single_dataset",
    "evaluate_all_svb",
    "evaluate_single_dataset_svb",
    "bkg_syst_C_2_train",
    "bkg_syst_C_3_evaluate",
}

LOG_DIR = "k8s_logs"
NAMESPACE = os.environ.get("NAUTILUS_NAMESPACE", "cms-cmu")
STORAGE_CLAIM = os.environ.get("NAUTILUS_STORAGE_CLAIM", "cms-cmu-storage")
IMAGE = os.environ.get(
    "NAUTILUS_IMAGE", "gitlab-registry.nrp-nautilus.io/cms-cmu/barista:classifier_latest"
)
WORKDIR = os.environ.get(
    "NAUTILUS_WORKDIR", os.getcwd()
)
CVMFS_CLAIM = os.environ.get("NAUTILUS_CVMFS_CLAIM", "cvmfs")
ENABLE_CVMFS = os.environ.get("NAUTILUS_ENABLE_CVMFS", "true").lower() in ("true", "1", "yes")


def n_gpus(resources):
    """Detect whether this rule needs a GPU."""
    for key in ("request_gpus", "gpus", "gpu"):
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
        return 1
    return 0


def should_submit(rule, resources, config):
    """Determine whether to dispatch to Nautilus or run locally."""
    if str(config.get("nautilus_all", "")).lower() in ("true", "1", "yes"):
        return True
    if n_gpus(resources) > 0:
        return True
    rules = set(DEFAULT_GPU_RULES)
    extra = config.get("nautilus_rules") or os.environ.get("NAUTILUS_RULES", "")
    if isinstance(extra, str):
        extra = [r.strip() for r in extra.split(",") if r.strip()]
    rules.update(extra)
    return rule in rules


def build_k8s_job_yaml(job_name, jobscript_content, gpus=1, cpus=6, mem_gb=16, labels=None):
    """Generate the Kubernetes Job manifest."""
    import base64
    b64_script = base64.b64encode(jobscript_content.encode("utf-8")).decode("utf-8")

    labels_block = ""
    if labels:
        lines = [f"    {k}: \"{v}\"" for k, v in labels.items() if v]
        if lines:
            labels_block = "  labels:\n" + "\n".join(lines)

    template_labels_block = ""
    if labels:
        lines = [f"        {k}: \"{v}\"" for k, v in labels.items() if v]
        if lines:
            template_labels_block = "    metadata:\n      labels:\n" + "\n".join(lines)

    if gpus > 0:
        req_cpu = max(int(cpus), 6)
        lim_cpu = req_cpu
        req_mem = f"{max(int(mem_gb), 28)}Gi"
        lim_mem = req_mem
        gpu_limits = f"nvidia.com/gpu: {gpus}\n            cpu: '{lim_cpu}'\n            memory: {lim_mem}"
        gpu_requests = f"nvidia.com/gpu: {gpus}\n            cpu: '{req_cpu}'\n            memory: {req_mem}"
    else:
        req_cpu = max(int(cpus), 1)
        lim_cpu = req_cpu
        gpu_limits = f"cpu: '{lim_cpu}'\n            memory: 8Gi"
        gpu_requests = f"cpu: '{req_cpu}'\n            memory: 8Gi"

    labels_section = f"\n{labels_block}" if labels_block else ""
    template_labels_section = f"\n{template_labels_block}" if template_labels_block else ""

    manifest = f"""apiVersion: batch/v1
kind: Job
metadata:
  name: {job_name}
  namespace: {NAMESPACE}{labels_section}
spec:
  backoffLimit: 0
  template:{template_labels_section}
    spec:
      restartPolicy: Never
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: kubernetes.io/hostname
                operator: NotIn
                values:
                - gpn-fiona-mizzou-8.rnet.missouri.edu
                - nautilus-ext-gpu01.fullerton.edu
                - nautilus-it-gpu05.fullerton.edu
                - fiona.offn.oscer.ou.edu
                - dtn-gpu2.kreonet.net
                - dtn-gpu1.kreonet.net
                - ry-gpu-15.sdsc.optiputer.net
              - key: nvidia.com/gpu.product
                operator: NotIn
                values:
                - NVIDIA-GeForce-GTX-1080-Ti
                - NVIDIA-GeForce-GTX-1080
                - NVIDIA-GeForce-RTX-2080-Ti
                - NVIDIA-TITAN-Xp
                - NVIDIA-TITAN-X-Pascal
                - Quadro-M4000
                - Tesla-T4
                - NVIDIA-A2
                - Tesla-V100-PCIE-16GB
                - Tesla-V100-SXM2-16GB
                - NVIDIA-A100-80GB-PCIe-MIG-1g.10gb
                - NVIDIA-RTX-A4000
                - RTX-A4000
                - NVIDIA-RTX-A4500
                - Quadro-RTX-5000
      containers:
      - name: worker
        image: {IMAGE}
        imagePullPolicy: IfNotPresent
        workingDir: {WORKDIR}
        command: ["bash", "-c"]
        args:
          - |
            set -e
            source /entrypoint.sh 2>/dev/null || true
            export PYTHONPATH=/workspace/shared/lib/python3.11/site-packages:{WORKDIR}:$PYTHONPATH
            export X509_USER_PROXY={WORKDIR}/proxy/x509_proxy
            export CLASSIFIER_CONFIG_PATHS=coffea4bees
            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
            cd {WORKDIR}
            if ! /opt/conda/envs/hep/bin/python -c "import snakemake" >/dev/null 2>&1; then
                /opt/conda/envs/hep/bin/pip install --no-cache-dir --quiet snakemake==9.27.0
            fi
            mkdir -p /opt/conda/envs/snakemake/bin
            ln -sf /opt/conda/envs/hep/bin/python /opt/conda/envs/snakemake/bin/python3.14
            ln -sf /opt/conda/envs/hep/bin/python /opt/conda/envs/snakemake/bin/python3
            echo "{b64_script}" | base64 -d > /tmp/jobscript.sh
            exec bash /tmp/jobscript.sh
        resources:
          limits:
            {gpu_limits}
          requests:
            {gpu_requests}
        volumeMounts:
        - mountPath: /workspace
          name: ceph-storage
        - mountPath: /dev/shm
          name: dshm
        - mountPath: /cvmfs
          name: cvmfs-storage
          readOnly: true
      volumes:
      - name: ceph-storage
        persistentVolumeClaim:
          claimName: {STORAGE_CLAIM}
      - name: dshm
        emptyDir:
          medium: Memory
      - name: cvmfs-storage
        persistentVolumeClaim:
          claimName: {CVMFS_CLAIM}
          readOnly: true
"""
    return manifest


def main():
    jobscript = sys.argv[-1]
    props = read_job_properties(jobscript)
    rule = props.get("rule", "unknown")
    jobid = props.get("jobid", 0)
    resources = props.get("resources", {}) or {}
    config = props.get("config", {}) or {}

    # If this is not a GPU rule, execute locally
    if not should_submit(rule, resources, config):
        print(f"[nautilus_submit] running rule '{rule}' locally", file=sys.stderr)
        res = subprocess.run(["/bin/bash", jobscript], stdout=sys.stderr)
        print(f"local_job_{rule}_{jobid}")
        sys.exit(res.returncode)

    gpus = n_gpus(resources) or 1
    threads = props.get("threads", 1)
    cpus = int(resources.get("cpu", resources.get("cpus_per_task", threads)))
    mem_mb = int(resources.get("mem_mb", 28000))
    mem_gb = max(28, mem_mb // 1024)

    os.makedirs(LOG_DIR, exist_ok=True)

    # Sanitize job name for Kubernetes (RFC 1123, max 63 characters)
    clean_rule = re.sub(r"[^a-z0-9-]", "-", rule.lower())[:20].strip("-")
    uid = uuid.uuid4().hex[:6]
    job_name = f"smk-{clean_rule}-{jobid}-{uid}"

    with open(jobscript, "r") as f:
        jobscript_content = f.read()

    roast_id = os.environ.get("ROAST_ID") or config.get("roast_id", "")
    roast_step = os.environ.get("ROAST_STEP", "")
    labels = {}
    if roast_id:
        labels["roast"] = roast_id
    if roast_step:
        labels["step"] = roast_step
    labels["rule"] = clean_rule
    labels["app"] = "roast-worker"

    manifest_yaml = build_k8s_job_yaml(
        job_name, jobscript_content, gpus=gpus, cpus=cpus, mem_gb=mem_gb, labels=labels
    )
    manifest_path = os.path.join(LOG_DIR, f"{job_name}.yaml")

    with open(manifest_path, "w") as f:
        f.write(manifest_yaml)

    env = os.environ.copy()
    env.pop("KUBECONFIG", None)
    try:
        subprocess.run(
            ["kubectl", "apply", "--validate=false", "-f", manifest_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        print(
            f"[nautilus_submit] kubectl apply failed ({e.returncode}):\n{e.stderr.decode()}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"[nautilus_submit] rule '{rule}' job {jobid} -> NRP Nautilus Job '{job_name}' ({gpus} GPU)",
        file=sys.stderr,
    )
    # Output job ID for Snakemake
    print(job_name)


if __name__ == "__main__":
    main()
