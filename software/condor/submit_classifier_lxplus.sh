#!/usr/bin/env bash
# Submit a command as an HTCondor GPU job on CERN lxplus, executed inside the classifier container
# through MY.SingularityImage (the job runs on AFS with the user's Kerberos credential).
#
# Called by `./run_container classifier <cmd>` on lxplus; can also be used directly:
#   software/condor/submit_classifier_lxplus.sh --image <cvmfs image> --workdir <barista dir> \
#       [--proxy <x509 proxy>] [--interactive] [--dry-run] --cmd "<shell command>"
#
# Environment knobs (defaults in parentheses):
#   CONDOR_JOB_FLAVOUR   (tomorrow)  espresso|microcentury|longlunch|workday|tomorrow|testmatch|nextweek
#   CONDOR_REQUEST_GPUS  (1)         number of GPUs; 0 submits a CPU-only job
#   CONDOR_REQUEST_CPUS  (4)
#   CONDOR_REQUEST_MEMORY (32GB)      CERN raises RequestCpus to memory/3GB (64GB -> 22 cores, which no
#                                    single-GPU A100 machine [16 cores, 118GB] can offer); stay <= 48GB
#   CONDOR_GOOD_GPUS     (0)         1 -> only A100/V100/H100/H200 (MIG slices are always excluded)
#   CONDOR_LOG_DIR       (<workdir>/condor_logs/classifier)  must NOT be under /eos: the standard
#                                    schedds refuse /eos paths for executable/log/output/error (use AFS)
#   CONDOR_EXTRA         ()          extra submit-file lines, separated by ';'
#
# Modelled on ttHbb_SPANet/scripts/submit_to_condor.py and jobs/classification.sh.
set -euo pipefail

IMAGE=""; WORKDIR="$PWD"; PROXY=""; INTERACTIVE=0; DRY_RUN=0; CMD=""
usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --image)       IMAGE="$2"; shift 2 ;;
        --workdir)     WORKDIR="$2"; shift 2 ;;
        --proxy)       PROXY="$2"; shift 2 ;;
        --cmd)         CMD="$2"; shift 2 ;;
        --interactive) INTERACTIVE=1; shift ;;
        --dry-run)     DRY_RUN=1; shift ;;
        -h|--help)     usage; exit 0 ;;
        *) echo "[submit_classifier_lxplus] unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done
[ -n "$IMAGE" ] || { echo "[submit_classifier_lxplus] --image is required" >&2; exit 2; }
if [ -z "$CMD" ] && [ "$INTERACTIVE" != 1 ]; then
    echo "[submit_classifier_lxplus] --cmd is required unless --interactive" >&2; exit 2
fi
case "$IMAGE" in
    /cvmfs/*) ;;
    *) echo "[submit_classifier_lxplus] WARNING: image '$IMAGE' is not under /cvmfs/unpacked.cern.ch; CERN HTCondor may not be able to start it" >&2 ;;
esac

JOB_FLAVOUR="${CONDOR_JOB_FLAVOUR:-tomorrow}"
REQUEST_GPUS="${CONDOR_REQUEST_GPUS:-1}"
REQUEST_CPUS="${CONDOR_REQUEST_CPUS:-4}"
REQUEST_MEMORY="${CONDOR_REQUEST_MEMORY:-32GB}"
GOOD_GPUS="${CONDOR_GOOD_GPUS:-0}"
LOG_DIR="${CONDOR_LOG_DIR:-${WORKDIR}/condor_logs/classifier}"
case "$LOG_DIR" in /eos/*)
    echo "[submit_classifier_lxplus] ERROR: CONDOR_LOG_DIR=$LOG_DIR is on /eos; the standard CERN schedds refuse /eos paths" >&2
    echo "[submit_classifier_lxplus]        for executable/log/output/error. Set CONDOR_LOG_DIR to an AFS directory." >&2
    exit 2 ;;
esac
JOB_DIR="${LOG_DIR}/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$JOB_DIR"

# --- job wrapper (runs inside the classifier image on the worker node) ---------------------------
cat > "${JOB_DIR}/job.sh" <<JOBEOF
#!/bin/bash
echo "[job] host=\$(hostname) date=\$(date) cluster=\${_CONDOR_JOB_AD:-}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,memory.total --format=csv || true
cd "${WORKDIR}" || { echo "[job] cannot cd to ${WORKDIR} (AFS token missing?)"; exit 1; }
set +u; { [ -f /entrypoint.sh ] && source /entrypoint.sh; } 2>/dev/null || true; set -u
export PYTHONUNBUFFERED=1
${PROXY:+export X509_USER_PROXY=\"${PROXY}\"}
ulimit -n 65536 2>/dev/null || true
set -eo pipefail
${CMD}
JOBEOF
chmod +x "${JOB_DIR}/job.sh"

# --- GPU requirements -------------------------------------------------------------------------------
REQUIREMENTS=""
if [ "$REQUEST_GPUS" -gt 0 ]; then
    if [ "$GOOD_GPUS" = 1 ]; then
        REQUIREMENTS='(regexp("A100", TARGET.GPUs_DeviceName) || regexp("V100", TARGET.GPUs_DeviceName) || regexp("H100", TARGET.GPUs_DeviceName) || regexp("H200", TARGET.GPUs_DeviceName)) && !regexp("MIG", TARGET.GPUs_DeviceName)'
    else
        # MIG slices expose a fraction of a GPU and break some CUDA setups
        REQUIREMENTS='!regexp("MIG", TARGET.GPUs_DeviceName)'
    fi
fi

# --- submit description ----------------------------------------------------------------------------
{
    echo "universe = vanilla"
    echo "executable = ${JOB_DIR}/job.sh"
    echo "output = ${JOB_DIR}/job.\$(ClusterId).\$(ProcId).out"
    echo "error = ${JOB_DIR}/job.\$(ClusterId).\$(ProcId).err"
    echo "log = ${JOB_DIR}/job.\$(ClusterId).log"
    echo "MY.SendCredential = True"
    echo "MY.SingularityImage = \"${IMAGE}\""
    echo "+JobFlavour = \"${JOB_FLAVOUR}\""
    echo "request_cpus = ${REQUEST_CPUS}"
    echo "request_gpus = ${REQUEST_GPUS}"
    echo "request_memory = ${REQUEST_MEMORY}"
    [ -n "$REQUIREMENTS" ] && echo "requirements = ${REQUIREMENTS}"
    echo "should_transfer_files = YES"
    echo "when_to_transfer_output = ON_EXIT"
    echo "transfer_output_files = \"\""
    if [ -n "${CONDOR_EXTRA:-}" ]; then
        IFS=';' read -ra _extra <<< "${CONDOR_EXTRA}"
        for line in "${_extra[@]}"; do [ -n "$line" ] && echo "$line"; done
    fi
    echo "queue"
} > "${JOB_DIR}/job.sub"

echo "[submit_classifier_lxplus] job directory: ${JOB_DIR}"
echo "[submit_classifier_lxplus] flavour=${JOB_FLAVOUR} gpus=${REQUEST_GPUS} cpus=${REQUEST_CPUS} memory=${REQUEST_MEMORY}"
if [ "$DRY_RUN" = 1 ]; then
    echo "--- job.sub ---"; cat "${JOB_DIR}/job.sub"; echo "--- job.sh ---"; cat "${JOB_DIR}/job.sh"
    exit 0
fi

if [ "$INTERACTIVE" = 1 ]; then
    echo "[submit_classifier_lxplus] requesting an interactive GPU session (condor_submit -interactive)..."
    exec condor_submit -interactive "${JOB_DIR}/job.sub"
fi

OUT=$(condor_submit "${JOB_DIR}/job.sub")
echo "$OUT"
CLUSTER_ID=$(echo "$OUT" | sed -n 's/.*submitted to cluster \([0-9]*\).*/\1/p')
if [ -n "$CLUSTER_ID" ]; then
    echo "[submit_classifier_lxplus] submitted cluster ${CLUSTER_ID}"
    echo "  monitor:  condor_q ${CLUSTER_ID}"
    echo "  stdout:   ${JOB_DIR}/job.${CLUSTER_ID}.0.out"
    echo "  stderr:   ${JOB_DIR}/job.${CLUSTER_ID}.0.err"
    echo "${CLUSTER_ID}" > "${JOB_DIR}/cluster_id"
fi
