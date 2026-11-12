#!/bin/bash
# Single-GPU classifier training or evaluation on NCSA Delta, submitted by the pull agent
# (software/delta/agent.py), which sets the DELTA_* variables and the account, partition and size.
# See docs/software/delta.md. Modelled on software/orchard/train.sbatch, without staging (inputs are read over
# XRootD, as roast trainings on ORCHARD do) and without a bucket mirror.
#
#   1. run the classifier image through apptainer + cvmfsexec (Delta has no /cvmfs)
#   2. train with a per-epoch checkpoint in the run directory (src/classifier/ml/training.py resumes
#      from it and skips models that are already saved)
#   3. 15 min before the time limit stop and requeue; the requeued job resumes from the last epoch.
#      An evaluation (DELTA_MAX_RESTARTS=0) is not checkpointed and is not requeued.
#
#SBATCH --signal=B:USR1@900
#SBATCH --requeue
#SBATCH --open-mode=append
set -uo pipefail

: "${DELTA_LABEL:?set by the agent}" "${DELTA_TRAIN:?}" "${DELTA_COMMON:?}" "${DELTA_BARISTA:?}" "${DELTA_RUN_DIR:?}"
: "${DELTA_TEMPLATE_FULL:?}"
DELTA_TASK="${DELTA_TASK:-train}"
DELTA_MAX_RESTARTS="${DELTA_MAX_RESTARTS:-12}"
DELTA_CLASSIFIER_CONFIG_PATHS="${DELTA_CLASSIFIER_CONFIG_PATHS:-coffea4bees}"
DELTA_IMAGE="${DELTA_IMAGE:-/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:classifier_latest}"

RUN="${DELTA_RUN_DIR}"
LOCAL="/tmp/${USER}/barista_${SLURM_JOB_ID}"          # node-local, gone after the job
CKPT="${RUN}/checkpoints"
LOGS="${RUN}/logs"
RESTART="${SLURM_RESTART_COUNT:-0}"
TAG="${SLURM_JOB_ID}.${RESTART}"
STEP_PID=""; BG_PIDS=()
APPTAINER="${DELTA_BARISTA}/software/bin/apptainer"  # cvmfsexec wrapper

log() { echo "[delta $(date '+%F %T')] $*"; }
mark() { echo "$1 $(date +%s)" >> "${LOGS}/timing_${TAG}.txt"; }

BINDS=( -B "${DELTA_BARISTA}:${DELTA_BARISTA}" -B "${RUN}:${RUN}" -B /tmp:/tmp
        -B /cvmfs/grid.cern.ch/etc/grid-security:/etc/grid-security )
for _d in "$(dirname "${DELTA_TRAIN}")" "$(dirname "${DELTA_COMMON}")"; do
    case "$_d/" in "${DELTA_BARISTA}/"*|/tmp/*) ;; *) BINDS+=( -B "${_d}:${_d}:ro" ) ;; esac
done

in_container() {  # run a command inside the classifier image, in the barista directory
    "$APPTAINER" exec --nv --cleanenv "${BINDS[@]}" --pwd "${DELTA_BARISTA}" \
        --env "X509_USER_PROXY=${X509_USER_PROXY:-}" \
        --env "USER=${USER}" --env "LOGNAME=${USER}" \
        --env X509_CERT_DIR=/etc/grid-security/certificates \
        "${DELTA_IMAGE}" \
        bash -c 'set +u; [ -f /entrypoint.sh ] && source /entrypoint.sh; set -u; ulimit -n 65536 2>/dev/null || true; export PYTHONUNBUFFERED=1; '"$1"
}

finish() {
    bash "${DELTA_BARISTA}/software/orchard/summarize.sh" "${RUN}" > "${RUN}/summary.txt" 2>&1
}

cleanup() {
    for p in "${BG_PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null; done
    rm -rf "${LOCAL}"
}

stop_step() {  # $1 = reason
    log "$1: stopping the ${DELTA_TASK} step"
    [ -n "$STEP_PID" ] && kill -TERM "$STEP_PID" 2>/dev/null && wait "$STEP_PID" 2>/dev/null
    mark interrupted
    finish
    cleanup
}
stop_and_requeue() {
    stop_step "$1"
    if [ "$RESTART" -lt "$DELTA_MAX_RESTARTS" ]; then
        log "requeueing (restart $((RESTART + 1)) of ${DELTA_MAX_RESTARTS}); the next run resumes from ${CKPT}"
        scontrol requeue "${SLURM_JOB_ID}" || log "ERROR: scontrol requeue failed"
    else
        log "ERROR: reached DELTA_MAX_RESTARTS=${DELTA_MAX_RESTARTS}"
    fi
    exit 1
}
trap 'stop_and_requeue "time limit in 15 min (SIGUSR1)"' USR1
# SIGTERM: scancel or preemption. Not requeued here (an scancel'ed job must stay cancelled); the
# driver requests a new attempt, which resumes from the checkpoint.
trap 'stop_step "preempted or cancelled (SIGTERM)"; exit 143' TERM

log "job ${SLURM_JOB_ID} restart ${RESTART} on $(hostname), partition ${SLURM_JOB_PARTITION:-?}, account ${SLURM_JOB_ACCOUNT:-?}, ${DELTA_TASK} ${DELTA_LABEL}"
mkdir -p "$CKPT" "$LOGS" "$LOCAL/eval_scratch" || exit 1
mark start
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/[delta] GPU: /'
cd "${DELTA_BARISTA}" || exit 1
for _p in "${X509_USER_PROXY:-}" "${HOME}/.roast/x509_proxy"; do
    if [ -n "$_p" ] && [ -s "$_p" ] && openssl x509 -checkend 3600 -noout -in "$_p" >/dev/null 2>&1; then
        install -m 600 "$_p" "${LOCAL}/x509_proxy" && export X509_USER_PROXY="${LOCAL}/x509_proxy"
        log "grid proxy from $_p"; break
    fi
done
[ -n "${X509_USER_PROXY:-}" ] && [ -f "${X509_USER_PROXY}" ] || log "WARNING: no grid proxy with an hour left; XRootD reads and EOS writes will fail"
mark staged

# GPU utilisation every 30 s (summarize.sh: mean utilisation, peak memory, power)
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw \
    --format=csv,nounits -l 30 >> "${LOGS}/gpu_${TAG}.csv" 2>/dev/null &
BG_PIDS+=($!)

log "${DELTA_TASK}: template {${DELTA_TEMPLATE_FULL}} ${DELTA_TRAIN} -from ${DELTA_COMMON}"
in_container "HCR_YIELDS_FILE='${RUN}/yields_loaded.json' \
    CLASSIFIER_CONFIG_PATHS='${DELTA_CLASSIFIER_CONFIG_PATHS}' \
    CLASSIFIER_CHECKPOINT_DIR='${CKPT}' \
    EVAL_SCRATCH='${LOCAL}/eval_scratch' \
    python -m src.classifier.task.main template '{${DELTA_TEMPLATE_FULL}}' '${DELTA_TRAIN}' -from '${DELTA_COMMON}' -flag debug" \
    >> "${RUN}/train.log" 2>&1 &
STEP_PID=$!; wait "$STEP_PID"; RC=$?; STEP_PID=""
mark trained

[ "$RC" -eq 0 ] && touch "${RUN}/train.done"
finish
cat "${RUN}/summary.txt"
cleanup
log "${DELTA_TASK} exit code ${RC}"
exit "$RC"
