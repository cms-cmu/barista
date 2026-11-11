#!/bin/bash
# Single-GPU classifier training (FvT, SvB, MvD, ...) on the CMU ORCHARD cluster.
# Submit with software/orchard/submit.sh, which sets the ORCHARD_* variables and the sbatch options;
# see docs/software/orchard.md.
#
# The batch script itself runs on the host (it needs gcloud); the staging and the training run inside
# the classifier image as Pyxis/Enroot job steps (srun --container-image).
#
#   1. restore the run directory from the bucket if /project/community was purged
#   2. stage the input ROOT files to the node-local SSD (/tmp) and rewrite the configs to point there
#   3. train with a per-epoch checkpoint in the run directory (src/classifier/ml/training.py resumes
#      from it, and skips models that are already saved)
#   4. mirror the run directory to the bucket every ORCHARD_SYNC_MINUTES and at exit
#   5. shortly before the time limit, or when preempted, stop the training and requeue the job; the
#      requeued job resumes from the last completed epoch
#
#SBATCH --job-name=barista-train
#SBATCH --partition=general
#SBATCH --qos=general_qos
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --requeue
#SBATCH --open-mode=append
set -uo pipefail

: "${ORCHARD_LABEL:?set by submit.sh}" "${ORCHARD_TRAIN:?}" "${ORCHARD_COMMON:?}" "${ORCHARD_BARISTA:?}"
: "${ORCHARD_RUN_DIR:?}" "${ORCHARD_IMAGE:?}"
ORCHARD_BUCKET="${ORCHARD_BUCKET:-}"                  # gs://bucket/prefix, empty = no bucket mirror
ORCHARD_SOURCE="${ORCHARD_SOURCE:-xrootd}"            # xrootd | gcs | none (read the inputs remotely)
ORCHARD_INCLUDE="${ORCHARD_INCLUDE:-}"                # regex: only stage matching root:// URLs
ORCHARD_EXCLUDE="${ORCHARD_EXCLUDE:-}"
ORCHARD_TEMPLATE="${ORCHARD_TEMPLATE:-}"              # extra template keys, e.g. "FvT: /path, x: y"
ORCHARD_EXTRA_ARGS="${ORCHARD_EXTRA_ARGS:-}"          # appended to the training command line
ORCHARD_MAX_RESTARTS="${ORCHARD_MAX_RESTARTS:-6}"
ORCHARD_SYNC_MINUTES="${ORCHARD_SYNC_MINUTES:-30}"
ORCHARD_STAGE_JOBS="${ORCHARD_STAGE_JOBS:-16}"
ORCHARD_CLASSIFIER_CONFIG_PATHS="${ORCHARD_CLASSIFIER_CONFIG_PATHS:-coffea4bees}"

HERE="${ORCHARD_BARISTA}/software/orchard"
LOCAL="/tmp/${USER}/barista_${SLURM_JOB_ID}"          # node-local SSD, gone after the job
DATA="/tmp/${USER}/barista_data"                      # shared by this user's jobs on the node
MODEL="${ORCHARD_RUN_DIR}/model"
CKPT="${ORCHARD_RUN_DIR}/checkpoints"
LOGS="${ORCHARD_RUN_DIR}/logs"
RESTART="${SLURM_RESTART_COUNT:-0}"
TAG="${SLURM_JOB_ID}.${RESTART}"
REMOTE="${ORCHARD_BUCKET:+${ORCHARD_BUCKET%/}/runs/${ORCHARD_LABEL}}"
STEP_PID=""; BG_PIDS=()

log() { echo "[orchard $(date '+%F %T')] $*"; }
mark() { echo "$1 $(date +%s)" >> "${LOGS}/timing_${TAG}.txt"; }

in_container() {  # run a command inside the classifier image, in the barista directory
    srun --ntasks=1 --container-image="${ORCHARD_IMAGE}" --no-container-entrypoint \
        --container-mounts="${ORCHARD_BARISTA}:${ORCHARD_BARISTA},${ORCHARD_RUN_DIR}:${ORCHARD_RUN_DIR},/tmp:/tmp" \
        --container-workdir="${ORCHARD_BARISTA}" \
        bash -c 'set +u; [ -f /entrypoint.sh ] && source /entrypoint.sh; set -u; ulimit -n 65536 2>/dev/null || true; export PYTHONUNBUFFERED=1; '"$1"
}

sync_out() {  # mirror the run directory (model, checkpoints, logs) to the bucket
    [ -n "$REMOTE" ] || return 0
    gcloud storage rsync --recursive "${ORCHARD_RUN_DIR}" "${REMOTE}" > "${LOGS}/sync_${TAG}.log" 2>&1 \
        || { log "WARNING: bucket sync failed, see ${LOGS}/sync_${TAG}.log"; return 1; }
}

cleanup() {
    for p in "${BG_PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null; done
    rm -rf "${LOCAL}"
}

stop_and_requeue() {  # $1 = reason
    log "$1: stopping the training step"
    [ -n "$STEP_PID" ] && kill -TERM "$STEP_PID" 2>/dev/null && wait "$STEP_PID" 2>/dev/null
    mark interrupted
    sync_out
    cleanup
    if [ "$RESTART" -lt "$ORCHARD_MAX_RESTARTS" ]; then
        log "requeueing (restart $((RESTART + 1)) of ${ORCHARD_MAX_RESTARTS}); the next run resumes from ${CKPT}"
        scontrol requeue "${SLURM_JOB_ID}" || log "ERROR: scontrol requeue failed; resubmit by hand with submit.sh"
    else
        log "ERROR: reached ORCHARD_MAX_RESTARTS=${ORCHARD_MAX_RESTARTS}; resubmit by hand with submit.sh"
    fi
    exit 1
}
trap 'stop_and_requeue "time limit in 15 min (SIGUSR1)"' USR1
trap 'stop_and_requeue "preempted or cancelled (SIGTERM)"' TERM

# ── 1. directories, restore after a purge ────────────────────────────────────────────────────────
log "job ${SLURM_JOB_ID} restart ${RESTART} on $(hostname), partition ${SLURM_JOB_PARTITION:-?}, label ${ORCHARD_LABEL}"
if [ -n "$REMOTE" ] && [ ! -d "$CKPT" ] && [ ! -d "$MODEL" ] \
    && gcloud storage ls "${REMOTE}/" > /dev/null 2>&1; then
    log "run directory is empty but ${REMOTE} exists (purged?): restoring"
    mkdir -p "${ORCHARD_RUN_DIR}"
    gcloud storage rsync --recursive "${REMOTE}" "${ORCHARD_RUN_DIR}" || log "WARNING: restore failed"
fi
mkdir -p "$MODEL" "$CKPT" "$LOGS" "$LOCAL" "$DATA" || exit 1
mark start
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/[orchard] GPU: /'
cd "${ORCHARD_BARISTA}" || exit 1
[ -z "${X509_USER_PROXY:-}" ] && [ -f proxy/x509_proxy ] && export X509_USER_PROXY="${PWD}/proxy/x509_proxy"

# ── 2. stage the inputs ──────────────────────────────────────────────────────────────────────────
TRAIN="${ORCHARD_TRAIN}"; COMMON="${ORCHARD_COMMON}"
if [ "$ORCHARD_SOURCE" != none ]; then
    mapfile -t STAGED < <(python3 "${HERE}/stage_data.py" plan --stage "$LOCAL" --data "$DATA" --root "$PWD" \
        ${ORCHARD_INCLUDE:+--include "$ORCHARD_INCLUDE"} ${ORCHARD_EXCLUDE:+--exclude "$ORCHARD_EXCLUDE"} \
        "$TRAIN" "$COMMON")
    [ "${#STAGED[@]}" -eq 2 ] || { log "ERROR: staging plan failed"; cleanup; exit 1; }
    TRAIN="${STAGED[0]}"; COMMON="${STAGED[1]}"
    FETCH="python3 ${HERE}/stage_data.py fetch --stage ${LOCAL} --data ${DATA} --source ${ORCHARD_SOURCE} --jobs ${ORCHARD_STAGE_JOBS}"
    if [ "$ORCHARD_SOURCE" = gcs ]; then
        $FETCH --gcs-prefix "${ORCHARD_BUCKET%/}/data" &
    else
        in_container "$FETCH" &                      # xrdcp and the grid CA certificates are in the image
    fi
    STEP_PID=$!; wait "$STEP_PID"; RC=$?; STEP_PID=""
    [ "$RC" -eq 0 ] || { log "ERROR: staging failed (rc=${RC})"; sync_out; cleanup; exit 1; }
    cp "${LOCAL}/manifest.txt" "${LOGS}/manifest_${TAG}.txt"
    log "inputs staged: $(du -sh "$DATA" | cut -f1) in ${DATA}"
fi
mark staged

# ── 3. monitors: GPU utilisation every 30 s, bucket mirror every ORCHARD_SYNC_MINUTES ────────────
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw \
    --format=csv,nounits -l 30 >> "${LOGS}/gpu_${TAG}.csv" 2>/dev/null &
BG_PIDS+=($!)
if [ -n "$REMOTE" ]; then
    ( while sleep $((ORCHARD_SYNC_MINUTES * 60)); do sync_out; done ) &
    BG_PIDS+=($!)
fi

# ── 4. train ─────────────────────────────────────────────────────────────────────────────────────
TEMPLATE="model: ${MODEL}${ORCHARD_TEMPLATE:+, ${ORCHARD_TEMPLATE}}"
log "training: template {${TEMPLATE}} ${TRAIN} -from ${COMMON} ${ORCHARD_EXTRA_ARGS}"
in_container "HCR_YIELDS_FILE='${ORCHARD_RUN_DIR}/yields_loaded.json' \
    CLASSIFIER_CONFIG_PATHS='${ORCHARD_CLASSIFIER_CONFIG_PATHS}' \
    CLASSIFIER_CHECKPOINT_DIR='${CKPT}' \
    python -m src.classifier.task.main template '{${TEMPLATE}}' '${TRAIN}' -from '${COMMON}' -flag debug ${ORCHARD_EXTRA_ARGS}" \
    >> "${ORCHARD_RUN_DIR}/train.log" 2>&1 &
STEP_PID=$!; wait "$STEP_PID"; RC=$?; STEP_PID=""
mark trained

# ── 5. save and report ───────────────────────────────────────────────────────────────────────────
[ "$RC" -eq 0 ] && touch "${ORCHARD_RUN_DIR}/train.done"
bash "${HERE}/summarize.sh" "${ORCHARD_RUN_DIR}" | tee "${LOGS}/summary_${TAG}.txt"
sync_out && log "run directory mirrored to ${REMOTE:-<no bucket>}"
cleanup
log "training exit code ${RC}"
exit "$RC"
