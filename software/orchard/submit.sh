#!/usr/bin/env bash
# Submit a single-GPU classifier training to the CMU ORCHARD cluster. Run from the barista directory
# on orchard-login-001. See docs/software/orchard.md.
#
#   software/orchard/submit.sh [options] <label> <wfs_dir | train.yml>
#
#   <label>            names the run: outputs go to $ORCHARD_PROJECT/runs/<label> and the bucket
#   <wfs_dir>          directory holding train.yml, with ../common.yml (as wfs_base in the workflow
#                      configs), e.g. coffea4bees/classifier/config/workflows/HH4b_Run3/FvT
#
#   --preempt          use the preempt (scavenger) partition; the job is requeued when it is killed
#   --common FILE      common.yml (default: <wfs_dir>/../common.yml)
#   --source SRC       xrootd (default) | gcs | none   where the inputs are staged from
#   --include RE       only stage root:// URLs matching RE      --exclude RE   skip those matching RE
#   --template STR     extra template keys, e.g. "FvT: /path/to/friend"
#   --args STR         extra arguments for src.classifier.task.main (e.g. '-setting ...')
#   --time T           wall time (default 12:00:00, the general-queue maximum)
#   --cpus N           CPUs (default 24)        --mem M   memory (default 200G)
#   --dry-run          print the sbatch command and exit
#   any other option is passed to sbatch
#
# Environment (defaults in parentheses):
#   ORCHARD_PROJECT (/project/community/$USER/barista)   run directories and the image; PURGED biweekly
#   ORCHARD_BUCKET  (gs://cmu-gpucloud-$USER/barista)    long-term mirror; "none" disables it
#   ORCHARD_IMAGE   ($ORCHARD_PROJECT/images/barista_classifier_latest.sqsh)   see setup_image.sh
#   ORCHARD_PARTITION / ORCHARD_QOS                      override the partition and QOS
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BARISTA="$(cd "${HERE}/../.." && pwd)"
export ORCHARD_PROJECT="${ORCHARD_PROJECT:-/project/community/${USER}/barista}"
export ORCHARD_BUCKET="${ORCHARD_BUCKET:-gs://cmu-gpucloud-${USER}/barista}"
[ "$ORCHARD_BUCKET" = none ] && ORCHARD_BUCKET=""
export ORCHARD_IMAGE="${ORCHARD_IMAGE:-${ORCHARD_PROJECT}/images/barista_classifier_latest.sqsh}"

PARTITION="${ORCHARD_PARTITION:-general}"; QOS="${ORCHARD_QOS-general_qos}"
TIME="12:00:00"; CPUS=24; MEM="200G"; DRY=0; COMMON=""; POS=(); EXTRA=()
export ORCHARD_SOURCE="${ORCHARD_SOURCE:-xrootd}" ORCHARD_INCLUDE="${ORCHARD_INCLUDE:-}" ORCHARD_EXCLUDE="${ORCHARD_EXCLUDE:-}"
export ORCHARD_TEMPLATE="${ORCHARD_TEMPLATE:-}" ORCHARD_EXTRA_ARGS="${ORCHARD_EXTRA_ARGS:-}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --preempt)  PARTITION="${ORCHARD_PARTITION:-preempt}"; QOS="${ORCHARD_QOS-}"; shift ;;
        --common)   COMMON="$2"; shift 2 ;;
        --source)   ORCHARD_SOURCE="$2"; shift 2 ;;
        --include)  ORCHARD_INCLUDE="$2"; shift 2 ;;
        --exclude)  ORCHARD_EXCLUDE="$2"; shift 2 ;;
        --template) ORCHARD_TEMPLATE="$2"; shift 2 ;;
        --args)     ORCHARD_EXTRA_ARGS="$2"; shift 2 ;;
        --time)     TIME="$2"; shift 2 ;;
        --cpus)     CPUS="$2"; shift 2 ;;
        --mem)      MEM="$2"; shift 2 ;;
        --dry-run)  DRY=1; shift ;;
        -h|--help)  sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*)         EXTRA+=("$1"); shift ;;
        *)          POS+=("$1"); shift ;;
    esac
done
[ "${#POS[@]}" -eq 2 ] || { echo "usage: $0 [options] <label> <wfs_dir | train.yml>  (-h for help)" >&2; exit 2; }
export ORCHARD_LABEL="${POS[0]}"
WF="${POS[1]}"; [ -d "$WF" ] && WF="${WF%/}/train.yml"
[ -f "$WF" ] || { echo "[orchard] no such workflow file: $WF" >&2; exit 2; }
COMMON="${COMMON:-$(dirname "$(dirname "$WF")")/common.yml}"
[ -f "$COMMON" ] || { echo "[orchard] no such common.yml: $COMMON (use --common)" >&2; exit 2; }
case "$ORCHARD_SOURCE" in xrootd|gcs|none) ;; *) echo "[orchard] --source must be xrootd, gcs or none" >&2; exit 2 ;; esac
if [ "$ORCHARD_SOURCE" = gcs ] && [ -z "$ORCHARD_BUCKET" ]; then echo "[orchard] --source gcs needs ORCHARD_BUCKET" >&2; exit 2; fi
if [ "$ORCHARD_SOURCE" != gcs ] && [ ! -f "${BARISTA}/proxy/x509_proxy" ] && [ -z "${X509_USER_PROXY:-}" ]; then
    echo "[orchard] WARNING: no grid proxy at proxy/x509_proxy; XRootD reads will fail" >&2
fi
[ "$DRY" = 1 ] || [ -e "$ORCHARD_IMAGE" ] || { echo "[orchard] image not found: $ORCHARD_IMAGE (run software/orchard/setup_image.sh)" >&2; exit 2; }

export ORCHARD_TRAIN="$WF" ORCHARD_COMMON="$COMMON" ORCHARD_BARISTA="$BARISTA"
export ORCHARD_RUN_DIR="${ORCHARD_PROJECT}/runs/${ORCHARD_LABEL}"
CMD=(sbatch --job-name="barista-${ORCHARD_LABEL}" --partition="$PARTITION" ${QOS:+--qos="$QOS"}
    --gres=gpu:1 --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME"
    --output="${ORCHARD_RUN_DIR}/logs/slurm-%j.out" --export=ALL
    ${EXTRA[@]+"${EXTRA[@]}"} "${HERE}/train.sbatch")
echo "[orchard] label=${ORCHARD_LABEL} partition=${PARTITION} qos=${QOS:-<default>} source=${ORCHARD_SOURCE}"
echo "[orchard] run dir ${ORCHARD_RUN_DIR}  bucket ${ORCHARD_BUCKET:-<none>}"
if [ "$DRY" = 1 ]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
mkdir -p "${ORCHARD_RUN_DIR}/logs"
"${CMD[@]}"
