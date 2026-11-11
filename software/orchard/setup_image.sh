#!/usr/bin/env bash
# Put the classifier image where ORCHARD jobs can use it (Enroot squashfs under /project/community).
# /project/community is purged every two weeks, so the image is also kept in the bucket:
#
#   image present            -> nothing to do
#   image in the bucket      -> copy it back                       (seconds to minutes)
#   otherwise, or --update   -> enroot import from the registry, then upload to the bucket
#
# Run on orchard-login-001, or inside an interactive job if the login node is too small for the import:
#   srun -p general -q general_qos --cpus-per-task=16 --mem=64G --time=1:00:00 software/orchard/setup_image.sh
set -euo pipefail

ORCHARD_PROJECT="${ORCHARD_PROJECT:-/project/community/${USER}/barista}"
ORCHARD_BUCKET="${ORCHARD_BUCKET:-gs://cmu-gpucloud-${USER}/barista}"
[ "$ORCHARD_BUCKET" = none ] && ORCHARD_BUCKET=""
# enroot spelling of docker://gitlab-registry.cern.ch/cms-cmu/barista:classifier_latest
SOURCE_IMAGE="${ORCHARD_SOURCE_IMAGE:-docker://gitlab-registry.cern.ch#cms-cmu/barista:classifier_latest}"
IMAGE="${ORCHARD_IMAGE:-${ORCHARD_PROJECT}/images/barista_classifier_latest.sqsh}"
REMOTE="${ORCHARD_BUCKET:+${ORCHARD_BUCKET%/}/images/$(basename "$IMAGE")}"
UPDATE=0; [ "${1:-}" = --update ] && UPDATE=1

mkdir -p "$(dirname "$IMAGE")"
if [ "$UPDATE" = 0 ] && [ -s "$IMAGE" ]; then
    echo "[orchard] image present: $IMAGE"; exit 0
fi
if [ "$UPDATE" = 0 ] && [ -n "$REMOTE" ] && gcloud storage ls "$REMOTE" > /dev/null 2>&1; then
    echo "[orchard] restoring $IMAGE from $REMOTE"
    gcloud storage cp "$REMOTE" "$IMAGE"; exit 0
fi

# the layer cache and the unpacking scratch do not fit in the 25 GB home quota
SCRATCH="$( [ -w /tmp ] && echo "/tmp/${USER}/enroot" || echo "${ORCHARD_PROJECT}/enroot" )"
export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-${SCRATCH}/cache}" ENROOT_TEMP_PATH="${ENROOT_TEMP_PATH:-${SCRATCH}/tmp}"
mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_TEMP_PATH"
echo "[orchard] importing $SOURCE_IMAGE -> $IMAGE"
rm -f "${IMAGE}.new"
enroot import -o "${IMAGE}.new" "$SOURCE_IMAGE"
mv -f "${IMAGE}.new" "$IMAGE"
rm -rf "$SCRATCH"
if [ -n "$REMOTE" ]; then
    echo "[orchard] uploading to $REMOTE"
    gcloud storage cp "$IMAGE" "$REMOTE"
fi
ls -lh "$IMAGE"
