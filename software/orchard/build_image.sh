#!/usr/bin/env bash
# Build the classifier image on ORCHARD from software/dockerfiles/ml/env_ml.yml, for when the CERN
# registry cannot be reached from the cluster (setup_image.sh imports the released image, which is
# the preferred route). Same conda environment ("hep") and /entrypoint.sh as Dockerfile.ml; without
# the VOMS clients and the rucio configuration, which a training does not use (the grid proxy is
# made elsewhere and copied over, e.g. with `roast proxy`).
#
# Needs the node-local SSD, so run it as a job from the barista directory:
#   sbatch -p general -q general_qos --cpus-per-task=16 --mem=64G --time=2:00:00 \
#          -o build_image-%j.out software/orchard/build_image.sh
set -euo pipefail

BARISTA="${ORCHARD_BARISTA:-${SLURM_SUBMIT_DIR:-$PWD}}"
ORCHARD_PROJECT="${ORCHARD_PROJECT:-/project/community/${USER}/barista}"
ORCHARD_BUCKET="${ORCHARD_BUCKET:-gs://cmu-gpucloud-${USER}/barista}"
[ "$ORCHARD_BUCKET" = none ] && ORCHARD_BUCKET=""
IMAGE="${ORCHARD_IMAGE:-${ORCHARD_PROJECT}/images/barista_classifier_latest.sqsh}"
BASE="${ORCHARD_BASE_IMAGE:-docker://condaforge/miniforge3:latest}"
ENV_YML="${BARISTA}/software/dockerfiles/ml/env_ml.yml"
[ -f "$ENV_YML" ] || { echo "[orchard] $ENV_YML not found: run from the barista directory" >&2; exit 2; }
[ -d "/tmp/${USER}" ] || mkdir -p "/tmp/${USER}"
df --output=avail -BG /tmp | tail -1 | grep -q -E '^ *[0-9]{3,}G' || { echo "[orchard] /tmp has < 100 GB: run this in a job, not on the login node" >&2; exit 2; }

W="/tmp/${USER}/enroot_build"
export ENROOT_CACHE_PATH="$W/cache" ENROOT_DATA_PATH="$W/data" ENROOT_TEMP_PATH="$W/tmp" ENROOT_RUNTIME_PATH="$W/run"
rm -rf "$W"; mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH" "$ENROOT_TEMP_PATH" "$ENROOT_RUNTIME_PATH"
trap 'chmod -R u+w "$W" 2>/dev/null; rm -rf "$W"' EXIT

echo "[orchard] base image $BASE"
enroot import -o "$W/base.sqsh" "$BASE"
enroot create --name barista_build "$W/base.sqsh"
enroot start --root --rw --mount "${ENV_YML}:/tmp/env.yml" barista_build bash -c '
    set -euo pipefail
    export CONDA_OVERRIDE_CUDA=12.6
    mamba env create -y -f /tmp/env.yml
    mamba clean --all --yes
    printf "%s\n" "eval \"\$(/opt/conda/bin/conda shell.bash hook)\"" "conda activate hep" > /entrypoint.sh
    . /entrypoint.sh
    python -c "import torch, uproot, awkward, XRootD, fsspec_xrootd; print(\"torch\", torch.__version__, \"cuda build\", torch.version.cuda)"
'
mkdir -p "$(dirname "$IMAGE")"
rm -f "${IMAGE}.new"
enroot export --output "${IMAGE}.new" barista_build
mv -f "${IMAGE}.new" "$IMAGE"
ls -lh "$IMAGE"
if [ -n "$ORCHARD_BUCKET" ]; then
    gcloud storage cp "$IMAGE" "${ORCHARD_BUCKET%/}/images/$(basename "$IMAGE")"
    echo "[orchard] copied to ${ORCHARD_BUCKET%/}/images/"
fi
