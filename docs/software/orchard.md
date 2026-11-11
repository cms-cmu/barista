# Classifier training on the CMU ORCHARD cluster

ORCHARD is CMU's Google-cloud GPU cluster (Slurm; nodes are `a3-megagpu-8g`: 8 × H100 80 GB,
208 cores, 1.8 TB RAM, Ubuntu 24.04). The scripts in `software/orchard/` run a single-GPU
FvT / SvB / MvD training fit there, staged on the node-local SSD, checkpointed every epoch and
requeued across the 12-hour limit or a preemption.

Cluster documentation (CMU login): <https://cmu-enterprise.atlassian.net/wiki/spaces/ORCHARD/overview>.
Support: `ORCD-COMMUNITY@andrew.cmu.edu`.

!!! warning "Status"
    Written from the ORCHARD wiki on 2026-11-10 and not yet run on the cluster. The items under
    [To confirm on the cluster](#to-confirm-on-the-cluster) are assumptions until the first job.

## The cluster in one table

| | |
|---|---|
| Login | `orchard-login-001`, through Google IAP with your Andrew ID (no plain SSH) |
| Queues (Community) | `general`: `-p general -q general_qos`, 1 GPU per job, 12 h maximum. `preempt`: scavenger, jobs can be killed at any time |
| Containers | Enroot + Pyxis (`srun --container-image=...`); no Apptainer, no CVMFS |
| `/home` | 25 GB hard quota, on login and compute nodes |
| `/project/community/<andrew-id>` | 10 TB shared by all Community users, **emptied completely every two weeks** (calendar on the wiki) |
| `/tmp` | 6 TB SSD local to each compute node, fastest; not on the login node; gone after the job |
| `gs://cmu-gpucloud-<andrew-id>` | personal GCS bucket, the only long-term storage; billed to your Oracle string per GB and per transaction; no backups |
| Maintenance | Wednesdays 10:00–14:00 when announced: do not expect jobs to survive it |

## Login

Install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) on your laptop
(`brew install --cask google-cloud-sdk`), then

```bash
gcloud auth login                              # browser, Andrew ID (not a personal Google account)
gcloud config set project cmu-gpu-cloud
gcloud config set compute/zone us-central1-c
gcloud compute ssh orchard-login-001 --tunnel-through-iap
gcloud compute scp --tunnel-through-iap file.txt orchard-login-001:      # copy a file
```

Do not touch `~/.ssh/authorized_keys` or the generated key on the cluster: node-to-node access
(and therefore every job) depends on them.

## One-time setup on the login node

```bash
# 1. bucket credentials (prints a URL to open in a browser; paste the code back)
mkdir -p ~/.config/gcloud && chmod 700 ~/.config/gcloud
gcloud auth login --update-adc
gcloud storage ls gs://cmu-gpucloud-$USER

# 2. code. $HOME survives the purge, /project/community does not. The two .git directories
#    are ~0.7 GB; keep outputs out of the checkout.
cd ~ && git clone https://gitlab.cern.ch/cms-cmu/barista.git && cd barista
git clone https://gitlab.cern.ch/cms-cmu/coffea4bees.git

# 3. classifier image -> /project/community/$USER/barista/images/*.sqsh (+ a copy in the bucket)
software/orchard/setup_image.sh

# 4. grid proxy, only needed to stage from XRootD. There are no grid tools on the host: make the
#    proxy where you have your certificate (LPC, lxplus, laptop) and copy it over.
voms-proxy-init -voms cms -rfc --valid 168:00 -out x509_proxy                      # elsewhere
gcloud compute scp --tunnel-through-iap x509_proxy orchard-login-001:barista/proxy/x509_proxy
```

`setup_image.sh` imports `gitlab-registry.cern.ch/cms-cmu/barista:classifier_latest`, the same
image used on LPC, falcon and lxplus, so the software environment is pinned to
`software/dockerfiles/ml/env_ml.yml`. After a purge, rerun it: it restores the image from the
bucket instead of importing again. `setup_image.sh --update` re-imports a newer image.

## Submitting a training

From the barista directory on the login node:

```bash
# general queue: 12 h, requeues itself and resumes if the fit is not finished
software/orchard/submit.sh fvt_run3 coffea4bees/classifier/config/workflows/HH4b_Run3/FvT

# scavenger queue: may be killed at any time, resumes from the last completed epoch
software/orchard/submit.sh --preempt fvt_run3 coffea4bees/classifier/config/workflows/HH4b_Run3/FvT

software/orchard/submit.sh --dry-run ...     # print the sbatch command
software/orchard/submit.sh -h                # all options
```

The second argument is a `wfs_base` directory (holding `train.yml`, with `../common.yml`) or a
`train.yml` path; the label names the run. Templates that need more than `model` take
`--template "FvT: <path>, ..."`. The workflow files are used unchanged: the training command is
the one in `src/classifier/workflow/Snakefile` (rule `train`).

What the job does (`software/orchard/train.sbatch`):

1. **Restore.** If the run directory is missing but the bucket has it (after a purge), copy it back.
2. **Stage.** `stage_data.py` collects every `root://` URL in the metadata YAML / friend JSON files
   that `train.yml` and `common.yml` name, copies those files to `/tmp/$USER/barista_data/<host>/<path>`
   and writes copies of the configs that point there. The repository is not modified.
3. **Train** inside the classifier image with `CLASSIFIER_CHECKPOINT_DIR=<run dir>/checkpoints`.
4. **Mirror** the run directory to the bucket every 30 minutes and at exit.
5. **Requeue.** Fifteen minutes before the time limit (`SIGUSR1`), or on preemption (`SIGTERM`), the
   training step is stopped, the run directory mirrored, and the job requeued (at most
   `ORCHARD_MAX_RESTARTS`, default 6). The requeued job stages again and resumes.

Resuming is the training code's own: each stage writes `checkpoint_<model>_<stage>.pt` at the end
of every epoch and continues from it, and a model whose final file exists is skipped. A restart
therefore loses at most the epoch in progress, plus the staging time.

### Where the inputs come from

| `--source` | Use |
|---|---|
| `xrootd` (default) | `xrdcp` from the URLs in the metadata (FNAL EOS, CMU Tier-3, ...), 16 parallel streams. Needs `proxy/x509_proxy`. |
| `gcs` | `gcloud storage rsync` from `$ORCHARD_BUCKET/data/<host>/<path>`. No proxy, and fastest from a compute node. |
| `none` | no staging: read over XRootD during the training, as on the other clusters |

To fill the bucket mirror once, so that later jobs (and preempt restarts) do not depend on the grid:

```bash
srun -p general -q general_qos --cpus-per-task=16 --mem=32G --time=4:00:00 --pty bash   # /tmp is on compute nodes only
S=/tmp/$USER/stage; D=/tmp/$USER/barista_data; W=coffea4bees/classifier/config/workflows/HH4b_Run3
python3 software/orchard/stage_data.py plan  --stage $S --data $D $W/FvT/train.yml $W/common.yml
srun --container-image=/project/community/$USER/barista/images/barista_classifier_latest.sqsh \
     --container-mounts=$PWD:$PWD,/tmp:/tmp --container-workdir=$PWD --no-container-entrypoint \
     bash -c "source /entrypoint.sh; X509_USER_PROXY=$PWD/proxy/x509_proxy python3 software/orchard/stage_data.py fetch --stage $S --data $D"
python3 software/orchard/stage_data.py push  --stage $S --data $D --gcs-prefix gs://cmu-gpucloud-$USER/barista/data
```

`--include REGEX` / `--exclude REGEX` restrict what is staged (for example one year for a smoke
test: `--include '2022_EE'`); URLs that are not selected stay remote. The bucket is charged for
storage and per transaction, so stage whole datasets with one `rsync` (as the script does) and
do not rsync it in a tight loop.

## Outputs

```
/project/community/$USER/barista/runs/<label>/      # PURGED every two weeks
    model/            final models + result.json (the IO output of the training)
    checkpoints/      per-epoch checkpoints, removed by the training when a model is final
    train.log         training log, appended across restarts
    yields_loaded.json
    train.done        written when the training exits 0
    logs/             slurm-<job>.out, gpu_<job>.<restart>.csv, timing_*.txt, summary_*.txt, manifest_*.txt
gs://cmu-gpucloud-$USER/barista/runs/<label>/        # mirror of the above: the copy to rely on
gs://cmu-gpucloud-$USER/barista/images/              # classifier image
gs://cmu-gpucloud-$USER/barista/data/                # optional mirror of the input ROOT files
```

Treat `/project/community` as scratch: anything not in the bucket (or copied off the cluster) is
lost at the next purge. To take a model elsewhere:

```bash
gcloud storage rsync --recursive gs://cmu-gpucloud-$USER/barista/runs/<label>/model ./<label>   # any machine, after gcloud auth login
```

Evaluation (friend trees to EOS) is not set up on ORCHARD; copy the model to EOS and evaluate on
the usual clusters.

## Monitoring and timing

```bash
squeue -u $USER
tail -f /project/community/$USER/barista/runs/<label>/train.log
software/orchard/summarize.sh /project/community/$USER/barista/runs/<label>
```

`summarize.sh` reports staging and training wall time summed over restarts (= GPU-hours, one GPU
per job), mean GPU utilisation and peak GPU memory from the 30-second `nvidia-smi` log, and the
per-stage `run N epochs in ...` lines of the training log.

## Settings

Environment variables read by `submit.sh` (see `-h`) and passed to the job:

| Variable | Default | |
|---|---|---|
| `ORCHARD_PROJECT` | `/project/community/$USER/barista` | run directories and image |
| `ORCHARD_BUCKET` | `gs://cmu-gpucloud-$USER/barista` | long-term mirror; `none` disables it |
| `ORCHARD_IMAGE` | `$ORCHARD_PROJECT/images/barista_classifier_latest.sqsh` | |
| `ORCHARD_PARTITION`, `ORCHARD_QOS` | `general`, `general_qos` (`preempt`, none with `--preempt`) | |
| `ORCHARD_MAX_RESTARTS` | 6 | requeues before giving up |
| `ORCHARD_SYNC_MINUTES` | 30 | bucket mirror period |
| `ORCHARD_STAGE_JOBS` | 16 | parallel `xrdcp` streams |

No credentials belong in the repository: the gcloud login lives in `~/.config/gcloud` (mode 700)
and the grid proxy in `proxy/` (git-ignored).

## To confirm on the cluster

- the QOS (if any) that the `preempt` partition wants, and whether preempted jobs are requeued by
  Slurm itself (`scontrol show partition preempt`, `sacctmgr show qos`);
- that `scontrol requeue` of a job from its own batch script is allowed;
- CPU and memory limits per GPU on `general` (the defaults here ask for 24 cores and 200 GB, about
  one eighth of a node);
- that `enroot import` of the 10+ GB classifier image works on the login node (otherwise run
  `setup_image.sh` in an `srun` job, as its header shows), and that the CERN registry lets it pull
  anonymously;
- that compute nodes reach `cmseos.fnal.gov:1094` (XRootD staging), and `gitlab.cern.ch`.
