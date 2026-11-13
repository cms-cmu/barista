# Classifier training on NCSA Delta

NCSA Delta (ACCESS) has 4×A100-40GB nodes (`gpuA100x4`, 100 nodes), and also A40, A100-80GB and H200
nodes. Barista uses it in two ways:

| mode | what runs on Delta | needs |
|------|--------------------|-------|
| **`--train-on delta --eval-on delta`** (this page) | only the classifier trainings and evaluations | a `-delta-gpu` account; the pull agent |
| `roast new --delta` (whole step on Delta) | all of phases C, D and BKG_C, CPU rules included | a `-delta-gpu` **and** a `-delta-cpu` account, a roast checkout on Delta |

The first mode is the ORCHARD model ([orchard.md](orchard.md)): the step's Snakemake driver stays on
cmslpc (or falcon), and `src/classifier/workflow/Snakefile` swaps `train` / `evaluate` /
`evaluate_split` for `train_delta` / `evaluate_delta` / `evaluate_split_delta`. Those are local rules
that ship the job to Delta and wait for it. Inputs are read over XRootD, and the models, friend trees and
`result.json` go to the template's EOS paths, so `analyze`, plots, galleries and every later rule run on
the driver unchanged. No CPU allocation is used on Delta.

## How the driver reaches Delta: an EOS spool and a pull agent

Delta asks for a password and NCSA Duo on **every** login, and SSH keys are not accepted. No driver
can log in by itself, so nothing does:

```
cmslpc driver (train_delta rule)             EOS spool                          Delta login node
software/delta/remote.py         --->  root://cmseos.fnal.gov//store/user/   <---  software/delta/agent.py (tmux)
  code bundles, request, then waits        <lpc-user>/barista_delta                  builds code, sbatch, sacct,
  for status/<rid>.json, copies logs                                                 publishes status + logs
                                                                                       |
                                                                         software/delta/train.sbatch on gpuA100x4
```

* **Driver** (`software/delta/remote.py`, run by the rule on cmslpc):
  1. Uploads the code. The agent publishes the commits its bare repositories hold, and the driver
     uploads only a git bundle of what is missing (usually a few KB) plus `git diff HEAD` for hot
     patches.
  2. Uploads the untracked workflow files the job names (`output/<model>/wfs/train.yml`,
     `common.yml`) and a request JSON.
  3. Polls the agent's status, copying `train.log` back as it grows.
  4. At the end, copies `train.log`, `yields_loaded.json` and the run summary back and touches
     `train.done`, only if the fit finished.

  The request id is kept in `<model_dir>/delta.json` (`delta_<task>.json` for evaluations). A
  restarted driver (`roast resume`) therefore reattaches instead of submitting again. A failed job is
  requested once more and resumes from its last epoch checkpoint.
* **Agent** (`software/delta/agent.py`, started with `software/delta/agent.sh` in tmux):
  * Every 60 s it builds requested code states under `$DELTA_HOME/code/<key>` and submits new
    requests with `sbatch` on `$DELTA_GPU_ACCOUNT`.
  * It follows its jobs with `sacct` and writes `status/<rid>.json`, `runs/<label>/train.log` and,
    at the end, `yields_loaded.json` and `summary.txt` to the spool.
  * It makes only outbound calls (XRootD and local Slurm). Its state is in
    `$DELTA_HOME/agent_state.json`, so a restarted agent picks its jobs up again.
* **Job** (`software/delta/train.sbatch`):
  * Runs the classifier image from `/cvmfs/unpacked.cern.ch` through apptainer and cvmfsexec (Delta
    has no `/cvmfs`).
  * Checkpoints every epoch into the run directory (`$DELTA_HOME/runs/<label>/checkpoints`).
  * On `SIGUSR1`, 15 min before the wall-time limit, it stops the fit and runs `scontrol requeue`.
    The requeued job resumes from the last epoch, up to `max_restarts` times (default 12).
    Evaluations are not checkpointed and are not requeued.
  * Records GPU utilisation every 30 s (`logs/gpu_<job>.<restart>.csv`). The summary comes from
    `software/orchard/summarize.sh`.

Nothing secret goes through the spool: the grid proxy stays on each host (`~/.roast/x509_proxy`).

## One-time setup (example: project `bjsy`, user `johnda`, LPC user `jda102`)

You run the logins yourself (password + Duo). Everything else can go through one SSH master
connection that you open once and that lasts up to 10 h:

```bash
# laptop: one Duo login, reused by every later ssh/scp/roast command for 10 h
ssh -o ControlMaster=yes -o ControlPath=~/.ssh/cm-delta -o ControlPersist=10h johnda@login.delta.ncsa.illinois.edu
```

and in `~/.ssh/config` (so `roast proxy --push --to delta` can use it):

```
Host delta
    HostName login.delta.ncsa.illinois.edu
    User johnda
    ControlPath ~/.ssh/cm-delta
```

On Delta:

```bash
# 1. which project: read by run_container, software/bin/apptainer and the agent (software/delta/env.sh)
mkdir -p ~/.config/barista
cat > ~/.config/barista/delta.env <<'EOF'
DELTA_PROJECT=bjsy
DELTA_SPOOL=root://cmseos.fnal.gov//store/user/jda102/barista_delta
EOF
#   defaults: DELTA_GPU_ACCOUNT=<project>-delta-gpu, DELTA_CPU_ACCOUNT=<project>-delta-cpu,
#   DELTA_SOFTWARE=/projects/<project>/software, DELTA_HOME=/projects/<project>/$USER/barista_delta

# 2. cvmfsexec and grid certificates (another project's /projects is not readable to you)
mkdir -p /projects/bjsy/software && cd /projects/bjsy/software
git clone https://github.com/cvmfs/cvmfsexec.git && cd cvmfsexec && ./makedist default
cd /tmp && /projects/bjsy/software/cvmfsexec/cvmfsexec grid.cern.ch -- bash -c \
  'mkdir -p /projects/bjsy/software/grid-security && cp -rL /cvmfs/grid.cern.ch/etc/grid-security/{certificates,vomsdir,vomses} /projects/bjsy/software/grid-security/'

# 3. a barista checkout for the agent, plus the agent's bare repositories. Delta cannot clone from
#    gitlab.cern.ch without credentials, so copy full bundles in from a laptop
#    (git bundle create barista.bundle --all; same for coffea4bees; scp via the master connection):
H=/projects/bjsy/johnda/barista_delta; mkdir -p $H/git
git clone --bare barista.bundle $H/git/barista.git; git clone --bare coffea4bees.bundle $H/git/coffea4bees.git
mkdir -p /projects/bjsy/johnda/workingArea/HH4b && cd /projects/bjsy/johnda/workingArea/HH4b
git clone barista.bundle barista && git clone coffea4bees.bundle barista/coffea4bees
#    (or, with no repositories on Delta, `software/delta/remote.py --bootstrap` on cmslpc uploads the
#    bundles to the spool and the agent clones them itself)

# 4. pixi environment (xrdfs/xrdcp for the agent): installs pixi in /projects/bjsy/johnda/.pixi
cd /projects/bjsy/johnda/workingArea/HH4b/barista && ./run_container snakemake --version
```

Grid proxy: make it where you have your certificate (cmslpc: `bin/roast proxy`), then copy it to
Delta's `~/.roast/x509_proxy`, either with `bin/roast proxy --push --to delta` (once `roast init
--delta-user johnda --delta-project bjsy` has a `delta` host, and the master connection is open) or
by hand. Renew it weekly. The agent warns in its log, and in the heartbeat the drivers print, when
less than 24 h is left.

## Running

Start the agent (once, and again after a Delta maintenance or login-node reboot):

```bash
ssh delta     # dt-login0N: remember which node; tmux sessions are per node
cd /projects/bjsy/johnda/workingArea/HH4b/barista
tmux new -d -s delta-agent 'software/delta/agent.sh 2>&1 | tee -a /projects/bjsy/johnda/barista_delta/agent.log'
#   options: --partition gpuA40x4 (half price), --time (default 8:00:00), --cpus/--mem, --max-restarts, --once
```

Then a roast with trainings and evaluations on Delta and everything else on cmslpc:

```bash
bin/roast new --config coffea4bees/workflows/config/<config>.yml --phases B,C,C4,D,F \
    --cmslpc --train-on delta --eval-on delta --label <label>
bin/roast submit <id> --step C    # on cmslpc; the train/evaluate rules wait for Delta
# switch an existing roast: bin/roast submit <id> --step C --train-on delta --eval-on delta
```

Per-workflow settings go in a `delta:` block of the workflow config (all optional): `spool`, `time`,
`partition`, `cpus`, `mem`, `max_restarts`, `poll`, and `eval_<key>` overrides for the evaluations.

Status: the driver log (`logs/<step>.log` of the roast) prints every state change and the agent's
heartbeat. On Delta, run `tail -f /projects/bjsy/johnda/barista_delta/agent.log` and `squeue --me`. Each
run directory `$DELTA_HOME/runs/<label>/` holds `train.log`, `logs/slurm-*.out`, the GPU monitor files
and `summary.txt`.

## Measured (2026-11-12/13): one production FvT fit

The 2025 nominal FvT (22.95 M events, 3 folds × 20 epochs + finetune) on 1 A100 / 16 cores / 62 GB,
driven from cmslpc with `--train-on delta`:

* **≈ 3.3 A100-h as one job** (~10 min load over XRootD, ~2.9 h training, epochs falling from 35 to
  4.5 min as the batch size grows). 3.9 GPU-h as run in 4 requeued chunks.
* GPU utilisation ~49 %, 6.9 GB of GPU memory. Host memory 62.6 GB, right at the quarter-node share,
  so larger inputs need more memory, and more memory means higher billing (see below).
* **Queue waits: 0–1 s on a fresh account, then 3.4 h, 6.0 h and 7.3 h** for the next three starts,
  once ~1.5 GPU-h had been used. The fit took 20.6 h of wall-clock, of which 16.6 h was queue.
  Ale's `bjsv` jobs, on an account with ~6 GPU-h used, waited > 47 h.
* Why: priority is dominated by fair-tree fairshare (weight 10000, QOS flag `NoDecay`). A few GPU-h of
  usage put an account below most of the thousands of pending jobs, and usage is never forgotten. Only
  age (≤ +1000) and backfill holes for short jobs are left.

So: run each fit as **one long job** (agent default `--time 8:00:00`). The checkpoint and requeue cycle
works, but each requeue costs another wait. Submit campaigns early, and keep quick iteration on
falcon/ORCHARD.

## Accounts, partitions, costs

`accounts` lists your allocations; `jobcharge -a <account> --detail` lists per-job charges.

* Charging is `PriorityFlags=MAX_TRES`: a job pays for the largest of its GPU, CPU and memory
  shares. On `gpuA100x4` one GPU is billed 1000 units = 1 GPU-hour per hour, and so are 16 cores
  (62.5 each) or 62 GB (16 per GB). The agent's default (1 GPU, 16 cores, 62 GB) is exactly one
  quarter-node: **1 GPU-h per wall-clock hour**. More cores or memory than that raise the charge.

| partition | GPU | GPU-h per GPU-hour | max time | preemption |
|-----------|-----|--------------------|----------|------------|
| `gpuA100x4` (default) | A100-SXM4 40 GB | 1 | 48 h | no |
| `gpuA40x4` | A40 48 GB | 0.5 | 48 h | no |
| `gpuA100x4-preempt` | A100 40 GB | 0.5 | 48 h | `REQUEUE`, 300 s grace |
| `gpuA100x8` | A100 80 GB | 1.5 | 48 h | no |
| `gpuH200x8` | H200 | 3 | 48 h | no |
| `gpuA100x4-interactive` | A100 40 GB | 2 | 1 h | no |

* The `cpu` partition needs a `-delta-cpu` account (an ACCESS exchange into "NCSA Delta CPU").
  `--train-on delta` never uses it. `roast new --delta` and interactive `./run_container` shells on
  Delta do.
* Preempt partitions: a preempted job gets `SIGTERM` and Slurm requeues it (`PreemptMode=REQUEUE`).
  `train.sbatch` saves its state on `SIGTERM`, and the requeued job resumes from the checkpoint.

## Storage (per project)

| path | quota | use |
|------|-------|-----|
| `/u/<user>` | 100 GB, 750k files | `~/.config/barista/delta.env`, `~/.roast/x509_proxy` |
| `/projects/<project>` | 500 GB | software (cvmfsexec, certificates), checkouts, pixi, `barista_delta/` (code states, run directories with checkpoints) |
| `/work/nvme/<project>` | 500 GB | `run_container` cache |
| `/work/hdd/<project>` | 1 TB | spare |
| `/tmp` on GPU nodes | 1.5 TB, per job | per-job scratch (`EVAL_SCRATCH`), proxy copy |

`quota` shows the usage. Each code state is about 0.5 GB. Prune old ones in
`$DELTA_HOME/code` (and their spool `code/<key>`) now and then.

## Ale's mode: whole steps on Delta (`roast new --delta`)

`roast new --delta` moves phases C, D and BKG_C to Delta. A roast checkout there runs Snakemake in
tmux on the login node, with `software/snakemake/profiles/delta`: GPU rules on the `-delta-gpu`
account, and CPU rules (`analyze`, plots, `evaluate_all*`, ...) on the `cpu` partition with the
`-delta-cpu` account. The profile names project `bjsv`. For another project, `run_container` writes
a copy with that project's accounts, QOS and binds into `.delta_profile/delta`, from `DELTA_PROJECT`
(environment or `~/.config/barista/delta.env`). Set up with `bin/roast init --delta-user <user>
--delta-project <project>`; the step scripts then export `DELTA_PROJECT`.

## Known issues

* Every new SSH login needs Duo. Keep the master connection open while you run `roast proxy --push
  --to delta` or start the agent. The agent itself needs none.
* tmux sessions live on one login node (`dt-login01..04`). `ssh login.delta...` may land on another:
  use `ssh dt-login04.delta.ncsa.illinois.edu` to get back to the agent.
* The agent stops with the login node (maintenance). Its state survives: restart it and it picks up
  the running jobs again. Drivers keep waiting and print the agent's last heartbeat every hour.
* `profiles/delta` (whole-step mode) lists no resources for `evaluate_split` / `evaluate_combine`.
  With eval splits, those rules would get the classifier Snakefile's falcon defaults (partition
  `work`, `mps:25`).
