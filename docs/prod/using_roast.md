# Using roast

`roast` runs the analysis workflows as **production batches** whose results can be traced
back to the exact code that made them, and puts those results somewhere the whole group
can read.

A *roast* pins three things: a `barista` commit, a `coffea4bees` commit, and the workflow
config file. It then gets its **own git checkout on each machine**, runs the Snakemake
phases there, and publishes the products. Because the checkout is made from the pinned
commits and is separate from your development tree, a result never depends on what
happened to be in your working directory that afternoon.

Every roast has an id of the form `<label>_<YYYYMMDD>_<barista7>-<coffea4bees7>`, for
example `nominal_run2_20260919_a180b1a-a16559c`. Any unique prefix of an id works as an
argument, so `nominal_run2_20260919` is usually enough to type.

Finished roasts are listed in the [cupping notes](index.md).

---

## One-time setup

```bash
bin/roast init --cmslpc-user <lpc-user> --falcon-user <falcon-user> \
               --cern-user <cern-user> --owner "Your Name"
```

This writes `~/.config/roast/config.json` and prints what it filled in. Check two things
in that file:

* `hosts.cmslpc.ssh` is the `cmslpc-el9` gateway, so the load balancer picks an
  interactive node per roast and the roast remembers it. Only the driver process and its
  `tmux` window are tied to a node: the filesystem is shared across interactive nodes and
  HTCondor runs central schedds, so a checkout, its logs and its jobs are reachable from
  anywhere. If a roast's node goes down, the next `submit` or `resume` moves that roast to
  a live one through the gateway and says so. `roast init --cmslpc-node <node>` pins every
  roast to one node instead, which you rarely want.
* `hosts.*.reference` points at your existing barista clone on each machine. New roast
  checkouts are cloned from it, which is far faster than cloning from GitLab.

You also need a grid proxy on the LPC, renewed about weekly:

```bash
bin/roast proxy            # voms-proxy-init on cmslpc, then copy the proxy to the other hosts
bin/roast proxy --check    # how much time is left: on this node, on shared home, on the other hosts
bin/roast proxy --push     # no new proxy: copy the existing one again (--to falcon,orchard for some)
```

`voms-proxy-init` writes into `/tmp`, which on the LPC belongs to one interactive node, while
the login alias is round-robin. The proxy you just made is therefore often not on the node a
given roast is pinned to, which shows up later as CERN EOS answering "unauthorized identity"
even though you have a valid proxy. `roast proxy` also drops a copy in `~/.roast/x509_proxy`
on shared home, and the commands that reach EOS take whichever copy still has time left.

The other machines (falcon, Bridges-2, Nautilus, ORCHARD) have no VOMS client or no certificate,
so `roast proxy` then copies that proxy to every other host in `hosts`: to `~/.roast/x509_proxy`
and to `<reference>/proxy/x509_proxy`, which `run_container` binds (on Nautilus, only the latter,
through the storage-helper pod). A host that cannot be reached is reported and the rest still get
it; `--no-push` keeps the proxy on the LPC. A host reached with `gcloud compute ssh` instead of
ssh, such as CMU ORCHARD, is declared with a `gcloud` block:

```json
"orchard": {"gcloud": {"instance": "orchard-login-001", "zone": "us-central1-c", "project": "cmu-gpu-cloud"},
            "reference": "~/barista"}
```

For looking at histograms locally, see [Local pourover](#local-pourover) below, which
needs a one-time virtual environment.

---

## The Roast Production Sequence

The full lifecycle of an analysis production follows a mandatory 10-step protocol:

```
[0. Validate Config] ──▶ [1. Roast New] ──▶ [2. Roast Checkout] ──▶ [3. Dry-run (-n)]
                                                                           │
                                                                           ▼
[7. Copy Products] ◀── [6. Step C Rule] ◀── [5. Monitor Status] ◀── [4. Submit Step]
       │
       ▼
[8. Roast Publish] ──▶ [9. Archive & Clean]
```

| Step | Command | Purpose |
|---|---|---|
| **0. Preflight Validation** | `bin/roast validate <config.yml>` | Audits YAML for duplicate parameters, divergence, missing files, and era mismatches. |
| **1. Create Manifest** | `bin/roast new --config <config.yml> --phases ...` | Pins `barista` and `coffea4bees` git SHAs, merges base configs, records validation. |
| **2. Host Checkouts** | `bin/roast checkout <id> --host <host>` | Clones clean isolated checkouts at pinned SHAs on each compute host. |
| **3. Dry-Run Verification** | `bin/roast submit <id> --step <step> -n` | Previews DAG and Snakemake job stats without executing commands. |
| **4. Live Step Submission** | `bin/roast submit <id> --step <step>` | Launches detached tmux job; generates rulegraph SVG and summary log. |
| **5. Status Tracking** | `bin/roast status <id>` / `bin/roast monitor` | Monitors driver processes, Condor workers, Slurm jobs, or Nautilus pods. |
| **6. Step C Full Lifecycle** | Submit `C` (GPU), then immediately `C4` (CPU) | Mandatory closure step for FvT (trained strictly in Sideband, never Signal Region). |
| **7. Copy Products** | `bin/roast copy <id>` | Synchronizes outputs across remote nodes and central EOS storage. |
| **8. Publish Results** | `bin/roast publish <id>` | Deploys web dashboard with inline SVG DAGs, cutflow tables, and cupping notes to CERNBox. |
| **9. Archive & Clean** | `bin/roast archive <id>` / `bin/roast clean <id>` | Archives products to EOS and cleans up ephemeral cluster checkouts. |

---

## 0. Preflight Configuration Validation

Before creating or launching a production, audit the workflow YAML configuration:

```bash
bin/roast validate coffea4bees/workflows/config/nominal_run3.yml
# To treat warnings as fatal errors:
bin/roast validate coffea4bees/workflows/config/nominal_run3.yml --strict
# To validate all standard configs:
bin/roast validate --all
```

The validator checks:
* **Duplicate parameter definitions:** Catches identical keys defined across different sections (e.g. `require_trigWeight`, `year_eras`, `blind`) that may diverge when modified.
* **Parameter divergence:** Verifies that duplicate definitions have identical values.
* **Era consistency:** Validates that dataset eras match between top-level `year_eras` and processor/classifier settings.
* **Filesystem integrity:** Audits paths to selection configs, friend trees, and calibration payloads.

`bin/roast new` automatically executes this preflight check and records the validation results (`valid`, `errors`, `warnings`) in `roast.json`. If fatal errors are found, roast creation is refused (override with `--no-validate` only if strictly necessary).

---

## 1. Running a phase

Commit and push your code first. `roast` records the current `HEAD` of both repositories,
and ships exactly those commits to the machines, so uncommitted edits are not in the run.

```bash
bin/roast new --config coffea4bees/workflows/config/nominal_run2.yml \
              --phases B --label nominal_run2
bin/roast checkout <id> --host cmslpc      # isolated clone at the pinned commits
bin/roast submit   <id> --step B -n        # dry run: build the DAG, run nothing
bin/roast submit   <id> --step B -t        # small test slice, writes to output/<label>_test/
bin/roast submit   <id> --step B           # the real run
```

### Visual DAG Rulegraphs and Summaries
Whenever `roast submit` runs a step:
1. It automatically runs `snakemake --rulegraph ... | dot -Tsvg` to capture the step's DAG rulegraph into `logs/{step}_rulegraph.svg`.
2. Upon exit 0, it runs `snakemake --summary ...` to capture rule execution timestamps, input/output file states, and software commands into `logs/{step}_summary.txt`.
3. When `roast publish` deploys the results to CERNBox, `generate_roast_index_html` embeds these SVG rulegraphs directly into the browser dashboard with clickable links to execution summaries.

Phases map to machines the way the
[workflow documentation](../bbbb/workflows.md) describes:

| Phase | A | B | C | D | E | F |
|---|---|---|---|---|---|---|
| runs on | cmslpc | cmslpc | falcon / nautilus / cmslpc¹ | falcon / nautilus / cmslpc¹ | cmslpc | cmslpc |

¹ with `--cmslpc`: the GPU work of C and D then runs on CMU ORCHARD, see below.

Passing `--nautilus` to `bin/roast new` maps the GPU phases (`C`, `D`, `BKG_C`) to run as Kubernetes Jobs on the **NRP Nautilus** cluster instead of Falcon (see [Using roast on Nautilus](using_roast_nautilus.md)).

### GPU training and evaluation on CMU ORCHARD

The classifier trainings and evaluations of a roast can run on the CMU ORCHARD H100s while the rest
of every step stays on its usual host. A nominal Run-3 2025 FvT trains in 2 h 08 min there, against
3 h 10 min on a falcon MPS share. How it works, and the cluster itself:
[Classifier training on the CMU ORCHARD cluster](../software/orchard.md).

```bash
# trainings (and evaluations) on ORCHARD, the rest of C/D on falcon
bin/roast new --config coffea4bees/workflows/config/<config>.yml --phases B,C,C4,D,F \
    --train-on orchard --eval-on orchard
# no falcon: C and D on cmslpc, their trainings and evaluations on ORCHARD
bin/roast new --config coffea4bees/workflows/config/<config>.yml --phases B,C,C4,D,F --cmslpc
# switch an existing roast (saved in roast.json; `host` switches back)
bin/roast submit <id> --step C --train-on orchard --eval-on orchard
```

Then `checkout`, `submit`, `status` and `resume` as for any roast. The step's Snakemake gets
`train_host=orchard` / `eval_host=orchard`; its `train` / `evaluate` rules are replaced by local
rules that run `software/orchard/remote_train.py` on the step's host. That script ships the code,
submits the job on ORCHARD with the models and friend trees going to the roast's EOS paths, waits,
and copies `train.log` and `yields_loaded.json` back. Everything after (evaluation if it stays local,
`analyze`, plots, C4, the next phase) is unchanged. BKG_C cannot use `--cmslpc` (its
`evaluate_synthetic` is a GPU rule of its own); it can use `--train-on orchard` on falcon.

**Before submitting** (each check takes seconds; the rule refuses to start without the first three):

| Check | How | If it fails |
|---|---|---|
| The roast's barista sha includes ORCHARD support (master since 2026-11-11, b61f5552) | `git merge-base --is-ancestor b61f5552 <barista sha>` | new roast from an up-to-date tree |
| gcloud on the step's host, logged in as the Andrew ID | `ssh <host> '~/google-cloud-sdk/bin/gcloud compute ssh orchard-login-001 --zone us-central1-c --project cmu-gpu-cloud --tunnel-through-iap --command hostname'` prints `orchard-login-001` | the owner runs `gcloud auth login <andrew-id>@andrew.cmu.edu --no-launch-browser` there (browser + Duo: not something an agent can do) |
| A grid proxy with time left on ORCHARD (models and friend trees are written to EOS) | `bin/roast proxy --check` | `bin/roast proxy` (or `--push --to orchard`) |
| The classifier image on ORCHARD (`/project/community` is wiped every two weeks) | `gcloud compute ssh ... --command 'ls -l /project/community/$USER/barista/images'` | on ORCHARD: `software/orchard/setup_image.sh` (restores from the bucket) |

**Watching it.** The step log shows the ORCHARD side as `[orchard ...]` lines (code shipped, job id,
completion). The job id is also in `output/<model>/orchard.json` (`orchard_evaluate*.json` for
evaluations) in the checkout. On ORCHARD the run directory is
`/project/community/<andrew-id>/barista/runs/<roast-id>__output_<model>[__evaluate]/`, with
`train.log`, `logs/slurm-<job>.out` and `software/orchard/summarize.sh <run dir>` for wall time and GPU use:

```bash
gcloud compute ssh orchard-login-001 --zone us-central1-c --project cmu-gpu-cloud --tunnel-through-iap \
    --command 'squeue -u $USER; tail -5 /project/community/$USER/barista/runs/<run>/train.log'
```

**Queueing.** The `general` queue runs one job per user at a time (12 h maximum, 5 submitted at
most; more wait in `remote_train.py`). Trainings of one roast therefore run one after the other on
ORCHARD; the default 4 h limit backfills within minutes, and a fit that needs longer requeues itself
and resumes from its last epoch. Evaluations are not checkpointed and are not requeued: give long
ones `orchard: {eval_time: "8:00:00"}` in the workflow config.

**When something goes wrong:**

* *The step was stopped or the host rebooted while ORCHARD was training.* `bin/roast resume`: the rule
  finds `orchard.json` and reattaches to the running job (or fetches its result) instead of
  submitting another. Do not `scancel` it and resubmit by hand, and do not delete its ORCHARD run
  directory while it runs: two jobs for one model race, and the one that starts second finds the
  saved models, skips the training and rewrites `result.json` without the training history that
  `analyze` plots.
* *`no grid proxy with 4 h left`*, *`no classifier image`*, *`no gcloud on this host`*: the table above.
* *`ORCHARD has no copy of barista|coffea4bees yet`*: ORCHARD keeps git copies in `~/.orchard_git`
  and receives only missing commits. If they are lost, rerun once, from a laptop:
  `python3 software/orchard/remote_train.py --bootstrap` in a barista checkout with coffea4bees.
  Not on the falcon login node: it kills processes above 60 % CPU or 4 GB
  (`/usr/local/bin/falcon-watchdog.sh`).
* *The ORCHARD job failed.* The rule resubmits once (resuming from the last epoch), then fails with
  the end of `slurm-<job>.out` in the step log; `train.log` is copied into the checkout either way.
  Once fixed, `bin/roast resume`.
* *A model must be retrained although it exists on EOS.* The training skips models it finds at the
  output path: remove them there first.
* *Maintenance.* ORCHARD is down for announced maintenance on Wednesdays 10:00–14:00; the rule waits
  up to 2 h for the cluster to answer, then fails, and `resume` picks it up again.

`--phases B,C,D,F` sets up several steps at once; each is submitted separately, and you
move to the next when the previous one reports `exit=0`. Run `checkout` for each machine
when a roast spans them. Workflows outside the A–F scheme are added with
`--step host:Snakefile[:targets]`, for example:

```bash
bin/roast new --config coffea4bees/workflows/config/run3_SvB_c6mvd.yaml --label svb_c6mvd \
  --step falcon:coffea4bees/workflows/Snakefile_Run3_SvB_training.smk:output/Run3_quadjet_run2/SvB/train.done
```

To reproduce someone else's run with different code, take the config and shas from their
manifest and pass `--barista` and `--coffea4bees` explicitly.

### Who a run belongs to

A workflow config in git names no one. Where it would name a person it writes a placeholder:

```yaml
handoff:
  eos_base: "{eos_prod}/{roast_id}/handoff"
fvt:
  eos_base: "{eos_prod}/{roast_id}"
  plot_base: "{web_prod}/{roast_id}/classifier"
```

`{eos_prod}` is your production area on FNAL EOS and `{web_prod}` your CERNBox web area.
`roast submit` reads both from `~/.config/roast/config.json`, the file `roast init` wrote,
and passes them alongside `{roast_id}`, so you never configure them twice. Running snakemake
by hand without roast works too: `helpers/common.smk` falls back to
`~/.config/coffea4bees/profile.yml` and then to values derived from your account names, so a
dry run needs no setup at all.

The point of resolving this at submit time rather than baking it into the captured config is
that a roast stays shareable. Take a colleague's manifest and config, run them under your own
profile, and the outputs land in your area rather than theirs. What each run actually resolved
to is recorded in its manifest and shown on its page, so provenance still says where the
results went.

### Run-scoped output paths

A workflow config may contain the placeholder `{roast_id}`, as `nominal_run2.yml` does for
the classifier input friend trees:

```yaml
classifier_inputs:
  config:
    make_classifier_input: root://cmseos.fnal.gov//store/user/<you>/HH4b_prod/{roast_id}/classifier_inputs/
```

`roast submit` passes `--config roast_id=<id>`, and `helpers/common.smk` substitutes it
everywhere in the config, so each production run writes to its own EOS directory and
cannot overwrite an earlier one. Outside roast the placeholder falls back to the config's
`label`, so running Snakemake by hand still works.

### Variant configs (`base:`)

A variant of an existing production can hold only its differences, with a top-level
`base:` naming the config it modifies (path relative to the barista root), as
`nominal_run3_quadjet_run2.yml` does over `nominal_run3.yml`:

```yaml
base: coffea4bees/workflows/config/nominal_run3.yml
label: "Run3_nominal_quadjet_run2"
analysis_config:
  config:
    candidates_selection_cfg: "coffea4bees/analysis/metadata/candidates_selection_thresholds_quadjet_run2.yml"
```

`roast new` merges the variant over its base (dicts recursively, lists and scalars replaced;
a base may itself have a `base:`) and captures the merged, self-contained result as
`roasts/<id>/config.yml`. `roast.json` records each base's path and sha256 under
`config.base`. Later edits to the base therefore never change an existing roast, and the
captured file loses the sources' comments. Snakemake merges repeated `--configfile` the
same way, so to run a variant by hand, pass the base first and then the variant.
Capturing a layered config needs PyYAML; plain configs are still copied verbatim.

---

## Watching a run

```bash
bin/roast status                 # one line per roast: where every one of them stands
bin/roast status <id>            # that roast in full, step by step
bin/roast status -s              # force the one-liner; -d forces the detail
bin/roast status --active        # hide roasts whose checkouts have been cleaned
bin/roast log    <id> --step B   # that step's log, without a tmux window
bin/roast attach <id> --step B   # drop into the tmux window on that machine
bin/roast monitor                # HTML dashboard of every live roast, in your browser
```

With no arguments, `status` gives a line per roast: a mark per step, and what is running
now. `✓` finished cleanly, `●` running, `✗` failed, `·` not started, `?` a driver that died
quietly, `!` a machine that did not answer.

`monitor` serves the same information as a self-refreshing page on `http://127.0.0.1:8765/`:
a card per roast, and per step a state pill, a progress bar (snakemake `N of M steps`) and
the last log line, with condor workers, slurm jobs or nautilus pods listed underneath.

```bash
bin/roast monitor                       # every roast with a live checkout; opens a browser
bin/roast monitor <id> [<id> ...]       # only these roasts
bin/roast monitor --no-open --port 9000 --interval 120
```

A background thread polls each host every `--interval` seconds (default 60; one ssh per host
per roast), and the page re-reads that cache every 5 s, so a slow host never freezes it. A
roast whose steps are all `exit=0` is not polled again until you press *Refresh now*. It
listens on localhost only and needs the same ssh/kubectl access as `status`, so renew the
proxy and any ssh session first: a password prompt would stall the polling thread.

Roasts are grouped into the same families as the [cupping notes](index.md) sub-pages, newest
first inside a family, so a production reads the same way here as on the website.

```
nominal
  nominal_run2_20260919_a180b1a-a16559c       B✓ C✓ C4✓ D✓ F✓    published  https://barista.docs.cern.ch/prod/nominal_run2_20260919_a180b1a-a16559c/
  nominal_run3_30x_20260924_8cc39b3-a512d88   C● C4· D· F·       C on falcon: 15888483/69615572 entries

svbfeynnet
  svbfeynnet_run3_20260923_5b01004-31a4b4d    SvBFeynN✓          cleaned 2026-10-06  https://barista.docs.cern.ch/prod/svbfeynnet_run3_20260923_5b01004-31a4b4d/
```

Once nothing is running, a roast carries a link to its cupping note, whether it has been
cleaned or not. While a step is still going, that space shows what the step is doing instead.
A cleaned roast is answered from the manifest alone, with no machine contacted, which is what
keeps `status` quick once a production is finished. Naming any published roast shows all its
links: the cupping note, the CERNBox results and the EOS archive.

Naming a roast gives the detail: one line per step, with that step's batch jobs underneath.

```
B    cmslpc  exit=0    tmux=0  3 of 3 steps (100%) done     === roast ... step B exit 0 ...
C    falcon  exit=0    tmux=0  5 of 5 steps (100%) done     === roast ... step C exit 0 ...
          slurm 42442 train        COMPLETED 04:35:59   (after 2 failed attempts)
          slurm 42468 evaluate     COMPLETED 01:22:47
C4   cmslpc  exit=0    tmux=0  20 of 20 steps (100%) done   === roast ... step C4 exit 0 ...
D    falcon  running   tmux=1                               Job 1 submitted with SLURM jobid 42722
          slurm 42722 train        RUNNING   29:39/8:00:00  rogue02  cpu=8 mem=62.50G gres/mps:50
                      ⠇ 34/603 batch training loss=0.4254
F    cmslpc  not started
     cmslpc  condor  Total for query: 0 jobs; 0 idle, 0 running, 0 held
     falcon  cluster work* 2 node(s) mixed gpu:1,mps:100
     falcon  1 finished job not part of any step, 1 failed (--all to list)
```

Steps are listed in the order the roast declares them, not grouped by machine, because a
roast is a pipeline and where a phase happens to run matters less than how far along it is.
Each line starts with the step, then the machine.

The state column means:

| state | meaning |
|---|---|
| `not started` | no log yet |
| `running` | a Snakemake driver process for this roast is alive |
| `error` | the driver died after a job failed |
| `stalled` | the driver died with no error in the log, e.g. the node rebooted |
| `exit=N` | finished, with that exit code |

`status` also lists this roast's own batch jobs. On the LPC they are dask workers, and
HTCondor spooling overwrites the directory they were submitted from, so the surviving
identifier is the dask scheduler address in each worker's arguments, which the step log
records when it starts one. Workers are therefore grouped by scheduler and shown under
the step they serve:

```
MakeMixedData cmslpc  running   tmux=1  4 of 10 steps (40%) done
                 condor    200 workers  190 idle, 10 running   tcp://131.225.191.81:10079
```

A group no step log claims is listed separately, as `condor jobs from a roast this
checkout does not know`. That is the common case when a roast's manifest lives on another
branch, and it is worth seeing: those jobs are competing for the same pool slots.
On falcon the equivalent information comes from the scheduler's own record of the
submitting directory: HTCondor batches by state on the LPC, and
on falcon each Slurm job with its rule name, elapsed time against its limit, node,
resources, and the last line of that job's log, which for a training is the live loss.

Each job is listed under the step that submitted it, which `status` works out from the
step log that announced the job id. That matters when two phases run a rule of the same
name, as Phases C and D both do with `train`: their attempts stay apart instead of one
hiding the other. Anything else submitted from the checkout, a one-off command you ran by hand for instance,
belongs to no phase. Such a job is reported at the very bottom, well clear of the last
step: if it is still running it is listed, because it is holding resources, and if it has
finished it is only counted, since it is history. `status --all` lists those too.

There is one line per rule, showing its latest attempt. Snakemake resubmits a rule that
fails, so the scheduler holds several job ids for it; the attempts that were replaced are
folded into a note like `(after 3 failed attempts)` instead of being listed beside the one
that succeeded. A failure that was never retried still appears on its own.

### Reading a step's log

`status` shows only the last line and `attach` needs the tmux window to still be there, so
`log` is the one to reach for when a step has already finished, or when you want to read
rather than watch. Everything it shows is scoped to the **most recent run** of that step,
not the whole history of resubmissions appended to the same file.

```bash
bin/roast log <id> --step B             # the last 50 lines (-n to change)
bin/roast log <id> --step B -f          # follow the live log (ctrl-c to stop)
bin/roast log <id> --step B --stats     # the last "Job stats:" block
bin/roast log <id> --step B --errors    # only the failure lines
```

`--stats` prints the job counts and the reasons Snakemake gives for them. It is the only
way to see a **dry run's** job count: `-n` produces no `N of M steps done` lines, so
`status` leaves the progress column empty for one, and `exit=0` from a dry run is
indistinguishable from a real one. Check the count there before dropping the `-n`.

`--errors` matches rule failures, `WorkflowError`, missing inputs or outputs, non-zero exits,
OOM kills and the step's own exit line. It deliberately does **not** match `Traceback`: dask
tears its client down noisily at the end of every successful job, so a dozen benign stacks
per wave of jobs would bury the real failure. `-f --errors` follows and filters at once,
which is the cheap way to sit on a long run without watching it.

## When something breaks

```bash
bin/roast resume <id> --step B
```

`resume` releases the Snakemake lock and reruns with `--rerun-incomplete`, keeping the
arguments of the last submit, so finished targets are not redone. Use it after a node
reboot, an out-of-memory kill on falcon, or a job failure you have since fixed **in the
configuration**.

A failure in the *code* is different: the checkout is pinned, so a code fix means a new
roast. That is the point of the tool, and it is cheap, since the new checkout is cloned
locally and Snakemake skips products that already exist.

A failed step leaves its `tmux` window open with a shell in the checkout, so
`bin/roast attach <id> --step B` puts you where the failure happened. The full log of a
step is `logs/<step>.log` in the checkout, and the resolved shell command of every rule is
in it, because roast runs Snakemake with `--printshellcmds`. To rerun one rule by hand:

```bash
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseB.smk \
    --configfile roasts/<id>/config.yml -n -p --forcerun make_new_JCM
```

---

## Where the outputs go

```bash
bin/roast archive <id>    # heavy products to FNAL EOS
bin/roast publish <id>    # plots and logs to CERNBox, and write the catalogue page
```

Run `archive` first, so the page that `publish` writes already carries the EOS location.
Both take `-n` to list what they would copy without copying, and both skip files already
at the destination with the same size, so re-running after another step is cheap.

| where | what | how to get at it |
|---|---|---|
| **FNAL EOS**, `/store/user/<you>/HH4b_prod/<id>/` | merged `.coffea` histograms, fitted JCM weights, friend-tree manifests, the config and the manifest | `xrdfs root://cmseos.fnal.gov ls -R /store/user/<you>/HH4b_prod/<id>` |
| **CERNBox**, `https://<you>.web.cern.ch/<you>/HH4b/prod/<id>/` | plots, cutflows, limits, the step log, the roast manifest | open it in a browser |
| **[Cupping notes](index.md)** | a page per family, each with one row per roast: both commit hashes linked to GitLab, the config, the steps, and a link into the CERNBox area | this documentation site |

Archiving deliberately keeps only merged products: per-dataset `singlefiles`, per-dataset
classifier-input manifests, per-job logs, Dask reports and memory profiles stay on the
machine. Publishing keeps the browsable artefacts under 50 MB. Both selections are
configurable per user under `publish` and `archive` in `~/.config/roast/config.json`, or
per roast under `publish_rules` and `archive_rules` in its `roast.json`.

The catalogue is only as complete as what is committed. After archiving/publishing:

```bash
bin/roast commit <id>
# or with a custom message and automatic push:
bin/roast commit <id> -m "Phase B results" --push
```

`roast commit` stages only `roasts/<id>/` (leaving active work in the tree and other roasts untouched), creates a structured commit message detailing which steps and archiving/publishing actions are complete, and optionally pushes to your remote branch.

Once merged to `master`, the `pages` job publishes the site within a minute of the merge, independently of the analysis CI.

Results live in each person's own CERNBox area, so nobody needs write access to anyone
else's; the catalogue is the shared index that ties them together.

---

## Local pourover

To look at a step's histograms interactively with the plot configuration that step
actually used:

```bash
bin/roast pourover <id> --step F
```

This reads the `makePlots` command from the step's log, so you get the same histogram file
and the same plot metadata that produced the published plots, not a guess. It copies both
to `output/roasts/<id>/` on your laptop and serves pourOver against them. The copy is
incremental: the first call moves the file, later calls take a couple of seconds.

```bash
bin/roast pourover <id> --step B --list       # what the step plotted; pick with --which or --match
bin/roast pourover <id> --step F --port 5001  # several sessions at once
bin/roast pourover <id> --step F --no-pull    # offline, from what you already have
bin/roast pourover <id> --step F --extra="--pregallery -j 8"
```

pourOver runs in the background and prints its URL and how to stop it. Add
`--extra=--foreground` to keep it in the terminal instead.

It needs a virtual environment once:

```bash
python -m venv ~/python-environments/pourover
source ~/python-environments/pourover/bin/activate
pip install -r coffea4bees/plots/requirements-pourover.txt
pip install -e .
```

Point `pourover.python` in `~/.config/roast/config.json` somewhere else if you keep it
elsewhere. To pull products without serving anything, use `bin/roast pull <id> --only
'*.coffea'`.

---

## Housekeeping

```bash
bin/roast ls                 # every roast, its machines, steps and whether it is published
bin/roast show <id>          # the full manifest
bin/roast clean <id>         # what cleaning would remove, a dry run
bin/roast clean <id> --yes   # remove the checkouts, keeping the results and the record
bin/roast clean --all        # every roast that qualifies
bin/roast rm  <id>           # what deleting would remove, a dry run
bin/roast rm  <id> --yes     # delete locally, on both machines, on EOS and on CERNBox
```

### `clean` and `rm` are not the same thing

`clean` reclaims space. It removes the checkouts on cmslpc and falcon, each of which carries
its own pixi environment and job logs, and leaves everything that makes the run readable: the
manifest, the cupping note, the EOS archive and the published CERNBox area. `roast checkout
<id>` brings the checkout back, because the manifest pins both commits.

It refuses unless all of this holds:

* every step finished with `exit=0`, asked of the machines rather than taken from the manifest;
* the roast was archived and published, and neither reported a failure;
* both EOS trees still exist and are not empty, confirmed by listing them.

Whether a roast was really published is read from its history, which records only the copies
that ran, so refreshing a cupping note with `publish --docs-only` neither counts as publishing
nor erases the real publish that came before it. If the published-tree check reports an
unauthorized identity, that is a missing credential rather than missing data: run `bin/roast
proxy` on cmslpc and try again.

`rm` is the opposite. It deletes the roast everywhere, the record included, and `--keep-eos`,
`--keep-cernbox` and `--keep-hosts` spare individual locations. It refuses while a step is
still running.

## Things that catch people out

* **Commit before `roast new`.** It records `HEAD`, not your working tree. It warns when
  the tree is dirty but does not stop.
* **Never run production in your development tree.** The checkouts live outside the
  directories that `mutagen` keeps in sync between machines, precisely so that checking
  out a pinned commit cannot rewrite files on your laptop and on the cluster.
* **The test slice writes elsewhere.** `submit -t` redirects `output_path` to
  `<path>_test/`, so a test can never leave products that make the real run think it has
  nothing to do.
* **Friend trees must go to EOS.** Condor workers cannot see the node-local `output/`
  directory; a workflow that writes friend trees there fails when merging them. This is
  what the `{roast_id}` EOS path in the config is for.
* **Watch the proxy before a long run.** `archive` and `publish` need it too, and a run
  longer than the proxy's remaining life will finish with failed copies.

## Under the hood

`roast` is `src/tools/roast.py`, standard-library Python with a `bin/roast` wrapper, and
its reference documentation is in [src/tools](../src/tools.md). It talks to the machines
over ssh, runs each step in a detached `tmux` session called `roast`, and copies files with
`xrdcp` and `rsync`. It is covered by `src/tests/roast.py` in the `code_roast_barista` CI
job, which needs no cluster and reports in seconds.
