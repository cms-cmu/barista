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

* `hosts.cmslpc.ssh` names one LPC interactive node (`cmslpc307` by default). Pass
  `--cmslpc-node` to pick another. Your home area is shared across nodes, so this only
  decides where the driver process and its `tmux` session live.
* `hosts.*.reference` points at your existing barista clone on each machine. New roast
  checkouts are cloned from it, which is far faster than cloning from GitLab.

You also need a grid proxy on the LPC, renewed about weekly:

```bash
bin/roast proxy            # voms-proxy-init on cmslpc
bin/roast proxy --check    # how much time is left
```

For looking at histograms locally, see [Local pourover](#local-pourover) below, which
needs a one-time virtual environment.

---

## Running a phase

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

Phases map to machines the way the
[workflow documentation](../bbbb/workflows.md) describes:

| Phase | A | B | C | D | E | F |
|---|---|---|---|---|---|---|
| runs on | cmslpc | cmslpc | falcon | falcon | cmslpc | cmslpc |

`--phases B,C,D,F` sets up several steps at once; each is submitted separately, and you
move to the next when the previous one reports `exit=0`. Run `checkout` for both machines
when a roast spans them. Workflows outside the A–F scheme are added with
`--step host:Snakefile[:targets]`, for example:

```bash
bin/roast new --config coffea4bees/workflows/config/run3_SvB_c6mvd.yaml --label svb_c6mvd \
  --step falcon:coffea4bees/workflows/Snakefile_Run3_SvB_training.smk:output/Run3_quadjet_run2/SvB/train.done
```

To reproduce someone else's run with different code, take the config and shas from their
manifest and pass `--barista` and `--coffea4bees` explicitly.

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

---

## Watching a run

```bash
bin/roast status <id>            # all steps, on every machine the roast uses
bin/roast log    <id> --step B   # that step's log, without a tmux window
bin/roast attach <id> --step B   # drop into the tmux window on that machine
```

`status` prints one line per step:

```
cmslpc  B   exit=0    tmux=0  29 of 29 steps (100%) done   === roast ... step B exit 0 ...
falcon  C   running   tmux=1  3 of 7 steps (42%) done      Job 1 submitted with SLURM jobid 41037
falcon  slurm   41037 train      RUNNING   22:01/8:00:00  rogue01  cpu=8 mem=62.50G gres/mps:50
falcon                ⠼ 104/1630 batch training loss=0.8293
```

The state column means:

| state | meaning |
|---|---|
| `not started` | no log yet |
| `running` | a Snakemake driver process for this roast is alive |
| `error` | the driver died after a job failed |
| `stalled` | the driver died with no error in the log, e.g. the node rebooted |
| `exit=N` | finished, with that exit code |

Below the steps, `status` lists this roast's own batch jobs, matched through the
scheduler's record of the submitting directory: HTCondor batches by state on the LPC, and
on falcon each Slurm job with its rule name, elapsed time against its limit, node,
resources, and the last line of that job's log, which for a training is the live loss.

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
| **[Cupping notes](index.md)** | one row per roast with both commit hashes linked to GitLab, the config, the steps, and a link into the CERNBox area | this documentation site |

Archiving deliberately keeps only merged products: per-dataset `singlefiles`, per-dataset
classifier-input manifests, per-job logs, Dask reports and memory profiles stay on the
machine. Publishing keeps the browsable artefacts under 50 MB. Both selections are
configurable per user under `publish` and `archive` in `~/.config/roast/config.json`, or
per roast under `publish_rules` and `archive_rules` in its `roast.json`.

The catalogue is only as complete as what is committed. After publishing:

```bash
git add roasts/<id> docs/prod
git commit -m "roast <id>: Phase B results"
```

and merge to `master`. The `pages` job publishes the site within a minute of the merge,
independently of the analysis CI.

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
bin/roast rm  <id>           # what deleting would remove, a dry run
bin/roast rm  <id> --yes     # delete locally, on both machines, on EOS and on CERNBox
```

`rm` refuses while a step is still running, and `--keep-eos`, `--keep-cernbox` and
`--keep-hosts` spare individual locations. Old checkouts are worth removing once a roast is
archived and published, since each one carries its own environment and outputs.

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
