---
name: test-coffea4bees
description: |
  Push the current coffea4bees branch, open or reuse a GitLab MR to master,
  monitor the CI pipeline, and find and fix any failures.
version: 1.1.0
argument-hint: "[branch-name] [--no-push]"
---

# /test-coffea4bees — Trigger and Monitor coffea4bees CI Pipeline

Push the current (or specified) branch of the **coffea4bees** sub-repo to GitLab,
ensure a Merge Request to `master` exists (creating one if not), then monitor the
triggered CI pipeline. On failure, read job logs, identify root causes, fix code,
push again, and repeat until the pipeline is green or you are blocked.

All git operations run inside `coffea4bees/` (a separate git repo from barista).

## Execution Notes

**Always use newline-separated Bash commands.** The Bash tool runs each call in a fresh
shell. Variables set by `source ~/.aliases_local` do not propagate when chained with
`&&` or `;` on the same line. Every multi-step shell block must use newlines:

```bash
# correct
source ~/.aliases_local
curl -H "PRIVATE-TOKEN: $GITLAB_TOKEN" ...

# broken — token will be empty
source ~/.aliases_local && curl -H "PRIVATE-TOKEN: $GITLAB_TOKEN" ...
source ~/.aliases_local; curl -H "PRIVATE-TOKEN: $GITLAB_TOKEN" ...
```

This applies to every `curl` call in this skill.

**GitLab API responses may contain ANSI escape codes** that break `json.load()`. Always
strip them before parsing JSON:

```python
import re
raw = sys.stdin.read()
clean = re.sub(r'\x1b\[[0-9;]*[mK]', '', raw)
data = json.loads(clean)
```

## Arguments

- No arguments — use the current coffea4bees branch, push, and run.
- `branch-name` — use that branch instead of the current one.
- `--no-push` — skip the push; assume the branch is already up-to-date on the remote.

## Constants

```
GITLAB_BASE   = https://gitlab.cern.ch
PROJECT_PATH  = cms-cmu/coffea4bees
PROJECT_ID    = cms-cmu%2Fcoffea4bees     # URL-encoded for API calls
TARGET_BRANCH = master
REPO_DIR      = coffea4bees/              # relative to barista root
```

Token is read from the environment: `$GITLAB_TOKEN`.
If unset, abort immediately:
> "GITLAB_TOKEN is not set. Export it in your shell: export GITLAB_TOKEN=<your-token>"

---

## Step 1: Resolve Branch

```bash
cd coffea4bees && git rev-parse --abbrev-ref HEAD
```

If a plain (non-flag) argument was passed in `$ARGUMENTS`, use that as `BRANCH` instead.

If the branch is `master`, abort:
> "Refusing to open an MR from master to master. Check out a feature branch first."

---

## Step 2: Derive MR Title Slug

Generate a short kebab-case descriptor for the MR title from the branch name:

1. Strip common prefixes: `feature/`, `feat/`, `fix/`, `bugfix/`, `hotfix/`, `chore/`.
2. Replace underscores and spaces with hyphens.
3. Lowercase everything.
4. Truncate to 40 characters at a word boundary.

Example: branch `add_MvD_weights` → slug `add-mvd-weights`
Example: branch `fix/ttbar-estimation` → slug `ttbar-estimation`

MR title: `WIP: CI test — <slug>`

---

## Step 3: Push (unless --no-push)

First, check for uncommitted changes to tracked files:

```bash
cd coffea4bees && git status --short
```

If any tracked files show as modified (`M`) or staged, stop and report:
> "There are uncommitted changes to tracked files. Commit or stash them before pushing so
> the CI tests the right code."
> List the dirty files and ask the user whether to commit them or abort.

If the working tree is clean (only untracked `??` files are fine), proceed:

```bash
cd coffea4bees && git push origin $BRANCH
```

If the push fails (non-zero exit), report the error and abort — do not force-push.

---

## Step 4: Ensure MR Exists

Check if an open MR already exists from `$BRANCH` to `master`:

```bash
curl -s --header "PRIVATE-TOKEN: $GITLAB_TOKEN" \
  "https://gitlab.cern.ch/api/v4/projects/cms-cmu%2Fcoffea4bees/merge_requests?state=opened&source_branch=$BRANCH&target_branch=master"
```

- If one is found, extract its `iid` and `web_url`. Report it and proceed.
- If none found, create one:

```bash
curl -s -X POST \
     --header "PRIVATE-TOKEN: $GITLAB_TOKEN" \
     --header "Content-Type: application/json" \
     --data "{\"source_branch\": \"$BRANCH\", \"target_branch\": \"master\", \"title\": \"WIP: CI test — $SLUG\"}" \
  "https://gitlab.cern.ch/api/v4/projects/cms-cmu%2Fcoffea4bees/merge_requests"
```

Extract `iid` and `web_url` from the response and report the MR URL.

---

## Step 5: Get Pipeline ID

After push + MR exist, wait for a pipeline to appear. Poll up to 10 times, 15 seconds apart:

```bash
curl -s --header "PRIVATE-TOKEN: $GITLAB_TOKEN" \
  "https://gitlab.cern.ch/api/v4/projects/cms-cmu%2Fcoffea4bees/merge_requests/$MR_IID/pipelines"
```

Take the pipeline with the most recent `created_at`. Extract its `id` and `web_url`.
Report: `Pipeline #<id> — <web_url>`

If no pipeline appears after 10 attempts, abort:
> "No pipeline triggered after 150s. The MR may not match any workflow:rules."

---

## Step 6: Monitor Pipeline

Poll pipeline status every 60 seconds:

```bash
curl -s --header "PRIVATE-TOKEN: $GITLAB_TOKEN" \
  "https://gitlab.cern.ch/api/v4/projects/cms-cmu%2Fcoffea4bees/pipelines/$PIPELINE_ID"
```

Terminal statuses: `success`, `failed`, `canceled`, `skipped`.
Running statuses: `created`, `waiting_for_resource`, `preparing`, `pending`, `running`.

Each poll cycle, also fetch job statuses for a running summary:

```bash
curl -s --header "PRIVATE-TOKEN: $GITLAB_TOKEN" \
  "https://gitlab.cern.ch/api/v4/projects/cms-cmu%2Fcoffea4bees/pipelines/$PIPELINE_ID/jobs?per_page=100"
```

Print one line per cycle: `[elapsed] status — N passed / N running / N pending / N failed`

When the pipeline reaches a terminal status, print a final job table:
```
PASSED  analysis-test
PASSED  skimmer-test
FAILED  analysis-systematics-test
...
```

---

## Step 7: Handle Results

### If `success`

Report: "Pipeline passed." Stop.

### If `failed`

For each failed job:

**7a. Fetch the log:**
```bash
curl -s --header "PRIVATE-TOKEN: $GITLAB_TOKEN" \
  "https://gitlab.cern.ch/api/v4/projects/cms-cmu%2Fcoffea4bees/jobs/$JOB_ID/trace"
```
Read the last ~200 lines to identify the error.

**7b. Classify the failure:**
- **Pre-existing / unrelated** — error is unrelated to files changed on `$BRANCH` vs master,
  or the job was already failing on master. Note it and skip.
- **Fixable** — error traces to code in files modified on this branch.

To check which files were changed on this branch:
```bash
cd coffea4bees && git diff --name-only origin/master...$BRANCH
```

**7c. Fix and iterate:**
- Read the relevant file(s), diagnose the root cause, apply the fix.
- If the root cause is unclear from the log alone, say so explicitly and stop rather than guessing.
- After all fixes are applied, return to Step 3 (push and re-trigger).

**Iteration cap:** stop after 3 push/fix cycles. If still failing, report remaining failures
and ask the user how to proceed.

---

## Step 8: Final Report

```
Pipeline #<id>: <status>
  Passed:      N
  Pre-existing failures: N  (not fixed — unrelated to this branch)
  Fixed:       N
  Unresolved:  N

MR: <web_url>
```

For each unresolved failure, print the job name and a one-line error summary from the log.
