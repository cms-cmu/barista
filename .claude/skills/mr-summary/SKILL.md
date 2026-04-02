---
name: mr-summary
description: Use when creating a GitLab merge request for barista, coffea4bees, or bbreww — generates a title and summary from the diff against origin/master.
argument-hint: "[barista|coffea4bees|bbreww]"
---

# /mr-summary — Generate MR Title and Summary

Inspect the diff between the current branch and `origin/master` for the specified
repository and produce a ready-to-paste GitLab MR title and description.

## Arguments

- No argument — use the barista repository (current working directory).
- `coffea4bees` — use the `coffea4bees/` subdirectory relative to the barista root.
- `bbreww` — use the `bbreww/` subdirectory relative to the barista root.

## Step 1: Resolve Repository Path

Start from the barista root (the directory containing this `.claude/` folder).

| Argument      | Path (relative to barista root) |
|---------------|---------------------------------|
| (none)        | `.`                             |
| `coffea4bees` | `coffea4bees/`                  |
| `bbreww`      | `bbreww/`                       |

## Step 2: Gather Diff Information

Run both commands in the resolved directory:

```bash
git log origin/master..HEAD --oneline
git diff origin/master...HEAD --stat
```

If `git log` returns no commits, abort:
> "No commits ahead of origin/master on this branch. Nothing to summarize."

## Step 3: Generate the Summary

From the commit messages and diff stat, produce a markdown block with:

1. **A short title** (≤ 70 characters) — imperative mood, describes the main change.
2. **A summary section** — 3–6 bullet points grouped by theme (e.g., new features,
   refactors, config/metadata changes). Each bullet should be concrete and specific,
   referencing file or module names where helpful. Avoid generic phrases like
   "various improvements".

Do **not** include a test plan — the user will add one if needed.

## Output Format

Print the result as a fenced markdown block so it is easy to copy:

~~~
```markdown
## <title>

- <bullet 1>
- <bullet 2>
...
```
~~~

Keep the summary factual and concise. One tight paragraph of bullets is better
than a long list of trivial changes.
