# /session-summary — Session Summary

Generate a concise summary of the current session and save it to the
project's `.claude/session-summaries/` directory for git tracking.

## Purpose

Provide future Claude instances with fast, structured context about
what happened in this session — decisions made, files changed, and
anything left open.

## Steps

### Step 1: Determine save path

- Use the current working directory as the project root
- Target directory: `<project-root>/.claude/session-summaries/`
- Create it if it doesn't exist: `mkdir -p <project-root>/.claude/session-summaries/`
- Filename: `<slug>-YYYY-MM-DD.md` where `<slug>` is a 2-3 word kebab-case summary of the session (e.g. `fix-background-model-2026-03-12.md`, `issue-bot-testing-2026-03-12.md`)
- Choose the slug from the dominant topic of the session — concrete and specific, not generic ("misc" or "work" are not acceptable)
- If the file already exists (unlikely with slug), append `-2`, `-3`, etc.

### Step 2: Generate the summary

Review the full conversation and produce a summary in this format:

```markdown
# Session — <slug> (YYYY-MM-DD)

## What we did
- <action taken> (be specific — what was built, fixed, changed, tested)
- ...

## Decisions
- <decision> — <brief rationale>
- ...

## Files changed
- `path/to/file` — <what changed>
- (none) if no files were modified

## Open threads
- <unresolved issue or deferred item>
- (none) if everything is resolved
```

Guidelines for content:
- "What we did" = concrete actions, not just topics. "Built X", "Fixed Y", "Tested Z".
- "Decisions" = choices that would otherwise be lost. Include the *why* when it matters.
- "Files changed" = every file created or modified this session, with a one-line description.
- "Open threads" = anything explicitly deferred, broken, or needing follow-up.
- Keep each bullet to one line. No padding.

### Step 3: Write the file

Write the summary to the resolved path from Step 1.

### Step 4: Ensure git tracking

Check if `.claude/session-summaries/` is gitignored:
```bash
git -C <project-root> check-ignore -q .claude/session-summaries/ 2>/dev/null && echo "ignored" || echo "tracked"
```

If ignored, note it to the user — they may want to add an exception or
move the directory. Do not modify `.gitignore` without asking.

If the project root is not a git repo, skip this step silently.

### Step 5: Confirm to user

Output one line:
```
Saved: .claude/session-summaries/<filename>
```

Then stop. Do not summarize the summary or add commentary.
