# Session — 2026-03-12 (5)

## What we did
- Diagnosed why `./run_container snakemake -h` returned no output (snakemake crashing silently with SIGBUS, exit 135)
- Traced crash chain: `snakemake → snakemake.report → yte → numpy → libopenblas 0.3.30 → SIGBUS`
- Confirmed root cause: `libopenblas 0.3.30` on conda-forge was compiled with AVX-512 as a baseline; this node is AMD EPYC (AVX2 only), so library initialization crashes
- Confirmed numpy/openblas enter transitively via both `panoptes-ui` and `snakemake-logger-plugin-snkmt` (both → peppy → pandas → numpy)
- Fixed by pinning `libopenblas = "<0.3.30"` and removing `panoptes-ui` from `software/pixi/pixi.toml`
- Reinstalled pixi environment; verified `snakemake --version` and `--help` now work
- Configured `.claude/settings.json` with `"allow": ["Bash(*)"]` so Bash commands run without permission prompts

## Decisions
- Removed `panoptes-ui` — web monitoring server, not a core workflow tool, needlessly pulls in pandas/numpy/openblas
- Pinned `libopenblas = "<0.3.30"` rather than switching BLAS implementations — minimal targeted change
- Used project-level `Bash(*)` allow rule rather than `bypassPermissions` — auto-approves shell commands while keeping other safety checks

## Files changed
- `software/pixi/pixi.toml` — removed `panoptes-ui`; added `libopenblas = "<0.3.30"` pin with comment
- `pixi.toml` — regenerated from template
- `pixi.lock` — regenerated after fix
- `.claude/settings.json` — created; sets `"allow": ["Bash(*)"]`

## Open threads
- `software/pixi/pixi.toml` and `pixi.lock` should be committed so CI picks up the fix
- CI runner nodes should be checked — if also AMD EPYC without AVX-512, they had the same crash before this fix
