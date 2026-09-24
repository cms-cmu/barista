# Patches applied to the pinned checkout of svbfeynnet_run3_20260923_5b01004-31a4b4d

The roast is pinned to barista `5b01004b` and coffea4bees `31a4b4da`. One file was replaced in the
cmslpc checkout by `scp` after the pin, rather than making a new roast. It is the only difference
from the pinned commits.

| file | as run | why |
|---|---|---|
| `coffea4bees/analysis/metadata/HH4b_make_friend_SvBFeynNet_Run3.yml` | coffea4bees `6aaebd47` (md5 `a5e87d3be75b02585538c94d858683dc`), copied here as `HH4b_make_friend_SvBFeynNet_Run3.effective.yml` | see below |

It went in two stages, both committed to coffea4bees on `RoastingNominalRun3`:

1. `da5abd8e`: `runner.friend_file: friends_HH4b.yml` plus `friends_include: [is_parking]`. Without it no friend
   is injected, and MC 2023_preBPix (mixed parking, fraction 0.6875) raises in `assign_is_parking`. Only
   `is_parking` is injected, so the existing SvB_FeynNet friend is never read.
2. `6aaebd47`: `runner.condor: true`, `condor_cores: 1`. Without it runner.py uses a local process pool.
   The first submit ran all 8 jobs on the cmslpc307 login node (load 204, one OOM kill). It was stopped with
   Ctrl-C (exit 130) and the step was resumed on condor.

Already in the pin (`31a4b4da`): `friend_base` moved from `jda102/XX4b/2025_v3` to
`jda102/XX4b/2026_Run3_SvBFeynNet`, so that the friend files still referenced by the previous JSON were
not overwritten in place.

Submitted as `roast resume ... --extra="--forcerun install_SvBFeynNet_friend_json"`. The forcerun is
needed because `rule all`'s only input is a git-tracked file, which already exists in every checkout.

## Diff 31a4b4da..6aaebd47

```diff
diff --git a/analysis/metadata/HH4b_make_friend_SvBFeynNet_Run3.yml b/analysis/metadata/HH4b_make_friend_SvBFeynNet_Run3.yml
index 8db4940f..28b5b4f5 100644
--- a/analysis/metadata/HH4b_make_friend_SvBFeynNet_Run3.yml
+++ b/analysis/metadata/HH4b_make_friend_SvBFeynNet_Run3.yml
@@ -1,11 +1,20 @@
 runner:
   workers: 4
   worker_memory: 8GB
+  # Without this runner.py uses a local process pool: the Snakefile's run_on_condor
+  # param is never passed on, and 8 local jobs overload an LPC login node.
+  condor: true
+  condor_cores: 1
   friend_metafile: make_friend_SvB_FeynNet
   # Fresh directory: the 2025_v3 friends are still referenced by committed JSONs
   # (and pinned roasts); writing there would overwrite them in place.
   friend_base: &friend_base root://cmseos.fnal.gov//store/user/jda102/XX4b/2026_Run3_SvBFeynNet
   # friend_base: &friend_base /tmp/SvBFeynNet_test   # local path for testing
+  # MC 2023_preBPix (mixed parking) reads is_parking from its friend tree. Only that
+  # friend is injected: loading the existing SvB_FeynNet friend would defeat the point.
+  friend_file: coffea4bees/metadata/friends/friends_HH4b.yml
+
+friends_include: [is_parking]
 
 config:
   blind: false
```
