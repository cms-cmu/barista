# Plan: apply_MvD in processor_HH4b.py

## Status: IMPLEMENTED ✓

## Goal

Apply MvD classifier weights (+ JCM) to `mixeddata_all` events when running the
analysis processor, so that MvD-weighted mixed_all histograms can be compared to
data 4b.

## Constraints

- MvD applies **only** to `mixeddata_all` datasets, not other mixed types (`mix_v`,
  `mix_noTT_v`, `mix_pz_v`)
- Reuse the **existing `JCM` parameter** — user will never run FvT JCM and MvD JCM
  simultaneously, so no new JCM parameter needed
- Load MvD scores from **friend trees** produced by the evaluate workflow (not inline
  inference)
- Follow the FvT pattern as closely as possible

## MvD weight

`MvD = (p_d4 - p_t4) / p_mix4`

Applied to fourTag `mixeddata_all` events. Reweights mixed_all to look like data 4b.
JCM (4b) accounts for the jet combinatoric probability of the hemisphere-mixed events.

## Changes Required

### 1. `coffea4bees/analysis/helpers/processor_config.py`

Add a narrower flag distinguishing `mixeddata_all` from other mixed types:

```python
config["isMixedDataAll"] = "mixeddata_all" in dataset
```

Place after the existing `isMixedData` line (line 12).

### 2. `coffea4bees/analysis/processors/processor_HH4b.py`

**Constructor** — add parameter:
```python
apply_MvD: bool = False,
```

Store it:
```python
self.apply_MvD = apply_MvD
```

**`load_MvD` method** — analogous to `load_FvT`, loads the MvD friend tree:
```python
def load_MvD(self, event):
    event["MvD"] = self.friends["MvD"]
```
(exact implementation depends on how friends dict is structured — mirror `load_FvT`)

**In the process method** — load MvD friend when needed:
```python
with self._stage("load_friend_MvD"):
    if self.apply_MvD and self.config["isMixedDataAll"]:
        self.load_MvD(event)
```

Pass `apply_MvD` and `isMixedDataAll` through to `add_pseudotagweights` calls.

### 3. `coffea4bees/analysis/helpers/event_weights.py`

**`add_pseudotagweights` signature** — add parameters:
```python
apply_MvD: bool = False,
isMixedDataAll: bool = False,
```

**New MvD weight block** — add after the existing FvT block. When
`isMixedDataAll and apply_MvD` and JCM is set, apply JCM to fourTag events and
multiply by MvD:

```python
if apply_MvD and isMixedDataAll:
    fourTag = event["fourTag"]

    if JCM:
        # JCM for 4b mixed_all (same JCM callable, conditioned on fourTag)
        jcm_weight = np.ones(len(event), dtype=float)
        jcm_weight[fourTag], _ = JCM(
            ak.num(event[fourTag]["Jet_untagged_loose"], axis=1),
            event.event[fourTag]
        )
        weights.add("JCM", jcm_weight)
        list_weight_names.append("JCM")

    mvd_weight = np.where(fourTag, event.MvD.MvD, 1.0)
    weights.add("MvD", mvd_weight)
    list_weight_names.append("MvD")
```

Note: the existing FvT JCM block applies to `threeTag`; the MvD JCM block applies
to `fourTag`. They are mutually exclusive by the user's constraint.

## Friend Tree Format

The MvD evaluate workflow produces a friend tree containing at minimum:
- `MvD` — the per-event weight `(p_d4 - p_t4) / p_mix4`
- `p_mix4`, `p_d4`, `p_t4` — class probabilities

The friend tree key in the `friends` dict should be `"MvD"` (to be confirmed when
looking at evaluate.yml output structure).

## Implementation Notes

- MvD friend tree field is `MvD` (confirmed from `baseline.py:output_definition`)
- `jetCombinatoricModel_SB_.yml` in `Run3_MvD/` has all standard JCM parameters —
  reuses the existing `jetCombinatoricModel` class unchanged
- JCM for 4b: called with `ak.num(event[fourTag]["Jet_untagged_loose"], axis=1)`,
  same interface as 3b JCM but conditioned on `fourTag` mask
- `setMvDVars` not needed for initial testing — `event.MvD.MvD` gives the weight directly
- MvD block returns early from `add_pseudotagweights`, so the 3b JCM/FvT path is
  never reached for mixeddata_all (the two are mutually exclusive by design)
- Confirmed working via import checks: all flags in signatures, `isMixedDataAll`
  correctly True for `mixeddata_all_*` and False for `mix_v*`

## Files Changed

- `coffea4bees/analysis/helpers/processor_config.py` — added `isMixedDataAll` flag
- `coffea4bees/analysis/helpers/load_friend.py` — added `read_MvD_friend()`
- `coffea4bees/analysis/helpers/event_weights.py` — added `apply_MvD`/`isMixedDataAll`
  params and MvD weight block (JCM on fourTag + MvD weight, early return)
- `coffea4bees/analysis/processors/processor_HH4b.py` — added `apply_MvD` param,
  `load_MvD` method, `load_friend_MvD` stage, passes new flags to
  `include_pseudotag_in_weight`

## Usage

In your analysis config YAML:
```yaml
apply_MvD: true
apply_FvT: false     # don't apply FvT to mixeddata_all
apply_JCM: true
JCM_file: coffea4bees/analysis/weights/JCM/Run3_MvD/jetCombinatoricModel_SB_.yml
```

MvD friend tree is already registered in `coffea4bees/metadata/friends_HH4b.yml`
for all Run3 eras (under the `friends_Run3` anchor):
```yaml
MvD: root://cmseos.fnal.gov//store/user/jda102/HH4b_Run3_v2/friend/MvD/result.json@@analysis.0.merged
```
The `@@analysis.0.merged` key matches the `kfold.Merge` output structure, same as FvT.
`load_MvD` is only called when `isMixedDataAll=True`, so having the entry in the shared
anchor does not affect data/ttbar/signal processing.

## Files Changed (full list)

- `coffea4bees/analysis/helpers/processor_config.py` — added `isMixedDataAll` flag
- `coffea4bees/analysis/helpers/load_friend.py` — added `read_MvD_friend()`
- `coffea4bees/analysis/helpers/event_weights.py` — added `apply_MvD`/`isMixedDataAll`
  params and MvD weight block (JCM on fourTag + MvD weight, early return)
- `coffea4bees/analysis/processors/processor_HH4b.py` — added `apply_MvD` param,
  `load_MvD` method, `load_friend_MvD` stage, passes new flags to
  `include_pseudotag_in_weight`
- `coffea4bees/metadata/friends_HH4b.yml` — added MvD friend tree entry for Run3

## Open Threads

- Test the processor on a small mixeddata_all sample with apply_MvD=true
- Verify MvD-weighted mixed_all histograms look reasonable vs data 4b
- JCM chi²/ndf=98 is high — MvD fit should improve this once classifier is applied
