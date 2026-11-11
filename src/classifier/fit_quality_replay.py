#!/usr/bin/env python
"""Replay an existing HCR training to compute its fit-quality benchmarks without retraining.

Takes the training workflow (train.yml) a classifier was trained with and the directory of its
saved fold models, and writes a replay workflow that

  * loads each fold's saved model (--pretrained-weights, matched by ``_offset{N}_``),
  * runs zero training and no finetuning epochs, so only the pre-training "Baseline" benchmark
    runs: HCRModel.validate over the fold's training and validation splits with every
    configured benchmark (ROCs, scalars and the FvT/MvD fit-quality),
  * writes result.json (and a throw-away model copy) to a fresh output directory.

Then summarize with ``python -m src.classifier.fvt_benchmark --result <out>/result.json
--stage baseline``.

Faithfulness check: the input pass that builds the k-fold split also recomputes the input
mean/std (initMeanStd) over the training split, overwriting the stored ones. With the same
inputs, k-fold seed and offsets they are the same; confirm by comparing the replayed
Baseline loss/AUCs with the original training's final-epoch values
(``fvt_benchmark --stage final`` on the original result.json, or its loss-auc-table page).

Usage (inside the classifier container, on a GPU node):
  python -m src.classifier.fit_quality_replay \\
      --train-wf <wfs_base>/train.yml --common <common.yml> \\
      --pretrained root://.../classifier/FvT_nominal \\
      --output <scratch>/replay_FvT_2024 [--floor 0.01] [--run]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shlex
import subprocess
import sys

import yaml

SCHEDULE_FLAGS = ("--training", "--finetuning")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-wf", required=True, help="train.yml the model was trained with (after roast overrides)")
    p.add_argument("--common", default=None, help="common.yml used at training time (-from)")
    p.add_argument("--pretrained", required=True, help="directory holding the saved fold models (*_offset{N}_*.pkl)")
    p.add_argument("--output", required=True, help="fresh output directory for the replay (never the original)")
    p.add_argument("--floor", type=float, default=None,
                   help="fit-quality source floor (analysis FvT_pd3_floor / mvd_pmix4_floor); default: benchmark default")
    p.add_argument("--template", default="",
                   help='extra template keys besides "model", e.g. "FvT: root://..." if train.yml uses them')
    p.add_argument("--inputs-from", default=None, metavar="CHECKOUT",
                   help="roast barista checkout whose files the training read: relative 'coffea4bees/...' "
                        "paths in the dataset options are pointed there (pinned metadata), while the code "
                        "runs from the current checkout (which has the fit-quality benchmark)")
    p.add_argument("--reference", default=None,
                   help="the original training's result.json: compare its final-epoch per-fold loss/AUCs "
                        "with the replayed Baseline ones (consistency.json)")
    p.add_argument("--archived-wf", default=None,
                   help="the train.yml archived by the original run, to compare with --train-wf "
                        "(e.g. when --train-wf was regenerated from the captured config)")
    p.add_argument("--tolerance", type=float, default=2e-3,
                   help="max |replay - original| per fold for loss and AUCs to call the replay faithful")
    p.add_argument("--run", action="store_true", help="run the replay and the summary, not just write the workflow")
    return p.parse_args(argv)


def _flag(item) -> str | None:
    if isinstance(item, str) and item.startswith("--"):
        return item.split()[0]
    return None


def rewrite_model_options(options: list, pretrained: str, floor: float | None) -> list:
    """Zero-epoch training, no finetuning, warm start from ``pretrained``."""
    out = []
    i = 0
    while i < len(options):
        item = options[i]
        flag = _flag(item)
        has_kwargs = i + 1 < len(options) and isinstance(options[i + 1], dict)
        if flag == "--training":
            out += ["--training FixedStep", {"epoch": 0}]
            i += 2 if has_kwargs else 1
            continue
        if flag in ("--finetuning", "--pretrained-weights", "--fit-quality-floor"):
            i += 2 if (flag == "--finetuning" and has_kwargs) else 1
            continue
        out.append(item)
        i += 1
    if not any(_flag(o) == "--training" for o in out):
        out += ["--training FixedStep", {"epoch": 0}]
    out.append(f"--pretrained-weights {pretrained}")
    if floor is not None:
        out.append(f"--fit-quality-floor {floor}")
    return out


def _repoint(item, checkout: str):
    if isinstance(item, str):
        return re.sub(r"(?<![\w/:.])coffea4bees/", checkout.rstrip("/") + "/coffea4bees/", item)
    if isinstance(item, dict):
        return {k: _repoint(v, checkout) for k, v in item.items()}
    if isinstance(item, list):
        return [_repoint(v, checkout) for v in item]
    return item


def build_replay(train_wf: dict, pretrained: str, floor: float | None, inputs_from: str | None = None) -> dict:
    wf = copy.deepcopy(train_wf)
    if inputs_from:
        for d in wf.get("dataset") or []:
            d["option"] = _repoint(d.get("option") or [], os.path.abspath(os.path.expanduser(inputs_from)))
    models = wf.get("model") or []
    if not models:
        raise ValueError("train workflow has no 'model' section")
    for m in models:
        if not str(m.get("module", "")).endswith(".Train"):
            raise ValueError(f"model module {m.get('module')!r} is not a Train task")
        m["option"] = rewrite_model_options(m.get("option") or [], pretrained, floor)
    return wf


def stage_models(pretrained: str, dest: str) -> str:
    """Copy the fold models to a local directory.

    HCRTraining resolves ``--pretrained-weights <dir>`` with a local ``glob`` on
    ``*_offset{N}_*.pkl``, which cannot see root:// (EOS) paths.
    """
    import fsspec

    fs, path = fsspec.core.url_to_fs(pretrained)
    files = sorted(fs.glob(f"{path.rstrip('/')}/*_offset*_*.pkl"))
    if not files:
        raise SystemExit(f"no *_offset*_*.pkl models under {pretrained}")
    os.makedirs(dest, exist_ok=True)
    for f in files:
        local = os.path.join(dest, os.path.basename(f))
        if not os.path.exists(local):
            fs.get(f, local)
    print(f"staged {len(files)} models from {pretrained} -> {dest}")
    return dest


def _read_text(path: str) -> str:
    if "://" in path:
        import fsspec

        with fsspec.open(path, "rt") as f:
            return f.read()
    with open(path) as f:
        return f.read()


def _exists(path: str) -> bool:
    if "://" in path:
        import fsspec

        fs, p = fsspec.core.url_to_fs(path)
        return fs.exists(p)
    return os.path.exists(path)


def _per_fold(result: dict, stage: str) -> dict:
    """{offset: {"loss": x, "AUC: <name>": y, ...}} from the validation benchmarks of `stage`."""
    from src.classifier.fvt_benchmark import _select_benchmarks

    folds = {}
    for m in result.get("models", []):
        offset = (m.get("metadata") or {}).get("offset")
        for bm in _select_benchmarks([m], stage):
            val = bm.get("validation", {})
            row = {}
            if "loss" in val.get("scalars", {}):
                row["loss"] = val["scalars"]["loss"]
            for r in val.get("roc", []):
                if r.get("AUC") is not None:
                    row[f"AUC: {r['name']}"] = r["AUC"]
            folds[offset] = row
    return folds


def consistency(reference: dict, replay: dict, tolerance: float) -> dict:
    """Per-fold original final-epoch vs replayed Baseline validation loss/AUCs."""
    ref, rep = _per_fold(reference, "final"), _per_fold(replay, "baseline")
    rows, worst = [], 0.0
    for offset in sorted(set(ref) | set(rep), key=str):
        a, b = ref.get(offset, {}), rep.get(offset, {})
        for k in sorted(set(a) & set(b)):
            d = b[k] - a[k]
            worst = max(worst, abs(d))
            rows.append({"offset": offset, "metric": k, "original": a[k], "replay": b[k], "diff": d})
    missing = sorted(str(o) for o in set(ref) ^ set(rep))
    return {
        "tolerance": tolerance,
        "max_abs_diff": worst,
        "n_compared": len(rows),
        "missing_folds": missing,
        "faithful": bool(rows) and not missing and worst <= tolerance,
        "rows": rows,
    }


def _check_common(path: str | None):
    if not path:
        return
    with open(path) as f:
        text = f.read()
    if "disable_benchmark" in text and "true" in text.lower().split("disable_benchmark", 1)[1][:20]:
        raise SystemExit(f"{path} disables benchmarks; the replay needs them (remove disable_benchmark)")


def main(argv=None):
    args = parse_args(argv)
    out = args.output.rstrip("/")
    if "://" in out:
        raise SystemExit("--output must be a local directory (the fresh-directory guard cannot check remote paths)")
    if out == args.pretrained.rstrip("/"):
        raise SystemExit("--output must not be the original model directory")
    if os.path.exists(out) and any(
        f.startswith("checkpoint_") or f.endswith((".pt", ".pkl")) or f == "result.json" for f in os.listdir(out)
    ):
        raise SystemExit(f"{out} already holds a result/checkpoint; use a fresh directory "
                         "(a stale checkpoint would be resumed instead of the saved model)")
    os.makedirs(out, exist_ok=True)
    # local copies of the inputs, so -from and the record never depend on a remote read
    train_wf_text = _read_text(args.train_wf)
    with open(os.path.join(out, "input_train.yml"), "w") as f:
        f.write(train_wf_text)
    common = None
    if args.common:
        common = os.path.join(out, "input_common.yml")
        with open(common, "w") as f:
            f.write(_read_text(args.common))
    _check_common(common)
    if args.archived_wf:
        record = {"train_wf": args.train_wf, "archived_wf": args.archived_wf}
        if _exists(args.archived_wf):
            record["identical"] = yaml.safe_load(_read_text(args.archived_wf)) == yaml.safe_load(train_wf_text)
        else:
            record["identical"] = None  # nothing archived to compare with
        with open(os.path.join(out, "train_wf_check.json"), "w") as f:
            json.dump(record, f, indent=2)
        print(f"train workflow vs archived: {record['identical']}")

    pretrained = args.pretrained
    if "://" in pretrained:
        pretrained = stage_models(pretrained, os.path.join(os.path.abspath(out), "pretrained"))

    wf = yaml.safe_load(train_wf_text)
    replay = build_replay(wf, pretrained, args.floor, args.inputs_from)

    replay_path = os.path.join(out, "replay_train.yml")
    with open(replay_path, "w") as f:
        f.write(f"# fit-quality replay of {args.train_wf}\n# pretrained: {args.pretrained}\n")
        yaml.safe_dump(replay, f, sort_keys=False)

    template = f"model: {out}" + (f", {args.template}" if args.template else "")
    cmd = ["python", "-m", "src.classifier.task.main", "template", "{" + template + "}", replay_path]
    if common:
        cmd += ["-from", common]
    cmd += ["-flag", "debug"]
    summary = [sys.executable, "-m", "src.classifier.fvt_benchmark",
               "--result", os.path.join(out, "result.json"), "--stage", "baseline", "--json"]

    # Every path the training task writes goes to `out`: IO output (result.json, the re-saved
    # model, diagnostics) via the template, stage checkpoints via CLASSIFIER_CHECKPOINT_DIR, the
    # loaded-yields dump via HCR_YIELDS_FILE. CLASSIFIER_FORCE_RETRAIN stops the "model already
    # trained" shortcut (find_existing_model), which would skip the Baseline benchmark.
    replay_env = {
        "CLASSIFIER_CHECKPOINT_DIR": os.path.abspath(out),
        "HCR_YIELDS_FILE": os.path.join(os.path.abspath(out), "yields_loaded.json"),
        "CLASSIFIER_FORCE_RETRAIN": "1",
    }
    print(f"replay workflow: {replay_path}")
    print("train :", " ".join(f"{k}={shlex.quote(v)}" for k, v in replay_env.items()),
          " ".join(shlex.quote(c) for c in cmd))
    print("summary:", " ".join(shlex.quote(c) for c in summary))
    if not args.run:
        return
    env = dict(os.environ, **replay_env)
    with open(os.path.join(out, "replay.log"), "w") as log:
        rc = subprocess.call(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    if rc:
        raise SystemExit(f"replay failed (exit {rc}); see {out}/replay.log")
    with open(os.path.join(out, "fit_quality.json"), "w") as f:
        subprocess.check_call(summary, stdout=f)
    subprocess.check_call(summary[:-1])  # human-readable table
    if args.reference:
        reference = json.loads(_read_text(args.reference))
        with open(os.path.join(out, "result.json")) as f:
            replayed = json.load(f)
        cons = consistency(reference, replayed, args.tolerance)
        with open(os.path.join(out, "consistency.json"), "w") as f:
            json.dump(cons, f, indent=2)
        print(f"replay vs original: max |diff| {cons['max_abs_diff']:.2e} over {cons['n_compared']} "
              f"fold metrics -> {'FAITHFUL' if cons['faithful'] else 'NOT FAITHFUL'}")


if __name__ == "__main__":
    main()
