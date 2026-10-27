"""
Split evaluation for the classifier workflow (``eval_splits`` in the workflow config).

:func:`split_workflow` derives one split's evaluate.yml from the full one: the split's dataset
options replace those of every dataset module, and the IO output moves to ``<output>/split_<name>``.
It needs only yaml/shlex, since snakemake runs it on the submit host (the NRP orchestrator image has
no barista dependencies).

``python -m src.classifier.workflow.eval_splits <split output dir> ...`` sums the split friend trees
(``result.json@@analysis.0.merged``) into ``<output>/result.json`` with the same layout, so the
consumers of a split evaluation read it exactly like an unsplit one.
"""

from __future__ import annotations

import copy
import posixpath
import shlex

SPLIT_PREFIX = "split_"
RESULT = "result.json"


def _flag(opt) -> str | None:
    if isinstance(opt, str):
        tokens = shlex.split(opt)
        if tokens and tokens[0].startswith("-"):
            return tokens[0]
    return None


def override_options(options: list, overrides: dict) -> list:
    """Replace each flag in ``overrides`` in a module option list.

    A value of ``None`` removes the flag, ``""`` passes it with no values, a list passes each item.
    """
    options = [o for o in (options or []) if _flag(o) not in overrides]
    for flag, value in overrides.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = " ".join(shlex.quote(str(v)) for v in value)
        options.append(f"{flag} {value}".rstrip())
    return options


def _template_mapping(template_str: str) -> dict:
    import yaml

    mapping = yaml.safe_load("{" + template_str + "}") if template_str else {}
    return {str(k): str(v) for k, v in (mapping or {}).items()}


def split_workflow(wf: dict, name: str, overrides: dict, template_str: str = "") -> tuple[dict, str]:
    """Return ``(split evaluate workflow, split output dir)`` for split ``name``."""
    wf = copy.deepcopy(wf)
    datasets = wf.get("dataset") or []
    if not datasets:
        raise ValueError("evaluate workflow has no dataset modules to split")
    for module in datasets:
        module["option"] = override_options(module.get("option"), overrides or {})

    mapping = _template_mapping(template_str)
    out_dir = None
    for setting in wf.get("setting") or []:
        if setting.get("module") != "IO":
            continue
        for opt in setting.get("option") or []:
            if isinstance(opt, dict) and "output" in opt:
                base = str(opt["output"]).format(**mapping).rstrip("/")
                out_dir = posixpath.join(base, f"{SPLIT_PREFIX}{name}")
                opt["output"] = out_dir
    if out_dir is None:
        raise ValueError("evaluate workflow has no IO output setting to split")
    if "{" in out_dir or "}" in out_dir:
        raise ValueError(f"unresolved template field in the IO output: {out_dir}")
    return wf, out_dir


def combine(split_dirs: list[str]) -> str:
    """Sum the split friend trees into ``<output>/result.json``; return its path."""
    import json
    import logging
    from functools import reduce
    import operator as op

    import fsspec

    from src.data_formats.root import Friend
    from src.utils.json import DefaultEncoder

    split_dirs = [d.rstrip("/") for d in split_dirs]
    bases = {posixpath.dirname(d) for d in split_dirs}
    if len(bases) != 1 or not all(posixpath.basename(d).startswith(SPLIT_PREFIX) for d in split_dirs):
        raise ValueError(f"split outputs must be <output>/{SPLIT_PREFIX}<name> of one output: {split_dirs}")
    base = bases.pop()

    results, friends = [], []
    for d in split_dirs:
        with fsspec.open(posixpath.join(d, RESULT), "rt") as f:
            result = json.load(f)
        friend = Friend.from_json(result["analysis"][0]["merged"])
        logging.info(f"{d}: {friend.name} {friend.n_entries} entries, {friend.n_fragments} fragments")
        results.append(result)
        friends.append(friend)
    # Friend.__add__ checks the names and branches; the chunks must not overlap
    merged = reduce(op.add, friends)
    n_split = sum(f.n_entries for f in friends)
    if merged.n_entries != n_split:
        raise ValueError(f"splits overlap: {merged.n_entries} merged entries vs {n_split} in the splits")

    combined = {
        "analysis": [{"merged": merged}],
        "predictions": [p for r in results for p in (r.get("predictions") or [])],
        "splits": [posixpath.join(d, RESULT) for d in split_dirs],
    }
    out = posixpath.join(base, RESULT)
    with fsspec.open(out, "wt") as f:
        f.write(json.dumps(combined, cls=DefaultEncoder))
    logging.info(f"{out}: {merged.name} {merged.n_entries} entries from {len(split_dirs)} splits")
    return out


if __name__ == "__main__":
    import logging
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if len(sys.argv) < 2:
        sys.exit("usage: python -m src.classifier.workflow.eval_splits <split output dir> ...")
    print(combine(sys.argv[1:]))
