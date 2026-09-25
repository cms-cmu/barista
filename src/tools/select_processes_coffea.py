"""Keep only some processes of a coffea histogram file, local or remote.

Reads a ``.coffea`` output (lz4-compressed cloudpickle, as written by coffea.util.save) from a
local path or any fsspec URL (e.g. ``root://cmseos.fnal.gov//store/...``, read over xrootd, no
copy), and writes a local file holding only the requested processes:

  * every histogram under a ``*hists*`` key is sliced to those categories of its ``process``
    axis (histograms with none of them, or without a ``process`` axis, are dropped);
  * every other dict (cutflows, ``reproducible``, ...) keeps the entries whose dataset key
    starts with one of the processes (``TTToHadronic__2022_EE``, ``TTToHadronic_2022_EEE``, ...).

The result merges like a singlefile of those processes (src/tools/merge_coffea_files.py). Used
to reuse another roast's merged, archived products without reprocessing, e.g. its Phase B.1
ttbar MC in a Phase C.4-only roast.

Usage:
    python src/tools/select_processes_coffea.py root://.../computeJCM/histAll_wJCM.coffea \
        -o output/.../hist__TTbar_upstream_wJCM.coffea -p TTToHadronic TTToSemiLeptonic TTTo2L2Nu
"""

import argparse
import logging
import sys
from pathlib import Path

# 'src' parent on the path so cloudpickle can resolve modules pickled with 'src.*' references
_src_parent = str(Path(__file__).resolve().parent.parent.parent)
if _src_parent not in sys.path:
    sys.path.insert(0, _src_parent)

logger = logging.getLogger("select_processes_coffea")


def load_any(url: str):
    """coffea.util.load, but through fsspec so a remote URL is read in place."""
    import cloudpickle
    import fsspec
    import lz4.frame
    with fsspec.open(url, "rb") as raw, lz4.frame.open(raw) as f:
        return cloudpickle.load(f)


def select_hist(h, processes: list):
    if "process" not in getattr(getattr(h, "axes", None), "name", ()):
        return None
    keep = [p for p in processes if p in list(h.axes["process"])]
    if not keep:
        return None
    return h[{"process": keep}]


def matches(key, processes: list) -> bool:
    return isinstance(key, str) and any(key == p or key.startswith(p + "_") for p in processes)


def select(output: dict, processes: list) -> dict:
    out = {}
    for key, val in output.items():
        if "hists" in key and isinstance(val, dict):
            kept = {name: s for name, h in val.items() if (s := select_hist(h, processes)) is not None}
            logger.info(f"{key}: kept {len(kept)} of {len(val)} histograms")
            out[key] = kept
        elif isinstance(val, dict):
            kept = {k: v for k, v in val.items() if matches(k, processes)}
            logger.info(f"{key}: kept {sorted(kept)}")
            out[key] = kept
        else:
            out[key] = val
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="local .coffea file or fsspec URL (root://...)")
    ap.add_argument("-o", "--output", required=True, help="local output .coffea file")
    ap.add_argument("-p", "--processes", nargs="+", required=True, help="process names to keep")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    from coffea.util import save
    logger.info(f"reading {args.input}")
    out = select(load_any(args.input), args.processes)
    n_hists = sum(len(v) for k, v in out.items() if "hists" in k and isinstance(v, dict))
    if not n_hists:
        raise SystemExit(f"none of {args.processes} found in any histogram of {args.input}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save(out, args.output)
    logger.info(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
