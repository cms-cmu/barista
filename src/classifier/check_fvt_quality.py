#!/usr/bin/env python3
"""FvT Fit Quality Gate Check.

Evaluates FvT closure / transfer weight chi2/ndf against a maximum threshold.
If the model passes, touches an output token file and exits 0.
If the model fails (poor closure, diverging weights), logs a detailed diagnostic
and exits 1, halting the workflow before downstream steps (Phase C.4, Phase D, Phase F)
can run.
"""

import argparse
import math
import sys
from pathlib import Path

# Add repo root to python path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.classifier.training_summary import parse_fit_quality_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", required=True, help="Path to model directory")
    parser.add_argument(
        "--max-chi2",
        type=float,
        default=5.0,
        help="Maximum allowed chi2/ndf for w_FvT closure (default: 5.0)",
    )
    parser.add_argument(
        "--metric",
        default="chi2_ndf_fvt",
        help="Primary closure metric to check (default: chi2_ndf_fvt)",
    )
    parser.add_argument(
        "--output-token",
        required=True,
        help="Path to touch on successful quality gate passage",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        default=True,
        help="Pass with warning if fit quality benchmarks were not recorded (default: True)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    models_dir = Path(args.models_dir).resolve()
    token_path = Path(args.output_token).resolve()

    # Locate result.json
    res_candidates = [
        models_dir / "result.json",
        models_dir / "models" / "result.json",
    ]
    res_file = None
    for cand in res_candidates:
        if cand.exists():
            res_file = cand
            break

    if not res_file:
        # Check subdirectories if ensemble
        for sub in sorted(models_dir.glob("mix_*")):
            if (sub / "result.json").exists():
                res_file = sub / "result.json"
                break
            if (sub / "models" / "result.json").exists():
                res_file = sub / "models" / "result.json"
                break

    if not res_file:
        if args.allow_missing:
            print(
                f"[WARNING] FvT Quality Gate: No result.json found in {models_dir}. "
                "Fit quality benchmark was not recorded. Bypassing gate.",
                file=sys.stderr,
            )
            token_path.parent.mkdir(parents=True, exist_ok=True)
            token_path.touch()
            sys.exit(0)
        else:
            print(
                f"[ERROR] FvT Quality Gate: No result.json found in {models_dir}!",
                file=sys.stderr,
            )
            sys.exit(1)

    fq = parse_fit_quality_json(res_file)

    # Determine metric to check
    metric_name = args.metric
    val = fq.get(metric_name)

    # Fallback to composite or pt4 closure if fvt_sb was not recorded
    if val is None:
        for fb in ["composite_chi2_ndf", "chi2_ndf_pt4", "chi2_ndf_pt1"]:
            if fq.get(fb) is not None:
                val = fq.get(fb)
                metric_name = fb
                print(
                    f"[INFO] Metric '{args.metric}' not recorded, using fallback '{fb}' = {val:.3f}",
                    file=sys.stderr,
                )
                break

    if val is None:
        if args.allow_missing:
            print(
                f"[WARNING] FvT Quality Gate: No closure metrics found in {res_file}. "
                "Fit quality benchmark empty. Bypassing gate.",
                file=sys.stderr,
            )
            token_path.parent.mkdir(parents=True, exist_ok=True)
            token_path.touch()
            sys.exit(0)
        else:
            print(
                f"[ERROR] FvT Quality Gate: Closure metrics missing in {res_file}!",
                file=sys.stderr,
            )
            sys.exit(1)

    max_chi2 = args.max_chi2
    is_failed = math.isnan(val) or math.isinf(val) or (val > max_chi2)

    summary_html = models_dir / "training_summary.html"
    analyze_html = models_dir / "plots" / "analyze" / "index.html"

    if is_failed:
        print("\n" + "=" * 80, file=sys.stderr)
        print("[ERROR] FvT Fit Quality Gate FAILED!", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print(
            f"  Metric:         {metric_name}\n"
            f"  Measured Value: {val:.3f}\n"
            f"  Threshold:      <= {max_chi2:.2f}\n",
            file=sys.stderr,
        )
        print(
            "  The FvT transfer reweighting does not sufficiently close in the Sideband.\n"
            "  Continuing downstream (Phase C.4 closure, Phase D / SvB training, Phase F)\n"
            "  would corrupt background modeling.\n",
            file=sys.stderr,
        )
        print("  All diagnostic outputs and websites have been generated for inspection:", file=sys.stderr)
        if summary_html.exists():
            print(f"    - Training Summary: {summary_html}", file=sys.stderr)
        if analyze_html.exists():
            print(f"    - Analyze Gallery:  {analyze_html}", file=sys.stderr)
        print("=" * 80 + "\n", file=sys.stderr)
        sys.exit(1)

    # Gate passed
    print(
        f"[INFO] FvT Fit Quality Gate PASSED: {metric_name} = {val:.3f} <= {max_chi2:.2f}",
        file=sys.stderr,
    )
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.touch()
    sys.exit(0)


if __name__ == "__main__":
    main()
