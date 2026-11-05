#!/usr/bin/env python
"""HTML report of FvT / MvD fit-quality benchmarks (replayed or online).

Two steps, so the plotting (matplotlib) can run where the classifier image is and the page
assembly anywhere:

  entry:   one training's result.json (+ the replay's consistency / workflow checks) ->
           <out>/entry.json and <out>/plots/*.png: the closure and transfer histograms summed
           over folds, with a ratio panel, and every metric per fold.
  summary: several entry directories -> one self-contained-ish HTML page (tables + the plots,
           linked relative to the page).

  python -m src.classifier.fit_quality_report entry --result replay/result.json --stage baseline \\
      --consistency replay/consistency.json --wf-check replay/train_wf_check.json \\
      --name fvt_2223 --title "FvT 2022+2023" --out report/fvt_2223
  python -m src.classifier.fit_quality_report summary --entries report/* --out report/index.html

Generalization: per fold k, gap_k = validation - training for a metric (both from the same
benchmark pass); the table compares the mean gap with the fold-to-fold spread of the
validation metric (sigma_fold, the k-fold estimate of its statistical noise) and with the
standard error of the mean gap, std(gap_k)/sqrt(K).
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import sys

HIST_TITLES = {
    "pt4_sb": ("4th candidate-jet p$_T$ [GeV] (SB)", "4b data", "model"),
    "pt1_sb": ("leading candidate-jet p$_T$ [GeV] (SB)", "4b data", "model"),
    "fvt_sb": ("transfer weight w (SB)", "4b data", "model"),
    "t3_sb": ("p$_{t3}$ (SB)", "t3 MC", "source x p$_{t3}$/p$_{src}$"),
    "t4_sb": ("p$_{t4}$ (SB)", "t4 MC", "source x p$_{t4}$/p$_{src}$"),
    "t3_sr": ("p$_{t3}$ (SR)", "t3 MC", "source x p$_{t3}$/p$_{src}$"),
    "t4_sr": ("p$_{t4}$ (SR)", "t4 MC", "source x p$_{t4}$/p$_{src}$"),
}
MODEL_LABEL = {"d3": "3b data x w + 4b ttbar", "mix4": "mixed x w + 4b ttbar"}


# ----------------------------------------------------------------------------- reading
def _stages(model: dict, stage: str):
    from src.classifier.fvt_benchmark import _select_benchmarks

    return list(_select_benchmarks([model], stage))


def _fold_rows(result: dict, stage: str) -> list[dict]:
    """One dict per fold: offset, and for each split its scalars, AUCs and fit_quality."""
    folds = []
    for m in result.get("models", []):
        bms = _stages(m, stage)
        if not bms:
            continue
        bm = bms[-1]
        row = {"offset": (m.get("metadata") or {}).get("offset")}
        for split in ("training", "validation"):
            b = bm.get(split, {})
            fq = (b.get("fit_quality") or [{}])[0]
            metrics = {}
            if "loss" in b.get("scalars", {}):
                metrics["loss"] = b["scalars"]["loss"]
            for r in b.get("roc", []):
                if r.get("AUC") is not None:
                    metrics[f"AUC {r['name']}"] = r["AUC"]
            for k, v in (fq.get("proxies") or {}).items():
                if k.startswith("loss_") and v is not None:
                    metrics[k] = v
            row[split] = {"metrics": metrics, "fit_quality": fq}
        folds.append(row)
    return sorted(folds, key=lambda r: (r["offset"] is None, r["offset"]))


def _mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def _std(v):
    v = [x for x in v if x is not None]
    if len(v) < 2:
        return None
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def _sum_hists(fqs: list[dict]) -> dict:
    """Sum the per-fold validation histograms (sum w and sum w^2 add)."""
    out = {}
    for fq in fqs:
        for k, h in (fq.get("histograms") or {}).items():
            if k not in out:
                out[k] = {key: list(h[key]) for key in h if isinstance(h[key], list)}
                continue
            for key in ("target", "model", "target_sumw2", "model_sumw2"):
                if key in h and key in out[k]:
                    out[k][key] = [a + b for a, b in zip(out[k][key], h[key])]
    return out


def _chi2_ndf(h: dict):
    t, m = h.get("target", []), h.get("model", [])
    t2, m2 = h.get("target_sumw2"), h.get("model_sumw2")
    st, sm = sum(t), sum(m)
    if not t or st <= 0 or sm <= 0:
        return None, 0
    k = st / sm
    chi2, n = 0.0, 0
    for i in range(len(t)):
        var = (t2[i] if t2 else abs(t[i])) + k**2 * (m2[i] if m2 else abs(m[i]) * k)
        if t[i] > 0 and var > 0:
            chi2 += (t[i] - k * m[i]) ** 2 / var
            n += 1
    return (chi2 / (n - 1), n - 1) if n > 1 else (None, 0)


# ----------------------------------------------------------------------------- entry
def _plot(hname: str, h: dict, source: str, path: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    edges = np.array(h["bin_edges"])
    centers, widths = 0.5 * (edges[1:] + edges[:-1]), np.diff(edges)
    t, m = np.array(h["target"]), np.array(h["model"])
    t2 = np.array(h.get("target_sumw2") or np.abs(t))
    m2 = np.array(h.get("model_sumw2") or np.abs(m))
    k = t.sum() / m.sum() if m.sum() > 0 else 1.0
    title, tl, ml = HIST_TITLES.get(hname, (hname, "target", "model"))
    if hname in ("pt4_sb", "pt1_sb", "fvt_sb"):
        ml = MODEL_LABEL.get(source, ml)
    chi2, ndf = _chi2_ndf(h)

    fig, (ax, rx) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, facecolor="white",
                                 gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})
    ax.stairs(k * m, edges, color="#1f77b4", lw=1.8, label=f"{ml} (x{k:.3f})")
    ax.fill_between(np.repeat(edges, 2)[1:-1], np.repeat(k * (m - np.sqrt(m2)), 2),
                    np.repeat(k * (m + np.sqrt(m2)), 2), color="#1f77b4", alpha=0.25, step=None, lw=0)
    ax.errorbar(centers, t, yerr=np.sqrt(t2), fmt="o", ms=3.5, color="black", label=tl)
    ax.set_ylabel("events / bin")
    ax.set_yscale("log" if t.max() > 50 * max(t[t > 0].min(), 1e-9) else "linear")
    txt = f"$\\chi^2$/ndf = {chi2:.2f} ({ndf})" if chi2 is not None else "empty"
    ax.text(0.97, 0.95, txt, transform=ax.transAxes, ha="right", va="top")
    ax.legend(loc="upper right", bbox_to_anchor=(0.97, 0.86), frameon=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(m > 0, t / (k * m), np.nan)
        rerr = np.where(m > 0, np.sqrt(t2) / (k * m), np.nan)
        band = np.where(m > 0, np.sqrt(m2) / m, np.nan)
    rx.fill_between(np.repeat(edges, 2)[1:-1], np.repeat(1 - band, 2), np.repeat(1 + band, 2),
                    color="#1f77b4", alpha=0.25, lw=0)
    rx.errorbar(centers, ratio, yerr=rerr, fmt="o", ms=3.5, color="black")
    rx.axhline(1, color="grey", lw=1)
    rx.set_ylim(0.8, 1.2)
    rx.set_ylabel("target / model")
    rx.set_xlabel(title)
    for a in (ax, rx):
        a.set_facecolor("white")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def make_entry(args):
    from src.classifier.fvt_benchmark import closure_train_val

    with open(args.result) as f:
        result = json.load(f)
    folds = _fold_rows(result, args.stage)
    if not folds:
        raise SystemExit(f"no '{args.stage}' benchmarks in {args.result}")
    val_fqs = [r["validation"]["fit_quality"] for r in folds if r["validation"]["fit_quality"]]
    source = (val_fqs[0].get("source") if val_fqs else None) or "d3"

    # generalization: gap per fold vs fold spread
    names = sorted({k for r in folds for k in r["validation"]["metrics"]} &
                   {k for r in folds for k in r["training"]["metrics"]})
    gen = []
    for n in names:
        val = [r["validation"]["metrics"].get(n) for r in folds]
        tr = [r["training"]["metrics"].get(n) for r in folds]
        gaps = [v - t for v, t in zip(val, tr) if v is not None and t is not None]
        sf = _std(val)
        se = (_std(gaps) / math.sqrt(len(gaps))) if _std(gaps) is not None else None
        g = _mean(gaps)
        gen.append({"metric": n, "val": _mean(val), "train": _mean(tr), "gap": g, "sigma_fold": sf,
                    "se_gap": se, "gap_over_sigma_fold": (g / sf) if (g is not None and sf) else None,
                    "per_fold_gap": gaps})

    def fold_mean(path):
        vals = []
        for fq in val_fqs:
            d = fq
            for p in path:
                d = d.get(p) if isinstance(d, dict) else None
            vals.append(d if isinstance(d, (int, float)) and not isinstance(d, bool) else None)
        return _mean(vals), _std(vals)

    tables = {
        "weights": {k: fold_mean(("weights", k)) for k in ("neg_fraction", "p99", "max", "mean", "std")},
        "ttbar": {k: fold_mean(("ttbar_mc_comparison", k)) for k in
                  ("calib_t4_prob_to_mc", "calib_t3_prob_to_mc", "transfer_d3_to_t4_to_mc",
                   "transfer_d3_to_t3_to_mc", "true_ratio_t4_t3", "pred_ratio_t4_t3")},
        "data_calibration": {k: fold_mean(("data_calibration", k)) for k in
                             ("calib_d4_prob_to_data", "calib_d3_prob_to_data")},
        "closure_fold_mean": {k: fold_mean(("closure", k)) for k in
                              ("chi2_ndf_pt4", "chi2_ndf_pt1", "chi2_ndf_fvt", "norm_closure_sb")},
        "regional": {k: (_mean([r["validation"]["metrics"].get(k) for r in folds]),
                         _std([r["validation"]["metrics"].get(k) for r in folds]))
                     for k in names if k.startswith("loss_") or " SB" in k or " SR" in k},
    }
    summed = _sum_hists(val_fqs)
    os.makedirs(os.path.join(args.out, "plots"), exist_ok=True)
    hist_summary = {}
    for hname, h in summed.items():
        if not h.get("target") or sum(h["target"]) <= 0:
            continue
        png = f"plots/{hname}.png"
        _plot(hname, h, source, os.path.join(args.out, png))
        chi2, ndf = _chi2_ndf(h)
        hist_summary[hname] = {"png": png, "chi2_ndf": chi2, "ndf": ndf,
                               "norm_model_over_target": (sum(h["model"]) / sum(h["target"])) if sum(h["target"]) else None}

    def load(p):
        if p and os.path.exists(p):
            with open(p) as f:
                return json.load(f)
        return None

    cons, wfc = load(args.consistency), load(args.wf_check)
    entry = {
        "name": args.name, "title": args.title or args.name, "source": source,
        "floor": (val_fqs[0].get("source_floor") if val_fqs else None),
        "n_folds": len(folds), "stage": args.stage,
        "faithful": cons.get("faithful") if cons else None,
        "max_abs_diff": cons.get("max_abs_diff") if cons else None,
        "train_wf_identical": wfc.get("identical") if wfc else None,
        "generalization": gen, "tables": tables, "hists": hist_summary,
        "closure_train_val": closure_train_val(result.get("models", []), args.stage),
        "per_fold": [{"offset": r["offset"], **{f"val {k}": v for k, v in r["validation"]["metrics"].items()}}
                     for r in folds],
    }
    with open(os.path.join(args.out, "entry.json"), "w") as f:
        json.dump(entry, f, indent=2)
    print(f"wrote {args.out}/entry.json and {len(hist_summary)} plots")


# ----------------------------------------------------------------------------- summary
CSS = """
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:24px;color:#222;background:#fff;max-width:1400px}
h1{font-size:22px} h2{font-size:18px;margin-top:32px;border-bottom:1px solid #ddd;padding-bottom:4px}
h3{font-size:15px;margin-top:20px} p.note{color:#555;font-size:13px;max-width:1000px}
table{border-collapse:collapse;font-size:13px;margin:8px 0}
th,td{border:1px solid #ddd;padding:3px 8px;text-align:right;white-space:nowrap}
th{background:#f4f4f4} td.l,th.l{text-align:left}
.bad{color:#b00020;font-weight:600} .ok{color:#1b7f3b}
.grid{display:flex;flex-wrap:wrap;gap:8px} .grid figure{margin:0;width:440px}
.grid img{width:100%;border:1px solid #eee} figcaption{font-size:12px;color:#555;text-align:center}
"""


def _f(v, fmt="{:.4g}"):
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else '<span class="bad">NO</span>'
    if isinstance(v, (int, float)):
        return fmt.format(v)
    return html.escape(str(v))


def _pm(pair, fmt="{:.4g}"):
    m, s = pair if isinstance(pair, (list, tuple)) else (pair, None)
    if m is None:
        return "–"
    return _f(m, fmt) + (f" ± {_f(s, fmt)}" if s is not None else "")


def _table(headers, rows):
    out = ["<table><tr>" + "".join(f'<th class="{"l" if i == 0 else ""}">{h}</th>' for i, h in enumerate(headers)) + "</tr>"]
    for r in rows:
        out.append("<tr>" + "".join(f'<td class="{"l" if i == 0 else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>")
    return "\n".join(out + ["</table>"])


def make_summary(args):
    entries = []
    for d in args.entries:
        p = os.path.join(d, "entry.json")
        if os.path.exists(p):
            with open(p) as f:
                e = json.load(f)
            e["_dir"] = os.path.relpath(d, os.path.dirname(os.path.abspath(args.out)) or ".")
            entries.append(e)
    if not entries:
        raise SystemExit("no entry.json found")
    S = []
    S.append(f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(args.title)}</title>"
             f"<style>{CSS}</style></head><body><h1>{html.escape(args.title)}</h1>")
    S.append("<p class='note'>Fit-quality benchmarks (src/classifier/ml/benchmarks/fit_quality.py) of each training, "
             "evaluated on its own validation split per k-fold and averaged over folds (± = fold-to-fold standard "
             "deviation). Closure histograms compare, in the SB, the 4b data with the background model the analysis "
             "uses there (source × w + 4b ttbar MC, raw weights; source = 3b data for FvT, mixed/declustered data for "
             "MvD); χ² uses both sides' Σw² with the model normalised to the data, and the plots sum the folds.</p>")

    S.append("<h2>Replay faithfulness</h2>")
    S.append(_table(["run", "folds", "replay = original (max |Δ| val loss/AUC)", "train.yml = archived", "weight floor"],
                    [[html.escape(e["title"]), e["n_folds"],
                      (_f(e["faithful"]) + f" ({_f(e['max_abs_diff'], '{:.1e}')})") if e["faithful"] is not None else "–",
                      _f(e["train_wf_identical"]) if e["train_wf_identical"] is not None else "– (none archived)",
                      _f(e["floor"])] for e in entries]))

    S.append("<h2>Sideband closure and component transfer (χ²/ndf, folds summed)</h2>")
    hn = ["pt4_sb", "pt1_sb", "fvt_sb", "t3_sb", "t4_sb", "t3_sr", "t4_sr"]
    S.append(_table(["run", "norm model/data (SB)"] + hn,
                    [[html.escape(e["title"]), _f((e["hists"].get("pt1_sb") or {}).get("norm_model_over_target"))] +
                     [(_f(e["hists"][h]["chi2_ndf"], "{:.2f}") + f" ({e['hists'][h]['ndf']})") if h in e["hists"] else "–"
                      for h in hn] for e in entries]))

    S.append("<h2>Sideband closure: training vs validation split</h2>")
    S.append("<p class='note'>χ²/ndf of each fold's closure histogram on its training and on its validation events "
             "(mean ± std over the folds, ddof=1), and of the folds summed. A fixed mis-modelling grows χ²/ndf − 1 in "
             "proportion to the number of events, and the training split holds K−1 times the validation events, so the "
             "two splits are compared through the training χ²/ndf normalized to the fold's validation size, "
             "1 + (χ²/ndf<sub>train</sub> − 1)·N<sub>val</sub>/N<sub>train</sub> (compare with χ²/ndf val), and "
             "the excess per 100k 4b SB events, (χ²/ndf − 1)/(N/10⁵): equal excess = "
             "the same bias on seen and unseen events (under-/mis-fit); training excess well below validation = "
             "overtraining (train/val ratio &lt; 1).</p>")
    ctv_rows = []
    for e in entries:
        for hname, c in (e.get("closure_train_val") or {}).items():
            tr, va = c.get("training") or {}, c.get("validation") or {}

            def pmk(d, key):
                return _pm((d.get(f"{key}_mean"), d.get(f"{key}_std")), "{:.2f}")

            ratio = c.get("excess_ratio_train_over_val")
            ctv_rows.append([html.escape(e["title"]), hname, pmk(tr, "chi2_ndf"), pmk(tr, "chi2_ndf_valsize"),
                             pmk(va, "chi2_ndf"),
                             _f(tr.get("chi2_ndf_summed"), "{:.2f}"), _f(va.get("chi2_ndf_summed"), "{:.2f}"),
                             pmk(tr, "excess_per_100k"), pmk(va, "excess_per_100k"),
                             (f'<span class="bad">{ratio:.2f}</span>' if ratio is not None and ratio < 0.5
                              else _f(ratio, "{:.2f}")),
                             ", ".join(f"{x:.2f}" for x in tr.get("chi2_ndf_per_fold", [])),
                             ", ".join(f"{x:.2f}" for x in va.get("chi2_ndf_per_fold", []))])
    if ctv_rows:
        S.append(_table(["run", "hist", "χ²/ndf train", "χ²/ndf train (normalized to val size)", "χ²/ndf val",
                         "train Σfolds", "val Σfolds",
                         "excess/100k train", "excess/100k val", "excess train/val",
                         "per-fold train", "per-fold val"], ctv_rows))
    else:
        S.append("<p class='note'>(entries predate this table; rerun <code>fit_quality_report entry</code>)</p>")

    S.append("<h2>Transfer weights</h2>")
    wk = ["neg_fraction", "p99", "max", "mean", "std"]
    S.append(_table(["run"] + wk, [[html.escape(e["title"])] + [_pm(e["tables"]["weights"][k]) for k in wk] for e in entries]))
    S.append("<p class='note'>Over the source events of the validation split (max/p99/mean/std from a sample of up to "
             "50k weights per fold; with the run's analysis floor on the source probability).</p>")

    S.append("<h2>ttbar preservation and calibration</h2>")
    tk = ["calib_t4_prob_to_mc", "transfer_d3_to_t4_to_mc", "calib_t3_prob_to_mc", "transfer_d3_to_t3_to_mc",
          "true_ratio_t4_t3", "pred_ratio_t4_t3"]
    dk = ["calib_d4_prob_to_data", "calib_d3_prob_to_data"]
    S.append(_table(["run"] + tk + dk, [[html.escape(e["title"])] + [_pm(e["tables"]["ttbar"][k]) for k in tk] +
                                        [_pm(e["tables"]["data_calibration"][k]) for k in dk] for e in entries]))

    S.append("<h2>Regional loss and discrimination (validation)</h2>")
    rk = sorted({k for e in entries for k in e["tables"]["regional"]})
    S.append(_table(["run"] + rk, [[html.escape(e["title"])] + [_pm(e["tables"]["regional"].get(k)) for k in rk]
                                   for e in entries]))

    S.append("<h2>Generalization: train–validation gap vs k-fold spread</h2>")
    S.append("<p class='note'>gap = mean over folds of (validation − training); σ<sub>fold</sub> = standard deviation of the "
             "validation metric across folds (its statistical noise as seen by the k-fold); SE = std of the per-fold gaps / √K. "
             "|gap| ≫ σ<sub>fold</sub> (and ≫ SE) means the model fits its training events better than the fold-to-fold "
             "noise explains, i.e. overtraining; |gap| ≲ σ<sub>fold</sub> means no measurable overtraining. "
             "For losses a positive gap is the overtraining direction, for AUCs a negative one.</p>")
    for e in entries:
        S.append(f"<h3>{html.escape(e['title'])}</h3>")
        S.append(_table(["metric", "validation", "training", "gap", "σ_fold", "SE(gap)", "gap/σ_fold", "per-fold gaps"],
                        [[html.escape(g["metric"]), _f(g["val"], "{:.5f}"), _f(g["train"], "{:.5f}"),
                          _f(g["gap"], "{:+.5f}"), _f(g["sigma_fold"], "{:.5f}"), _f(g["se_gap"], "{:.5f}"),
                          (f'<span class="bad">{g["gap_over_sigma_fold"]:+.2f}</span>'
                           if g["gap_over_sigma_fold"] is not None and abs(g["gap_over_sigma_fold"]) > 2
                           else _f(g["gap_over_sigma_fold"], "{:+.2f}")),
                          ", ".join(f"{x:+.5f}" for x in g["per_fold_gap"])] for g in e["generalization"]]))

    S.append("<h2>Plots</h2>")
    for e in entries:
        S.append(f"<h3>{html.escape(e['title'])}</h3><div class='grid'>")
        for h in hn:
            if h in e["hists"]:
                S.append(f"<figure><a href='{e['_dir']}/{e['hists'][h]['png']}'><img src='{e['_dir']}/{e['hists'][h]['png']}'></a>"
                         f"<figcaption>{h}: χ²/ndf {_f(e['hists'][h]['chi2_ndf'], '{:.2f}')}</figcaption></figure>")
        S.append("</div>")
    S.append("</body></html>")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(S))
    print(f"wrote {args.out} ({len(entries)} runs)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("entry")
    e.add_argument("--result", required=True)
    e.add_argument("--stage", default="baseline", choices=["baseline", "final"])
    e.add_argument("--consistency", default=None)
    e.add_argument("--wf-check", default=None)
    e.add_argument("--name", required=True)
    e.add_argument("--title", default=None)
    e.add_argument("--out", required=True)
    s = sub.add_parser("summary")
    s.add_argument("--entries", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--title", default="Fit-quality benchmarks")
    args = p.parse_args(argv)
    (make_entry if args.cmd == "entry" else make_summary)(args)


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    main()
