"""Derive and validate the b-tagging SF calibration.

The calibration is a per-sample map in (Njet, HT),

    calib = N(without b-tag SF) / N(with b-tag SF),

derived for the central b-tag SF and for each of its variations. N.B. no b-tag requirement must
be applied to the events used to derive it.

Input: a coffea file whose ``hists`` have the categorical axes ``process``, ``year`` and
``variation`` and follow this naming convention (``<map>`` is set with ``--sf-hist``):

    <map>_noBtagSF          (Njet, HT), weights without the b-tag SF, variation "central"
    <map>_btagSF            (Njet, HT), weights with the b-tag SF, one entry per SF variation
    <map>_btagSFcross       (Njet, HT), optional, weight = w_noSF * sqrt(SF); its variances give
                            sum(w^2 SF), needed for the statistical uncertainty of the ratio
    <var>_noBtagSF          (var, Njet, HT), diagnostic variable, without the b-tag SF
    <var>_btagSF            (var, Njet, HT), diagnostic variable, with the b-tag SF
    <var>_btagSF_calib      (var, Njet, HT), optional, calibration applied in the processor

The last two axes of every histogram are the (Njet, HT) axes of the map. Since the calibration
is constant inside a (Njet, HT) bin, the calibrated distribution of any diagnostic variable is
obtained exactly from <var>_btagSF by reweighting these bins: no second pass on the events is
needed for the closure test.

Usage:
    python -m src.tools.compute_btagSF_calibration -i hists.coffea -o outdir -c   # derive maps
    python -m src.tools.compute_btagSF_calibration -i hists.coffea -o outdir -v   # closure plots
"""

import argparse
import logging
import os

import hist
import numpy as np
import yaml
from coffea.util import load

from src.physics.btag_calibration import (
    CENTRAL,
    CORRECTION_NAME,
    INPUT_HT,
    INPUT_NJET,
    INPUT_SAMPLE,
    INPUT_VARIATION,
)

NO_SF = "noBtagSF"
SF = "btagSF"
SF_CROSS = "btagSFcross"
SF_CALIB = "btagSF_calib"
CATEGORIES = ("process", "year", "variation")


def fold_flow(arr: np.ndarray) -> np.ndarray:
    """Add the under/overflow of every axis to the first/last bin and drop the flow bins.

    The correction is evaluated with ``flow="clamp"``, so out-of-range events get the value of
    the edge bin; folding makes the derivation consistent with that.
    """
    arr = np.array(arr, dtype=float)
    for ax in range(arr.ndim):
        arr = np.moveaxis(arr, ax, 0)
        arr[1] += arr[0]
        arr[-2] += arr[-1]
        arr = np.moveaxis(arr[1:-1], 0, ax)
    return arr


def select(h: hist.Hist, process: str, year: str, variation: str = CENTRAL):
    """Values and variances (flow folded) of ``h`` for one process, year and variation."""
    for name, value in zip(CATEGORIES, (process, year, variation)):
        if value not in h.axes[name]:
            return None, None
    sub = h[{"process": process, "year": year, "variation": variation}]
    return fold_flow(sub.values(flow=True)), fold_flow(sub.variances(flow=True))


def compute_ratio(num, num_var, den, den_var, cross_var=None):
    """Ratio no-SF / SF and its statistical uncertainty.

    Numerator and denominator are the same events with different weights, so they are fully
    correlated. With sum(w^2 SF) available (``cross_var``) the uncertainty is the exact
    linearized one, var(R) = [sum(w^2) - 2 R sum(w^2 SF) + R^2 sum(w^2 SF^2)] / D^2.
    Without it the yields are treated as independent, which overestimates the uncertainty.
    """
    ok = (num > 0) & (den > 0)
    safe_den = np.where(ok, den, 1.0)
    ratio = np.where(ok, num / safe_den, 1.0)
    if cross_var is not None:
        var = (num_var - 2 * ratio * cross_var + ratio**2 * den_var) / safe_den**2
    else:
        var = num_var / safe_den**2 + (num / safe_den**2) ** 2 * den_var
    err = np.where(ok, np.sqrt(np.clip(var, 0, None)), 0.0)
    return ratio, err


class CalibrationInput:
    """Access to the map histograms of a coffea file."""

    def __init__(self, hists: dict, sf_hist: str):
        self.hists = hists
        self.sf_hist = sf_hist
        try:
            self.h_no = hists[f"{sf_hist}_{NO_SF}"]
            self.h_sf = hists[f"{sf_hist}_{SF}"]
        except KeyError as e:
            raise KeyError(f"Histogram {e} not found in the input file (see --sf-hist)") from e
        self.h_cross = hists.get(f"{sf_hist}_{SF_CROSS}")
        if self.h_cross is None:
            logging.warning(
                f"{sf_hist}_{SF_CROSS} not found: the statistical uncertainty of the calibration "
                "is computed assuming uncorrelated yields and is overestimated"
            )
        self.processes = list(self.h_sf.axes["process"])
        self.years = list(self.h_sf.axes["year"])
        self.variations = list(self.h_sf.axes["variation"])
        if CENTRAL not in self.variations:
            raise ValueError(f"Variation '{CENTRAL}' not found in {sf_hist}_{SF}")
        self.njet_axis, self.ht_axis = self.h_sf.axes[-2], self.h_sf.axes[-1]

    def yields(self, process, year, variation=CENTRAL):
        """(num, num_var, den, den_var, cross_var) for one map; None if the sample is empty."""
        num, num_var = select(self.h_no, process, year)
        den, den_var = select(self.h_sf, process, year, variation)
        if num is None or den is None or num.sum() == 0:
            return None
        cross_var = None
        if self.h_cross is not None and variation == CENTRAL:
            _, cross_var = select(self.h_cross, process, year)
        return num, num_var, den, den_var, cross_var

    def calibration(self, process, year, variation=CENTRAL):
        y = self.yields(process, year, variation)
        if y is None:
            return None, None
        return compute_ratio(*y)


def write_calibration(inp: CalibrationInput, year: str, output_dir: str, min_neff: float):
    """Write the correctionlib file of one year. Returns the maps, keyed by (process, variation)."""
    import correctionlib.convert
    import correctionlib.schemav2 as cs

    maps = {}
    samples = []
    for process in inp.processes:
        ratio, err = inp.calibration(process, year)
        if ratio is None:
            continue
        samples.append(process)
        num, num_var, *_ = inp.yields(process, year)
        neff = np.where(num_var > 0, num**2 / np.where(num_var > 0, num_var, 1.0), 0.0)
        low = (neff < min_neff) & (num > 0)
        if low.any():
            logging.warning(
                f"{process} {year}: {int(low.sum())}/{low.size} (Njet, HT) bins have fewer than "
                f"{min_neff:g} effective MC events (min {neff[num > 0].min():.1f}); consider merging bins"
            )
        empty = int((num <= 0).sum())
        if empty:
            logging.warning(f"{process} {year}: {empty}/{num.size} (Njet, HT) bins are empty, calibration set to 1")
        for variation in inp.variations:
            r, e = inp.calibration(process, year, variation)
            maps[(process, variation)] = (r, e)

    if not samples:
        logging.warning(f"No sample with events for year {year}: no file written")
        return maps

    sfhist = hist.Hist(
        hist.axis.StrCategory(samples, name=INPUT_SAMPLE, label="MC sample"),
        hist.axis.StrCategory(inp.variations, name=INPUT_VARIATION, label="b-tag SF variation"),
        hist.axis.Variable(inp.njet_axis.edges, name=INPUT_NJET, label="Jet multiplicity"),
        hist.axis.Variable(inp.ht_axis.edges, name=INPUT_HT, label="HT [GeV]"),
        storage=hist.storage.Double(),
    )
    for i, process in enumerate(samples):
        for j, variation in enumerate(inp.variations):
            sfhist.view()[i, j] = maps[(process, variation)][0]

    sfhist.name = CORRECTION_NAME
    sfhist.label = "out"
    correction = correctionlib.convert.from_histogram(sfhist, flow="clamp")
    correction.description = (
        "b-tagging SF calibration: N(no b-tag SF) / N(b-tag SF) in bins of (Njet, HT), "
        f"per MC sample and b-tag SF variation, year {year}"
    )
    cset = cs.CorrectionSet(schema_version=2, description=f"b-tagging SF calibration {year}", corrections=[correction])

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"btagSF_calibrationSF_{year}.json")
    with open(path, "w") as f:
        f.write(cset.model_dump_json(exclude_unset=True, indent=1))
    logging.info(f"Wrote {path} ({len(samples)} samples, {len(inp.variations)} variations)")
    return maps


# ---------------------------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------------------------
def _setup_style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import mplhep as hep

        plt.style.use(hep.style.CMS)
    except ImportError:
        pass
    return plt


def plot_maps(inp: CalibrationInput, maps: dict, year: str, output_dir: str, variations: list[str]):
    """2D maps of the calibration and of its statistical uncertainty."""
    plt = _setup_style()
    xe, ye = inp.njet_axis.edges, inp.ht_axis.edges
    # Draw the bins with equal width: the HT binning is very non-uniform
    xi, yi = np.arange(len(xe)), np.arange(len(ye))
    for (process, variation), (ratio, err) in maps.items():
        if variation not in variations:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(22, 9))
        for ax, values, title, fmt in (
            (axes[0], ratio, "Calibration", "{:.3f}"),
            (axes[1], err, "Stat. uncertainty", "{:.3f}"),
        ):
            mesh = ax.pcolormesh(xi, yi, values.T, cmap="cividis")
            fig.colorbar(mesh, ax=ax)
            for ix in range(len(xe) - 1):
                for iy in range(len(ye) - 1):
                    ax.text(ix + 0.5, iy + 0.5, fmt.format(values[ix, iy]), ha="center", va="center",
                            fontsize=9, color="white" if values[ix, iy] < np.mean(values) else "black")
            ax.set_xticks(xi, [f"{e:g}" for e in xe])
            ax.set_yticks(yi, [f"{e:g}" for e in ye])
            ax.set_xlabel("Jet multiplicity")
            ax.set_ylabel("$H_T$ [GeV]")
            ax.set_title(f"{title}: {process} {year} ({variation})", fontsize=16)
        outdir = os.path.join(output_dir, "maps", year)
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, f"btagSF_calibration_{process}_{year}_{variation}.png"), dpi=120, bbox_inches="tight")
        fig.savefig(os.path.join(outdir, f"btagSF_calibration_{process}_{year}_{variation}.pdf"), bbox_inches="tight")
        plt.close(fig)


def diagnostic_variables(hists: dict, sf_hist: str) -> list[str]:
    """Variables with both a <var>_noBtagSF and a <var>_btagSF histogram."""
    variables = []
    for name in hists:
        if not name.endswith(f"_{NO_SF}"):
            continue
        var = name[: -len(f"_{NO_SF}")]
        if var == sf_hist or f"{var}_{SF}" not in hists:
            continue
        if hists[name].ndim != len(CATEGORIES) + 3:
            continue
        variables.append(var)
    return variables


def plot_shapes(axis, entries: dict, title: str, path: str):
    """Overlay 1D distributions with a ratio panel to the first entry.

    entries: {label: (values, variances)}
    """
    plt = _setup_style()
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})
    edges = axis.edges
    centers = 0.5 * (edges[1:] + edges[:-1])
    colors = ["black", "tab:red", "tab:blue", "tab:green"]
    styles = ["-", "-", "--", ":"]
    ref = None
    for (label, (values, variances)), color, ls in zip(entries.items(), colors, styles):
        ax.stairs(values, edges, label=f"{label} ({values.sum():.4g})", color=color, linestyle=ls, linewidth=2)
        ax.errorbar(centers, values, yerr=np.sqrt(variances), fmt="none", color=color, linewidth=1)
        if ref is None:
            ref = values
            rax.axhline(1.0, color=color, linewidth=1)
            continue
        r = np.divide(values, ref, out=np.full_like(values, np.nan), where=ref > 0)
        rax.stairs(r, edges, color=color, linestyle=ls, linewidth=2)
    ax.set_ylabel("Events")
    ax.set_yscale("log")
    ax.legend(title=title, fontsize=14, title_fontsize=14)
    rax.set_ylabel("Ratio to no SF")
    rax.set_ylim(0.7, 1.3)
    rax.set_xlabel(axis.label or axis.name)
    rax.grid(axis="y", alpha=0.4)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def validate(inp: CalibrationInput, output_dir: str):
    """Closure test: no SF vs SF vs SF x calibration, for every diagnostic variable."""
    variables = diagnostic_variables(inp.hists, inp.sf_hist)
    if not variables:
        logging.warning("No diagnostic histograms (<var>_noBtagSF / <var>_btagSF) found")
    summary = {}
    for year in inp.years:
        for process in inp.processes:
            calib, _ = inp.calibration(process, year)
            if calib is None:
                continue
            num, _, den, _, _ = inp.yields(process, year)
            entry = {
                "noBtagSF": float(num.sum()),
                "btagSF": float(den.sum()),
                "btagSF_calib": float((den * calib).sum()),
            }
            entry["btagSF_over_noBtagSF"] = entry["btagSF"] / entry["noBtagSF"]
            entry["btagSF_calib_over_noBtagSF"] = entry["btagSF_calib"] / entry["noBtagSF"]

            for var in variables:
                h_no, h_sf = inp.hists[f"{var}_{NO_SF}"], inp.hists[f"{var}_{SF}"]
                no, no_var = select(h_no, process, year)
                sf, sf_var = select(h_sf, process, year)
                if no is None or sf is None:
                    continue
                entries = {
                    "no b-tag SF": (no.sum(axis=(1, 2)), no_var.sum(axis=(1, 2))),
                    "b-tag SF": (sf.sum(axis=(1, 2)), sf_var.sum(axis=(1, 2))),
                    "b-tag SF x calibration": ((sf * calib).sum(axis=(1, 2)), (sf_var * calib**2).sum(axis=(1, 2))),
                }
                # Calibration applied event by event in the processor, if available: it must
                # agree with the offline reweighting above.
                h_applied = inp.hists.get(f"{var}_{SF_CALIB}")
                if h_applied is not None:
                    applied, applied_var = select(h_applied, process, year)
                    if applied is not None and applied.sum() != 0:
                        applied_1d = applied.sum(axis=(1, 2))
                        entries["b-tag SF x calibration (processor)"] = (applied_1d, applied_var.sum(axis=(1, 2)))
                        offline = entries["b-tag SF x calibration"][0]
                        scale = np.abs(offline).max() or 1.0
                        diff = float(np.abs(applied_1d - offline).max() / scale)
                        entry.setdefault("processor_vs_offline_max_rel_diff", {})[var] = diff
                plot_shapes(
                    h_no.axes[len(CATEGORIES)], entries, f"{process} {year}",
                    os.path.join(output_dir, "closure", year, var, f"{var}_{process}_{year}_calibrated.png"),
                )
            summary.setdefault(year, {})[process] = entry
            logging.info(
                f"{process} {year}: SF/noSF = {entry['btagSF_over_noBtagSF']:.4f}, "
                f"SF x calib / noSF = {entry['btagSF_calib_over_noBtagSF']:.6f}"
            )

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "btagSF_calibration_yields.yml")
    with open(path, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    logging.info(f"Wrote {path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", required=True, help="Input coffea file with the calibration histograms")
    parser.add_argument("-o", "--output", required=True, help="Output folder")
    parser.add_argument("-c", "--compute", action="store_true", help="Compute the calibration and write one correctionlib JSON per year")
    parser.add_argument("-v", "--validate", action="store_true", help="Closure plots and yield table")
    parser.add_argument("--sf-hist", default="Njet_HT", help="Name (without suffix) of the map histograms")
    parser.add_argument("--min-neff", type=float, default=100.0, help="Warn for bins with fewer effective MC events")
    parser.add_argument("--plot-variations", default=CENTRAL, help="Comma-separated variations for the 2D map plots, or 'all'")
    parser.add_argument("--no-plots", action="store_true", help="Do not produce the 2D map plots")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not (args.compute or args.validate):
        parser.error("Nothing to do: use -c and/or -v")

    inp = CalibrationInput(load(args.input)["hists"], args.sf_hist)
    logging.info(f"Processes: {inp.processes}\nYears: {inp.years}\nVariations: {inp.variations}")

    if args.compute:
        variations = inp.variations if args.plot_variations == "all" else args.plot_variations.split(",")
        for year in inp.years:
            maps = write_calibration(inp, year, args.output, args.min_neff)
            if not args.no_plots:
                plot_maps(inp, maps, year, args.output, variations)
    if args.validate:
        validate(inp, args.output)


if __name__ == "__main__":
    main()
