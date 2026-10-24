from __future__ import annotations

import io
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import chain, cycle
from typing import TYPE_CHECKING

import fsspec

from src.classifier.config.setting import IO, ResultKey
from src.classifier.task import Analysis, ArgParser

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


class LossROC(Analysis):
    argparser = ArgParser()

    def analyze(self, results: list[dict]):
        return _collect_loss_roc(results=results)


def _walk_benchmark_hyperparameter(history):
    for stage in history:
        name = stage["name"]
        if "benchmarks" in stage:
            yield stage["benchmarks"], {"stage": name}
        elif "training" in stage:
            for epoch in stage["training"]:
                pars = epoch["hyperparameters"] | {"stage": name}
                if lr := pars.get("learning rate", None):
                    pars["learning rate"] = lr[0]
                yield epoch["benchmarks"], pars


def _dict_list():
    return defaultdict(list)


def _dict_dict():
    return defaultdict(dict)


@dataclass
class GroupData:
    phase: pd.DataFrame
    classifiers: list = field(default_factory=list)
    data: dict = field(default_factory=dict)
    rocs: dict = field(default_factory=_dict_dict)


class _collect_loss_roc:
    _target = {"scalars", "roc"}

    @classmethod
    def _collect_data(cls, results: list):
        import pandas as pd
        import src.data_formats.numpy as npext

        plot = {}
        datasets = set()
        int64 = {"epoch"}

        groups: list[GroupData] = []

        for model in chain.from_iterable(map(lambda x: x[ResultKey.models], results)):
            name = model["name"].replace("__", ",").replace("_", ":")
            _data = defaultdict(list)
            _rocs = defaultdict(_dict_list)
            _phases = []
            for benchmark, hyperparameter in _walk_benchmark_hyperparameter(
                model["history"]
            ):
                if not benchmark or any(
                    not (cls._target <= set(v)) for v in benchmark.values()
                ):
                    continue
                datasets.update(benchmark)
                for k, v in benchmark.items():
                    # roc
                    aucs = {
                        f"AUC: {r['name']}": r["AUC"]
                        for r in v["roc"]
                        if r["AUC"] is not None
                    }
                    plot.update(aucs)
                    for r in v["roc"]:
                        if r["FPR"] is None or r["TPR"] is None:
                            _rocs[f"ROC: {r['name']}"][k].append(
                                pd.DataFrame(
                                    {
                                        "FPR": pd.array([], dtype="float64"),
                                        "TPR": pd.array([], dtype="float64"),
                                    }
                                )
                            )
                        else:
                            _rocs[f"ROC: {r['name']}"][k].append(
                                pd.DataFrame(
                                    {
                                        "FPR": npext.from_.base64(r["FPR"]),
                                        "TPR": npext.from_.base64(r["TPR"]),
                                    }
                                )
                            )
                    # scalars
                    scalars = v["scalars"]
                    plot.update(scalars)
                    # update data
                    _data[k].append(scalars | aucs)
                _phases.append(hyperparameter)
                for k, v in hyperparameter.items():
                    if isinstance(v, int):
                        int64.add(k)
            _phases = pd.DataFrame(_phases)
            for k in int64:
                _phases[k] = _phases[k].astype(pd.Int64Dtype())
            group = None
            for target_group in groups:
                if target_group.phase.equals(_phases):
                    group = target_group
                    break
            if group is None:
                group = GroupData(phase=_phases)
                groups.append(group)
            group.classifiers.append(name)
            group.data |= {(name, k): pd.DataFrame(v) for k, v in _data.items()}
            for k, v in _rocs.items():
                group.rocs[k] |= {(name, kk): vv for kk, vv in v.items()}

        groups.sort(
            key=lambda grp: (
                tuple(grp.phase.columns),
                tuple(map(tuple, grp.phase.values)),
            )
        )
        return plot, datasets, groups

    def __new__(cls, results: list, inline_resources=False):
        plot, datasets, groups = cls._collect_data(results)

        jobs = []
        for idx, group in enumerate(groups):
            jobs.append(
                _generate_group_diagnostics_and_gallery(
                    group=idx,
                    classifiers=sorted(group.classifiers),
                    plot_data=group.data,
                    rocs=group.rocs,
                    phase=group.phase,
                )
            )
        return jobs


class _generate_group_diagnostics_and_gallery:
    def __init__(self, group: int, classifiers: list[str], plot_data: dict, rocs: dict, phase):
        self._group = group + 1
        self._classifiers = classifiers
        self._plot_data = plot_data
        self._rocs = rocs
        self._phase = phase

    def __call__(self):
        import logging
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        try:
            import mplhep as hep
            plt.style.use(hep.style.CMS)
        except Exception:
            pass

        from jinja2 import Environment, FileSystemLoader
        from src.classifier.monitor import Index
        from src.storage.eos import EOS

        report_dir = IO.report / "HCR"
        report_dir.mkdir(recursive=True)

        # 1. Filter training stage
        if "stage" in self._phase and (self._phase["stage"] == "Training").any():
            mask = self._phase["stage"] == "Training"
        else:
            mask = slice(None)

        if "epoch" in self._phase:
            epochs = np.asarray(self._phase.loc[mask, "epoch"].values, dtype=int)
        else:
            epochs = None

        val_losses, tr_losses = [], []
        lr_series = []
        for c in self._classifiers:
            val_df = self._plot_data.get((c, "validation"))
            tr_df = self._plot_data.get((c, "training"))
            if val_df is not None and "loss" in val_df and tr_df is not None and "loss" in tr_df:
                val_losses.append(np.asarray(val_df.loc[mask, "loss"].values, dtype=float))
                tr_losses.append(np.asarray(tr_df.loc[mask, "loss"].values, dtype=float))
                if "learning rate" in self._phase:
                    lr_series.append(np.asarray(self._phase.loc[mask, "learning rate"].values, dtype=float))

        if not val_losses or not tr_losses:
            logging.warning(f"No loss data available for Group {self._group}")
            return

        val_mat = np.array(val_losses)
        tr_mat = np.array(tr_losses)
        k_folds, n_epochs = val_mat.shape
        if epochs is None or len(epochs) != n_epochs:
            epochs = np.arange(1, n_epochs + 1)

        mean_val = np.mean(val_mat, axis=0)
        std_val = np.std(val_mat, axis=0, ddof=1) if k_folds > 1 else np.zeros_like(mean_val)
        mean_tr = np.mean(tr_mat, axis=0)
        gap = mean_val - mean_tr
        z_pull = np.where(std_val > 1e-7, gap / std_val, 0.0) if k_folds > 1 else np.zeros_like(mean_val)

        best_idx = int(np.argmin(mean_val))
        best_ep = int(epochs[best_idx])
        best_val = float(mean_val[best_idx])
        best_tr = float(mean_tr[best_idx])
        best_gap = float(gap[best_idx])
        best_std = float(std_val[best_idx])
        best_z = float(z_pull[best_idx]) if k_folds > 1 else 0.0

        if k_folds > 1:
            if best_z <= 1.0:
                status_text, status_badge = "Optimal (Z ≤ 1.0σ)", "optimal"
            elif best_z <= 2.0:
                status_text, status_badge = "Mild Divergence (1.0σ < Z ≤ 2.0σ)", "mild"
            else:
                status_text, status_badge = "Overtrained (Z > 2.0σ)", "overtrained"
        else:
            status_text, status_badge = "Single Fold", "optimal"

        logging.info(
            f"[K-Fold Loss Analysis Group {self._group}] K={k_folds} folds: "
            f"Best Ep={best_ep}, Val Loss={best_val:.5f}, Tr Loss={best_tr:.5f}, "
            f"Gap={best_gap:+.5f}, Fold σ={best_std:.5f}, Z={best_z:+.2f}σ ({status_text})"
        )

        gallery_items = []

        def _save_fig(fig, filename, name, group_name):
            out_path = report_dir / filename
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
            plt.close(fig)
            with fsspec.open(out_path, "wb") as f:
                f.write(buf.getvalue())
            gallery_items.append({
                "path": filename,
                "group": group_name,
                "name": name,
                "kind": "img",
            })

        # -------------------------------------------------------------
        # Plot 1: K-Fold Loss & Statistical Divergence Pull
        # -------------------------------------------------------------
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 7.5), sharex=True, gridspec_kw={"height_ratios": [2, 1.2]}
        )
        for i in range(k_folds):
            ax1.plot(epochs, val_mat[i], color="gray", linestyle=":", alpha=0.55,
                     label="Individual Folds (Val)" if i == 0 else None)
        ax1.plot(epochs, mean_val, color="#1f77b4", marker="o", markersize=5, linewidth=2,
                 label=f"Mean Val Loss (min: {best_val:.5f} @ ep {best_ep})")
        if k_folds > 1:
            ax1.fill_between(epochs, mean_val - std_val, mean_val + std_val, color="#1f77b4",
                             alpha=0.18, label=r"Fold Spread ($\pm 1\sigma_{\mathrm{fold}}$)")
        ax1.plot(epochs, mean_tr, color="#ff7f0e", marker="s", markersize=4, linewidth=1.8,
                 linestyle="--", label=f"Mean Train Loss ({best_tr:.5f} @ ep {best_ep})")
        ax1.axvline(best_ep, color="#d62728", linestyle="--", alpha=0.8, label=f"Best Epoch ({best_ep})")
        ax1.set_ylabel(r"Loss $\mathcal{L}$", fontsize=13)
        ax1.set_title(f"Cross-Validation Loss & Statistical Divergence (K={k_folds})", fontsize=14, fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="upper right", fontsize=9.5, framealpha=0.95)

        if k_folds > 1:
            max_z = max(3.2, float(np.max(z_pull)) * 1.2)
            ax2.axhspan(-0.5, 1.0, color="green", alpha=0.14, label=r"Optimal ($Z \leq 1.0\sigma$)")
            ax2.axhspan(1.0, 2.0, color="yellow", alpha=0.14, label=r"Mild Divergence ($1.0\sigma < Z \leq 2.0\sigma$)")
            ax2.axhspan(2.0, max_z, color="red", alpha=0.10, label=r"Overtrained ($Z > 2.0\sigma$)")
            ax2.axhline(0, color="gray", linestyle="-", linewidth=0.8)
            ax2.axhline(1.0, color="green", linestyle="--", linewidth=1.0)
            ax2.axhline(2.0, color="orange", linestyle="--", linewidth=1.0)
            ax2.plot(epochs, z_pull, color="#2ca02c", marker="D", markersize=5, linewidth=2,
                     label=f"Divergence Pull Z ({best_z:+.2f}σ @ ep {best_ep})")
            ax2.set_ylabel(r"Pull $Z = \Delta L / \sigma$", fontsize=11)
            ax2.set_ylim(-0.5, max_z)
        else:
            ax2.plot(epochs, gap, color="purple", marker="o", linewidth=2, label=r"Generalization Gap $\Delta L$")
            ax2.set_ylabel(r"Gap $\Delta L$", fontsize=11)

        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_xlim(float(epochs[0]) - 0.2, float(epochs[-1]) + 0.2)
        ax2.set_xticks(epochs)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="upper left", fontsize=9, framealpha=0.95)
        plt.tight_layout()
        _save_fig(fig, f"kfold-loss-divergence-{self._group}.png", "Loss & Divergence Pull", "1. Loss & Divergence")

        # -------------------------------------------------------------
        # Plot 2: Learning Rate Schedule vs Epoch
        # -------------------------------------------------------------
        if lr_series:
            mean_lr = np.mean(lr_series, axis=0)
            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.plot(epochs, mean_lr, color="#9467bd", marker="o", linewidth=2, label="Learning Rate")
            ax.axvline(best_ep, color="#d62728", linestyle="--", alpha=0.8, label=f"Best Epoch ({best_ep})")
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("Learning Rate", fontsize=12)
            ax.set_title("Learning Rate Evolution vs Epoch", fontsize=13, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="upper right", fontsize=9.5, framealpha=0.95)
            plt.tight_layout()
            _save_fig(fig, f"learning-rate-{self._group}.png", "Learning Rate Schedule", "1. Loss & Divergence")

        # -------------------------------------------------------------
        # Plot 3: Per-Fold Val Loss & Best Epoch Consistency
        # -------------------------------------------------------------
        if k_folds > 1:
            fig, (ax_bar1, ax_bar2) = plt.subplots(1, 2, figsize=(10, 4))
            fold_min_vals = [float(np.min(val_mat[i])) for i in range(k_folds)]
            fold_best_eps = [int(epochs[np.argmin(val_mat[i])]) for i in range(k_folds)]
            fold_labels = [f"Fold {i}" for i in range(k_folds)]

            ax_bar1.bar(fold_labels, fold_min_vals, color="#1f77b4", alpha=0.85, edgecolor="#0f4c81")
            ax_bar1.set_ylabel("Min Validation Loss", fontsize=11)
            ax_bar1.set_title("Fold Min Val Loss", fontsize=12, fontweight="bold")
            ax_bar1.grid(axis="y", linestyle="--", alpha=0.5)

            ax_bar2.bar(fold_labels, fold_best_eps, color="#ff7f0e", alpha=0.85, edgecolor="#c65d00")
            ax_bar2.set_ylabel("Best Epoch", fontsize=11)
            ax_bar2.set_title("Fold Best Epoch", fontsize=12, fontweight="bold")
            ax_bar2.grid(axis="y", linestyle="--", alpha=0.5)

            plt.tight_layout()
            _save_fig(fig, f"fold-consistency-{self._group}.png", "Fold Consistency", "3. Fold Performance")

        # -------------------------------------------------------------
        # Plot 3: AUC Evolution vs Epoch
        # -------------------------------------------------------------
        # Discover all AUC metrics tracked in group data
        auc_keys = []
        if len(self._classifiers) > 0:
            first_val_df = self._plot_data.get((self._classifiers[0], "validation"))
            if first_val_df is not None:
                auc_keys = [col for col in first_val_df.columns if col.startswith("AUC:")]

        summary_aucs = {}
        for auc_metric in sorted(auc_keys):
            auc_name = auc_metric.replace("AUC: ", "")
            fig, ax = plt.subplots(figsize=(9, 5))
            
            val_auc_mat = []
            tr_auc_mat = []
            for c in self._classifiers:
                v_df = self._plot_data.get((c, "validation"))
                t_df = self._plot_data.get((c, "training"))
                if v_df is not None and auc_metric in v_df:
                    val_auc_mat.append(np.asarray(v_df.loc[mask, auc_metric].values, dtype=float))
                if t_df is not None and auc_metric in t_df:
                    tr_auc_mat.append(np.asarray(t_df.loc[mask, auc_metric].values, dtype=float))

            if val_auc_mat:
                v_mat = np.array(val_auc_mat)
                m_val = np.mean(v_mat, axis=0)
                s_val = np.std(v_mat, axis=0, ddof=1) if len(val_auc_mat) > 1 else np.zeros_like(m_val)
                best_val_auc = float(m_val[best_idx]) if best_idx < len(m_val) else float(m_val[-1])
                best_std_auc = float(s_val[best_idx]) if best_idx < len(s_val) else 0.0

                summary_aucs[auc_name] = f"{best_val_auc:.4f}" + (f" ± {best_std_auc:.4f}" if best_std_auc > 0 else "")

                for i in range(len(val_auc_mat)):
                    ax.plot(epochs, v_mat[i], color="gray", linestyle=":", alpha=0.45,
                            label="Individual Folds (Val)" if i == 0 else None)
                ax.plot(epochs, m_val, color="#1f77b4", marker="o", markersize=4, linewidth=2,
                        label=f"Val AUC ({best_val_auc:.4f} @ ep {best_ep})")
                if len(val_auc_mat) > 1:
                    ax.fill_between(epochs, m_val - s_val, m_val + s_val, color="#1f77b4",
                                    alpha=0.18, label=r"Fold Spread ($\pm 1\sigma$)")

            if tr_auc_mat:
                t_mat = np.array(tr_auc_mat)
                m_tr = np.mean(t_mat, axis=0)
                best_tr_auc = float(m_tr[best_idx]) if best_idx < len(m_tr) else float(m_tr[-1])
                ax.plot(epochs, m_tr, color="#ff7f0e", marker="s", markersize=3.5, linewidth=1.8,
                        linestyle="--", label=f"Train AUC ({best_tr_auc:.4f} @ ep {best_ep})")

            ax.axvline(best_ep, color="#d62728", linestyle="--", alpha=0.8, label=f"Best Epoch ({best_ep})")
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel(f"AUC ({auc_name})", fontsize=12)
            ax.set_title(f"AUC Evolution: {auc_name}", fontsize=13, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="lower right", fontsize=9.5, framealpha=0.95)
            plt.tight_layout()

            safe_name = auc_name.replace(" ", "_").replace("/", "_").lower()
            _save_fig(fig, f"auc-evolution-{safe_name}-{self._group}.png", f"AUC: {auc_name}", "2. AUC Evolution")

        # -------------------------------------------------------------
        # Plot 4+: ROC Curves (Linear and Semi-Log Rejection at Best Epoch)
        # -------------------------------------------------------------
        for roc_key, roc_splits in sorted(self._rocs.items()):
            roc_title = roc_key.replace("ROC: ", "")
            fig, (ax_roc, ax_rej) = plt.subplots(1, 2, figsize=(12, 5.2))

            for split_name, color, ls in [("validation", "#1f77b4", "-"), ("training", "#ff7f0e", "--")]:
                fold_aucs = []
                fold_dfs = []
                for c in self._classifiers:
                    series = roc_splits.get((c, split_name), [])
                    # Filter for training stage
                    tr_series = [df for df, is_tr in zip(series, mask) if is_tr]
                    if tr_series:
                        # Extract curve at best epoch index
                        chosen_df = tr_series[best_idx] if best_idx < len(tr_series) else tr_series[-1]
                        if chosen_df is not None and not chosen_df.empty and "FPR" in chosen_df and "TPR" in chosen_df:
                            fpr = chosen_df["FPR"].values
                            tpr = chosen_df["TPR"].values
                            if len(fpr) > 1:
                                _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
                                fold_auc = float(np.abs(_trapz(tpr, fpr)))
                                fold_aucs.append(fold_auc)
                                fold_dfs.append(chosen_df)
                                if k_folds > 1 and split_name == "validation":
                                    ax_roc.plot(fpr, tpr, color="gray", linestyle=":", alpha=0.4)

                # Representative fold / mean curve
                if fold_dfs:
                    rep_df = fold_dfs[0]
                    fpr = rep_df["FPR"].values
                    tpr = rep_df["TPR"].values
                    mean_auc = np.mean(fold_aucs) if fold_aucs else 0.0
                    std_auc = np.std(fold_aucs) if len(fold_aucs) > 1 else 0.0
                    label_str = f"{split_name.capitalize()} (AUC: {mean_auc:.4f}" + (f" ± {std_auc:.4f})" if std_auc > 0 else ")")

                    # If summary_aucs doesn't already have this ROC, add it
                    if split_name == "validation" and roc_title not in summary_aucs:
                        summary_aucs[roc_title] = f"{mean_auc:.4f}" + (f" ± {std_auc:.4f}" if std_auc > 0 else "")

                    ax_roc.plot(fpr, tpr, color=color, linestyle=ls, linewidth=2, label=label_str)

                    # Background rejection: 1 / (FPR + eps)
                    eps = 1e-4
                    valid_mask = (fpr > 0) & (tpr > 0.1)
                    if np.any(valid_mask):
                        rej = 1.0 / np.maximum(fpr[valid_mask], eps)
                        ax_rej.plot(tpr[valid_mask], rej, color=color, linestyle=ls, linewidth=2, label=label_str)

            ax_roc.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, alpha=0.5)
            ax_roc.set_xlabel("False Positive Rate (FPR)", fontsize=11)
            ax_roc.set_ylabel("True Positive Rate (TPR)", fontsize=11)
            ax_roc.set_title(f"ROC: {roc_title} (Epoch {best_ep})", fontsize=12, fontweight="bold")
            ax_roc.grid(True, linestyle="--", alpha=0.5)
            ax_roc.legend(loc="lower right", fontsize=9.5, framealpha=0.95)

            ax_rej.set_yscale("log")
            ax_rej.set_xlabel("Signal Efficiency (TPR)", fontsize=11)
            ax_rej.set_ylabel("Background Rejection (1 / FPR)", fontsize=11)
            ax_rej.set_title(f"Rejection: {roc_title} (Epoch {best_ep})", fontsize=12, fontweight="bold")
            ax_rej.grid(True, linestyle="--", alpha=0.5, which="both")
            ax_rej.legend(loc="upper right", fontsize=9.5, framealpha=0.95)

            plt.tight_layout()
            safe_name = roc_title.replace(" ", "_").replace("/", "_").lower()
            _save_fig(fig, f"roc-{safe_name}-{self._group}.png", f"ROC: {roc_title}", "4. ROC & Rejection Curves")

        # -------------------------------------------------------------
        # 5. Assemble Summary Dashboard Data & Render Single HTML
        # -------------------------------------------------------------
        summary_cards = [
            {
                "title": "Validation Loss",
                "value": f"{best_val:.5f}",
                "sub": f"Minimum reached at Epoch {best_ep}",
            },
            {
                "title": "Generalization Gap",
                "value": f"{best_gap:+.5f}",
                "sub": f"Fold Spread σ = {best_std:.5f}",
            },
            {
                "title": "Overtraining Status",
                "value": f"{best_z:+.2f}σ",
                "badge": status_text,
                "badge_type": status_badge,
            },
        ]
        if summary_aucs:
            main_roc_name = list(summary_aucs.keys())[0]
            summary_cards.append({
                "title": f"ROC AUC ({main_roc_name})",
                "value": summary_aucs[main_roc_name],
                "sub": f"{k_folds}-Fold Mean @ Best Ep",
            })

        summary_table = [
            {"metric": "Number of Folds (K)", "value": str(k_folds), "note": "Cross-validation partitions"},
            {"metric": "Best Epoch", "value": str(best_ep), "note": "Epoch with lowest mean validation loss"},
            {"metric": "Min Validation Loss ⟨L_val⟩", "value": f"{best_val:.5f}", "note": "Mean across folds"},
            {"metric": "Training Loss ⟨L_train⟩", "value": f"{best_tr:.5f}", "note": "At best epoch"},
            {"metric": "Generalization Gap ΔL", "value": f"{best_gap:+.5f}", "note": "⟨L_val⟩ - ⟨L_train⟩"},
            {"metric": "Fold Spread σ_fold", "value": f"{best_std:.5f}", "note": "Sample standard deviation across folds"},
            {"metric": "Statistical Divergence Pull Z", "value": f"{best_z:+.2f}σ", "note": "ΔL / σ_fold (Optimal: Z ≤ 1.0σ)"},
        ]
        for r_name, r_auc in summary_aucs.items():
            summary_table.append({
                "metric": f"Validation AUC: {r_name}",
                "value": str(r_auc),
                "note": "Evaluation at best/last epoch",
            })

        template_dir = EOS(__file__).parent.parent.parent.parent / "monitor" / "template" / "html"
        if not (template_dir / "classifier_report.html").exists:
            # Fallback to repo root src/ path if __file__ is in a different relative location
            fallback_dir = EOS("src/classifier/monitor/template/html")
            if (fallback_dir / "classifier_report.html").exists:
                template_dir = fallback_dir
        env = Environment(loader=FileSystemLoader(str(template_dir)))
        tpl = env.get_template("classifier_report.html")

        gallery_html = tpl.render(
            title=f"Classifier Benchmark Report - Group {self._group}",
            summary_cards=summary_cards,
            summary_table=summary_table,
            items_json=json.dumps(gallery_items),
        )

        index_html_path = report_dir / f"index_group_{self._group}.html"
        with fsspec.open(index_html_path, "wt") as f:
            f.write(gallery_html)

        # Also write directly as main index.html in the report root for single-group runs
        if self._group == 1:
            root_items = [
                dict(it, path=f"HCR/{it['path']}") for it in gallery_items
            ]
            root_html = tpl.render(
                title=f"Classifier Benchmark Report - Group {self._group}",
                summary_cards=summary_cards,
                summary_table=summary_table,
                items_json=json.dumps(root_items),
            )
            with fsspec.open(IO.report / "index.html", "wt") as f:
                f.write(root_html)

        # Register in central Monitor Index
        Index.add("Classifier Reports", f"Group {self._group} Benchmark Gallery", index_html_path)
