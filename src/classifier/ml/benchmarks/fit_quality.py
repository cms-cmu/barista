from __future__ import annotations

import copy
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from src.classifier.ml import BatchType


class FvTFitQuality:
    """Benchmark evaluated during validation to verify FvT reweighting health and closure.

    The reweighted ("source") class defaults to 3b data (``d3``, FvT). For the MvD the
    source is the mixed data (``source="mix4"``): the transfer weight is then
    (p_d4 - p_t4) / p_mix4 and the t3 terms drop out (no t3 class). The output keys keep
    the FvT names (d3, chi2_t4_sb, ...) so one reader handles both.

    ``source_floor`` lower-bounds the source probability in the weight denominator; set it
    to the analysis floor (e.g. FvT_pd3_floor / mvd_pmix4_floor = 0.01) to benchmark the
    weights the analysis actually applies.

    Sideband closure (pt4, pt1, FvT-weight histograms, norm_closure_sb) compares, in the SB
    only, the 4b data (target) with the background model the analysis uses there:
    source x w + 4b ttbar MC (model), with the raw (possibly negative) weights. Every histogram
    keeps sum(w) and sum(w^2) per bin; the chi2 uses both sides' errors with the model
    normalised to the target.

    Jet kinematics are read from ``Input.CanJet`` (feature-major: [event, (pt, eta, phi, mass),
    jet]). HCR.dataPrep rewrites that tensor in place (log(1+pt), eta sign), so the model's
    validate pass must give the network a copy when this benchmark is attached (HCRModel does).
    The leading / 4th jet are the highest / lowest pt of the four candidate jets.
    """

    HISTS = ("pt4_sb", "pt1_sb", "fvt_sb", "t3_sb", "t4_sb", "t3_sr", "t4_sr")

    def __init__(self, name: str = "FvT_fit_quality", source: str = "d3", source_floor: float = 1e-6):
        self.name = name
        self.source = source
        self.source_floor = source_floor
        self.bins = {
            "pt4_sb": np.linspace(20, 150, 27),    # 5 GeV bins
            "pt1_sb": np.linspace(40, 300, 27),    # 10 GeV bins
            "fvt_sb": np.linspace(-1.0, 8.0, 46),  # 0.2 wide
            "t3_sb": np.linspace(0, 1, 21),
            "t4_sb": np.linspace(0, 1, 21),
            "t3_sr": np.linspace(0, 1, 21),
            "t4_sr": np.linspace(0, 1, 21),
            "svb_sr": np.linspace(0, 1, 41),       # 40 bins in [0, 1] (bkg_syst mixeddata SR only)
        }
        self.reset()

    def reset(self):
        self.sum_w_t4_true = 0.0
        self.sum_w_t3_true = 0.0
        self.sum_w_d4_true = 0.0
        self.sum_w_d3_true = 0.0

        self.sum_w_t4_pred = 0.0
        self.sum_w_t3_pred = 0.0
        self.sum_w_d4_pred = 0.0
        self.sum_w_d3_pred = 0.0

        self.sum_w_d3_to_t4 = 0.0
        self.sum_w_d3_to_t3 = 0.0

        self.sum_loss_sb = 0.0
        self.sum_w_loss_sb = 0.0
        self.sum_loss_sr = 0.0
        self.sum_w_loss_sr = 0.0

        self.n_w_neg = 0
        self.n_w_total = 0
        self._sampled_weights = []

        # {hist: {"target"/"model": [sum w, sum w^2]}}
        self.h = {
            k: {side: [np.zeros(len(b) - 1), np.zeros(len(b) - 1)] for side in ("target", "model")}
            for k, b in self.bins.items()
        }
        self.sum_w_target_sb = 0.0
        self.sum_w_model_sb = 0.0
        self._has_svb = False

    def copy(self) -> FvTFitQuality:
        return copy.deepcopy(self)

    def _fill(self, hist: str, side: str, values, weights):
        values = np.asarray(values, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        if values.size == 0:
            return
        edges = self.bins[hist]
        s, _ = np.histogram(values, bins=edges, weights=weights)
        s2, _ = np.histogram(values, bins=edges, weights=weights**2)
        self.h[hist][side][0] += s
        self.h[hist][side][1] += s2

    def update(self, batch: BatchType):
        import torch
        import torch.nn.functional as F
        from src.classifier.config.setting.HCR import Input, MassRegion, Output
        from src.classifier.config.state.label import MultiClass

        probs = batch[Output.class_prob].detach().float().cpu()
        labels = batch[Input.label].detach().cpu()
        weights = batch[Input.weight].detach().float().cpu() if Input.weight in batch else torch.ones(len(probs))

        d4_idx = MultiClass.index("d4") if "d4" in MultiClass.labels else None
        # "d3" below is the reweighted source class: 3b data (FvT) or mixed data (MvD)
        d3_idx = MultiClass.index(self.source) if self.source in MultiClass.labels else None
        t4_idx = MultiClass.index("t4") if "t4" in MultiClass.labels else None
        t3_idx = MultiClass.index("t3") if "t3" in MultiClass.labels else None

        zeros = torch.zeros(len(probs))
        pd4 = probs[:, d4_idx] if d4_idx is not None else zeros
        pd3 = probs[:, d3_idx] if d3_idx is not None else zeros
        pt4 = probs[:, t4_idx] if t4_idx is not None else zeros
        pt3 = probs[:, t3_idx] if t3_idx is not None else zeros

        def is_label(idx):
            return (labels == idx) if idx is not None else torch.zeros(len(labels), dtype=torch.bool)

        is_d4, is_d3, is_t4, is_t3 = is_label(d4_idx), is_label(d3_idx), is_label(t4_idx), is_label(t3_idx)

        # True MC and Data yields
        self.sum_w_t4_true += float(weights[is_t4].sum())
        self.sum_w_t3_true += float(weights[is_t3].sum())
        self.sum_w_d4_true += float(weights[is_d4].sum())
        self.sum_w_d3_true += float(weights[is_d3].sum())

        # Predicted probability yields (sum_i w_i * p_c(x_i))
        self.sum_w_t4_pred += float((weights * pt4).sum())
        self.sum_w_t3_pred += float((weights * pt3).sum())
        self.sum_w_d4_pred += float((weights * pd4).sum())
        self.sum_w_d3_pred += float((weights * pd3).sum())

        # Transfer weight for the source: (pd4 - pt4) / pd3
        denom = torch.clamp(pd3, min=self.source_floor)
        num = pd4 - pt4 if t4_idx is not None else pd4
        w_fvt = num / denom

        # Data-driven ttbar components from the source: w_i * (pt4 / pd3) and w_i * (pt3 / pd3)
        if d3_idx is not None:
            w_d3_events = weights[is_d3]
            self.sum_w_d3_to_t4 += float((w_d3_events * (pt4[is_d3] / denom[is_d3])).sum())
            self.sum_w_d3_to_t3 += float((w_d3_events * (pt3[is_d3] / denom[is_d3])).sum())

            w_d3 = w_fvt[is_d3].numpy()
            if len(w_d3) > 0:
                self.n_w_neg += int(np.sum(w_d3 < 0))
                self.n_w_total += len(w_d3)
                if len(self._sampled_weights) < 50000:
                    self._sampled_weights.extend(w_d3[: min(1000, len(w_d3))])

        # Regions
        has_region = Input.region in batch
        region = batch[Input.region].detach().cpu() if has_region else None
        is_sr = ((region & MassRegion.SR) != 0) if has_region else torch.zeros(len(labels), dtype=torch.bool)
        is_sb = ((region & MassRegion.SB) != 0) if has_region else torch.ones(len(labels), dtype=torch.bool)

        # Regional loss (SB: all classes; SR: without the 4b data class, as in training)
        if Output.class_raw in batch:
            c_score = batch[Output.class_raw].detach().float().cpu()
            if torch.any(is_sb):
                w_sb = weights[is_sb]
                ce_sb = F.cross_entropy(c_score[is_sb], labels[is_sb], reduction="none")
                self.sum_loss_sb += float((ce_sb * w_sb).sum())
                self.sum_w_loss_sb += float(w_sb.sum())
            if torch.any(is_sr):
                if d4_idx is not None:
                    valid_sr = is_sr & (labels != d4_idx)
                    if torch.any(valid_sr):
                        w_sr = weights[valid_sr]
                        no_d4_idx = [*range(len(MultiClass.trainable_labels))]
                        no_d4_idx = no_d4_idx[:d4_idx] + no_d4_idx[d4_idx + 1:]
                        no_d4_y = labels[valid_sr]
                        no_d4_y = torch.where(no_d4_y > d4_idx, no_d4_y - 1, no_d4_y)
                        ce_sr = F.cross_entropy(c_score[valid_sr][:, no_d4_idx], no_d4_y, reduction="none")
                        self.sum_loss_sr += float((ce_sr * w_sr).sum())
                        self.sum_w_loss_sr += float(w_sr.sum())
                else:
                    w_sr = weights[is_sr]
                    ce_sr = F.cross_entropy(c_score[is_sr], labels[is_sr], reduction="none")
                    self.sum_loss_sr += float((ce_sr * w_sr).sum())
                    self.sum_w_loss_sr += float(w_sr.sum())

        # Component transfer: MC t3 (t4) distribution of p_t3 (p_t4) vs the source x p_t/p_d3
        def np_(mask):
            return mask.numpy()

        for reg, in_reg in (("sb", is_sb), ("sr", is_sr)):
            if d3_idx is not None and t3_idx is not None:
                m_t, m_s = np_(is_t3 & in_reg), np_(is_d3 & in_reg)
                self._fill(f"t3_{reg}", "target", pt3.numpy()[m_t], weights.numpy()[m_t])
                self._fill(f"t3_{reg}", "model", pt3.numpy()[m_s],
                           weights.numpy()[m_s] * (pt3.numpy()[m_s] / denom.numpy()[m_s]))
            if d3_idx is not None and t4_idx is not None:
                m_t, m_s = np_(is_t4 & in_reg), np_(is_d3 & in_reg)
                self._fill(f"t4_{reg}", "target", pt4.numpy()[m_t], weights.numpy()[m_t])
                self._fill(f"t4_{reg}", "model", pt4.numpy()[m_s],
                           weights.numpy()[m_s] * (pt4.numpy()[m_s] / denom.numpy()[m_s]))

        # Sideband closure: 4b data vs source x w + 4b ttbar MC (SB only; the 4b SR is not trained on)
        if Input.CanJet in batch and d4_idx is not None and d3_idx is not None:
            cj = batch[Input.CanJet].detach().float().cpu().numpy()
            cj = cj.reshape(cj.shape[0], -1, 4)                 # [event, feature, jet]
            jet_pt = np.sort(cj[:, 0, :], axis=1)              # ascending
            pt1, pt4_jet = jet_pt[:, -1], jet_pt[:, 0]
            w = weights.numpy()
            wf = w_fvt.numpy()
            target = np_(is_d4 & is_sb)
            model_src = np_(is_d3 & is_sb)
            model_tt = np_(is_t4 & is_sb)

            for side, mask, ew in (
                ("target", target, w[target]),
                ("model", model_src, w[model_src] * wf[model_src]),
                ("model", model_tt, w[model_tt]),
            ):
                if not np.any(mask):
                    continue
                self._fill("pt4_sb", side, pt4_jet[mask], ew)
                self._fill("pt1_sb", side, pt1[mask], ew)
                self._fill("fvt_sb", side, wf[mask], ew)
                if side == "target":
                    self.sum_w_target_sb += float(np.sum(ew))
                else:
                    self.sum_w_model_sb += float(np.sum(ew))

        # Signal-region SvB_MA.ps closure (opt-in for bkg_syst mixeddata only; never loaded in nominal)
        svb_key = getattr(Input, "svb", None)
        if svb_key is not None and svb_key in batch and d4_idx is not None and d3_idx is not None:
            svb_vals = batch[svb_key].detach().float().cpu().numpy().reshape(-1)
            w = weights.numpy()
            wf = w_fvt.numpy()
            for side, mask, ew in (
                ("target", np_(is_d4 & is_sr), w[np_(is_d4 & is_sr)]),
                ("model", np_(is_d3 & is_sr), w[np_(is_d3 & is_sr)] * wf[np_(is_d3 & is_sr)]),
                ("model", np_(is_t4 & is_sr), w[np_(is_t4 & is_sr)]),
            ):
                if np.any(mask):
                    self._fill("svb_sr", side, svb_vals[mask], ew)
                    self._has_svb = True

    @staticmethod
    def _chi2_ndf(target, model):
        """Shape chi2/ndf: model normalised to the target, errors from both sides' sum w^2.
        Bins with no target entries are skipped. Returns (chi2/ndf, ndf); (None, 0) if empty."""
        (t, t2), (m, m2) = target, model
        st, sm = np.sum(t), np.sum(m)
        if st <= 0 or sm <= 0:
            return None, 0
        k = st / sm
        var = t2 + k**2 * m2
        mask = (t > 0) & (var > 0)
        ndf = int(np.sum(mask)) - 1
        if ndf < 1:
            return None, 0
        chi2 = float(np.sum((t[mask] - k * m[mask]) ** 2 / var[mask]))
        return chi2 / ndf, ndf

    def to_json(self) -> dict:
        w_arr = np.array(self._sampled_weights) if self._sampled_weights else np.array([1.0])
        neg_frac = (self.n_w_neg / self.n_w_total) if self.n_w_total > 0 else 0.0
        w_p99 = float(np.percentile(w_arr, 99))
        w_max = float(np.max(w_arr))
        w_mean = float(np.mean(w_arr))
        w_std = float(np.std(w_arr))

        calib_t4_prob = self.sum_w_t4_pred / max(1e-9, self.sum_w_t4_true)
        calib_t3_prob = self.sum_w_t3_pred / max(1e-9, self.sum_w_t3_true)
        transfer_d3_to_t4 = self.sum_w_d3_to_t4 / max(1e-9, self.sum_w_t4_true)
        transfer_d3_to_t3 = self.sum_w_d3_to_t3 / max(1e-9, self.sum_w_t3_true)
        true_ratio_t4_t3 = self.sum_w_t4_true / max(1e-9, self.sum_w_t3_true)
        pred_ratio_t4_t3 = self.sum_w_t4_pred / max(1e-9, self.sum_w_t3_pred)
        calib_d4_prob = self.sum_w_d4_pred / max(1e-9, self.sum_w_d4_true)
        calib_d3_prob = self.sum_w_d3_pred / max(1e-9, self.sum_w_d3_true)

        active_hists = self.HISTS + (("svb_sr",) if self._has_svb else ())
        chi2 = {k: self._chi2_ndf(self.h[k]["target"], self.h[k]["model"]) for k in active_hists}
        if self._has_svb:
            svb_gt0p8 = self.bins["svb_sr"][:-1] >= 0.8 - 1e-9
            chi2_svb_gt0p8, ndf_svb_gt0p8 = self._chi2_ndf(
                (self.h["svb_sr"]["target"][0][svb_gt0p8], self.h["svb_sr"]["target"][1][svb_gt0p8]),
                (self.h["svb_sr"]["model"][0][svb_gt0p8], self.h["svb_sr"]["model"][1][svb_gt0p8]),
            )
        else:
            chi2_svb_gt0p8, ndf_svb_gt0p8 = None, 0

        def r(v, n=3):
            return None if v is None else round(v, n)

        closure_parts = [chi2[k][0] for k in ("pt4_sb", "pt1_sb", "fvt_sb") if chi2[k][0] is not None]
        composite = sum(closure_parts) / len(closure_parts) if closure_parts else None
        norm_closure_sb = (self.sum_w_model_sb / self.sum_w_target_sb) if self.sum_w_target_sb > 0 else None
        loss_sb = (self.sum_loss_sb / self.sum_w_loss_sb) if self.sum_w_loss_sb > 0 else 0.0
        loss_sr = (self.sum_loss_sr / self.sum_w_loss_sr) if self.sum_w_loss_sr > 0 else 0.0

        gate_neg = bool(neg_frac < 0.005)
        gate_explosion = bool(w_p99 < 5.0 and w_max < 20.0)
        gate_ttbar = bool(0.70 <= calib_t4_prob <= 1.30 and 0.70 <= transfer_d3_to_t4 <= 1.30)

        def hist_json(k):
            d = {
                "bin_edges": [round(float(x), 4) for x in self.bins[k]],
                "target": [round(float(x), 4) for x in self.h[k]["target"][0]],
                "target_sumw2": [round(float(x), 4) for x in self.h[k]["target"][1]],
                "model": [round(float(x), 4) for x in self.h[k]["model"][0]],
                "model_sumw2": [round(float(x), 4) for x in self.h[k]["model"][1]],
                "chi2_ndf": r(chi2[k][0]),
                "ndf": chi2[k][1],
            }
            if k == "svb_sr":
                d["chi2_ndf_gt0p8"] = r(chi2_svb_gt0p8)
                d["ndf_gt0p8"] = ndf_svb_gt0p8
            return d

        return {
            "name": self.name,
            "source": self.source,
            "source_floor": self.source_floor,
            "proxies": {
                "loss_sb": round(loss_sb, 5),
                "loss_sr": round(loss_sr, 5),
                "chi2_fvt_sb": r(chi2["fvt_sb"][0]),
                "chi2_t3_sb": r(chi2["t3_sb"][0]),
                "chi2_t4_sb": r(chi2["t4_sb"][0]),
                "chi2_t3_sr": r(chi2["t3_sr"][0]),
                "chi2_t4_sr": r(chi2["t4_sr"][0]),
            },
            "histograms": {k: hist_json(k) for k in active_hists},
            "weights": {
                "neg_fraction": round(neg_frac, 5),
                "p99": round(w_p99, 3),
                "max": round(w_max, 3),
                "mean": round(w_mean, 3),
                "std": round(w_std, 3),
            },
            "ttbar_mc_comparison": {
                "true_mc_t4_weight": round(self.sum_w_t4_true, 2),
                "true_mc_t3_weight": round(self.sum_w_t3_true, 2),
                "true_ratio_t4_t3": round(true_ratio_t4_t3, 4),
                "pred_ratio_t4_t3": round(pred_ratio_t4_t3, 4),
                "calib_t4_prob_to_mc": round(calib_t4_prob, 4),
                "calib_t3_prob_to_mc": round(calib_t3_prob, 4),
                "transfer_d3_to_t4_to_mc": round(transfer_d3_to_t4, 4),
                "transfer_d3_to_t3_to_mc": round(transfer_d3_to_t3, 4),
            },
            "data_calibration": {
                "calib_d4_prob_to_data": round(calib_d4_prob, 4),
                "calib_d3_prob_to_data": round(calib_d3_prob, 4),
            },
            "closure": {
                "chi2_ndf_pt4": r(chi2["pt4_sb"][0]),
                "chi2_ndf_pt1": r(chi2["pt1_sb"][0]),
                "chi2_ndf_fvt": r(chi2["fvt_sb"][0]),
                **(
                    {
                        "chi2_ndf_svb_sr": r(chi2["svb_sr"][0]),
                        "chi2_ndf_svb_sr_gt0p8": r(chi2_svb_gt0p8),
                    }
                    if self._has_svb
                    else {}
                ),
                "norm_closure_sb": r(norm_closure_sb, 4),
                "composite_chi2_ndf": r(composite),
            },
            "verdict": {
                "gate_negative_weights": gate_neg,
                "gate_weight_explosion": gate_explosion,
                "gate_ttbar_preservation": gate_ttbar,
                "all_gates_passed": gate_neg and gate_explosion and gate_ttbar,
                "primary_score": r(composite),
            },
        }
