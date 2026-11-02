from __future__ import annotations

import copy
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from src.classifier.ml import BatchType


class FvTFitQuality:
    """Benchmark evaluated during validation to verify FvT reweighting health and closure."""

    def __init__(self, name: str = "FvT_fit_quality"):
        self.name = name
        self.bins_pt4 = np.linspace(20, 150, 27)  # 5 GeV bins
        self.bins_pt1 = np.linspace(40, 300, 27)  # 10 GeV bins
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

        self.n_w_neg = 0
        self.n_w_total = 0
        self._sampled_weights = []

        self.hist_target_pt4 = np.zeros(len(self.bins_pt4) - 1, dtype=np.float64)
        self.hist_model_pt4 = np.zeros(len(self.bins_pt4) - 1, dtype=np.float64)
        self.hist_target_pt1 = np.zeros(len(self.bins_pt1) - 1, dtype=np.float64)
        self.hist_model_pt1 = np.zeros(len(self.bins_pt1) - 1, dtype=np.float64)

        self.total_target_sb = 0
        self.total_model_sb = 0

    def copy(self) -> FvTFitQuality:
        return copy.deepcopy(self)

    def update(self, batch: BatchType):
        import torch
        from src.classifier.config.setting.HCR import Input, Output
        from src.classifier.config.state.label import MultiClass

        probs = batch[Output.class_prob].detach().cpu()
        labels = batch[Input.label].detach().cpu()
        weights = batch[Input.weight].detach().cpu() if Input.weight in batch else torch.ones(len(probs))

        d4_idx = MultiClass.index("d4") if "d4" in MultiClass.labels else None
        d3_idx = MultiClass.index("d3") if "d3" in MultiClass.labels else None
        t4_idx = MultiClass.index("t4") if "t4" in MultiClass.labels else None
        t3_idx = MultiClass.index("t3") if "t3" in MultiClass.labels else None

        pd4 = probs[:, d4_idx] if d4_idx is not None else torch.zeros(len(probs))
        pd3 = probs[:, d3_idx] if d3_idx is not None else torch.zeros(len(probs))
        pt4 = probs[:, t4_idx] if t4_idx is not None else torch.zeros(len(probs))
        pt3 = probs[:, t3_idx] if t3_idx is not None else torch.zeros(len(probs))

        # True MC and Data yields
        if t4_idx is not None:
            self.sum_w_t4_true += float(weights[labels == t4_idx].sum())
        if t3_idx is not None:
            self.sum_w_t3_true += float(weights[labels == t3_idx].sum())
        if d4_idx is not None:
            self.sum_w_d4_true += float(weights[labels == d4_idx].sum())
        if d3_idx is not None:
            self.sum_w_d3_true += float(weights[labels == d3_idx].sum())

        # Predicted probability yields (sum_i w_i * p_c(x_i))
        self.sum_w_t4_pred += float((weights * pt4).sum())
        self.sum_w_t3_pred += float((weights * pt3).sum())
        self.sum_w_d4_pred += float((weights * pd4).sum())
        self.sum_w_d3_pred += float((weights * pd3).sum())

        # FvT transfer weight for 3b: (pd4 - pt4) / pd3
        denom = torch.clamp(pd3, min=1e-6)
        num = pd4 - pt4 if t4_idx is not None else pd4
        w_fvt = num / denom

        # Data-driven ttbar components from 3b data:
        # w_i * (pt4 / pd3) and w_i * (pt3 / pd3)
        if d3_idx is not None:
            is_d3 = labels == d3_idx
            w_d3_events = weights[is_d3]
            pd3_d3 = denom[is_d3]
            pt4_d3 = pt4[is_d3]
            pt3_d3 = pt3[is_d3]

            self.sum_w_d3_to_t4 += float((w_d3_events * (pt4_d3 / pd3_d3)).sum())
            self.sum_w_d3_to_t3 += float((w_d3_events * (pt3_d3 / pd3_d3)).sum())

            w_d3 = w_fvt[is_d3].numpy()
            if len(w_d3) > 0:
                self.n_w_neg += int(np.sum(w_d3 < 0))
                self.n_w_total += len(w_d3)
                if len(self._sampled_weights) < 50000:
                    self._sampled_weights.extend(w_d3[: min(1000, len(w_d3))])

        # Kinematic closure (Target d4 vs Model d3 with w_fvt)
        # In the training dataset, all d4 events are Sideband events (SR is blinded).
        if Input.CanJet in batch and d4_idx is not None and d3_idx is not None:
            can_jets = batch[Input.CanJet].detach().cpu().numpy()
            if can_jets.ndim == 2:
                can_jets = can_jets.reshape(can_jets.shape[0], -1, 4)

            is_target = (labels == d4_idx).numpy()
            is_model = (labels == d3_idx).numpy()

            if np.any(is_target):
                pt4_t = can_jets[is_target, 0, 3]
                pt1_t = can_jets[is_target, 0, 0]
                w_t = weights[is_target].numpy()
                h_t4, _ = np.histogram(pt4_t, bins=self.bins_pt4, weights=w_t)
                h_t1, _ = np.histogram(pt1_t, bins=self.bins_pt1, weights=w_t)
                self.hist_target_pt4 += h_t4
                self.hist_target_pt1 += h_t1
                self.total_target_sb += len(pt4_t)

            if np.any(is_model):
                pt4_m = can_jets[is_model, 0, 3]
                pt1_m = can_jets[is_model, 0, 0]
                w_m = weights[is_model].numpy() * np.maximum(w_fvt[is_model].numpy(), 0.0)
                h_m4, _ = np.histogram(pt4_m, bins=self.bins_pt4, weights=w_m)
                h_m1, _ = np.histogram(pt1_m, bins=self.bins_pt1, weights=w_m)
                self.hist_model_pt4 += h_m4
                self.hist_model_pt1 += h_m1
                self.total_model_sb += len(pt4_m)

    def to_json(self) -> dict:
        w_arr = np.array(self._sampled_weights) if self._sampled_weights else np.array([1.0])
        neg_frac = (self.n_w_neg / self.n_w_total) if self.n_w_total > 0 else 0.0
        w_p99 = float(np.percentile(w_arr, 99)) if len(w_arr) > 0 else 1.0
        w_max = float(np.max(w_arr)) if len(w_arr) > 0 else 1.0
        w_mean = float(np.mean(w_arr)) if len(w_arr) > 0 else 1.0
        w_std = float(np.std(w_arr)) if len(w_arr) > 0 else 0.0

        # ttbar MC comparison ratios
        calib_t4_prob = self.sum_w_t4_pred / max(1e-9, self.sum_w_t4_true)
        calib_t3_prob = self.sum_w_t3_pred / max(1e-9, self.sum_w_t3_true)
        transfer_d3_to_t4 = self.sum_w_d3_to_t4 / max(1e-9, self.sum_w_t4_true)
        transfer_d3_to_t3 = self.sum_w_d3_to_t3 / max(1e-9, self.sum_w_t3_true)

        true_ratio_t4_t3 = self.sum_w_t4_true / max(1e-9, self.sum_w_t3_true)
        pred_ratio_t4_t3 = self.sum_w_t4_pred / max(1e-9, self.sum_w_t3_pred)

        calib_d4_prob = self.sum_w_d4_pred / max(1e-9, self.sum_w_d4_true)
        calib_d3_prob = self.sum_w_d3_pred / max(1e-9, self.sum_w_d3_true)

        # Compute chi2 for kinematic distributions
        def _compute_chi2_ndf(h_target, h_model):
            sum_t = np.sum(h_target)
            sum_m = np.sum(h_model)
            if sum_t <= 0 or sum_m <= 0:
                return 1.0
            # Normalize model to target
            h_model_norm = h_model * (sum_t / sum_m)
            mask = (h_target > 10) & (h_model_norm > 0)
            if np.sum(mask) <= 1:
                return 1.0
            chi2 = float(np.sum((h_target[mask] - h_model_norm[mask]) ** 2 / h_target[mask]))
            ndf = int(np.sum(mask) - 1)
            return chi2 / ndf

        chi2_pt4 = _compute_chi2_ndf(self.hist_target_pt4, self.hist_model_pt4)
        chi2_pt1 = _compute_chi2_ndf(self.hist_target_pt1, self.hist_model_pt1)
        composite_chi2 = (chi2_pt4 + chi2_pt1) / 2.0

        # Hard Gates
        gate_neg = bool(neg_frac < 0.005)
        gate_explosion = bool(w_p99 < 5.0 and w_max < 20.0)
        gate_ttbar = bool(0.70 <= calib_t4_prob <= 1.30 and 0.70 <= transfer_d3_to_t4 <= 1.30)
        all_passed = gate_neg and gate_explosion and gate_ttbar

        return {
            "name": self.name,
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
                "chi2_ndf_pt4": round(chi2_pt4, 3),
                "chi2_ndf_pt1": round(chi2_pt1, 3),
                "composite_chi2_ndf": round(composite_chi2, 3),
            },
            "verdict": {
                "gate_negative_weights": gate_neg,
                "gate_weight_explosion": gate_explosion,
                "gate_ttbar_preservation": gate_ttbar,
                "all_gates_passed": all_passed,
                "primary_score": round(composite_chi2, 3),
            },
        }
