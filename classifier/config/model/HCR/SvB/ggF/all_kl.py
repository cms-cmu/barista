from __future__ import annotations

from typing import TYPE_CHECKING

from classifier.config.setting.cms import MC_HH_ggF
from classifier.config.setting.HCR import Input, Output
from classifier.config.state.label import MultiClass
from classifier.task import ArgParser

from ..._HCR import ROC_BIN, HCREval, HCRTrain, roc_nominal_selection

if TYPE_CHECKING:
    from classifier.ml import BatchType

_BKG = ("multijet", "ttbar")
_SIG = ("ZZ", "ZH", "ggF")


class _roc_select_sig:
    def __init__(self, sig: str):
        self.sig = sig

    def __call__(self, batch: BatchType):
        selected = self._select(batch)
        return {
            "y_pred": batch[Output.class_prob][selected],
            "y_true": batch[Input.label][selected],
            "weight": batch[Input.weight][selected],
        }

    def _select(self, batch: BatchType):
        import torch

        label = batch[Input.label]
        return torch.isin(label, label.new_tensor(MultiClass.indices(*_BKG, self.sig)))


class _roc_select_ggF(_roc_select_sig):
    def __init__(self, *labels: str, kl: float):
        self.bkg = labels
        self.kl = kl

    def _select(self, batch: BatchType):
        import torch

        label = batch[Input.label]
        return torch.isin(label, label.new_tensor(MultiClass.indices(*self.bkg))) | (
            (batch["kl"] == self.kl) & (label == MultiClass.index("ggF"))
        )


class _roc_cat_by_largest:
    def __init__(self, cat: str, sig: str, kl: float = None):
        self.cat = cat
        self.sig = sig
        self.kl = kl

    def __call__(self, batch):
        import torch

        y_pred = batch[Output.class_prob]
        y_true = batch[Input.label]
        weight = batch[Input.weight]

        # remove other signals
        sig = y_true == MultiClass.index(self.sig)
        if self.kl is not None:
            sig &= batch["kl"] == self.kl
        bkg = torch.isin(y_true, y_true.new_tensor(MultiClass.indices(*_BKG)))
        select = sig | bkg
        y_pred = y_pred[select]
        y_true = y_true[select]
        weight = weight[select]

        n_train = MultiClass.n_trainable()
        isig = MultiClass.index(self.cat)
        # find signal samples
        isigs = [i for i in MultiClass.indices(*_SIG) if i is not None]
        # find signal outputs
        isigs_train = sorted([i for i in isigs if i < n_train])
        if isig not in isigs_train:
            return {}
        if len(isigs_train) > 1:
            sig_pred = y_pred[:, isigs_train]
            # select if self.sig is largest
            selected = torch.argmax(sig_pred, dim=1) == isigs_train.index(isig)
            # set all signal label to self.sig
            y_true = y_true[selected]
            is_sig = torch.isin(y_true, y_true.new_tensor(isigs))
            y_true[is_sig] = isig
            y_pred = y_pred[selected]
            weight = weight[selected]
        return {
            "y_pred": y_pred,
            "y_true": y_true,
            "weight": weight,
        }


class Train(HCRTrain):
    model = "SvB_ggF-all"
    argparser = ArgParser(description="Train SvB with SM and BSM ggF signals.")
    argparser.add_argument(
        "--roc-signal-by-category",
        action="store_true",
        help="categorize events by the largest signal probability and create ROC curves for each category",
    )

    @staticmethod
    def loss(batch: BatchType):
        import torch.nn.functional as F

        c_score = batch[Output.class_raw]
        weight = batch[Input.weight]
        label = batch[Input.label]

        # calculate loss
        cross_entropy = F.cross_entropy(c_score, label, reduction="none")
        loss = (cross_entropy * weight).sum() / weight.sum()
        return loss

    @property
    def rocs(self):
        from classifier.ml.benchmarks.multiclass import ROC

        rocs = [
            ROC(
                name="background vs signal",
                selection=roc_nominal_selection,
                bins=ROC_BIN,
                pos=_BKG,
            )
        ]
        for sig in ("ZZ", "ZH", "ggF"):
            if sig in MultiClass.labels:
                rocs.append(
                    ROC(
                        name=f"background vs {sig}",
                        selection=_roc_select_sig(sig),
                        bins=ROC_BIN,
                        pos=_BKG,
                    )
                )
            if self.opts.roc_signal_by_category and sig in MultiClass.trainable_labels:
                if "ggF" in MultiClass.labels:
                    for kl in MC_HH_ggF.kl:
                        rocs.append(
                            ROC(
                                name=f"(P({sig}) largest) background vs ggF (kl={kl:.6g})",
                                selection=_roc_cat_by_largest(sig, "ggF", kl=kl),
                                bins=ROC_BIN,
                                pos=_BKG,
                            )
                        )
                for sig2 in ("ZZ", "ZH"):
                    if sig2 in MultiClass.labels:
                        rocs.append(
                            ROC(
                                name=f"(P({sig}) largest) background vs {sig2}",
                                selection=_roc_cat_by_largest(sig, sig2),
                                bins=ROC_BIN,
                                pos=_BKG,
                            )
                        )
        if "ggF" in MultiClass.labels:
            for kl in MC_HH_ggF.kl:
                rocs.append(
                    ROC(
                        name=f"background vs ggF (kl={kl:.6g})",
                        selection=_roc_select_ggF(*_BKG, kl=kl),
                        bins=ROC_BIN,
                        pos=_BKG,
                    )
                )
        if "ggF" in MultiClass.trainable_labels:
            for sig in ("ZZ", "ZH"):
                if sig in MultiClass.trainable_labels:
                    for kl in MC_HH_ggF.kl:
                        rocs.append(
                            ROC(
                                name=f"{sig} vs ggF (kl={kl:.6g})",
                                selection=_roc_select_ggF(sig, kl=kl),
                                bins=ROC_BIN,
                                pos=(sig,),
                                neg=("ggF",),
                                score="differ",
                            )
                        )
        if all(sig in MultiClass.trainable_labels for sig in ("ZZ", "ZH")):
            rocs.append(
                ROC(
                    name="ZZ vs ZH",
                    selection=roc_nominal_selection,
                    bins=ROC_BIN,
                    pos=("ZZ",),
                    neg=("ZH",),
                    score="differ",
                )
            )
        return rocs


class Eval(HCREval):
    model = "SvB_ggF-all"

    @staticmethod
    def output_definition(batch: BatchType):
        output = {
            "q_1234": ...,
            "q_1324": ...,
            "q_1423": ...,
            "p_multijet": ...,
            "p_ttbar": ...,
            "p_bkg": batch["p_multijet"] + batch["p_ttbar"],
        }
        for sig in ("ZZ", "ZH", "ggF"):
            sig = f"p_{sig}"
            if sig in batch:
                output[sig] = ...
                if "p_sig" in output:
                    output["p_sig"] = output["p_sig"] + batch[sig]
                else:
                    output["p_sig"] = batch[sig]
        return output
