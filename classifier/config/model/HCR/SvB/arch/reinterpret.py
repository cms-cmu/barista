from classifier.config.setting.cms import MC_HH_ggF

from ..ggF import all_kl
from ..ggF.all_kl import (
    _BKG,
    _SIG,
    ROC_BIN,
    _roc_cat_by_largest,
    _roc_select_ggF,
    _roc_select_sig,
)


class Train(all_kl.Train):
    @property
    def rocs(self):
        from classifier.ml.benchmarks.multiclass import ROC

        rocs = []
        for sig in _SIG:
            for kl in MC_HH_ggF.kl:
                rocs.append(
                    ROC(
                        name=f"[P({sig}) largest, (1+P({sig})-P(bkg))/2] background vs ggF (kl={kl:.6g})",
                        selection=_roc_cat_by_largest(sig, "ggF", kl=kl),
                        bins=ROC_BIN,
                        neg=_BKG,
                        pos=(sig,),
                        score="differ",
                    )
                )
                rocs.append(
                    ROC(
                        name=f"[P({sig}) largest, 1-P(bkg)] background vs ggF (kl={kl:.6g})",
                        selection=_roc_cat_by_largest(sig, "ggF", kl=kl),
                        bins=ROC_BIN,
                        pos=_BKG,
                    )
                )
                rocs.append(
                    ROC(
                        name=f"[P({sig}) largest, P({sig})] background vs ggF (kl={kl:.6g})",
                        selection=_roc_cat_by_largest(sig, "ggF", kl=kl),
                        bins=ROC_BIN,
                        pos=(sig,),
                    )
                )
            for sig2 in ("ZZ", "ZH"):
                rocs.append(
                    ROC(
                        name=f"[P({sig}) largest, (1+P({sig})-P(bkg))/2] background vs {sig2}",
                        selection=_roc_cat_by_largest(sig, sig2),
                        bins=ROC_BIN,
                        neg=_BKG,
                        pos=(sig,),
                        score="differ",
                    )
                )
                rocs.append(
                    ROC(
                        name=f"[P({sig}) largest, 1-P(bkg)] background vs {sig2}",
                        selection=_roc_cat_by_largest(sig, sig2),
                        bins=ROC_BIN,
                        pos=_BKG,
                    )
                )
                rocs.append(
                    ROC(
                        name=f"[P({sig}) largest, P({sig})] background vs {sig2}",
                        selection=_roc_cat_by_largest(sig, sig2),
                        bins=ROC_BIN,
                        pos=(sig,),
                    )
                )
        for kl in MC_HH_ggF.kl:
            rocs.append(
                ROC(
                    name=f"((1+P(ggF)-P(bkg))/2) background vs ggF (kl={kl:.6g})",
                    selection=_roc_select_ggF(*_BKG, kl=kl),
                    bins=ROC_BIN,
                    neg=_BKG,
                    pos=("ggF",),
                    score="differ",
                )
            )
        for sig in ("ZZ", "ZH", "ggF"):
            rocs.append(
                ROC(
                    name=f"((1+P({sig})-P(bkg))/2) background vs {sig}",
                    selection=_roc_select_sig(sig),
                    bins=ROC_BIN,
                    neg=_BKG,
                    pos=(sig,),
                    score="differ",
                )
            )
        return rocs
