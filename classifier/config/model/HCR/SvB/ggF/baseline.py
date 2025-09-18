from __future__ import annotations

from classifier.task import ArgParser

from . import one_kl, all_kl


class Train(one_kl.Train):
    model = "SvB_ggF-baseline"
    argparser = ArgParser(description="Train SvB with SM ggF signal.")
    argparser.remove_argument("--signal-ggf-kl")
    defaults = {"signal_ggf_kl": 1.0}


class Eval(all_kl.Eval):
    model = "SvB_ggF-baseline"
