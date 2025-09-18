from classifier.task import ArgParser

from . import one_kl


class Train(one_kl.Train):
    model = "SvB_ggF-kl-0"
    argparser = ArgParser(description="Train SvB with ggF kl=0 signal.")
    argparser.remove_argument("--signal-ggf-kl")
    defaults = {"signal_ggf_kl": 0.0}
