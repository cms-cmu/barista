from __future__ import annotations

from classifier.task import ArgParser

from . import sm_no_ZX


class Train(sm_no_ZX.Train):
    model = "SvB_ggF-kl-5-no-ZX"
    argparser = ArgParser(description="Train SvB without ZZ and ZH.")

    def remover(self):
        return sm_no_ZX._remove_ZX_ggF(5.0)
