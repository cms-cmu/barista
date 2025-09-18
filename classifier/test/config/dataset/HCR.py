from classifier.config.dataset import HCR

from .._root import LoadGroupedRootForTest


class Eval(LoadGroupedRootForTest, HCR.Eval):
    evaluable = True
