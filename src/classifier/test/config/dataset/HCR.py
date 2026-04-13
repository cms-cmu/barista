from functools import cache, cached_property

from src.classifier.config.dataset._root import LoadGroupedRoot
from src.classifier.config.dataset.HCR._group import add_single_label
from src.classifier.config.setting.df import Columns
from src.classifier.config.setting.HCR import Input, InputBranch
from src.classifier.config.setting.ml import KFold
from src.classifier.task import ArgParser


class SimplifiedTrain(LoadGroupedRoot):
    trainable = True
    evaluable = True

    argparser = ArgParser()

    def __init__(self):
        super().__init__()
        from ....df.io import ToTensor

        self._preprocessors_by_group = [add_single_label()]
        self._to_tensor = ToTensor()

        # fmt: off
        (
            self._to_tensor
            .add(KFold.offset, KFold.offset_dtype).columns(Columns.event)
            .add(Input.label, Columns.index_dtype).columns(Columns.label_index)
            .add(Input.weight, "float32").columns(Columns.weight)
            .add(Input.ancillary, "float32").columns(*InputBranch.feature_ancillary)
            .add(Input.CanJet, "float32").columns(*InputBranch.feature_CanJet, target=InputBranch.n_CanJet)
            .add(Input.NotCanJet, "float32").columns(*InputBranch.feature_NotCanJet, target=InputBranch.n_NotCanJet, pad_value=InputBranch.pad_value)
        )
        # fmt: on

    @cached_property
    def _branches(self):
        return set().union(
            InputBranch.feature_ancillary,
            InputBranch.feature_CanJet,
            InputBranch.feature_NotCanJet,
            [Columns.weight, Columns.event],
        )

    @cache
    def from_root(self, groups: frozenset[str]):
        from src.classifier.df.io import FromRoot

        pres = []
        for g in self._preprocessors_by_group:
            pres.extend(g(groups))
        pres.extend(self.preprocessors)

        return FromRoot(
            branches=self._branches.intersection,
            preprocessors=pres,
        )
