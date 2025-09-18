from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Callable

from classifier.config.setting.ml import SplitterKeys

from . import all_kl

if TYPE_CHECKING:
    from classifier.ml import BatchType
    from torch import BoolTensor


class Train(all_kl.Train):
    @abstractmethod
    def remover(self) -> Callable[[BatchType], BoolTensor]: ...

    def initializer(self, splitter, **kwargs):
        from classifier.ml.skimmer import Filter

        return super().initializer(
            splitter=Filter(**{SplitterKeys.training: self.remover()}) + splitter,
            **kwargs,
        )
