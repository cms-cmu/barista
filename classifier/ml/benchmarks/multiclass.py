from typing import Callable, Iterable, Literal, TypedDict

from classifier.config.state.label import MultiClass
from torch import Tensor

from ...algorithm.hist import RegularAxis
from ...algorithm.metrics.roc import FixedThresholdROC, linear_differ
from .. import BatchType


class ROCInputType(TypedDict):
    y_true: Tensor
    y_score: Tensor
    weights: Tensor | None


class ROC(FixedThresholdROC):
    def __init__(
        self,
        name: str,
        selection: Callable[[BatchType], ROCInputType],
        bins: RegularAxis | Iterable[float],
        pos: Iterable[str],
        neg: Iterable[str] = None,
        score: Literal["differ"] | None = None,
    ):
        self._name = name
        self._select = selection
        pos = MultiClass.indices(*pos)
        if neg is not None:
            neg = MultiClass.indices(*neg)
        match score:
            case "differ":
                score = linear_differ
        super().__init__(
            thresholds=bins,
            positive_classes=pos,
            negative_classes=neg,
            score_interpretation=score,
        )

    def update(self, batch: BatchType):
        super().update(**self._select(batch))

    def to_json(self):
        return {"name": self._name} | super().to_json()
