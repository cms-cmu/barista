from .hist import (
    AxisLike,
    Collection,
    Fill,
    FillError,
    FillLike,
    HistError,
    Label,
    LabelLike,
)
from .template import Systematic, Template

__all__ = [
    "Collection",
    "Template",
    "Fill",
    "Systematic",
    "Label",
    "LabelLike",
    "FillLike",
    "AxisLike",
    "FillError",
    "HistError",
]

H = Template._Hist
