from __future__ import annotations

from typing import Optional, Protocol

from .special import interface
from .task import Task


class Analysis(Task):
    @interface
    def analyze(self, results: list[dict]) -> list[Analyzer]:
        """
        Prepare analyzers.
        """
        ...


class Analyzer(Protocol):
    def __call__(self) -> Optional[dict[str]]:
        """
        Run analysis.
        """
        ...
