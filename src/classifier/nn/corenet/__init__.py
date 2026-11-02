"""CoReNet, the decay-chain-configurable Combinatorial Residual Network.

Copy of CoReNet v1.0.0 (commit 2de82c1, https://github.com/cms-cmu/CoReNet) plus the working-tree
changes of 2026-10-29 (labels section, context-flag pass-through, grouped leaf embedding),
``src/corenet/models/__init__.py``. Barista-specific differences: the variance helper is barista's
``src.classifier.algorithm.statistics.variance.TensorVariance`` (``blocks.py``) and the jet-multiplicity
offset follows HCR's rule (``network.py``); the module docstrings list them.
"""

from .config import ConfigError, NetworkSpec, load_config, parse_config
from .network import CoReNet

__all__ = ["CoReNet", "ConfigError", "NetworkSpec", "load_config", "parse_config"]
