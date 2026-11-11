"""
Vendored and decoupled core ROOT plotting routines from CMS HH combination (dhi).
Stripped of law, luigi, and workflow orchestrators.
"""

import sys
import os

# Register package paths
this_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(this_dir)

if this_dir not in sys.path:
    sys.path.insert(0, this_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Make plotlib importable
plotlib_dir = os.path.join(this_dir, "plotlib")
if plotlib_dir not in sys.path:
    sys.path.insert(0, plotlib_dir)

# Aliases for internal imports expecting 'dhi' and 'dhi_core'
import dhi_core as _dhi_core
sys.modules["dhi"] = _dhi_core
import dhi_core.config as _config
sys.modules["dhi.config"] = _config
import dhi_core.util as _util
sys.modules["dhi.util"] = _util
import dhi_core.hepdata_tools as _hdt
sys.modules["dhi.hepdata_tools"] = _hdt
import dhi_core.plots as _plots
sys.modules["dhi.plots"] = _plots
import dhi_core.plots.util as _plots_util
sys.modules["dhi.plots.util"] = _plots_util
