from ...hist_tools import H
from .vector import _PlotDiLorentzVector, _PlotLorentzVector


class _PlotCommon:
    charge = H((-2, 3, ('charge', 'Charge')))


class _PlotLepton(_PlotCommon, _PlotLorentzVector):
    ...


class _PlotDiLepton(_PlotCommon, _PlotDiLorentzVector):
    ...


class Lepton:
    plot = _PlotLepton
    plot_pair = _PlotDiLepton
