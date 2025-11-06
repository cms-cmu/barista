from ...hist_tools import H
import numpy as np
from .vector import _PlotDiLorentzVector, _PlotLorentzVector


class _PlotCommon:
    charge = H((-2, 3, ('charge', 'Charge')))


class _PlotLepton(_PlotCommon, _PlotLorentzVector):
    ...


class _PlotDiLepton(_PlotCommon, _PlotDiLorentzVector):
    ...

class _PlotLeptonMeT(_PlotLorentzVector):
    ...
    dr    = H((50, 0, 5, ('dr', R'$\Delta R$')))
    dphi  = H((30, -np.pi, np.pi, ('dphi', R'$\Delta\phi$')))
    mT    = H((50, 0, 300, ('mT', R'$m_{\mathrm{T}}$ [GeV]')))

    nu = _PlotLorentzVector(('...', R'Nu'), 'nu',         skip=['n'], bins={"mass": (50, 0, 100)})
    lep = _PlotLepton(('...', R'lepton'), 'lep',     skip=['n'], bins={"mass": (50, 0, 100)})



class Lepton:
    plot = _PlotLepton
    plot_pair = _PlotDiLepton
    plot_leptonMeT = _PlotLeptonMeT
