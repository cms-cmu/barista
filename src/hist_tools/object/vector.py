import awkward as ak
import numpy as np

from ...hist_tools import H, Template


class _PlotLorentzVector(Template):
    n       = H((0, 20,             ('n', 'Number')), n=ak.num)
    pt      = H((60, 0, 300,        ('pt', R'$p_{\mathrm{T}}$ [GeV]')))
    mass    = H((30, 0, 300,        ('mass', R'Mass [GeV]')))
    eta     = H((50, -5, 5,         ('eta', R'$\eta$')))
    phi     = H((30, -np.pi, np.pi, ('phi', R'$\phi$')))
    pz      = H((50, -1000, 1000,   ('pz', R'$p_{\mathrm{z}}$ [GeV]')))
    energy  = H((50, 0, 1500,       ('energy', R'Energy [GeV]')))


class _PlotDiLorentzVector(_PlotLorentzVector):
    dr    = H((50, 0, 5, ('dr', R'$\Delta R$')))
    dphi  = H((30, -np.pi, np.pi, ('dphi', R'$\Delta\phi$')))
    st    = H((50, 0, 1000, ('st', R'$S_{\mathrm{T}}$ [GeV]')))

    lead = _PlotLorentzVector(('...', R'Lead Cand'), 'lead',     skip=['n'], bins={"mass": (50, 0, 100)})
    subl = _PlotLorentzVector(('...', R'Subl Cand'), 'subl',     skip=['n'], bins={"mass": (50, 0, 100)})

class LorentzVector:
    plot = _PlotLorentzVector
    plot_pair = _PlotDiLorentzVector
