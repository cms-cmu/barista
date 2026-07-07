from ...hist_tools import H
from .vector import _PlotDiLorentzVector, _PlotLorentzVector


class _PlotCommon:
    ...


class _PlotJet(_PlotCommon, _PlotLorentzVector):
    deepjet_b = H((50, 0, 1, ('btagScore', 'btagScore $b$')))
    deepjet_c = H((50, 0, 1, ('btagDeepFlavCvL', 'DeepJet $c$ vs $uds+g$')),
                  (50, 0, 1, ('btagDeepFlavCvB', 'DeepJet $c$ vs $b$')))
    id_pileup = H(([0b000, 0b100, 0b110, 0b111], ('puId', 'Pileup ID')))
    id_jet = H(([0b000, 0b010, 0b110], ('jetId', 'Jet ID')))


class _PlotJetExtra(_PlotJet):
    # Extra per-jet inputs (ParticleNet discriminants, energy fractions, jet
    # composition). Off by default: only filled when a caller explicitly uses
    # Jet.plot_extra (e.g. the canJet collection behind a config flag). Binning
    # follows the signal-vs-background shape study. HF shower-shape vars and
    # rawFactor are intentionally not plotted.
    btagPNetCvB     = H((50, 0,    0.7, ('btagPNetCvB',     R'ParticleNet $c$ vs $b$')))
    btagPNetCvL     = H((50, 0.3,  1.0, ('btagPNetCvL',     R'ParticleNet $c$ vs $uds+g$')))
    btagPNetQvG     = H((50, 0.1,  0.8, ('btagPNetQvG',     R'ParticleNet $q$ vs $g$')))
    btagPNetTauVJet = H((50, 0,    0.3, ('btagPNetTauVJet', R'ParticleNet $\tau$ vs jet')))
    PNetRegPtRawRes = H((50, 0.05, 0.30, ('PNetRegPtRawRes', R'ParticleNet $p_{\mathrm{T}}$ resolution')))
    nSVs            = H((6, -0.5,  5.5, ('nSVs',            R'Number of secondary vertices')))
    nConstituents   = H((50, -0.5, 49.5, ('nConstituents',  R'Number of constituents')))
    area            = H((50, 0.3,  0.6, ('area',            R'Jet area')))
    chHEF           = H((50, 0,    1.0, ('chHEF',           R'Charged hadron energy fraction')))
    neHEF           = H((50, 0,    0.6, ('neHEF',           R'Neutral hadron energy fraction')))
    chEmEF          = H((50, 0,    0.7, ('chEmEF',          R'Charged EM energy fraction')))
    neEmEF          = H((50, 0,    0.8, ('neEmEF',          R'Neutral EM energy fraction')))
    muEF            = H((50, 0,    0.6, ('muEF',            R'Muon energy fraction')))


class _PlotDiJet(_PlotCommon, _PlotDiLorentzVector):
    lead = _PlotJet(('...', R'Lead Cand'), 'lead',     skip=['n'], bins={"mass": (50, 0, 100)})
    subl = _PlotJet(('...', R'Subl Cand'), 'subl',     skip=['n'], bins={"mass": (50, 0, 100)})

class Jet:
    plot = _PlotJet
    plot_extra = _PlotJetExtra
    plot_pair = _PlotDiJet
