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


class _PlotDiJet(_PlotCommon, _PlotDiLorentzVector):
    lead = _PlotJet(('...', R'Lead Cand'), 'lead',     skip=['n'], bins={"mass": (50, 0, 100)})
    subl = _PlotJet(('...', R'Subl Cand'), 'subl',     skip=['n'], bins={"mass": (50, 0, 100)})

class Jet:
    plot = _PlotJet
    plot_pair = _PlotDiJet
