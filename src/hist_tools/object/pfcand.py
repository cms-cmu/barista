from ...hist_tools import H
from .vector import  _PlotLorentzVector


class _PlotCommon:
    ...


class _PlotPFCand(_PlotCommon, _PlotLorentzVector):

    # fields ['trkP', 'trkPhi',]

    pdgId            = H((50, -250,    250, ('pdgId',             'pdgId')))
    nHits            = H((40, -0.5,   39.5, ('numberOfHits',      'number of Hits')))
    nPix             = H((10, -0.5,    9.5, ('numberOfPixelHits', 'number of Pixel Hits')))
    puppiWeight      = H((50,    0,    1.0, ('puppiWeight',       'puppiWeight')))
    puppiWeightNoLep = H((50,    0,    1.0, ('puppiWeightNoLep',  'puppiWeightNoLep')))
    charge           = H((3,    -1.5,  1.5, ('charge',            'charge')))
    lostInnerHits    = H((4,    -0.5,  3.5, ('lostInnerHits',     'lostInnerHits')))
    #lostOuterHits    = H((10,   -0.5,  9.5, ('lostOuterHits',     'lostOuterHits')))
    pvAssocQuality   = H((10,   -0.5,  9.5, ('pvAssocQuality',    'pvAssocQuality')))
    #trkAlgo          = H((10,   -0.5,  9.5, ('trkAlgo',           'trkAlgo')))
    trkHighPurity    = H((2,    -0.5,  1.5, ('trkHighPurity',     'trkHighPurity')))
    trkQuality       = H((10,   -0.5,  9.5, ('trkQuality',        'trkQuality')))
    trkPt            = H((50,   -0.5,  9.5, ('trkPt',             'trkPt')))
    trkChi2          = H((50,   -0.5,  10.5,('trkChi2',           'trkChi2')))
    vtxChi2          = H((50,   -0.5,  10.5,('vtxChi2',           'vtxChi2')))
    trkEta           = H((50,   -2.5,  2.5, ('trkEta',            'trkEta')))
    d0               = H((50,   -2,    2,   ('d0',                'd0')))
    d0Err            = H((50,    0,    0.2, ('d0Err',             'd0Err')))
    dz               = H((50,   -5,    5,   ('dz',                'dz')))
    dzErr            = H((50,    0,    0.2, ('dzErr',             'dzErr')))





class PFCand:
    plot = _PlotPFCand
