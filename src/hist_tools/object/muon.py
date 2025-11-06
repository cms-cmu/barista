from ...hist_tools import H
from .lepton import _PlotDiLepton, _PlotLepton



class _PlotCommon:
    ip3d              = H((50,    0,      2      , ("ip3d",            'ip3d')))
    ip3d_l            = H((50,    0,     20      , ("ip3d",            'ip3d')))
    sip3d             = H((50,    -0.1,  10      , ("sip3d",           'sip3d')))
    sip3d_l           = H((50,    -0.1, 100      , ("sip3d",           'sip3d')))
    pfRelIso04_all    = H((50,    0,      2      , ("pfRelIso04_all",  'pfRelIso04_all')))
    pfRelIso03_all    = H((50,    0,      2      , ("pfRelIso03_all",  'pfRelIso03_all')))
    pfRelIso03_chg    = H((50,    0,      2      , ("pfRelIso03_chg",  'pfRelIso03_chg')))
    dxy               = H((50,    -2,     2      , ("dxy",             'dxy')))
    dxyErr            = H((50,    0,      0.2    , ("dxyErr",          'dxyErr')))
    tkRelIso          = H((50,    0,      2      , ("tkRelIso",        'tkRelIso')))
    pdgId             = H((31, -15.5,    15.5    , ("pdgId",           'pdgId')))
    looseId           = H((5,  -0.5,      4.5    , ("looseId",         'looseId')))
    mediumId          = H((5,  -0.5,      4.5    , ("mediumId",        'mediumId')))
    tightId           = H((5,  -0.5,      4.5    , ("tightId",         'tightId')))
    softId            = H((5,  -0.5,      4.5    , ("softId",          'softId')))
    highPtId          = H((5,  -0.5,      4.5    , ("highPtId",        'highPtId')))
    mediumPromptId    = H((5,  -0.5,      4.5    , ("mediumPromptId",  'mediumPromptId')))
    mvaId             = H((10,  -0.5,     9.5    , ("mvaId",           'mvaId')))
    pfIsoId           = H((10, -0.5,      9.5    , ("pfIsoId",         'pfIsoId')))
    tkIsoId           = H((10, -0.5,      9.5    , ("tkIsoId",         'tkIsoId')))
    jetIdx            = H((20, -1.5,     18.5    , ("jetIdx",          'jetIdx')))
    genPartFlav       = H((25, -0.5,     24.5    , ("genPartFlav",     'genPartFlav')))



    # Other Vars:   'ptErr',  'selected'
    


class _PlotMuon(_PlotCommon, _PlotLepton):
    ...


class _PlotDiMuon(_PlotCommon, _PlotDiLepton):
    ...


class Muon:
    plot = _PlotMuon
    plot_pair = _PlotDiMuon
    skip_detailed_plots = ['ip3d', 'ip3d_l', 'sip3d', 'sip3d_l', 
                           'pfRelIso04_all',  'dxy', 'dxyErr', 
                           'tkRelIso', 'pdgId', 'looseId', 
                           'mediumId', 'tightId', 'softId', 'highPtId', 
                           'mediumPromptId', 'mvaId', 'pfIsoId', 'tkIsoId', 
                           'jetIdx', 'genPartFlav']
