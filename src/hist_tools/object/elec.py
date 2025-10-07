from ...hist_tools import H
from .lepton import _PlotDiLepton, _PlotLepton


class _PlotCommon:
    pdgId                      = H((25, -12.5,  12.5  , ("pdgId"                        ,'pdgId')))
    dr03HcalDepth1TowerSumEt   = H((50,   0,    10    , ("dr03HcalDepth1TowerSumEt"     ,'dr03HcalDepth1TowerSumEt')))
    dr03TkSumPt                = H((50,   0,    10    , ("dr03TkSumPt"                  ,'dr03TkSumPt')))
    hoe                        = H((50,   0,     0.5  , ("hoe"                          ,'hoe')))
    eInvMinusPInv              = H((50,  -0.2,   0.2  , ("eInvMinusPInv"                ,'eInvMinusPInv')))
    miniPFRelIso_all           = H((50,   0,     2    , ("miniPFRelIso_all"             ,'miniPFRelIso_all')))
    miniPFRelIso_chg           = H((50,   0,     2    , ("miniPFRelIso_chg"             ,'miniPFRelIso_chg')))
    r9                         = H((50,   0,     2    , ("r9"                           ,'r9')))
    scEtOverPt                 = H((50,  -1,     1    , ("scEtOverPt"                   ,'scEtOverPt')))                
    sieie                      = H((50,   0,     0.1  , ("sieie"                        ,'sieie')))
    pfRelIso03_all             = H((50,   0,     2    , ("pfRelIso03_all"               ,'pfRelIso03_all')))
    pfRelIso03_chg             = H((50,   0,     2    , ("pfRelIso03_chg"               ,'pfRelIso03_chg')))
    cutBased                   = H((10,  -0.5,   9.5  , ("cutBased"                     ,'cutBased')))
    convVeto                   = H((2,   -0.5,   1.5  , ("convVeto"                     ,'convVeto')))
    mvaFall17V2Iso             = H((50,  -1,     1    , ("mvaFall17V2Iso"               ,'mvaFall17V2Iso')))
    mvaFall17V2noIso           = H((50,  -1,     1    , ("mvaFall17V2noIso"             ,'mvaFall17V2noIso')))
    genPartFlav                = H((25,  -0.5,  24.5  , ("genPartFlav"                  ,'genPartFlav')))



class _PlotElec(_PlotCommon, _PlotLepton):
    ...


class _PlotDiElec(_PlotCommon, _PlotDiLepton):
    ...


class Elec:
    plot = _PlotElec
    plot_pair = _PlotDiElec
    skip_detailed_plots = ['pdgId', 'dr03HcalDepth1TowerSumEt', 'dr03TkSumPt', 
                           'hoe', 'eInvMinusPInv', 'miniPFRelIso_all', 'miniPFRelIso_chg', 
                           'r9', 'scEtOverPt', 'sieie',  'cutBased', 'convVeto', 
                           'mvaFall17V2Iso', 'mvaFall17V2noIso', 'genPartFlav']
