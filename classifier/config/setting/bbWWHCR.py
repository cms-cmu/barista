from enum import IntEnum
from classifier.task import GlobalSetting

class InputBranch(GlobalSetting):
    "Name of branches in the input root file"
    # B-jet features (2 b-jets)
    feature_bJetCand: list[str] = ["pt", "eta", "phi", "mass"]
    # Non-b jet features (2 non-b jets)  
    feature_nonbJetCand: list[str] = ["pt", "eta", "phi", "mass"]
    # Leading lepton features
    feature_leadingLep: list[str] = ["pt", "eta", "phi", "mass", "isE", "isM"]
    # MET features
    feature_MET: list[str] = ["pt", "phi"]
    
    n_bJetCand: int = 2
    n_nonbJetCand: int = 2
    n_leadingLep: int = 1
    n_MET: int = 1

    @classmethod
    def get__feature_bJetCand(cls, var: list[str]):
        return [f"bJetCand_{f}" for f in var]
    
    @classmethod
    def get__feature_nonbJetCand(cls, var: list[str]):
        return [f"nonbJetCand_{f}" for f in var]
    
    @classmethod
    def get__feature_leadingLep(cls, var: list[str]):
        return [f"leadingLep_{f}" for f in var]
    
    @classmethod
    def get__feature_MET(cls, var: list[str]):
        return [f"MET_{f}" for f in var]

class Input(GlobalSetting):
    "Name of the keys in the input batch."
    label: str = "label"
    weight: str = "weight"
    bJetCand: str = "bJetCand"
    nonbJetCand: str = "nonbJetCand" 
    leadingLep: str = "leadingLep"
    MET: str = "MET"
    SR: str = "SR"
    CR: str = "CR"

class Output(GlobalSetting):
    "Name of the keys in the output batch."
    class_raw: str = "class_raw"
    class_prob: str = "class_prob"

# Keep only the regions you're using
class MassRegion(IntEnum):
    SR = 0b01
    CR = 0b10