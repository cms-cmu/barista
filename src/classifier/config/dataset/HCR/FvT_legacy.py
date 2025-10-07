# TODO: this should be removed after new skim
from . import FvT, _picoAOD, legacy


class Train(FvT.Train, legacy._CommonTrain): ...


class TrainBaseline(_picoAOD.Background, Train): ...


class Eval(FvT.Eval, legacy.Eval): ...
