from src.classifier.task import GlobalState


class DefaultSetting(GlobalState):
    seed: int = ...
    n_threads: tuple[int, int] = ...
    """[intraop, interop]"""
