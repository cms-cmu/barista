from src.classifier.task import GlobalState


class DefaultSetting(GlobalState):
    seed: tuple[int, int] = None
    """[cpu, cuda]"""
    n_threads: tuple[int, int] = None
    """[intraop, interop]"""
