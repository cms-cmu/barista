from classifier.task import GlobalState

from ..setting import ml


class MultiClass(GlobalState):
    labels: list[str] = []
    trainable_labels: list[str] = []

    @classmethod
    def add(cls, *labels: str):
        for label in labels:
            if label not in cls.labels:
                if label in ml.MultiClass.nontrainable_labels:
                    cls.labels.append(label)
                else:
                    cls.labels.insert(0, label)
                    cls.trainable_labels.insert(0, label)

    @classmethod
    def index(cls, label: str):
        try:
            return cls.labels.index(label)
        except ValueError:
            return None

    @classmethod
    def indices(cls, *labels: str):
        return [cls.index(label) for label in labels]

    @classmethod
    def n_trainable(cls):
        return len(cls.trainable_labels)

    @classmethod
    def n_nontrainable(cls):
        return len(cls.labels) - cls.n_trainable()
