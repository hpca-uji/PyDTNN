from abc import ABC

from pydtnn.metrics.metric import Metric


class ConfusionMatrix[T](Metric[T], ABC):
    pass