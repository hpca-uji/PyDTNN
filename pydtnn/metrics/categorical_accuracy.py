from abc import ABC

from pydtnn.metrics.metric import Metric


class CategoricalAccuracy[T](Metric[T], ABC):
    pass
