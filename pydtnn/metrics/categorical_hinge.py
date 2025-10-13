from abc import ABC

from pydtnn.metrics.metric import Metric


class CategoricalHinge[T](Metric[T], ABC):
    pass
