from abc import ABC

from pydtnn.metrics.metric import Metric


class CategoricalMAE[T](Metric[T], ABC):
    pass
