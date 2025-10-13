from abc import ABC

from pydtnn.metrics.metric import Metric


class RegressionMAE[T](Metric[T], ABC):
    pass
