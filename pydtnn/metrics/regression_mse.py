from abc import ABC

from pydtnn.metrics.metric import Metric


class RegressionMSE[T](Metric[T], ABC):
    pass
