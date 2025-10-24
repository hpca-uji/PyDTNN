import numpy as np

from pydtnn.backends.cpu.metrics.metric_cpu import MetricCPU
from pydtnn.metrics.categorical_hinge import CategoricalHinge


class CategoricalHingeCPU(MetricCPU, CategoricalHinge[np.ndarray]):

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> np.ndarray:
        # pos = np.sum(y_targ * y_pred, axis=-1)
        # neg = np.max((1.0 - y_targ) * y_pred, axis=-1)
        # return np.mean(np.maximum(0.0, neg - pos + 1), axis=-1)

        pos = y_targ * y_pred
        pos = np.sum(pos, axis=-1)

        neg = -1 * y_targ
        neg += 1.0
        neg *= y_pred
        neg = np.max(neg, axis=-1)

        neg -= pos
        neg += 1
        maximum = np.maximum(0.0, neg)

        return np.mean(maximum, axis=-1)
