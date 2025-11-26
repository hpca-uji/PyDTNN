import numpy as np

from pydtnn.backends.cpu.metrics.metric import MetricCPU
from pydtnn.metrics.categorical_hinge import CategoricalHinge


class CategoricalHingeCPU(CategoricalHinge[np.ndarray], MetricCPU):

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        # pos = np.sum(y_targ * y_pred, axis=-1)
        # neg = np.max((1.0 - y_targ) * y_pred, axis=-1)
        # return np.mean(np.maximum(0.0, neg - pos + 1), axis=-1)

        pos:np.ndarray = np.multiply(y_targ, y_pred, dtype=self.model.dtype)
        pos = np.sum(pos, axis=-1, dtype=self.model.dtype)

        neg = np.multiply(-1, y_targ, dtype=self.model.dtype)
        np.add(neg, 1, out=neg, dtype=self.model.dtype)
        np.multiply(neg, y_pred, out= neg, dtype=self.model.dtype)
        neg:np.ndarray = np.max(neg, axis=-1)

        np.subtract(neg, pos, out=neg, dtype=self.model.dtype)
        np.add(neg, 1, out=neg, dtype=self.model.dtype)
        maximum = np.maximum(0.0, neg)

        maximum = np.mean(maximum, axis=-1)

        return maximum
