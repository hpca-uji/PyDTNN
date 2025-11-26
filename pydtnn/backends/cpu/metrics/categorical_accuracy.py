import numpy as np

from pydtnn.backends.cpu.metrics.metric import MetricCPU
from pydtnn.metrics.categorical_accuracy import CategoricalAccuracy


class CategoricalAccuracyCPU(CategoricalAccuracy[np.ndarray], MetricCPU):

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        b = y_targ.shape[0]
        # return np.sum(y_targ[np.arange(b), np.argmax(y_pred, axis=1)]) * 100 / b
        y = y_targ[np.arange(b), np.argmax(y_pred, axis=1)]
        y = np.sum(y, dtype=self.model.dtype)
        y *= 100 / b
        return y
