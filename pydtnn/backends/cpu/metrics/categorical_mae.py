import numpy as np

from pydtnn.backends.cpu.metrics.metric import MetricCPU
from pydtnn.metrics.categorical_mae import CategoricalMAE


class CategoricalMAECPU(CategoricalMAE[np.ndarray], MetricCPU):

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        # return np.sum(np.absolute(1 - y_pred[np.arange(b), np.argmax(y_targ, axis=1)]))
        error = np.subtract(y_pred, y_targ, dtype=self.model.dtype)
        np.absolute(error, out=error, dtype=self.model.dtype)
        return np.mean(error, dtype=self.model.dtype)
