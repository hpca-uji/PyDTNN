import numpy as np

from pydtnn.backends.cpu.metrics.metric import MetricCPU
from pydtnn.metrics.categorical_mae import CategoricalMAE


class CategoricalMAECPU(CategoricalMAE[np.ndarray], MetricCPU):

    def initialize(self) -> None:
        super().initialize()
        self.error:np.ndarray = np.zeros(self.shape, dtype=self.model.dtype, order="C")

        self.actual_size += self.error.size

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        error = self.error[:y_pred.shape[0]]
        # return np.sum(np.absolute(1 - y_pred[np.arange(b), np.argmax(y_targ, axis=1)]))
        np.subtract(y_pred, y_targ, dtype=self.model.dtype, out=error)
        np.absolute(error, out=error, dtype=self.model.dtype)
        return np.mean(error, dtype=self.model.dtype)
