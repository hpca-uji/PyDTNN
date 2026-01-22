import numpy as np

from pydtnn.backends.cpu.metrics.metric import MetricCPU
from pydtnn.metrics.categorical_accuracy import CategoricalAccuracy


class CategoricalAccuracyCPU(CategoricalAccuracy[np.ndarray], MetricCPU):

    def initialize(self) -> None:
        super().initialize()
        self._argmax = np.zeros(self.model.batch_size, dtype=self.model.dtype, order="C")

        self.actual_size += self._argmax.size # + arange_size = self.model.batch_size
    # ----

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        b = y_targ.shape[0]
        _argmax = self._argmax[:b]
        # return np.sum(y_targ[np.arange(b), np.argmax(y_pred, axis=1)]) * 100 / b
        
        np.argmax(y_pred, axis=1, out=_argmax)
        y = y_targ[np.arange(b), _argmax]
        y = np.sum(y, dtype=self.model.dtype)
        y *= 100 / b
        return y
