import numpy as np

from pydtnn.backends.cpu.metrics.metric_cpu import MetricCPU
from pydtnn.metrics.categorical_mse import CategoricalMSE


class CategoricalMSECPU(MetricCPU, CategoricalMSE[np.ndarray]):

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> np.ndarray:
        b = y_targ.shape[0]
        # return np.square(1 - y_pred[np.arange(b), np.argmax(y_targ, axis=1)]).mean()

        y = y_pred[np.arange(b), np.argmax(y_targ, axis=1)]
        y *= -1
        y += 1
        np.square(y, out=y, dtype=self.model.dtype, casting="unsafe")
        return y.mean(dtype=self.model.dtype)
