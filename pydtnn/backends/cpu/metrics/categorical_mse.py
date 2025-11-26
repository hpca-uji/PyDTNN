import numpy as np

from pydtnn.backends.cpu.metrics.metric import MetricCPU
from pydtnn.metrics.categorical_mse import CategoricalMSE


class CategoricalMSECPU(CategoricalMSE[np.ndarray], MetricCPU):

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        # return np.square(1 - y_pred[np.arange(b), np.argmax(y_targ, axis=1)]).mean()
        error = np.subtract(y_pred, y_targ, dtype=self.model.dtype)
        np.power(error, 2, out=error, dtype=self.model.dtype, casting="unsafe")
        return np.mean(error, dtype=self.model.dtype)
