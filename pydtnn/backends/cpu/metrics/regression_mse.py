import numpy as np

from pydtnn.backends.cpu.metrics.metric import MetricCPU
from pydtnn.metrics.regression_mse import RegressionMSE


class RegressionMSECPU(RegressionMSE[np.ndarray], MetricCPU):

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        # return np.square(y_targ - y_pred).mean()
        diff = np.subtract(y_targ, y_pred, dtype=self.model.dtype)
        np.square(diff, out=diff, dtype=self.model.dtype, casting="unsafe")
        return diff.mean(dtype=self.model.dtype)
