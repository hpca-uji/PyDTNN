import numpy as np

from pydtnn.backends.cpu.metrics.metric_cpu import MetricCPU
from pydtnn.metrics.regression_mse import RegressionMSE


class RegressionMSECPU(MetricCPU, RegressionMSE[np.ndarray]):

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> np.ndarray:
        # return np.square(y_targ - y_pred).mean()
        diff = np.subtract(y_targ, y_pred, dtype=self.model.dtype)
        np.square(diff, out=diff, dtype=self.model.dtype, casting="unsafe")
        return diff.mean(dtype=self.model.dtype)
