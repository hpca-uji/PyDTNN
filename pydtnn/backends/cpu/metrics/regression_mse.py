import numpy as np

from pydtnn.backends.cpu.metrics.metric import MetricCPU
from pydtnn.metrics.regression_mse import RegressionMSE


class RegressionMSECPU(RegressionMSE[np.ndarray], MetricCPU):

    def initialize(self) -> None:
        super().initialize()
        self.diff = np.zeros(self.shape, dtype=self.model.dtype, order="C")

        self.actual_size += self.diff.size
    # ----

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        diff = self.diff[:y_pred.shape[0]]
        # return np.square(y_targ - y_pred).mean()
        np.subtract(y_targ, y_pred, dtype=self.model.dtype, out=diff)
        np.square(diff, out=diff, dtype=self.model.dtype, casting="unsafe")
        return float(diff.mean(dtype=self.model.dtype))
    # ----
