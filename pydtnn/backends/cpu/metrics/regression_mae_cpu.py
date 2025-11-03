import numpy as np

from pydtnn.backends.cpu.metrics.metric_cpu import MetricCPU
from pydtnn.metrics.regression_mae import RegressionMAE


class RegressionMAECPU(MetricCPU, RegressionMAE[np.ndarray]):

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        # return np.sum(np.absolute(y_targ - y_pred))
        diff = np.subtract(y_targ, y_pred, dtype=self.model.dtype)
        np.absolute(diff, out=diff, dtype=self.model.dtype, casting="unsafe")
        return np.mean(diff, dtype=self.model.dtype)
