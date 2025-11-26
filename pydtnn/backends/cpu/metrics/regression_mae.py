import numpy as np

from pydtnn.backends.cpu.metrics.metric import MetricCPU
from pydtnn.metrics.regression_mae import RegressionMAE


class RegressionMAECPU(RegressionMAE[np.ndarray], MetricCPU):

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        # return np.sum(np.absolute(y_targ - y_pred))
        diff = np.subtract(y_targ, y_pred, dtype=self.model.dtype)
        np.absolute(diff, out=diff, dtype=self.model.dtype, casting="unsafe")
        return np.mean(diff, dtype=self.model.dtype)
