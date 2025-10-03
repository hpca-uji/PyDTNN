import numpy as np

from pydtnn.backends.cpu.metrics import MetricCPU
from pydtnn.metrics import RegressionMAE


class RegressionMAECPU(MetricCPU, RegressionMAE):

    def __call__(self, y_pred: np.ndarray, y_targ: np.ndarray) -> np.ndarray:
        #return np.sum(np.absolute(y_targ - y_pred))
        diff = y_targ - y_pred
        np.absolute(diff, out=diff, dtype=self.model.dtype, casting="unsafe")
        return np.sum(diff, dtype=self.model.dtype)
