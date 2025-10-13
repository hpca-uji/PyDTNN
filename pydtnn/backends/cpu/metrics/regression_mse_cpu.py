import numpy as np

from pydtnn.backends.cpu.metrics import MetricCPU
from pydtnn.metrics import RegressionMSE


class RegressionMSECPU(MetricCPU, RegressionMSE[np.ndarray]):

    def __call__(self, y_pred: np.ndarray, y_targ: np.ndarray) -> np.ndarray:
        # return np.square(y_targ - y_pred).mean()
        diff = y_targ - y_pred
        np.square(diff, out=diff, dtype=self.model.dtype, casting="unsafe")
        return diff.mean(dtype=self.model.dtype)
