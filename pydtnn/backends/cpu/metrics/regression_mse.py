from pydtnn.libs import numpy as np

from pydtnn.backends.cpu.metrics.metric import MetricCPU
from pydtnn.metrics.regression_mse import RegressionMSE


class RegressionMSECPU(RegressionMSE[np.ndarray], MetricCPU):

    def initialize(self) -> None:
        super().initialize()

        self.temp_memory_size += int(np.prod(self.shape)) * self.model.dtype.itemsize
        self.real_memory_size += self.temp_memory_size
    # ----

    def post_initialize(self) -> None:
        super().post_initialize()
        with self.model.memory:
            self.diff = self.model.memory.ndarray(self.shape, dtype=self.model.dtype)

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        y_targ = np.asarray(y_targ, dtype=self.model.dtype)
        diff = self.diff[:y_pred.shape[0]]
        # return np.square(y_targ - y_pred).mean()
        np.subtract(y_targ, y_pred, dtype=self.model.dtype, out=diff)
        np.square(diff, out=diff, dtype=self.model.dtype, casting="unsafe")
        return float(diff.mean(dtype=self.model.dtype))
    # ----
