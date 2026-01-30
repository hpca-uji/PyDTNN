import numpy as np

from pydtnn.backends.cpu.metrics.metric import MetricCPU
from pydtnn.metrics.regression_mse import RegressionMSE


class RegressionMSECPU(RegressionMSE[np.ndarray], MetricCPU):

    def initialize(self) -> None:
        super().initialize()

        self.temp_memory_size += int(np.prod(self.shape)) * self.model.dtype.itemsize
        if not self.model.use_memory_pool:
            self.diff: np.ndarray = np.zeros(self.shape, dtype=self.model.dtype, order="C")
        else:
            self.diff: np.ndarray = None  #type: ignore (It will be initialized later)

        self.real_memory_size += self.temp_memory_size
    # ----

    def post_initialize(self) -> None:
        super().post_initialize()
        self.diff = self.model.memory_pool.get_ndarray(self.shape, dtype=self.model.dtype)
        self.model.memory_pool.free_buffer(self.temp_memory_size)


    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        diff = self.diff[:y_pred.shape[0]]
        # return np.square(y_targ - y_pred).mean()
        np.subtract(y_targ, y_pred, dtype=self.model.dtype, out=diff)
        np.square(diff, out=diff, dtype=self.model.dtype, casting="unsafe")
        return float(diff.mean(dtype=self.model.dtype))
    # ----
