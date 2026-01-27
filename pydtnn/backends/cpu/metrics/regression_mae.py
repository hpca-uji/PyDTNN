import numpy as np

from pydtnn.backends.cpu.metrics.metric import MetricCPU
from pydtnn.metrics.regression_mae import RegressionMAE


class RegressionMAECPU(RegressionMAE[np.ndarray], MetricCPU):

    def initialize(self) -> None:
        super().initialize()

        self.temp_memory_size += int(np.prod(self.shape))
        if not self.model.use_memory_pool:
            self.diff: np.ndarray = np.zeros(self.shape, dtype=self.model.dtype, order="C")
        else:
            self.diff: np.ndarray = None  #type: ignore (It will be initialized later)

        self.real_memory_size += self.temp_memory_size
    # ----

    def post_initialize(self) -> None:
        super().post_initialize()
        self.diff = self.model.memory_pool.get_ndarray(self.shape)
        self.model.memory_pool.free_memory(self.temp_memory_size)

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        diff = self.diff[:y_pred.shape[0]]
        # return np.sum(np.absolute(y_targ - y_pred))
        np.subtract(y_targ, y_pred, dtype=self.model.dtype, out=diff)
        np.absolute(diff, out=diff, dtype=self.model.dtype, casting="unsafe")
        return float(np.mean(diff, dtype=self.model.dtype))
