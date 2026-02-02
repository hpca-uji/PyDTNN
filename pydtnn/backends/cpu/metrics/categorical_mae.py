from pydtnn.libs import numpy as np

from pydtnn.backends.cpu.metrics.metric import MetricCPU
from pydtnn.metrics.categorical_mae import CategoricalMAE


class CategoricalMAECPU(CategoricalMAE[np.ndarray], MetricCPU):

    def initialize(self) -> None:
        super().initialize()

        self.temp_memory_size += int(np.prod(self.shape)) * self.model.dtype.itemsize
        if not self.model.use_memory_pool:
            self.error: np.ndarray = np.zeros(self.shape, dtype=self.model.dtype)
        else:
            self.error: np.ndarray = None  # type: ignore (It will be initialized later)
        self.real_memory_size += self.temp_memory_size

    def post_initialize(self) -> None:
        super().post_initialize()
        self.error = self.model.memory.ndarray(self.shape, dtype=self.model.dtype)
        self.model.memory._free(self.temp_memory_size)

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        y_targ = np.asarray(y_targ, dtype=self.model.dtype)
        error = self.error[:y_pred.shape[0]]
        # return np.sum(np.absolute(1 - y_pred[np.arange(b), np.argmax(y_targ, axis=1)]))
        np.subtract(y_pred, y_targ, dtype=self.model.dtype, out=error)
        np.absolute(error, out=error, dtype=self.model.dtype)
        return float(np.mean(error, dtype=self.model.dtype))
