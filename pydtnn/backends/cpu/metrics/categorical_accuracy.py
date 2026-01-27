import numpy as np

from pydtnn.backends.cpu.metrics.metric import MetricCPU
from pydtnn.metrics.categorical_accuracy import CategoricalAccuracy


class CategoricalAccuracyCPU(CategoricalAccuracy[np.ndarray], MetricCPU):

    def initialize(self) -> None:
        super().initialize()
        self._argmax_shape = (self.model.batch_size, )
        self.temp_memory_size = int(np.prod(self._argmax_shape))
        if not self.model.use_memory_pool:
            self._argmax: np.ndarray = np.zeros(self._argmax_shape, dtype=np.int32, order="C")
        else:
            self._argmax: np.ndarray = None  # type: ignore (It will be initalized later)

        self.real_memory_size += self.temp_memory_size # + arange_size = self.model.batch_size
    # ----

    def post_initialize(self) -> None:
        super().post_initialize()
        self._argmax = np.asarray(self.model.memory_pool.get_ndarray(self._argmax_shape), dtype=np.int32, order="C", copy=None)
        self.model.memory_pool.free_memory(self.temp_memory_size)

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        b = y_targ.shape[0]
        _argmax = self._argmax[:b]
        # return np.sum(y_targ[np.arange(b), np.argmax(y_pred, axis=1)]) * 100 / b
        
        np.argmax(y_pred, axis=1, out=_argmax)
        y = y_targ[np.arange(b), _argmax]
        y = np.sum(y, dtype=self.model.dtype)
        y *= 100 / b
        return float(y)
