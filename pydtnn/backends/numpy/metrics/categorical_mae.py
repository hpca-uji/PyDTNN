import logging
logger = logging.getLogger(__name__)

from pydtnn.libs import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np

from pydtnn.backends.numpy.metrics.metric import MetricNumpy
from pydtnn.metrics.categorical_mae import CategoricalMAE
import math

class CategoricalMAENumpy(CategoricalMAE[np.ndarray], MetricNumpy):

    def _model_init(self) -> None:
        super()._model_init()

        self.error: np.ndarray = None  # type: ignore (It will be initialized later)
        self.tmp_memory_used += int(math.prod(self.shape)) * self.model.dtype.itemsize
        self.memory_used += self.tmp_memory_used

    def _post_init(self) -> None:
        super()._post_init()
        with self.model.memory:
            self.error = self.model.memory.ndarray(self.shape, dtype=self.model.dtype)

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        y_targ = np.asarray(y_targ, dtype=self.model.dtype, order="C")
        error = self.error[:y_pred.shape[0]]
        # return np.sum(np.absolute(1 - y_pred[np.arange(b), np.argmax(y_targ, axis=1)]))
        np.subtract(y_pred, y_targ, dtype=self.model.dtype, out=error)
        np.absolute(error, out=error, dtype=self.model.dtype)
        return float(np.mean(error, dtype=self.model.dtype))
