import logging
import math
from typing import TYPE_CHECKING

from pydtnn.backends.numpy.metrics.metric import MetricNumpy
from pydtnn.libs import numpy as np
from pydtnn.metrics.categorical_accuracy import CategoricalAccuracy

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


class CategoricalAccuracyNumpy(CategoricalAccuracy[np.ndarray], MetricNumpy):

    def _model_init(self) -> None:
        super()._model_init()
        self._argmax_shape = (self.model.batch_size, )
        self.tmp_memory_used = int(math.prod(self._argmax_shape)) * np.int32().itemsize
        self.memory_used += self.tmp_memory_used  # + arange_size = self.model.batch_size
    # ----

    def _post_init(self) -> None:
        super()._post_init()
        with self.model.memory:
            self._argmax = self.model.memory.ndarray(self._argmax_shape, dtype=np.int32)

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        y_targ = np.asarray(y_targ, dtype=self.model.dtype, order="C")
        b = y_targ.shape[0]
        _argmax = self._argmax[:b]
        # return np.sum(y_targ[np.arange(b), np.argmax(y_pred, axis=1)]) * 100 / b

        np.argmax(y_pred, axis=1, out=_argmax)
        y = y_targ[np.arange(b), _argmax]
        y = np.sum(y, dtype=self.model.dtype)
        y *= 100 / b
        return float(y)
