import math
from pydtnn.metrics.regression_mse import RegressionMSE
from pydtnn.backends.numpy.metrics.metric import MetricNumpy
from typing import TYPE_CHECKING
from pydtnn.libs import numpy as np
import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


class RegressionMSENumpy(RegressionMSE[np.ndarray], MetricNumpy):

    def _model_init(self) -> None:
        super()._model_init()

        self.tmp_memory_used += int(math.prod(self.shape)) * self.model.dtype.itemsize
        self.memory_used += self.tmp_memory_used
    # ----

    def _post_init(self) -> None:
        super()._post_init()
        with self.model.memory:
            self.diff = self.model.memory.ndarray(self.shape, dtype=self.model.dtype)

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        y_targ = np.asarray(y_targ, dtype=self.model.dtype, order="C")
        diff = self.diff[:y_pred.shape[0]]
        # return np.square(y_targ - y_pred).mean()
        np.subtract(y_targ, y_pred, dtype=self.model.dtype, out=diff)
        np.square(diff, out=diff, dtype=self.model.dtype, casting="unsafe")
        return float(diff.mean(dtype=self.model.dtype))
    # ----
