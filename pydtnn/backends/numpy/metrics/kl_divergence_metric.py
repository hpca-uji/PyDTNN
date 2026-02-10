from pydtnn.libs import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np

from pydtnn.backends.numpy.metrics.metric import MetricNumpy
from pydtnn.metrics.kl_divergence_metric import KLDivergenceMetric


class KLDivergenceMetricNumpy(KLDivergenceMetric[np.ndarray], MetricNumpy):

    def _model_init(self) -> None:
        super()._model_init()

        self.tmp_memory_used += int(np.prod(self.shape)) * self.model.dtype.itemsize
        self.memory_used += self.tmp_memory_used

    def _post_init(self) -> None:
        super()._post_init()
        with self.model.memory:
            self.loss = self.model.memory.ndarray(self.shape, dtype=self.model.dtype)

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        y_targ = np.asarray(y_targ, dtype=self.model.dtype, order="C")
        loss = self.loss[:y_pred.shape[0]]
        # loss = np.abs(y_pred * np.log(np.abs(y_pred / (y_targ + eps) + eps)))
        np.add(y_targ, self.eps, out=loss)
        np.divide(y_pred, loss, out=loss)
        np.add(loss, self.eps, out=loss)
        np.abs(loss, out=loss)
        np.log(loss, out=loss)
        np.multiply(y_pred, loss, out=loss)
        np.abs(loss, out=loss)

        loss = float(np.sum(loss) / y_pred.shape[0])
        return loss
