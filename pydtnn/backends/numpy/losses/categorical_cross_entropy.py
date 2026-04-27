import logging
import math
from typing import TYPE_CHECKING

from pydtnn.backends.numpy.losses.loss import LossNumpy
from pydtnn.libs import numpy as np
from pydtnn.losses.categorical_cross_entropy import CategoricalCrossEntropy

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


class CategoricalCrossEntropyNumpy(CategoricalCrossEntropy[np.ndarray], LossNumpy):

    def _model_init(self) -> None:
        super()._model_init()

        self._argmax_shape = (self.model.batch_size, )
        self._y_pred_op_shape = (self.model.batch_size, )
        self._y_pred_shape = self.shape

        self.tmp_memory_used += int(math.prod(self._argmax_shape)) * np.int32().itemsize
        self.tmp_memory_used += int(math.prod(self._y_pred_op_shape) + math.prod(self._y_pred_shape)) * self.model.dtype.itemsize

        # _y_pred_sliced_size = self.model.batch_size
        # + _y_pred_sliced_size

        self.memory_used += self.tmp_memory_used

    def _post_init(self) -> None:
        super()._post_init()
        with self.model.memory:
            self._argmax = self.model.memory.ndarray(self._argmax_shape, dtype=np.int32)
            self._y_pred_op = self.model.memory.ndarray(self._y_pred_op_shape, dtype=self.model.dtype)
            self._y_pred = self.model.memory.ndarray(self._y_pred_shape, dtype=self.model.dtype)

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray, batch_size: int) -> tuple[float, np.ndarray]:
        b = y_pred.shape[0]
        _argmax: np.ndarray = self._argmax[:b]
        _y_pred: np.ndarray = self._y_pred[:b]
        _y_pred_op: np.ndarray = self._y_pred_op[:b]
        dx: np.ndarray = self.dx[:b]
        dx.fill(0)

        # Common
        b_range: np.ndarray = np.arange(b)
        np.clip(y_pred, a_min=self.eps, a_max=(1 - self.eps), out=_y_pred)
        np.argmax(y_targ, axis=1, out=_argmax)

        # Loss
        np.log(_y_pred[b_range, _argmax], out=_y_pred_op)
        loss: float = float(-np.sum(_y_pred_op) / b)

        # DX
        # dx: np.ndarray = np.copy(y_targ)
        # dx_amax: np.ndarray = np.argmax(dx, axis=1)
        # dx[b_range, dx_amax] /= (-_y_pred_sliced[b_range, dx_amax] * batch_size)
        dx[:] = y_targ
        np.multiply(-1 * batch_size, _y_pred, out=_y_pred)
        dx[b_range, _argmax] /= _y_pred[b_range, _argmax]
        return loss, np.asarray(dx, dtype=self.model.dtype, order="C")
