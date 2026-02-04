from pydtnn.libs import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np

from pydtnn.backends.cpu.losses.loss import LossCPU
from pydtnn.losses.categorical_cross_entropy import CategoricalCrossEntropy


class CategoricalCrossEntropyCPU(CategoricalCrossEntropy[np.ndarray], LossCPU):

    def initialize(self) -> None:
        super().initialize()

        self._argmax_shape = (self.model.batch_size, )
        self._y_pred_op_shape = (self.model.batch_size, )
        self._y_pred_shape = self.shape

        self.temp_memory_size += int(np.prod(self._argmax_shape)) * np.int32().itemsize
        self.temp_memory_size += int(np.prod(self._y_pred_op_shape) + np.prod(self._y_pred_shape)) * self.model.dtype.itemsize

        # _y_pred_sliced_size = self.model.batch_size
        # + _y_pred_sliced_size

        self.real_memory_size += self.temp_memory_size

    def post_initialize(self) -> None:
        super().post_initialize()
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

        # Common
        b_range: np.ndarray = np.arange(b)
        np.clip(y_pred, a_min=self.eps, a_max=(1 - self.eps), out=_y_pred)
        np.argmax(y_targ, axis=1, out=_argmax)
        _y_pred_sliced = _y_pred[b_range, _argmax]

        # Loss
        np.log(_y_pred_sliced, out=_y_pred_op)
        loss: float = -np.sum(_y_pred_op) / b

        # DX
        # dx: np.ndarray = np.copy(y_targ)
        # dx_amax: np.ndarray = np.argmax(dx, axis=1)
        # dx[b_range, dx_amax] /= (-_y_pred_sliced[b_range, dx_amax] * batch_size)
        dx.fill(0)
        np.multiply(-1 * batch_size, _y_pred_sliced, out=_y_pred_sliced)
        dx[b_range, _argmax] = y_targ[b_range, _argmax] / _y_pred_sliced

        return loss, np.asarray(dx, dtype=self.model.dtype)
