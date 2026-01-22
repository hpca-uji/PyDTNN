import numpy as np

from pydtnn.backends.cpu.losses.loss import LossCPU
from pydtnn.losses.categorical_cross_entropy import CategoricalCrossEntropy


class CategoricalCrossEntropyCPU(CategoricalCrossEntropy[np.ndarray], LossCPU):

    def initialize(self) -> None:
        super().initialize()

        self._argmax = np.zeros(self.model.batch_size, dtype=np.int32, order="C")
        self._y_pred_op = np.zeros(self.model.batch_size, dtype=self.model.dtype, order="C")

        self._y_pred = np.zeros(self.shape, dtype=self.model.dtype, order="C")
        
        _y_pred_sliced_size = self.model.batch_size

        self.actual_size += self._argmax.size + self._y_pred.size + self._y_pred_op.size + self.dx.size + _y_pred_sliced_size

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray, batch_size: int) -> tuple[float, np.ndarray]:
        b = y_pred.shape[0]
        _argmax = self._argmax[:b]
        _y_pred = self._y_pred[:b]
        _y_pred_op = self._y_pred_op[:b]
        dx = self.dx[:b]
        
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

        return loss, np.asarray(dx, dtype=self.model.dtype, order="C", copy=None)
