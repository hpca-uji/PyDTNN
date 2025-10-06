import numpy as np

from pydtnn.backends.cpu.losses.loss_cpu import LossCPU
from pydtnn.losses import CategoricalCrossEntropy


class CategoricalCrossEntropyCPU(LossCPU, CategoricalCrossEntropy):

    def __call__(self, y_pred: np.ndarray, y_targ: np.ndarray, batch_size: int) -> tuple[float, np.ndarray]:
        y_pred: np.ndarray = np.clip(y_pred, a_min=self.eps, a_max=(1 - self.eps))
        b_range: np.ndarray = np.arange(y_pred.shape[0])
        loss: float = -np.sum(np.log(y_pred[b_range, np.argmax(y_targ, axis=1)])) / y_pred.shape[0]
        # NOTE/FIXME: The last line before the return raise an error if the model works in int8 due it is trying to store an float64 into a int8 [possible fix below].
        dx: np.ndarray = np.copy(y_targ)
        # dx:np.ndarray = y_targ.astype(np.float64, copy=True)
        dx_amax: np.ndarray = np.argmax(dx, axis=1)
        # NOTE/FIXME: This will raise an error if the model works in int8 due it is trying to store an float64 into a int8.
        dx[b_range, dx_amax] /= (-y_pred[b_range, dx_amax] * batch_size)
        return loss, dx
