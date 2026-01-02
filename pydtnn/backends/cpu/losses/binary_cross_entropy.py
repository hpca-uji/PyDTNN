import numpy as np

from pydtnn.backends.cpu.losses.loss import LossCPU
from pydtnn.losses.binary_cross_entropy import BinaryCrossEntropy


class BinaryCrossEntropyCPU(BinaryCrossEntropy[np.ndarray], LossCPU):
    def compute(self, _y_pred: np.ndarray, y_targ: np.ndarray, batch_size: int) -> tuple[float, np.ndarray]:
        assert len(y_targ.shape) == 2
        # Loss
        b = y_targ.shape[0]
        loss: float = -np.sum(np.log(np.maximum((1 - y_targ) - _y_pred, self.eps))) / b

        # Dx
        y_pred: np.ndarray = np.clip(_y_pred, a_min=self.eps, a_max=(1 - self.eps))
        dx: np.ndarray = (-(y_targ / y_pred) + ((1 - y_targ) / (1 - y_pred))) / batch_size
        return loss, np.asarray(dx, dtype=self.model.dtype, order="C", copy=None)
