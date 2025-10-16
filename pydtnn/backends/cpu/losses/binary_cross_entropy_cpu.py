import numpy as np

from pydtnn.backends.cpu.losses.loss_cpu import LossCPU
from pydtnn.losses import BinaryCrossEntropy


class BinaryCrossEntropyCPU(LossCPU, BinaryCrossEntropy):
    def __call__(self, y_pred: np.ndarray, y_targ: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        assert len(y_targ.shape) == 2
        b = y_targ.shape[0]
        loss: float = -np.sum(np.log(np.maximum((1 - y_targ) - y_pred, self.eps))) / b
        y_pred: np.ndarray = np.clip(y_pred, a_min=self.eps, a_max=(1 - self.eps))
        dx: np.ndarray = (-(y_targ / y_pred) + ((1 - y_targ) / (1 - y_pred))) / batch_size        
        return loss, np.asarray(dx, dtype=self.model.dtype, order="C", copy=None)
