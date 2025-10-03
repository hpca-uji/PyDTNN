import numpy as np

from pydtnn.backends.cpu.metrics import MetricCPU
from pydtnn.metrics import CategoricalMAE


class CategoricalMAECPU(MetricCPU, CategoricalMAE):

    def __call__(self, y_pred:np.ndarray, y_targ:np.ndarray) -> np.ndarray:
        b = y_targ.shape[0]
        #return np.sum(np.absolute(1 - y_pred[np.arange(b), np.argmax(y_targ, axis=1)]))
        y = y_pred[np.arange(b), np.argmax(y_targ, axis=1)]
        y *= -1
        y += 1
        np.absolute(y, out=y, dtype=self.model.dtype, casting="unsafe")
        return np.sum(y)
