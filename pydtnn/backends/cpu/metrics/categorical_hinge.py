import numpy as np

from pydtnn.backends.cpu.metrics.metric import MetricCPU
from pydtnn.metrics.categorical_hinge import CategoricalHinge


class CategoricalHingeCPU(CategoricalHinge[np.ndarray], MetricCPU):

    def initialize(self) -> None:
        super().initialize()
        self._pos:np.ndarray = np.zeros(self.shape, dtype=self.model.dtype, order="C")
        self._neg:np.ndarray = np.zeros(self.shape, dtype=self.model.dtype, order="C")

        self.pos_maxm:np.ndarray = np.zeros(self.model.batch_size, dtype=self.model.dtype, order="C")
        self.neg:np.ndarray = np.zeros(self.model.batch_size, dtype=self.model.dtype, order="C")

        self.actual_size += self._pos.size + self._neg.size + self.pos_maxm.size + self.neg.size
    # ----

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        _pos = self._pos[: y_pred.shape[0]]
        _neg = self._neg[: y_pred.shape[0]]
        pos_maxm = self.pos_maxm[: y_pred.shape[0]]
        neg = self.neg[: y_pred.shape[0]]

        # pos = np.sum(y_targ * y_pred, axis=-1)
        # neg = np.max((1.0 - y_targ) * y_pred, axis=-1)
        # return np.mean(np.maximum(0.0, neg - pos + 1), axis=-1)

        np.multiply(y_targ, y_pred, dtype=self.model.dtype, out=_pos)
        np.sum(_pos, axis=-1, dtype=self.model.dtype, out=pos_maxm)

        np.multiply(-1, y_targ, dtype=self.model.dtype, out=_neg)
        np.add(_neg, 1, out=_neg, dtype=self.model.dtype)
        np.multiply(_neg, y_pred, out= _neg, dtype=self.model.dtype)
        np.max(_neg, axis=-1, out=neg)

        np.subtract(neg, pos_maxm, out=neg, dtype=self.model.dtype)
        np.add(neg, 1, out=neg, dtype=self.model.dtype)
        np.maximum(0.0, neg, out=pos_maxm)

        maximum = np.mean(pos_maxm, axis=-1)

        return maximum
