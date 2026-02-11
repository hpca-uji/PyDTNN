from pydtnn.libs import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np

from pydtnn.backends.numpy.metrics.metric import MetricNumpy
from pydtnn.metrics.categorical_hinge import CategoricalHinge
import math

class CategoricalHingeNumpy(CategoricalHinge[np.ndarray], MetricNumpy):

    def _model_init(self) -> None:
        super()._model_init()

        self._pos_shape = self.shape
        self._neg_shape = self.shape
        self.pos_maxm_shape = (self.model.batch_size, )
        self.neg_shape = (self.model.batch_size, )
        self.tmp_memory_used += int(math.prod(self._pos_shape) + math.prod(self._neg_shape) + math.prod(self.pos_maxm_shape) + math.prod(self.neg_shape)) * self.model.dtype.itemsize
        self.memory_used += self.tmp_memory_used
    # ----

    def _post_init(self) -> None:
        super()._post_init()
        with self.model.memory:
            self._pos = self.model.memory.ndarray(self._pos_shape, dtype=self.model.dtype)
            self._neg = self.model.memory.ndarray(self._neg_shape, dtype=self.model.dtype)
            self.pos_maxm = self.model.memory.ndarray(self.pos_maxm_shape, dtype=self.model.dtype)
            self.neg = self.model.memory.ndarray(self.neg_shape, dtype=self.model.dtype)

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        y_targ = np.asarray(y_targ, dtype=self.model.dtype, order="C")
        _pos: np.ndarray = self._pos[: y_pred.shape[0]]
        _neg: np.ndarray = self._neg[: y_pred.shape[0]]
        pos_maxm: np.ndarray = self.pos_maxm[: y_pred.shape[0]]
        neg: np.ndarray = self.neg[: y_pred.shape[0]]

        # pos = np.sum(y_targ * y_pred, axis=-1)
        # neg = np.max((1.0 - y_targ) * y_pred, axis=-1)
        # return np.mean(np.maximum(0.0, neg - pos + 1), axis=-1)

        np.multiply(y_targ, y_pred, dtype=self.model.dtype, out=_pos)
        np.sum(_pos, axis=-1, dtype=self.model.dtype, out=pos_maxm)

        np.multiply(-1, y_targ, dtype=self.model.dtype, out=_neg)
        np.add(_neg, 1, out=_neg, dtype=self.model.dtype)
        np.multiply(_neg, y_pred, out=_neg, dtype=self.model.dtype)
        np.max(_neg, axis=-1, out=neg)

        np.subtract(neg, pos_maxm, out=neg, dtype=self.model.dtype)
        np.add(neg, 1, out=neg, dtype=self.model.dtype)
        np.maximum(0.0, neg, out=pos_maxm)

        maximum = float(np.mean(pos_maxm, axis=-1))

        return maximum
