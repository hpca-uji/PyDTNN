import numpy as np

from pydtnn.backends.cpu.metrics.metric import MetricCPU
from pydtnn.metrics.categorical_hinge import CategoricalHinge

class CategoricalHingeCPU(CategoricalHinge[np.ndarray], MetricCPU):

    def initialize(self) -> None:
        super().initialize()

        self._pos_shape = self.shape
        self._neg_shape = self.shape
        self.pos_maxm_shape = (self.model.batch_size, )
        self.neg_shape = (self.model.batch_size, )
        self.temp_memory_size += int(np.prod(self._pos_shape) + np.prod(self._neg_shape) + np.prod(self.pos_maxm_shape) + np.prod(self.neg_shape)) * self.model.dtype.itemsize

        if not self.model.use_memory_pool:
            self._pos: np.ndarray = np.zeros(self._pos_shape, dtype=self.model.dtype, order="C")
            self._neg: np.ndarray = np.zeros(self._neg_shape, dtype=self.model.dtype, order="C")
            self.pos_maxm: np.ndarray = np.zeros(self.pos_maxm_shape, dtype=self.model.dtype, order="C")
            self.neg: np.ndarray = np.zeros(self.neg_shape, dtype=self.model.dtype, order="C")
        else:
            self._pos: np.ndarray = None  # type: ignore (It will be initialized later)
            self._neg: np.ndarray = None  # type: ignore (It will be initialized later)
            self.pos_maxm: np.ndarray = None  # type: ignore (It will be initialized later)
            self.neg: np.ndarray = None  # type: ignore (It will be initialized later)

        self.real_memory_size += self.temp_memory_size
    # ----

    def post_initialize(self) -> None:
        super().post_initialize()
        self._pos = self.model.memory_pool.get_ndarray(self._pos_shape, dtype=self.model.dtype)
        self._neg = self.model.memory_pool.get_ndarray(self._neg_shape, dtype=self.model.dtype)
        self.pos_maxm = self.model.memory_pool.get_ndarray(self.pos_maxm_shape, dtype=self.model.dtype)
        self.neg = self.model.memory_pool.get_ndarray(self.neg_shape, dtype=self.model.dtype)

        self.model.memory_pool.free_buffer(self.temp_memory_size)

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
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
        np.multiply(_neg, y_pred, out= _neg, dtype=self.model.dtype)
        np.max(_neg, axis=-1, out=neg)

        np.subtract(neg, pos_maxm, out=neg, dtype=self.model.dtype)
        np.add(neg, 1, out=neg, dtype=self.model.dtype)
        np.maximum(0.0, neg, out=pos_maxm)

        maximum = float(np.mean(pos_maxm, axis=-1))

        return maximum
