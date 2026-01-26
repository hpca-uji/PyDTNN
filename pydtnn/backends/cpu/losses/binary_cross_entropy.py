import numpy as np

from pydtnn.backends.cpu.losses.loss import LossCPU
from pydtnn.losses.binary_cross_entropy import BinaryCrossEntropy


class BinaryCrossEntropyCPU(BinaryCrossEntropy[np.ndarray], LossCPU):

    def initialize(self) -> None:
        super().initialize()

        neg_targ_size = self.shape
        log_maximum_size = self.shape
        _y_pred_size = self.shape
        div_y_size = self.shape
        neg_pred_size = self.shape

        self.temp_size += int(5 * np.prod(self.shape))

        if not self.model.use_memory_pool:
            self.neg_targ: np.ndarray = np.zeros(neg_targ_size, dtype=self.model.dtype, order="C")
            self.log_maximum: np.ndarray = np.zeros(log_maximum_size, dtype=self.model.dtype, order="C")
            self._y_pred: np.ndarray = np.zeros(_y_pred_size, dtype=self.model.dtype, order="C")
            self.div_y: np.ndarray = np.zeros(div_y_size, dtype=self.model.dtype, order="C")
            self.neg_pred: np.ndarray = np.zeros(neg_pred_size, dtype=self.model.dtype, order="C")
        else:
            self.neg_targ: np.ndarray = None  # type: ignore (It will be initialized later)
            self.log_maximum: np.ndarray = None  # type: ignore (It will be initialized later)
            self._y_pred: np.ndarray = None  # type: ignore (It will be initialized later)
            self.div_y: np.ndarray = None  # type: ignore (It will be initialized later)
            self.neg_pred: np.ndarray = None  # type: ignore (It will be initialized later)

        self.actual_size += self.temp_size

    def post_initialize(self) -> None:
        super().post_initialize()
        self.neg_targ = self.model.memory_pool.get_ndarray(self.shape)
        self.log_maximum = self.model.memory_pool.get_ndarray(self.shape)
        self._y_pred = self.model.memory_pool.get_ndarray(self.shape)
        self.div_y = self.model.memory_pool.get_ndarray(self.shape)
        self.neg_pred = self.model.memory_pool.get_ndarray(self.shape)
        self.model.memory_pool.free_memory(self.temp_size)
    # ----

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray, batch_size: int) -> tuple[float, np.ndarray]:
        assert len(y_targ.shape) == 2
        
        b = y_targ.shape[0]

        dx: np.ndarray = self.dx[:b]
        neg_targ: np.ndarray = self.neg_targ[:b]
        log_maximum: np.ndarray = self.log_maximum[:b]
        _y_pred: np.ndarray = self._y_pred[:b]
        div_y: np.ndarray = self.div_y[:b]
        neg_pred: np.ndarray = self.neg_pred[:b]

        # Loss
        # loss: float = -np.sum(np.log(np.maximum((1 - y_targ) - _y_pred, eps))) / b
        np.subtract(1, y_targ, out=neg_targ)
        np.subtract(neg_targ, y_pred, out=log_maximum)
        np.maximum(log_maximum, self.eps, out=log_maximum)
        np.log(log_maximum, out=log_maximum)
        loss: float = -np.sum(log_maximum) / b

        # Dx
        np.clip(y_pred, a_min=self.eps, a_max=(1 - self.eps), out=_y_pred)
        
        # dx: np.ndarray = (-(y_targ / _y_pred) + ((1 - y_targ) / (1 - _y_pred))) / batch_size
        np.divide(y_targ, _y_pred, out=div_y)
        np.multiply(-1, div_y, out=div_y)
        #neg_targ = np.subtract(1, y_targ)  # Move above.
        np.subtract(1, _y_pred, out=neg_pred)
        np.divide(neg_targ, neg_pred, out=neg_pred)
        np.add(div_y, neg_pred, out=div_y)
        np.divide(div_y, batch_size, out=dx)

        return loss, np.asarray(dx, dtype=self.model.dtype, order="C", copy=None)
