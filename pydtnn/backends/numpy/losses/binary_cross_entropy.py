"""Binary Cross Entropy loss implementation for the NumPy backend."""

import logging
import math
from typing import TYPE_CHECKING

from pydtnn.backends.numpy.losses.abstract.loss import LossNumpy
from pydtnn.libs import numpy as np
from pydtnn.losses.binary_cross_entropy import BinaryCrossEntropy

__all__ = ("BinaryCrossEntropyNumpy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class BinaryCrossEntropyNumpy(BinaryCrossEntropy[np.ndarray], LossNumpy):
    """NumPy implementation of the Binary Cross Entropy loss function."""

    def _model_init(self) -> None:
        """Initialize model-specific memory requirements for the loss."""
        super()._model_init()

        self._argmax_shape = (self.model.batch_size,)
        self.tmp_memory_used += int(math.prod(self._argmax_shape) * np.int32().itemsize)

        # NOTE: 5 = |{self.neg_targ, self.log_maximum, self._y_pred, self.div_y, self.neg_pred}|
        self.tmp_memory_used += int(5 * math.prod(self.shape)) * self.model.dtype.itemsize
        self.memory_used += self.tmp_memory_used

        # The following attribute will be initialized later
        self._argmax: np.ndarray = None # pyright: ignore[reportAttributeAccessIssue]

    def _post_init(self) -> None:
        """Allocate memory buffers for intermediate calculations."""
        super()._post_init()
        with self.model.memory:
            self.neg_targ = self.model.memory.ndarray(self.shape, dtype=self.model.dtype)
            self.log_maximum = self.model.memory.ndarray(self.shape, dtype=self.model.dtype)
            self._y_pred = self.model.memory.ndarray(self.shape, dtype=self.model.dtype)
            self.div_y = self.model.memory.ndarray(self.shape, dtype=self.model.dtype)
            self.neg_pred = self.model.memory.ndarray(self.shape, dtype=self.model.dtype)
            self._argmax = self.model.memory.ndarray(self._argmax_shape, dtype=np.dtype(np.int32))

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Compute the binary cross entropy loss and its gradient.

        Args:
            y_pred: Predicted values.
            y_targ: Target ground truth values.

        Returns:
            A tuple containing the scalar loss and the gradient array.
        """
        assert len(y_targ.shape) == 2

        b = self.model.real_batch_size

        dx: np.ndarray = self.dx[:b]
        neg_targ: np.ndarray = self.neg_targ[:b]
        log_maximum: np.ndarray = self.log_maximum[:b]
        _y_pred: np.ndarray = self._y_pred[:b]
        div_y: np.ndarray = self.div_y[:b]
        neg_pred: np.ndarray = self.neg_pred[:b]
        _argmax: np.ndarray = self._argmax[:b]

        # Common:
        # (1 - y_targ)
        np.subtract(1, y_targ, out=neg_targ)
        # (1 - _y_pred)
        np.subtract(1, _y_pred, out=neg_pred)

        np.argmax(y_targ, axis=1, out=_argmax)
        sum_weights_argmax = np.sum(self.weights[_argmax])

        # Loss
        # loss: float = -(np.sum( (1 - y_targ) * np.log(np.maximum(1 - _y_pred, eps))) ) / b
        np.maximum(neg_pred, self.eps, out=log_maximum)
        np.log(log_maximum, out=log_maximum)
        np.multiply(neg_targ, log_maximum, out=log_maximum)
        loss: float = float(-np.sum(log_maximum)) / sum_weights_argmax

        # Dx
        np.clip(y_pred, a_min=self.eps, a_max=(1 - self.eps), out=_y_pred)

        # dx: np.ndarray = (-(y_targ / _y_pred) + ((1 - y_targ) / (1 - _y_pred))) / b
        np.divide(y_targ, _y_pred, out=div_y)
        np.multiply(-1, div_y, out=div_y)
        np.divide(neg_targ, neg_pred, out=neg_pred)
        np.add(div_y, neg_pred, out=div_y)
        np.divide(div_y, sum_weights_argmax, out=dx)

        return float(loss), np.asarray(dx, dtype=self.model.dtype, order="C")
