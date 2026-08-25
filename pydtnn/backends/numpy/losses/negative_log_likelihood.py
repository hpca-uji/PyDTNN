"""Categorical Cross Entropy loss implementation for the NumPy backend."""

import logging
import math
from typing import TYPE_CHECKING

from pydtnn.backends.numpy.losses.abstract.loss import LossNumpy
from pydtnn.libs import numpy as np
from pydtnn.losses.negative_log_likelihood import NegativeLogLikelihood

__all__ = ("NegativeLogLikelihoodNumpy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class NegativeLogLikelihoodNumpy(NegativeLogLikelihood[np.ndarray], LossNumpy):
    """NumPy implementation of the Negative Log Likelihood loss function."""

    def _model_init(self) -> None:
        """Initialize memory requirements and shapes for the loss computation."""
        super()._model_init()

        self._argmax_shape = (self.model.batch_size,)
        self._y_pred_op_shape = (self.model.batch_size,)
        self._y_pred_shape = self.shape

        self.tmp_memory_used += int(math.prod(self._argmax_shape) * np.int32().itemsize)
        self.tmp_memory_used += (
            int(math.prod(self._y_pred_op_shape) + math.prod(self._y_pred_shape))
            * self.model.dtype.itemsize
        )

        self.memory_used += self.tmp_memory_used

    def _post_init(self) -> None:
        """Allocate memory buffers for loss computation."""
        super()._post_init()
        with self.model.memory:
            self._argmax = self.model.memory.ndarray(self._argmax_shape, dtype=np.dtype(np.int32))
            self._y_pred_op = self.model.memory.ndarray(
                self._y_pred_op_shape, dtype=self.model.dtype
            )
            self._y_pred = self.model.memory.ndarray(self._y_pred_shape, dtype=self.model.dtype)

    def compute(
        self, y_pred: np.ndarray, y_targ: np.ndarray, batch_size: int
    ) -> tuple[float, np.ndarray]:
        """
        Compute the categorical cross entropy loss and gradients.

        Args:
            y_pred: Predicted probabilities.
            y_targ: Target labels in one-hot encoded format.
            batch_size: The size of the current batch.

        Returns:
            A tuple containing the scalar loss value and the gradient array.
        """
        b = y_pred.shape[0]
        _argmax: np.ndarray = self._argmax[:b]
        _y_pred_op: np.ndarray = self._y_pred_op[:b]
        dx: np.ndarray = self.dx[:b]
        dx.fill(0)

        # Common
        b_range: np.ndarray = np.arange(b)
        np.argmax(y_targ, axis=1, out=_argmax)
        sum_weights_argmax = np.sum(self.weights[_argmax])

        # Loss
        np.log(y_pred[b_range, _argmax], out=_y_pred_op)
        np.multiply(_y_pred_op, self.weights[_argmax], out=_y_pred_op)
        loss: float = float(-np.sum(_y_pred_op) / sum_weights_argmax)

        # DX
        dx[:] = y_targ
        np.multiply(self.weights[_argmax], y_pred, out=y_pred)
        np.multiply(-1, y_pred, out=y_pred)
        dx[b_range, _argmax] /= y_pred[b_range, _argmax]

        return loss, np.asarray(dx, dtype=self.model.dtype, order="C")

