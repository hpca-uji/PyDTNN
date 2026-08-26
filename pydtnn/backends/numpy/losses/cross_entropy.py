"""Categorical Cross Entropy loss implementation for the NumPy backend."""

import logging
import math
from typing import TYPE_CHECKING

from pydtnn.backends.numpy.losses.abstract.loss import LossNumpy
from pydtnn.libs import numpy as np
from pydtnn.losses.cross_entropy import CrossEntropy

__all__ = ("CrossEntropyNumpy")

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class CrossEntropyNumpy(CrossEntropy[np.ndarray], LossNumpy):
    """NumPy implementation of the Cross Entropy loss function."""

    def _model_init(self) -> None:
        """Initialize memory requirements and shapes for the loss computation."""
        super()._model_init()

        self._argmax_shape = (self.model.batch_size,)
        self._y_pred_op_shape = (self.model.batch_size,)
        self._y_pred_shape = self.shape
        self._max_x_shape = (self.model.batch_size, 1)
        self._sum_y_shape = (self.model.batch_size, 1)

        self.tmp_memory_used += int(math.prod(self._argmax_shape)) * np.int32().itemsize

        self.tmp_memory_used += (
            int(
                math.prod(self._y_pred_op_shape)
                + math.prod(self._y_pred_shape)
                + math.prod(self._max_x_shape)
                + math.prod(self._sum_y_shape)
            )
            * self.model.dtype.itemsize
        )

        self.memory_used += self.tmp_memory_used

    def _post_init(self) -> None:
        """Allocate memory buffers for loss computation."""
        super()._post_init()

        with self.model.memory:
            self._argmax = self.model.memory.ndarray(self._argmax_shape, dtype=np.dtype(np.int32))
            self._y_pred_op = self.model.memory.ndarray(self._y_pred_op_shape, dtype=self.model.dtype)
            self._y_pred = self.model.memory.ndarray(self._y_pred_shape, dtype=self.model.dtype)
            self._max_x = self.model.memory.ndarray(self._max_x_shape, dtype=self.model.dtype)
            self._sum_y = self.model.memory.ndarray(self._sum_y_shape, dtype=self.model.dtype)

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Compute the cross entropy loss and gradients.

        Args:
            y_pred: Predicted probabilities.
            y_targ: Target labels in one-hot encoded format.

        Returns:
            A tuple containing the scalar loss value and the gradient array.
        """
        b = self.model.real_batch_size
        _argmax = self._argmax[:b]
        _y_pred_op = self._y_pred_op[:b]
        _y_pred = self._y_pred[:b]
        dx = self.dx[:b]
        max_x = self._max_x[:b]
        sum_y = self._sum_y[:b]
        dx.fill(0)

        # Target class
        np.argmax(y_targ, axis=1, out=_argmax)
        sample_weights = self.weights[_argmax]
        sum_weights_argmax = np.sum(sample_weights)

        # Stable Softmax
        np.max(y_pred, axis=1, keepdims=True, out=max_x)
        np.subtract(y_pred, max_x, out=_y_pred, dtype=self.model.dtype)
        np.exp(_y_pred, out=_y_pred, dtype=self.model.dtype)
        np.sum(_y_pred, axis=1, keepdims=True, out=sum_y)

        # dx = softmax(logits) - target
        np.divide(_y_pred, sum_y, out=dx, dtype=self.model.dtype)
        np.subtract(dx, y_targ, out=dx, dtype=self.model.dtype)
        np.multiply(dx, sample_weights[:, None], out=dx, dtype=self.model.dtype)
        np.divide(dx, sum_weights_argmax, out=dx, dtype=self.model.dtype)

        # Loss
        # log(target_logit - max) -> log_softmax[target] -> weighted loss
        _y_pred_op[:, None] = np.take_along_axis(y_pred, _argmax[:, None], axis=1)
        np.subtract(_y_pred_op, max_x[:, 0], out=_y_pred_op, dtype=self.model.dtype)
        np.log(sum_y, out=sum_y)
        np.subtract(_y_pred_op, sum_y[:, 0], out=_y_pred_op, dtype=self.model.dtype)
        np.multiply(_y_pred_op, sample_weights, out=_y_pred_op)
        loss = float(-np.sum(_y_pred_op) / sum_weights_argmax)

        return loss, np.asarray(dx, dtype=self.model.dtype, order="C")
