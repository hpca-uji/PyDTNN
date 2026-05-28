"""Categorical hinge metric implementation for the NumPy backend."""

import logging
import math
from typing import TYPE_CHECKING

from pydtnn.backends.numpy.metrics.abstract.metric import MetricNumpy
from pydtnn.libs import numpy as np
from pydtnn.metrics.categorical_hinge import CategoricalHinge

__all__ = ("CategoricalHingeNumpy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


class CategoricalHingeNumpy(CategoricalHinge[np.ndarray], MetricNumpy):
    """NumPy implementation of the categorical hinge metric."""

    def _model_init(self) -> None:
        """Initialize model-specific parameters and memory requirements."""
        super()._model_init()

        self._pos_shape = self.shape
        self._neg_shape = self.shape
        self.pos_maxm_shape = (self.model.batch_size,)
        self.neg_shape = (self.model.batch_size,)
        self.tmp_memory_used += (
            int(
                math.prod(self._pos_shape)
                + math.prod(self._neg_shape)
                + math.prod(self.pos_maxm_shape)
                + math.prod(self.neg_shape)
            )
            * self.model.dtype.itemsize
        )
        self.memory_used += self.tmp_memory_used

    def _post_init(self) -> None:
        """Allocate memory buffers for metric computation."""
        super()._post_init()
        with self.model.memory:
            self._pos = self.model.memory.ndarray(self._pos_shape, dtype=self.model.dtype)
            self._neg = self.model.memory.ndarray(self._neg_shape, dtype=self.model.dtype)
            self.pos_maxm = self.model.memory.ndarray(self.pos_maxm_shape, dtype=self.model.dtype)
            self.neg = self.model.memory.ndarray(self.neg_shape, dtype=self.model.dtype)

    def compute(self, y_pred: np.ndarray, y_targ: np.ndarray) -> float:
        """Compute the categorical hinge loss between predictions and targets.

        Args:
            y_pred: Predicted values.
            y_targ: Ground truth values.

        Returns:
            The computed categorical hinge loss as a float.
        """
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

        maximum: np.ndarray = np.mean(pos_maxm, axis=-1)

        return maximum.item()
