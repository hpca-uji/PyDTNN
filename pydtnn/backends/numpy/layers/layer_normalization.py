"""
NumPy backend implementation of the Layer Normalization layer.
"""

import logging
from typing import TYPE_CHECKING

from pydtnn.backends.numpy.layers.layer import LayerNumpy
from pydtnn.layers.layer_normalization import LayerNormalization
from pydtnn.libs import numpy as np

__all__ = ("LayerNormalizationNumpy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


class LayerNormalizationNumpy(LayerNormalization[np.ndarray], LayerNumpy):
    """
    NumPy implementation of Layer Normalization.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass of layer normalization.

        Args:
            x: Input tensor of shape (batch_size, features).

        Returns:
            Normalized output tensor.
        """
        # TODO: Check how to initialize this parameters outside (in the initalization layer)
        mu = np.mean(x, axis=self.axis, keepdims=True)
        xc = x - mu
        var = np.mean(xc**2, axis=self.axis, keepdims=True)

        # self.std = np.sqrt(var + self.epsilon)
        self.std: np.ndarray = np.add(var, self.epsilon)
        np.sqrt(self.std, out=self.std, dtype=self.model.dtype)

        # self.xn = xc / self.std
        self.xn: np.ndarray = np.divide(xc, self.std, dtype=self.model.dtype)

        # y = self.gamma * self.xn + self.beta
        y: np.ndarray = np.multiply(self.gamma, self.xn, dtype=self.model.dtype)
        np.add(y, self.beta, out=y)

        return y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """
        Performs the backward pass of layer normalization.

        Args:
            dy: Gradient of the loss with respect to the output.

        Returns:
            Gradient of the loss with respect to the input.
        """
        self.dgamma = np.sum(dy * self.xn, axis=0)
        self.dbeta = np.sum(dy, axis=0)

        # if self.need_dx:
        # dy = dy * self.gamma
        np.multiply(dy, self.gamma, out=dy)

        # dx = dy - self.xn * np.mean(dy * self.xn, self.axis, keepdims=True)
        dx = np.mean(dy * self.xn, self.axis, keepdims=True)
        np.multiply(self.xn, dx, out=dx, dtype=self.model.dtype)
        np.subtract(dy, dx, out=dx)

        # dx -= np.mean(dy, self.axis, keepdims=True)
        _mean = np.mean(dy, self.axis, keepdims=True)
        np.subtract(dx, _mean, out=dx)

        # dx /= self.std
        np.divide(dx, self.std, out=dx, dtype=self.model.dtype)
        return dx
