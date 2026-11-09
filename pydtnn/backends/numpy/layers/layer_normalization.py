"""NumPy backend implementation of the Layer Normalization layer."""

import logging
from typing import TYPE_CHECKING, Any

from numpy import ndarray

from pydtnn.backends.numpy.layers.abstract.layer import LayerNumpy
from pydtnn.layers.layer_normalization import LayerNormalization
from pydtnn.libs import numpy as np
from pydtnn.utils.constants import ArrayShape

__all__ = ("LayerNormalizationNumpy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class LayerNormalizationNumpy(LayerNormalization[np.ndarray], LayerNumpy):
    """NumPy implementation of Layer Normalization."""

    def __init__(self, axis: tuple | int = (-2, -1), beta: float = 0,
                 gamma: float = 1, epsilon: float = 0.00001,
                 sync_stats: bool = False) -> None:
        """Layer Normalization init"""
        super().__init__(axis, beta, gamma, epsilon, sync_stats)

    def _model_init(self, prev_shape: ArrayShape, x: ndarray[tuple[Any, ...], np.dtype[Any]]) -> None:
        super()._model_init(prev_shape, x)
        shape_mu = list(self.shape)
        # try:
        for axis in self.axis:
            shape_mu[axis] = 1
        # except Exception as e:
        #     print(e)
        #     breakpoint()

        # TODO: CHECK THIS!!
        # print("MODEL INIT LAYER NORM.")
        # TODO: Check how to initialize this parameters outside (in the initalization step)
        self._mu_mean: np.ndarray = np.zeros((self.model.batch_size, *shape_mu), dtype=self.model.dtype, order="C")
        self._xc: np.ndarray = np.zeros((self.model.batch_size, *self.shape), dtype=self.model.dtype, order="C")
        self._var: np.ndarray = np.zeros((self.model.batch_size, *self.shape), dtype=self.model.dtype, order="C")
        self._xn: np.ndarray = np.zeros((self.model.batch_size, *self.shape), dtype=self.model.dtype, order="C")
        self.y: np.ndarray = np.zeros((self.model.batch_size, *self.shape), dtype=self.model.dtype, order="C")
        self._std: np.ndarray = np.zeros((self.model.batch_size, *self.shape), dtype=self.model.dtype, order="C")
        self._dgamma: np.ndarray = np.zeros((self.model.batch_size, *self.shape), dtype=self.model.dtype, order="C")
        self._dbeta: np.ndarray = np.zeros((self.model.batch_size, *self.shape), dtype=self.model.dtype, order="C")
        self._temp: np.ndarray = np.zeros((self.model.batch_size, *self.shape), dtype=self.model.dtype, order="C")
        self._dx: np.ndarray = np.zeros((self.model.batch_size, *self.shape), dtype=self.model.dtype, order="C")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass of layer normalization.

        Args:
            x: Input tensor of shape (batch_size, features).

        Returns:
            Normalized output tensor.
        """
        mu = self._mu_mean[x.shape[0]:]
        xc = self._xc[x.shape[0]:]
        var = self._var[x.shape[0]:]
        y = self.y[x.shape[0]:]
        self.xn = self._xn[x.shape[0]:]
        self.std = self._std[x.shape[0]:]

        # TODO: CHECK THIS!
        # breakpoint()
        # mu = np.mean(x, axis=self.axis, keepdims=True)
        # xc = x - mu
        np.mean(x, axis=self.axis, keepdims=True, out=mu)
        np.subtract(x, mu, out=xc)

        # var = np.mean(xc**2, axis=self.axis, keepdims=True)
        np.power(xc, 2, dtype=self.model.dtype, out=var)
        np.mean(var, axis=self.axis, keepdims=True, out=var)

        # self.std = np.sqrt(var + self.epsilon)
        np.add(var, self.epsilon, out=self.std)
        np.sqrt(self.std, out=self.std, dtype=self.model.dtype)

        # self.xn = xc / self.std
        np.divide(xc, self.std, dtype=self.model.dtype, out=self.xn)

        # y = self.gamma * self.xn + self.beta
        np.multiply(self.gamma, self.xn, dtype=self.model.dtype, out=y)
        np.add(y, self.beta, out=y)

        return np.asarray(y, dtype=self.model.dtype, order="C")

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """
        Performs the backward pass of layer normalization.

        Args:
            dy: Gradient of the loss with respect to the output.

        Returns:
            Gradient of the loss with respect to the input.
        """
        self.dgamma = self._dgamma[dy.shape[0]:]
        self.dbeta = self._dbeta[dy.shape[0]:]
        dx = self._dx[dy.shape[0]:]
        temp = self._temp[dy.shape[0]:]
        _mean = self._mu_mean[dy.shape[0]:]

        # self.dgamma = np.sum(dy * self.xn, axis=0)
        np.multiply(dy, self.xn, out=temp)
        np.sum(temp, axis=0, out=self.dgamma)
        # self.dbeta = np.sum(dy, axis=0)
        np.sum(dy, axis=0, out=self.dbeta)

        # if self.need_dx:
        # dy = dy * self.gamma
        np.multiply(dy, self.gamma, out=dy)

        # dx = dy - self.xn * np.mean(dy * self.xn, self.axis, keepdims=True)
        np.mean(temp, self.axis, keepdims=True, out=_mean)
        np.multiply(self.xn, dx, out=dx, dtype=self.model.dtype)
        np.subtract(dy, dx, out=dx)

        # dx -= np.mean(dy, self.axis, keepdims=True)
        np.mean(dy, self.axis, keepdims=True, out=_mean)
        np.subtract(dx, _mean, out=dx)

        # dx /= self.std
        np.divide(dx, self.std, out=dx, dtype=self.model.dtype)
        return np.asarray(dx, dtype=self.model.dtype, order="C")
