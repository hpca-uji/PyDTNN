"""
Layer normalization implementation for PyDTNN.
"""
import logging

import numpy as np

from pydtnn.layers.layer import Layer
from pydtnn.utils.constants import Array

__all__ = ("LayerNormalization",)

logger = logging.getLogger(__name__)


# https://melfm.github.io/posts/2018-08-Understanding-Normalization/


class LayerNormalization[T: Array](Layer[T]):
    """
    Applies Layer Normalization over a mini-batch of inputs.
    """
    def __init__(self, axis=(-2, -1), beta: float = 0.0, gamma: float = 1.0, epsilon: float = 1e-5, sync_stats: bool = False):
        """
        Initializes the LayerNormalization layer.

        Args:
            axis: The axes along which to compute the mean and variance.
            beta: Initial value for the learnable shift parameter.
            gamma: Initial value for the learnable scale parameter.
            epsilon: Small value added to variance for numerical stability.
            sync_stats: Whether to synchronize statistics across devices.
        """
        super().__init__()
        if type(axis) is not tuple:
            self.axis = (axis,)
        else:
            self.axis = axis
        self.gamma_init_val = gamma
        self.beta_init_val = beta
        self.epsilon = epsilon
        self.grad_vars = {"beta": "dbeta", "gamma": "dgamma"}
        self.sync_stats = sync_stats
        # The next attributes will be initialized later
        self.gamma: np.ndarray = None  # type: ignore
        self.beta: np.ndarray = None  # type: ignore
        self.std: np.ndarray = None  # type: ignore
        self.xn: np.ndarray = None  # type: ignore
        self.dgamma: np.ndarray = None  # type: ignore
        self.dbeta: np.ndarray = None  # type: ignore

    def _model_init(self, prev_shape, x):
        """
        Initializes layer parameters based on the input shape.

        Args:
            prev_shape: Shape of the input tensor.
            x: Input tensor.
        """
        super()._model_init(prev_shape, x)
        self.shape = shape_ = prev_shape
        self.gamma = np.full(shape_, self.gamma_init_val, self.model.dtype)
        self.beta = np.full(shape_, self.beta_init_val, self.model.dtype)
        self.nparams = self.gamma.size + self.beta.size