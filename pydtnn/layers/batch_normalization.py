import numpy as np

from pydtnn.layers.layer import Layer

from typing import Callable

from pydtnn.utils.initializers import zeros

from pydtnn.utils.types import Array
from pydtnn.utils.types import ArrayShape


class BatchNormalization[T: Array](Layer[T]):

    def __init__(self, beta=0.0, gamma=1.0, momentum=0.9, epsilon=1e-5,
                 moving_mean_initializer: Callable = zeros,
                 moving_variance_initializer: Callable = zeros,
                 sync_stats=False):
        super().__init__()
        self.gamma_init_val = gamma
        self.beta_init_val = beta
        self.momentum = momentum
        self.epsilon = epsilon
        self.moving_mean_initializer: Callable[[ArrayShape, np.dtype], np.ndarray] = moving_mean_initializer
        self.moving_variance_initializer: Callable[[ArrayShape, np.dtype], np.ndarray] = moving_variance_initializer
        self.grad_vars = {"beta": "dbeta", "gamma": "dgamma"}
        self.sync_stats = sync_stats
        # The following attributes will be initialized later
        self.co = self.ci = self.hi = self.wi = 0
        self.spatial: bool = None  # type: ignore
        self.gamma: T = None  # type: ignore
        self.beta: T = None  # type: ignore
        self.running_mean: T = None  # type: ignore
        self.running_var: T = None  # type: ignore
        self.std: np.ndarray = None  # type: ignore
        self.xn: np.ndarray = None  # type: ignore
        self.dgamma: T = None  # type: ignore
        self.dbeta: T = None  # type: ignore
        self.inv_std: np.ndarray = None  # type: ignore

    def initialize(self, prev_shape: ArrayShape, x: T | None = None):
        super().initialize(prev_shape, x)
        self.shape = prev_shape
        self.spatial = len(self.shape) > 2
