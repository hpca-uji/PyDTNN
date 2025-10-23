import numpy as np

from pydtnn.layers.layer import Layer
from pydtnn.utils.tensor import decode_tensor
from typing import Callable

from pydtnn.initializers import zeros

from pydtnn.utils.types import Array
from pydtnn.utils.types import ArrayShape


class BatchNormalization[T: Array](Layer):

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
        # The next attributes will be initialized later
        self.spatial: bool = None
        self.co = self.ci = self.hi = self.wi = 0
        self.gamma: T = None
        self.beta: T = None
        self.running_mean: T = None
        self.running_var: T = None
        self.std: np.ndarray = None
        self.xn: np.ndarray = None
        self.dgamma: T = None
        self.dbeta: T = None
        self.inv_std: np.ndarray = None

    def initialize(self, prev_shape: ArrayShape, x: T | None = None):
        super().initialize(prev_shape, x)
        self.shape = shape_ = prev_shape
        self.spatial = len(self.shape) > 2
        if self.spatial:
            self.hi, self.wi, self.ci = decode_tensor(self.shape, self.model.tensor_format)
            shape_ = (self.ci,)
        else:
            self.ci = self.shape[0]
        self.gamma = np.full(shape_, self.gamma_init_val, dtype=self.model.dtype, order="C")
        self.beta = np.full(shape_, self.beta_init_val, dtype=self.model.dtype, order="C")
        self.running_mean = self.moving_mean_initializer(shape_, self.model.dtype)
        self.running_var = self.moving_variance_initializer(shape_, self.model.dtype)
        # self.inv_std = 1.0 / np.sqrt(self.running_var + self.epsilon)
        self.inv_std = np.sqrt(self.running_var + self.epsilon, dtype=self.model.dtype, order="C")
        np.reciprocal(self.inv_std, out=self.inv_std, dtype=self.model.dtype)
        self.nparams = self.gamma.size + self.beta.size + self.running_mean.size + self.running_var.size
