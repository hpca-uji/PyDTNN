import numpy as np

from abc import ABC

from pydtnn.layers.layer import Layer
from pydtnn.utils import decode_tensor
from typing import Callable

from pydtnn.initializers import zeros

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydtnn.model import Array


class BatchNormalization(Layer, ABC):

    def __init__(self, beta=0.0, gamma=1.0, momentum=0.9, epsilon=1e-5,
                 moving_mean_initializer: Callable = zeros,
                 moving_variance_initializer: Callable = zeros,
                 sync_stats=False):
        super().__init__()
        self.gamma_init_val = gamma
        self.beta_init_val = beta
        self.momentum = momentum
        self.epsilon = epsilon
        self.moving_mean_initializer: Callable[[tuple[int, ...], np.dtype], np.ndarray] = moving_mean_initializer
        self.moving_variance_initializer: Callable[[tuple[int, ...], np.dtype], np.ndarray] = moving_variance_initializer
        self.grad_vars = {"beta": "dbeta", "gamma": "dgamma"}
        self.sync_stats = sync_stats
        # The next attributes will be initialized later
        self.spatial: bool = None
        self.co = self.ci = self.hi = self.wi = 0
        self.gamma: "Array" = None
        self.beta: "Array" = None
        self.running_mean: "Array" = None
        self.running_var: "Array" = None
        self.std: np.ndarray = None
        self.xn: np.ndarray = None
        self.dgamma: "Array" = None
        self.dbeta: "Array" = None
        self.inv_std: np.ndarray = None

    def initialize(self, prev_shape: tuple[int, ...], x: Array | None = None):
        super().initialize(prev_shape, x)
        self.shape = shape_ = prev_shape
        self.spatial = len(self.shape) > 2
        if self.spatial:
            self.hi, self.wi, self.ci = decode_tensor(self.shape, self.model.tensor_format)
            shape_ = (self.ci,)
        else:
            self.ci = self.shape[0]
        self.gamma = np.full(shape_, self.gamma_init_val, self.model.dtype)
        self.beta = np.full(shape_, self.beta_init_val, self.model.dtype)
        self.running_mean = self.moving_mean_initializer(shape_, self.model.dtype)
        self.running_var = self.moving_variance_initializer(shape_, self.model.dtype)
        # self.inv_std = 1.0 / np.sqrt(self.running_var + self.epsilon)
        self.inv_std = np.sqrt(self.running_var + self.epsilon)
        np.reciprocal(self.inv_std, out=self.inv_std)
        self.nparams = self.gamma.size + self.beta.size + self.running_mean.size + self.running_var.size
