import numpy as np

from pydtnn.layers.layer import Layer
from pydtnn.utils.constants import Array


# https://melfm.github.io/posts/2018-08-Understanding-Normalization/

class LayerNormalization[T: Array](Layer[T]):
    def __init__(self, axis=(-2, -1), beta: float = 0.0, gamma: float = 1.0,
                 epsilon: float = 1e-5,
                 sync_stats: bool = False):
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
        self.gamma = self.beta = None
        self.std = self.xn = None
        self.dgamma = self.dbeta = None

    def initialize(self, prev_shape, x):
        super().initialize(prev_shape, x)
        self.shape = shape_ = prev_shape
        self.gamma = np.full(shape_, self.gamma_init_val, self.model.dtype)
        self.beta = np.full(shape_, self.beta_init_val, self.model.dtype)
        self.nparams = self.gamma.size + self.beta.size
