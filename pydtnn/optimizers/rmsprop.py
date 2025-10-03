from abc import ABC

import numpy as np

from .optimizer import Optimizer


class RMSProp(Optimizer, ABC):
    """
    RMSProp optimizer
    """

    def __init__(self, learning_rate: float = 1e-2, rho: float = 0.9, epsilon: float = 1e-7,
                 decay: float = 0.0, dtype: np.dtype = np.float32):
        super().__init__(learning_rate=learning_rate, dtype=dtype)
        self.rho = rho
        self.epsilon = epsilon
        self.decay = decay
