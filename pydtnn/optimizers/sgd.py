from abc import ABC

import numpy as np

from .optimizer import Optimizer


class SGD(Optimizer, ABC):
    """
    SGD Optimizer
    """

    def __init__(self, learning_rate: float = 1e-2, momentum: float = 0.9,
                 nesterov: bool = False, decay: float = 0.0, dtype: np.dtype = np.float32):
        super().__init__(learning_rate=learning_rate, dtype=dtype)
        self.momentum: float = momentum
        self.nesterov: bool = nesterov
        self.decay: float = decay
