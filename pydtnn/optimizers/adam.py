from abc import ABC

from .optimizer import Optimizer
import numpy as np
class Adam(Optimizer, ABC):
    """
    Adam optimizer
    """

    def __init__(self, learning_rate:float=1e-2, beta1:float=0.99, beta2:float=0.999,
                 epsilon:float=1e-7, decay:float=0.0, dtype:np.dtype=np.float32):
        super().__init__(learning_rate=learning_rate, dtype=dtype)
        self.beta1:float = beta1
        self.beta2:float = beta2
        self.epsilon:float = epsilon
        self.decay:float = decay
