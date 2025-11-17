from typing import TYPE_CHECKING

import numpy as np

from pydtnn.optimizers.optimizer import Optimizer
from pydtnn.utils.constants import Array

if TYPE_CHECKING:
    from pydtnn.model import Model


class Adam[T: Array](Optimizer[T]):
    """
    Adam optimizer
    """

    def __init__(self, learning_rate: float = 1e-2, beta1: float = 0.99, beta2: float = 0.999,
                 epsilon: float = 1e-7, decay: float = 0.0, dtype: np.dtype = np.dtype(np.float32)):
        super().__init__(learning_rate=learning_rate, dtype=dtype)
        self.beta1: float = beta1
        self.beta2: float = beta2
        self.epsilon: float = epsilon
        self.decay: float = decay

    @classmethod
    def from_model(cls, model: "Model") -> "Adam":
        return Adam(learning_rate=model.learning_rate,
                    beta1=model.optimizer_beta1,
                    beta2=model.optimizer_beta2,
                    epsilon=model.optimizer_epsilon,
                    decay=model.optimizer_decay,
                    dtype=model.dtype)
