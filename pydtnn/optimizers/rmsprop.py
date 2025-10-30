from typing import TYPE_CHECKING

import numpy as np

from pydtnn.optimizers.optimizer import Optimizer
from pydtnn.utils.types import Array

if TYPE_CHECKING:
    from pydtnn.model import Model


class RMSProp[T: Array](Optimizer[T]):
    """
    RMSProp optimizer
    """

    def __init__(self, learning_rate: float = 1e-2, rho: float = 0.9, epsilon: float = 1e-7,
                 decay: float = 0.0, dtype: np.dtype = np.dtype(np.float32)):
        super().__init__(learning_rate=learning_rate, dtype=dtype)
        self.rho = rho
        self.epsilon = epsilon
        self.decay = decay

    @classmethod
    def from_model(cls, model: "Model") -> "RMSProp":
        return RMSProp(learning_rate=model.learning_rate,
                       rho=model.rho,
                       epsilon=model.epsilon,
                       decay=model.decay,
                       dtype=model.dtype)
