from typing import TYPE_CHECKING

import numpy as np

from pydtnn.optimizers.optimizer import Optimizer
from pydtnn.utils.constants import Array

if TYPE_CHECKING:
    from pydtnn.model import Model


class SGD[T: Array](Optimizer[T]):
    """
    SGD Optimizer
    """

    def __init__(self, learning_rate: float = 1e-2, momentum: float = 0.9,
                 nesterov: bool = False, decay: float = 0.0, dtype: np.dtype = np.dtype(np.float32)):
        super().__init__(learning_rate=learning_rate, dtype=dtype)
        self.momentum: float = momentum
        self.nesterov: bool = nesterov
        self.decay: float = decay

    @classmethod
    def from_model(cls, model: "Model") -> "SGD":
        return SGD(learning_rate=model.learning_rate,
                   momentum=model.optimizer_momentum,
                   nesterov=model.optimizer_nesterov,
                   decay=model.optimizer_decay,
                   dtype=model.dtype)
