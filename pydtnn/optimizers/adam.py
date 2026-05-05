from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydtnn.optimizers.optimizer import Optimizer
from pydtnn.utils.constants import Array

__all__ = (
    "Adam",
)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pydtnn.model import Model


class Adam[T: Array](Optimizer[T]):
    """
    Adam optimizer
    """

    def __init__(self, learning_rate: float = 1e-2, beta1: float = 0.99, beta2: float = 0.999,
                 epsilon: float = 1e-7, decay: float = 0.0):
        super().__init__(learning_rate=learning_rate)
        self.beta1: float = beta1
        self.beta2: float = beta2
        self.epsilon: float = epsilon
        self.decay: float = decay

    def _show_props(self) -> dict:
        props = super()._show_props()

        props["beta1"] = self.beta1
        props["beta2"] = self.beta2
        props["epsilon"] = self.epsilon
        props["decay"] = self.decay

        return props

    @classmethod
    def from_model(cls, model: Model) -> Adam:
        return Adam(learning_rate=model.learning_rate,
                    beta1=model.optimizer_beta1,
                    beta2=model.optimizer_beta2,
                    epsilon=model.optimizer_epsilon,
                    decay=model.optimizer_decay)
