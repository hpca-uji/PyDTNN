from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydtnn.optimizers.optimizer import Optimizer
from pydtnn.utils.constants import Array

__all__ = ("RMSProp",)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pydtnn.model import Model


class RMSProp[T: Array](Optimizer[T]):
    """
    RMSProp optimizer
    """

    def __init__(self, learning_rate: float = 1e-2, rho: float = 0.9, epsilon: float = 1e-7, decay: float = 0.0):
        super().__init__(learning_rate=learning_rate)
        self.rho = rho
        self.epsilon = epsilon
        self.decay = decay

    def _show_props(self) -> dict:
        props = super()._show_props()

        props["rho"] = self.rho
        props["epsilon"] = self.epsilon
        props["decay"] = self.decay

        return props

    @classmethod
    def from_model(cls, model: Model) -> RMSProp:
        return RMSProp(learning_rate=model.learning_rate, rho=model.optimizer_rho, epsilon=model.optimizer_epsilon, decay=model.optimizer_decay)
