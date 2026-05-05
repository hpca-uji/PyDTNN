from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydtnn.optimizers.optimizer import Optimizer
from pydtnn.utils.constants import Array

__all__ = (
    "SGD",
)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pydtnn.model import Model


class SGD[T: Array](Optimizer[T]):
    """
    SGD Optimizer
    """

    def __init__(self, learning_rate: float = 1e-2, momentum: float = 0.9,
                 nesterov: bool = False, decay: float = 0.0):
        super().__init__(learning_rate=learning_rate)
        self.momentum: float = momentum
        self.nesterov: bool = nesterov
        self.decay: float = decay

    def _show_props(self) -> dict:
        props = super()._show_props()

        props["momentum"] = self.momentum
        props["nesterov"] = self.nesterov
        props["decay"] = self.decay

        return props

    @classmethod
    def from_model(cls, model: Model) -> SGD:
        return SGD(learning_rate=model.learning_rate,
                   momentum=model.optimizer_momentum,
                   nesterov=model.optimizer_nesterov,
                   decay=model.optimizer_decay)
