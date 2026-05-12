"""
RMSProp optimizer implementation for the PyDTNN framework.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydtnn.optimizers.optimizer import Optimizer
from pydtnn.utils.constants import Array

__all__ = ("RMSProp",)

"""
RMSProp optimizer implementation for the PyDTNN framework.
"""

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pydtnn.model import Model


class RMSProp[T: Array](Optimizer[T]):
    """
    RMSProp optimizer that maintains a moving average of squared gradients.
    """

    def __init__(self, learning_rate: float = 1e-2, rho: float = 0.9, epsilon: float = 1e-7, decay: float = 0.0):
        """
        Initialize the RMSProp optimizer.

        Args:
            learning_rate: Step size for parameter updates.
            rho: Discounting factor for the history of squared gradients.
            epsilon: Small constant for numerical stability.
            decay: Learning rate decay factor.
        """
        super().__init__(learning_rate=learning_rate)
        self.rho = rho
        self.epsilon = epsilon
        self.decay = decay

    def _show_props(self) -> dict:
        """
        Return a dictionary of optimizer properties.

        Returns:
            A dictionary containing optimizer configuration parameters.
        """
        props = super()._show_props()

        props["rho"] = self.rho
        props["epsilon"] = self.epsilon
        props["decay"] = self.decay

        return props

    @classmethod
    def from_model(cls, model: Model) -> RMSProp:
        """
        Create an RMSProp instance from a model configuration.

        Args:
            model: The model instance containing optimizer settings.

        Returns:
            An initialized RMSProp optimizer.
        """
        return RMSProp(learning_rate=model.learning_rate, rho=model.optimizer_rho, epsilon=model.optimizer_epsilon, decay=model.optimizer_decay)