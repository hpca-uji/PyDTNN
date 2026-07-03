"""Stochastic Gradient Descent (SGD) optimizer implementation for PyDTNN."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydtnn.optimizers.abstract.optimizer import Optimizer
from pydtnn.utils.constants import Array

__all__ = ("SGD",)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pydtnn.model import Model


class SGD[T: Array](Optimizer[T]):  # noqa: D101 (generics not detected)
    """Stochastic Gradient Descent optimizer with support for momentum, Nesterov acceleration, and weight decay."""

    def __init__(self, learning_rate: float = 1e-2, momentum: float = 0.9,
                 nesterov: bool = False, decay: float = 0.0) -> None:
        """
        Initialize the SGD optimizer.

        Args:
            learning_rate: The step size used for parameter updates.
            momentum: The momentum factor for accelerating gradients.
            nesterov: Whether to use Nesterov momentum.
            decay: The weight decay coefficient for L2 regularization.
        """
        super().__init__(learning_rate=learning_rate)
        self.momentum: float = momentum
        self.nesterov: bool = nesterov
        self.decay: float = decay

    def _show_props(self) -> dict:
        """
        Return a dictionary of optimizer properties for logging or inspection.

        Returns:
            A dictionary containing optimizer configuration parameters.
        """
        props = super()._show_props()

        props["momentum"] = self.momentum
        props["nesterov"] = self.nesterov
        props["decay"] = self.decay

        return props

    @classmethod
    def from_model(cls, model: Model) -> SGD:
        """
        Create an SGD instance from a model's configuration.

        Args:
            model: The model instance containing optimizer hyperparameters.

        Returns:
            An initialized SGD optimizer instance.
        """
        return SGD(
            learning_rate=model.learning_rate,
            momentum=model.optimizer_momentum,
            nesterov=model.optimizer_nesterov,
            decay=model.optimizer_decay,
        )
