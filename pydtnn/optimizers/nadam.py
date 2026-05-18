"""
Nadam optimizer implementation for PyDTNN.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydtnn.optimizers.abstract.optimizer import Optimizer
from pydtnn.utils.constants import Array

__all__ = ("Nadam",)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pydtnn.model import Model


class Nadam[T: Array](Optimizer[T]):
    """
    Nadam (Nesterov-accelerated Adaptive Moment Estimation) optimizer.

    Nadam is an extension of the Adam optimization algorithm that incorporates
    Nesterov momentum. It aims to improve convergence speed and stability by
    using future gradient information.
    """

    def __init__(self, learning_rate: float = 1e-2, beta1: float = 0.99, beta2: float = 0.999, epsilon: float = 1e-7, decay: float = 0.0):
        """
        Initialize the Nadam optimizer.

        Args:
            learning_rate: The step size used for parameter updates. Defaults to 1e-2.
            beta1: The exponential decay rate for the first moment estimates. Defaults to 0.99.
            beta2: The exponential decay rate for the second moment estimates. Defaults to 0.999.
            epsilon: A small constant for numerical stability to prevent division by zero. Defaults to 1e-7.
            decay: The learning rate decay factor. Defaults to 0.0.
        """
        super().__init__(learning_rate=learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.decay = decay

    def _show_props(self) -> dict:
        """
        Return a dictionary of optimizer properties.

        This method is intended for internal use or debugging to inspect the
        optimizer's current configuration.

        Returns:
            A dictionary containing optimizer configuration parameters, including
            learning rate, betas, epsilon, and decay.
        """
        props = super()._show_props()

        props["beta1"] = self.beta1
        props["beta2"] = self.beta2
        props["epsilon"] = self.epsilon
        props["decay"] = self.decay

        return props

    @classmethod
    def from_model(cls, model: Model) -> Nadam:
        """
        Create a Nadam optimizer instance from a model configuration.

        This class method allows initializing the Nadam optimizer using hyperparameters
        defined within a PyDTNN model object.

        Args:
            model: The model instance containing optimizer hyperparameters such as
                   learning_rate, optimizer_beta1, optimizer_beta2,
                   optimizer_epsilon, and optimizer_decay.

        Returns:
            An initialized Nadam optimizer instance configured with parameters
            from the provided model.
        """
        return Nadam(learning_rate=model.learning_rate, beta1=model.optimizer_beta1, beta2=model.optimizer_beta2, epsilon=model.optimizer_epsilon, decay=model.optimizer_decay)
