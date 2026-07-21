"""OkTopk optimizer implementation for distributed training."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydtnn.abstract.layerable import Layerable
from pydtnn.optimizers.abstract.optimizer import Optimizer
from pydtnn.utils.constants import Array

__all__ = ("OkTopk",)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pydtnn.model import Model


# TODO: Remove all row,col, use new flat structure, allows balanced partitioning


class OkTopk[T: Array](Optimizer[T]):  # noqa: D101 (generics not detected)
    """OkTopk Optimizer"""

    def __init__(
        self,
        learning_rate: float = 1e-2,
        momentum: float = 0.9,
        decay: float = 0.0,
        tau: int = 64,
        tau_prime: int = 32,
        density: float = 0.01,
        min_k_layer: int = 10,
    ) -> None:
        """
        Initialize the Ok-Topk optimizer.

        Args:
            learning_rate: Learning rate for the optimizer.
            momentum: Momentum factor.
            tau: Threshold parameter for local updates.
            tau_prime: Threshold parameter for global updates.
            density: Sparsity density for gradient compression.
            min_k_layer: Minimum number of elements per layer to apply top-k.
        """

        super().__init__(learning_rate=learning_rate)
        self.momentum = momentum
        self.decay = decay
        self.residuals = {}
        self.tau = tau
        self.tau_prime = tau_prime
        self.density = density
        self.min_k_layer = min_k_layer
        self.iterations = {}
        self.all_local_th = {}
        self.all_global_th = {}
        self.all_residuals = {}
        self.all_boundaries = {}

    def _show_props(self) -> dict:
        """
        Return a dictionary of optimizer properties.

        Returns:
            Dictionary containing optimizer configuration parameters.
        """
        props = super()._show_props()

        props["momentum"] = self.momentum
        props["decay"] = self.decay
        props["tau"] = self.tau
        props["tau_prime"] = self.tau_prime
        props["density"] = self.density
        props["min-k-layer"] = self.min_k_layer

        return props

    def _model_init(self, layers: list[Layerable]) -> None:
        """
        Initialize the optimizer with model layers.

        Args:
            list_layers: List of layers in the model.
        """
        super()._model_init(layers)

        if self.model.model_sync_freq >= 0:
            logger.warning("Optimizer does model sync but global model sync is also enabled!")

        if not self.model.shared_data:
            raise NotImplementedError(
                "OkTopK optimizer does not support Federated Learing (unbalanced datasets)!"
            )

    @classmethod
    def from_model(cls, model: Model) -> OkTopk:
        """
        Create an OkTopk instance from a model configuration.

        Args:
            model: The model instance to extract parameters from.

        Returns:
            An initialized OkTopk optimizer.
        """
        return OkTopk(
            learning_rate=model.learning_rate,
            momentum=model.optimizer_momentum,
            decay=model.optimizer_decay,
            tau=model.optimizer_tau,
            tau_prime=model.optimizer_tau_prime,
            density=model.optimizer_density,
            min_k_layer=model.oktopk_min_k,
        )
