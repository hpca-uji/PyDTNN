"""Fully connected layer implementation for PyDTNN."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydtnn.layers.layer import Layer
from pydtnn.utils.constants import Array, ArrayShape, Parameters
from pydtnn.utils.initializers import InitializerFunc, glorot_uniform, zeros

__all__ = ("FC",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pydtnn.activations.activation import Activation


class FC[T: Array](Layer[T]):
    """Fully connected (dense) layer."""

    def __init__(
        self, shape: ArrayShape = (1,), activation: type[Activation] | None = None, use_bias=True, weights_initializer: InitializerFunc = glorot_uniform, biases_initializer: InitializerFunc = zeros
    ):
        """Initializes the FC layer.

        Args:
            shape: Output shape of the layer.
            activation: Activation function class to apply.
            use_bias: Whether to include a bias term.
            weights_initializer: Initializer function for weights.
            biases_initializer: Initializer function for biases.
        """
        super().__init__(shape)
        self.act = activation
        self.use_bias = use_bias
        self.weights_initializer = weights_initializer
        self.biases_initializer = biases_initializer
        self.grad_vars = {Parameters.WEIGHTS: Parameters.DW}
        if self.use_bias:
            self.grad_vars[Parameters.BIASES] = Parameters.DB

    def _model_init(self, prev_shape: ArrayShape, x: T | None) -> None:
        """Initializes layer parameters based on input shape.

        Args:
            prev_shape: Shape of the input data.
            x: Optional input tensor.
        """
        super()._model_init(prev_shape, x)
        self.weights_shape = (*prev_shape, *self.shape)
