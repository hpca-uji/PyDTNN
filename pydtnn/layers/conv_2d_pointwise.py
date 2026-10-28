"""Pointwise 2D convolution layer implementation for the PyDTNN framework."""

from __future__ import annotations

import logging

from pydtnn.activations.abstract.activation import Activation
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.utils.constants import Array
from pydtnn.utils.initializers import InitializerFunc, glorot_uniform, zeros

__all__ = ("Conv2DPointwise",)

logger = logging.getLogger(__name__)


class Conv2DPointwise[T: Array](Conv2D[T]):  # noqa: D101 (generics not detected)
    """
    A 2D pointwise convolution layer that performs a 1x1 convolution across input channels.
    """

    def __init__(
        self,
        nfilters: int = 1,
        activation: type[Activation[T]] | None = None,
        use_bias: bool = True,
        weights_initializer: InitializerFunc = glorot_uniform,
        biases_initializer: InitializerFunc = zeros,
    ) -> None:
        """
        Initializes the Conv2DPointwise layer.

        Args:
            nfilters: Number of output filters.
            activation: Activation function class to apply.
            use_bias: Whether to include a bias term.
            weights_initializer: Function to initialize weights.
            biases_initializer: Function to initialize biases.
        """
        # NOTE: These are the values to make a Conv2D as a Pointwise.
        filter_shape = 1
        stride = 1
        padding = 0
        dilation = 1
        super().__init__(
            nfilters,
            filter_shape,
            padding,
            stride,
            dilation,
            activation,
            use_bias,
            weights_initializer,
            biases_initializer,
        )
