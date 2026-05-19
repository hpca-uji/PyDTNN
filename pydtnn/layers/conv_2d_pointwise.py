"""
Pointwise 2D convolution layer implementation for the PyDTNN framework.
"""

import logging
from typing import Callable, Optional

from pydtnn.activations.abstract.activation import Activation
from pydtnn.layers.abstract.conv_2d import AbstractConv2D
from pydtnn.utils.constants import Array
from pydtnn.utils.initializers import InitializerFunc, glorot_uniform, zeros

__all__ = ("Conv2DPointwise",)

logger = logging.getLogger(__name__)


class Conv2DPointwise[T: Array](AbstractConv2D[T]):
    """
    A 2D pointwise convolution layer that performs a 1x1 convolution across input channels.
    """

    def __init__(
        self,
        nfilters: int = 1,
        filter_shape: tuple[int, int] | int = (1, 1),
        padding: tuple[int, int] | int = 0,
        stride: tuple[int, int] | int = 1,
        dilation: tuple[int, int] | int = 1,
        activation: Optional[type["Activation"]] = None,
        use_bias=True,
        weights_initializer: InitializerFunc = glorot_uniform,
        biases_initializer: InitializerFunc = zeros,
    ):
        super().__init__(nfilters, filter_shape, padding, stride, dilation, activation, use_bias, weights_initializer, biases_initializer)
