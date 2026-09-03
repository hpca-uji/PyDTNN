"""Abstract base class for 2D convolutional layers."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from pydtnn.layers.abstract.layer import Layer
from pydtnn.utils.constants import Array, ArrayShape, Parameters
from pydtnn.utils.initializers import InitializerFunc, glorot_uniform, zeros

__all__ = ("AbstractConv2D",)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pydtnn.activations.abstract.activation import Activation


class AbstractConv2D[T: Array](Layer[T]):  # noqa: D101 (generics not detected)
    """Base class for 2D convolutional layers providing common configuration and initialization logic."""

    def __init__(
        self,
        nfilters: int = 1,
        filter_shape: tuple[int, int] | int = (3, 3),
        padding: tuple[int, int] | int = 0,
        stride: tuple[int, int] | int = 1,
        dilation: tuple[int, int] | int = 1,
        activation: type[Activation[T]] | None = None,
        use_bias: bool = True,
        weights_initializer: InitializerFunc = glorot_uniform,
        biases_initializer: InitializerFunc = zeros,
    ) -> None:
        """
        Initializes the 2D convolutional layer parameters.

        Args:
            nfilters: Number of output filters.
            filter_shape: Dimensions of the convolution kernel.
            padding: Padding applied to the input.
            stride: Stride of the convolution.
            dilation: Dilation factor for the kernel.
            activation: Activation function class.
            use_bias: Whether to include a bias term.
            weights_initializer: Initializer function for weights.
            biases_initializer: Initializer function for biases.
        """

        super().__init__()
        self.co = nfilters
        self.filter_shape = (
            (filter_shape, filter_shape) if isinstance(filter_shape, int) else filter_shape
        )
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.hpadding, self.wpadding = (padding, padding) if isinstance(padding, int) else padding
        self.hstride, self.wstride = (stride, stride) if isinstance(stride, int) else stride
        self.hdilation, self.wdilation = (
            (dilation, dilation) if isinstance(dilation, int) else dilation
        )
        self.act = activation
        self.use_bias = use_bias
        self.weights_initializer: InitializerFunc = weights_initializer
        self.biases_initializer: InitializerFunc = biases_initializer
        self.grad_vars = {Parameters.WEIGHTS: Parameters.DW}
        if self.use_bias:
            self.grad_vars[Parameters.BIASES] = Parameters.DB
        self.debug = False
        # The following attributes will be initialized later
        self.ci = self.hi = self.wi = self.kh = self.kw = self.ho = self.wo = 0
        self.weights_shape: ArrayShape = None  # pyright: ignore[reportAttributeAccessIssue]
        self.dw: T = None  # pyright: ignore[reportAttributeAccessIssue]
        self.db: T = None  # pyright: ignore[reportAttributeAccessIssue]
        # @warning: do not do this (affects the gpu version) self.forward = self.backward = None

    def _initializing_special_parameters(self) -> None:
        """Hook for subclasses to define or modify parameters required for initialization."""

    def _model_init(self, prev_shape: ArrayShape, x: T | None) -> None:
        """
        Initializes layer dimensions and output shape based on input shape.

        Args:
            prev_shape: Shape of the input tensor.
            x: Input tensor data.
        """
        super()._model_init(prev_shape, x)
        self.ci, self.hi, self.wi = self.model.decode_shape(prev_shape)
        self.kh, self.kw = self.filter_shape
        self._initializing_special_parameters()

        self.ho = (
            self.hi + 2 * self.hpadding - self.hdilation * (self.kh - 1) - 1
        ) // self.hstride + 1
        self.wo = (
            self.wi + 2 * self.wpadding - self.wdilation * (self.kw - 1) - 1
        ) // self.wstride + 1
        self.shape = self.model.encode_shape((self.co, self.ho, self.wo))

        # NOTE: self.weights_shape must be defined in "self._initializing_special_parameters"
        self.nparams = int(math.prod(self.weights_shape) + (self.co if self.use_bias else 0))

    def _show_props(self) -> dict:
        """
        Returns a dictionary of layer properties for debugging or logging.

        Returns:
            Dictionary containing layer configuration properties.
        """
        props = super()._show_props()

        props["padding"] = repr((self.hpadding, self.wpadding))
        props["stride"] = repr((self.hstride, self.wstride))
        props["dilation"] = repr((self.hdilation, self.wdilation))

        return props
