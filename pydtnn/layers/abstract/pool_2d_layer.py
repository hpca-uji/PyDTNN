"""
Abstract base class for 2D pooling layers in the PyDTNN framework.
"""
import logging
import math

from pydtnn.layers.layer import Layer, LayerError
from pydtnn.utils.constants import Array

__all__ = ("AbstractPool2DLayer",)

logger = logging.getLogger(__name__)


class AbstractPool2DLayer[T: Array](Layer[T]):
    """
    Base class for 2D pooling operations providing shared configuration and shape inference.
    """
    def __init__(self, pool_shape: tuple[int, int] | int = (2, 2), padding: tuple[int, int] | int = 0, stride: tuple[int, int] | int = 1, dilation: tuple[int, int] | int = 1):
        """
        Initializes the pooling layer parameters.

        Args:
            pool_shape: Height and width of the pooling window.
            padding: Padding applied to the input.
            stride: Stride of the pooling operation.
            dilation: Dilation factor for the pooling window.
        """
        super().__init__()
        self.pool_shape = (pool_shape, pool_shape) if isinstance(pool_shape, int) else pool_shape
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.hpadding, self.wpadding = (padding, padding) if isinstance(padding, int) else padding
        self.hstride, self.wstride = (stride, stride) if isinstance(stride, int) else stride
        self.hdilation, self.wdilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.ci = self.hi = self.wi = self.kh = self.kw = self.ho = self.wo = self.co = self.n = 0

    def _model_init(self, prev_shape, x: T | None):
        """
        Infers output dimensions and initializes layer state based on input shape.

        Args:
            prev_shape: Shape of the preceding layer's output.
            x: Optional input tensor.
        """
        super()._model_init(prev_shape, x)
        self.ci, self.hi, self.wi = self.model.decode_shape(prev_shape)
        if self.pool_shape[0] == 0:
            self.pool_shape = (self.hi, self.pool_shape[1])
        if self.pool_shape[1] == 0:
            self.pool_shape = (self.pool_shape[0], self.wi)
        self.kh, self.kw = self.pool_shape
        self.co = self.ci
        self.ho = (self.hi + 2 * self.hpadding - self.hdilation * (self.kh - 1) - 1) // self.hstride + 1
        self.wo = (self.wi + 2 * self.wpadding - self.wdilation * (self.kw - 1) - 1) // self.wstride + 1
        if not (self.ho > 0 and self.wo > 0):
            raise LayerError(f"Output dimensions must be greater than 0. ho: {self.ho}, wo: {self.wo}.")
        self.shape = self.model.encode_shape((self.co, self.ho, self.wo))
        self.n = math.prod(self.shape)

    def _show_props(self) -> dict:
        """
        Returns a dictionary of layer properties for inspection.

        Returns:
            Dictionary containing pooling configuration.
        """
        props = super()._show_props()

        props["pool"] = self.pool_shape
        props["padding"] = (self.hpadding, self.wpadding)
        props["stride"] = (self.hstride, self.wstride)
        props["dilation"] = (self.hdilation, self.wdilation)

        return props