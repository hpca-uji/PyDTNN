"""Abstract base class for standard 2D convolution layers using NumPy backend."""

import logging
from typing import TYPE_CHECKING

from pydtnn.backends.numpy.layers.abstract.conv_2d import AbstractConv2DNumpy
from pydtnn.libs import numpy as np
from pydtnn.utils.tensor import TensorFormat, format_transpose

__all__ = ("AbstractConv2DStandardNumpy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class AbstractConv2DStandardNumpy(AbstractConv2DNumpy):
    """
    Base class for standard 2D convolution layers.

    Implementing weight shape initialization and format-aware weight export/import.
    """

    def _initializing_special_parameters(self) -> None:
        """Initializes weight shapes based on the configured tensor format."""
        super()._initializing_special_parameters()
        match self.model.tensor_format:
            case TensorFormat.NCHW:
                self.weights_shape = (self.co, self.ci, *self.filter_shape)
            case TensorFormat.NHWC:
                self.weights_shape = (self.ci, *self.filter_shape, self.co)
            case _:
                raise NotImplementedError(f"{self.model.tensor_format} format not implemented.")

    def _export_weights_dw(self, key: str) -> np.ndarray:
        """
        Exports weights to NCHW format, transposing if necessary.

        Args:
            key: The attribute name of the weights to export.

        Returns:
            A NumPy array containing the weights in NCHW format.
        """
        value = getattr(self, key)

        match self.model.tensor_format:
            case TensorFormat.NHWC:
                # NHWC's src: ci, kh, kw, co
                # NCHW's dst: co, ci, kh, kw
                return np.asarray(
                    format_transpose(value, "IHWO", "OIHW"), dtype=np.float64, order="C", copy=True
                )
            case TensorFormat.NCHW:
                return np.asarray(value, dtype=np.float64, order="C", copy=True)
            case tensor_format:
                raise TypeError(f"Unsupported tensor format ({tensor_format})")

    def _import_weights_dw(self, key: str, value: np.ndarray) -> None:
        """
        Imports weights into the layer, transposing from NCHW if necessary.

        Args:
            key: The attribute name of the weights to update.
            value: The weight array to import.
        """
        ary = getattr(self, key)

        match self.model.tensor_format:
            case TensorFormat.NHWC:
                # NCHW's src: co, ci, kh, kw
                # NHWC's dst: ci, kh, kw, co
                ary[:] = format_transpose(value, "OIHW", "IHWO")
                return
            case TensorFormat.NCHW:
                ary[:] = value
                return
            case tensor_format:
                raise TypeError(f"Unsupported tensor format ({tensor_format})")
