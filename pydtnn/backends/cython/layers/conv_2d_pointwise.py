"""Cython implementation of 2D pointwise convolutional layers."""

import logging
from typing import TYPE_CHECKING

from pydtnn.backends.cython.layers.abstract.conv_2d import AbstractConv2DCython
from pydtnn.backends.cython.utils.pointwise_conv_nchw_cython import (
    bwd_pointwise_conv_cython_nchw, fwd_pointwise_conv_cython_nchw)
from pydtnn.backends.cython.utils.pointwise_conv_nhwc_cython import (
    bwd_pointwise_conv_cython_nhwc, fwd_pointwise_conv_cython_nhwc)
from pydtnn.backends.numpy.layers.conv_2d_pointwise import Conv2DPointwiseNumpy
from pydtnn.libs import numpy as np

__all__ = ("Conv2DPointwiseCython",)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class Conv2DPointwiseCython(Conv2DPointwiseNumpy, AbstractConv2DCython):
    """2D pointwise convolutional layer using Cython kernels for acceleration."""

    def _conv_fwd_nhwc(self, x: np.ndarray, y: np.ndarray) -> None:
        """Perform forward pointwise convolution in NHWC format."""
        fwd_pointwise_conv_cython_nhwc(
            x,
            self.weights,
            y,
            self.hpadding,
            self.wpadding,
            self.hstride,
            self.wstride,
        )

    def _conv_fwd_nchw(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Perform forward pointwise convolution in NCHW format."""
        fwd_pointwise_conv_cython_nchw(
            x,
            self.weights,
            y,
            self.hpadding,
            self.wpadding,
            self.hstride,
            self.wstride,
        )
        return y

    def _conv_bwd_nhwc(self, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
        """Perform backward pointwise convolution in NHWC format to compute gradients."""
        dx = dx.reshape(self.x.shape, copy=None)
        bwd_pointwise_conv_cython_nhwc(
            dy,
            self.x,
            self.weights,
            dx,
            self.dw,
            self.hpadding,
            self.wpadding,
            self.hstride,
            self.wstride,
        )
        return dx

    def _conv_bwd_nchw(self, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
        """Perform backward pointwise convolution in NCHW format to compute gradients."""
        dx = dx.reshape(self.x.shape, copy=None)
        bwd_pointwise_conv_cython_nchw(
            dy,
            self.x,
            self.weights,
            dx,
            self.dw,
            self.hpadding,
            self.wpadding,
            self.hstride,
            self.wstride,
        )
        return dx
