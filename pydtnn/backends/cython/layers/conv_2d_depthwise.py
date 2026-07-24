"""Cython implementation of 2D depthwise convolutional layers."""

import logging
from typing import TYPE_CHECKING

from pydtnn.backends.cython.layers.abstract.conv_2d import AbstractConv2DCython
from pydtnn.backends.cython.utils.depthwise_conv_nchw_cython import (
    depthwise_conv_backward_nchw_cython, depthwise_conv_nchw_cython)
from pydtnn.backends.cython.utils.depthwise_conv_nhwc_cython import (
    depthwise_conv_backward_nhwc_cython, depthwise_conv_nhwc_cython)
from pydtnn.backends.numpy.layers.conv_2d_depthwise import Conv2DDepthwiseNumpy
from pydtnn.libs import numpy as np

__all__ = ("Conv2DDepthwiseCython",)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class Conv2DDepthwiseCython(Conv2DDepthwiseNumpy, AbstractConv2DCython):
    """2D depthwise convolutional layer using Cython kernels for acceleration."""

    def _conv_fwd_nhwc(self, x: np.ndarray, y: np.ndarray) -> None:
        """Perform forward depthwise convolution in NHWC format."""
        depthwise_conv_nhwc_cython(
            x,
            self.weights,
            y,
            self.ho,
            self.wo,
            self.hpadding,
            self.wpadding,
            self.hstride,
            self.wstride,
            self.hdilation,
            self.wdilation,
        )

    def _conv_fwd_nchw(self, x: np.ndarray, y: np.ndarray) -> None:
        """Perform forward depthwise convolution in NCHW format."""
        depthwise_conv_nchw_cython(
            x,
            self.weights,
            y,
            self.ho,
            self.wo,
            self.hpadding,
            self.wpadding,
            self.hstride,
            self.wstride,
            self.hdilation,
            self.wdilation,
        )

    def _conv_bwd_nhwc(self, dx: np.ndarray, dy: np.ndarray) -> None:
        """Perform backward depthwise convolution in NHWC format to compute gradients."""
        depthwise_conv_backward_nhwc_cython(
            dy,
            self.x,
            self.weights,
            dx,
            self.dw,
            self.hpadding,
            self.wpadding,
            self.hstride,
            self.wstride,
            self.hdilation,
            self.wdilation,
        )

    def _conv_bwd_nchw(self, dx: np.ndarray, dy: np.ndarray) -> None:
        """Perform backward depthwise convolution in NCHW format to compute gradients."""
        depthwise_conv_backward_nchw_cython(
            dy,
            self.x,
            self.weights,
            dx,
            self.dw,
            self.hpadding,
            self.wpadding,
            self.hstride,
            self.wstride,
            self.hdilation,
            self.wdilation,
        )
