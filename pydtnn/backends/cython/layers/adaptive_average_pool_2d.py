"""
Cython implementation of the 2D adaptive average pooling layer.
"""

import logging
from typing import TYPE_CHECKING

from pydtnn.backends.cython.layers.abstract.pool_2d_layer import AbstractPool2DLayerCython
from pydtnn.backends.cython.utils.adaptive_avg_pooling_nchw_cython import (
    adaptive_avg_pooling_bwd_nchw_cython, adaptive_avg_pooling_fwd_nchw_cython)
from pydtnn.backends.cython.utils.adaptive_avg_pooling_nhwc_cython import (
    adaptive_avg_pooling_bwd_nhwc_cython, adaptive_avg_pooling_fwd_nhwc_cython)
from pydtnn.backends.numpy.layers.adaptive_average_pool_2d import AdaptiveAveragePool2DNumpy
from pydtnn.libs import numpy as np

__all__ = ("AdaptiveAveragePool2DCython",)


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import numpy as np


class AdaptiveAveragePool2DCython(AdaptiveAveragePool2DNumpy, AbstractPool2DLayerCython):
    """
    Cython-accelerated 2D adaptive average pooling layer.
    """

    def _fwd_nhwc(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Performs forward pass for NHWC layout using Cython.
        """
        adaptive_avg_pooling_fwd_nhwc_cython(x, y)  # type: ignore

    def _fwd_nchw(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Performs forward pass for NCHW layout using Cython.
        """
        adaptive_avg_pooling_fwd_nchw_cython(x, y)  # type: ignore

    def _bwd_nhwc(self, dx: np.ndarray, dy: np.ndarray) -> None:
        """
        Performs backward pass for NHWC layout using Cython.
        """
        adaptive_avg_pooling_bwd_nhwc_cython(dx, dy)  # type: ignore

    def _bwd_nchw(self, dx: np.ndarray, dy: np.ndarray) -> None:
        """
        Performs backward pass for NCHW layout using Cython.
        """
        adaptive_avg_pooling_bwd_nchw_cython(dx, dy)  # type: ignore
