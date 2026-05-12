"""CuPy backend implementation for 2D depthwise convolution layers."""
import logging
from typing import TYPE_CHECKING

from pydtnn.backends.cupy.layers.abstract.conv_2d import AbstractConv2DCupy
from pydtnn.backends.cupy.layers.layer import LayerCupy
from pydtnn.backends.numpy.layers.conv_2d_depthwise import Conv2DDepthwiseNumpy
from pydtnn.libs import numpy as np
from pydtnn.utils.constants import DTYPE2CTYPE, ArrayShape

__all__ = ("Conv2DDepthwiseCython",)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import numpy as np


class Conv2DDepthwiseCython(Conv2DDepthwiseNumpy, AbstractConv2DCupy, LayerCupy):
    """CuPy-accelerated 2D depthwise convolution layer."""
    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray | None = None) -> None:
        """Initialize model parameters and compile CUDA kernels."""
        super()._model_init(prev_shape, x)
        self.defines_replaces = {'"TYPE"': DTYPE2CTYPE[self.model.dtype], "TENSOR_FORMAT": str(self.model.tensor_format)}
        self.fwd = self._fwd_kernel()
        self.bwd = self._bwd_kernel()

    def _conv_fwd_nhwc(self, x: np.ndarray, y: np.ndarray) -> None:
        """Execute forward pass for NHWC format."""
        self.fwd(
            self.model.cuda_grid,
            self.model.cuda_block,
            (x, self.weights, y, x.shape[0], self.ci, self.hi, self.wi, self.ho, self.wo, self.kh, self.kw, self.hpadding, self.wpadding, self.hstride, self.wstride, self.hdilation, self.wdilation),
        )

    def _conv_fwd_nchw(self, x: np.ndarray, y: np.ndarray) -> None:
        """Execute forward pass for NCHW format."""
        self.fwd(
            self.model.cuda_grid,
            self.model.cuda_block,
            (x, self.weights, y, x.shape[0], self.ci, self.hi, self.wi, self.ho, self.wo, self.kh, self.kw, self.hpadding, self.wpadding, self.hstride, self.wstride, self.hdilation, self.wdilation),
        )

    def _conv_bwd_nhwc(self, dx: np.ndarray, dy: np.ndarray) -> None:
        """Execute backward pass for NHWC format."""
        self.bwd(
            self.model.cuda_grid,
            self.model.cuda_block,
            (
                dx,
                dy,
                self.x,
                self.weights,
                self.dw,
                dy.shape[0],
                self.ci,
                self.hi,
                self.wi,
                self.ho,
                self.wo,
                self.kh,
                self.kw,
                self.hpadding,
                self.wpadding,
                self.hstride,
                self.wstride,
                self.hdilation,
                self.wdilation,
            ),
        )

    def _conv_bwd_nchw(self, dx: np.ndarray, dy: np.ndarray) -> None:
        """Execute backward pass for NCHW format."""
        self.bwd(
            self.model.cuda_grid,
            self.model.cuda_block,
            (
                dx,
                dy,
                self.x,
                self.weights,
                self.dw,
                dy.shape[0],
                self.ci,
                self.hi,
                self.wi,
                self.ho,
                self.wo,
                self.kh,
                self.kw,
                self.hpadding,
                self.wpadding,
                self.hstride,
                self.wstride,
                self.hdilation,
                self.wdilation,
            ),
        )