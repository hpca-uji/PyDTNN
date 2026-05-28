"""CuPy implementation of the 2D Adaptive Average Pooling layer."""

import logging
from typing import TYPE_CHECKING

from pydtnn.backends.cupy.layers.abstract.layer import LayerCupy
from pydtnn.backends.cupy.layers.abstract.pool_2d_layer import AbstractPool2DLayerCupy
from pydtnn.backends.numpy.layers.adaptive_average_pool_2d import AdaptiveAveragePool2DNumpy
from pydtnn.libs import numpy as np
from pydtnn.utils.constants import DTYPE2CTYPE, ArrayShape

__all__ = ("AdaptiveAveragePool2DCupy",)


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import numpy as np


class AdaptiveAveragePool2DCupy(AdaptiveAveragePool2DNumpy, AbstractPool2DLayerCupy, LayerCupy):
    """CuPy-accelerated 2D adaptive average pooling layer."""

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray | None = None) -> None:
        """Initialize the layer model and compile CUDA kernels."""
        super()._model_init(prev_shape, x)
        self.defines_replaces = {
            '"TYPE"': DTYPE2CTYPE[self.model.dtype],
            "TENSOR_FORMAT": str(self.model.tensor_format),
        }
        # TODO / NOTE: See if it makes sense to generate both (NCHW, NHWC) kernels.
        self.fwd_kernel = self._fwd_kernel()
        self.bwd_kernel = self._bwd_kernel()

    def fwd(self, x: np.ndarray, y: np.ndarray) -> None:
        """Perform the forward pass using a CUDA kernel."""
        N = x.shape[0] * self.ci * self.ho * self.wo  # y.size
        self.fwd_kernel(
            self.model.cuda_grid,
            self.model.cuda_block,
            (x, y, x.shape[0], self.ci, self.hi, self.wi, self.ho, self.wo, N),
        )

    def bwd(self, dx: np.ndarray, dy: np.ndarray) -> None:
        """Perform the backward pass using a CUDA kernel."""
        N = dx.shape[0] * self.ci * self.hi * self.wi  # dx.size
        self.bwd_kernel(
            self.model.cuda_grid,
            self.model.cuda_block,
            (dx, dy, dx.shape[0], self.ci, self.hi, self.wi, self.ho, self.wo, N),
        )

    def _fwd_nhwc(self, x: np.ndarray, y: np.ndarray) -> None:
        """Forward pass for NHWC tensor format."""
        return self.fwd(x, y)

    def _fwd_nchw(self, x: np.ndarray, y: np.ndarray) -> None:
        """Forward pass for NCHW tensor format."""
        return self.fwd(x, y)

    def _bwd_nhwc(self, dx: np.ndarray, dy: np.ndarray) -> None:
        """Backward pass for NHWC tensor format."""
        return self.bwd(dx, dy)

    def _bwd_nchw(self, dx: np.ndarray, dy: np.ndarray) -> None:
        """Backward pass for NCHW tensor format."""
        return self.bwd(dx, dy)
