from pydtnn.utils.constants import ArrayShape, DTYPE2CTYPE
from typing import TYPE_CHECKING
from pydtnn.libs import numpy as np
from pydtnn.backends.numpy.layers.max_pool_2d import MaxPool2DNumpy
from pydtnn.backends.cupy.layers.layer import LayerCupy
from pydtnn.backends.cupy.layers.abstract.pool_2d_layer import AbstractPool2DLayerCupy
import logging
logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import numpy as np


class MaxPool2DCupy(MaxPool2DNumpy, AbstractPool2DLayerCupy, LayerCupy):
    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray | None = None) -> None:
        super()._model_init(prev_shape, x)
        self.fwd_kernel = self._fwd_kernel()
        self.bwd_kernel = self._bwd_kernel()
        #----

    def _fwd_max_pool_nchw(self, x: np.ndarray, y: np.ndarray) -> None:
        self.fwd_kernel(self.model.cuda_grid,
                        self.model.cuda_block,
                        (x, y, self.idx_max,
                         x.shape[0], self.ci, self.hi, self.wi,
                         self.kh, self.kw, self.ho, self.wo,
                         self.hpadding, self.wpadding,
                         self.hstride, self.wstride,
                         self.hdilation, self.wdilation, 
                         self.minval))
    # ----

    def _fwd_max_pool_nhwc(self, x: np.ndarray, y: np.ndarray) -> None:
        self.fwd_kernel(self.model.cuda_grid,
                        self.model.cuda_block,
                        (x, y, self.idx_max,
                         x.shape[0], self.ci, self.hi, self.wi,
                         self.kh, self.kw, self.ho, self.wo,
                         self.hpadding, self.wpadding,
                         self.hstride, self.wstride,
                         self.hdilation, self.wdilation, 
                         self.minval))
    # ----

    def _bwd_max_pool_nchw(self, dx: np.ndarray, dy: np.ndarray) -> None:
        self.bwd_kernel(self.model.cuda_grid,
                        self.model.cuda_block,
                        (dy, dx, self.idx_max,
                         dy.shape[0], self.hi, self.wi, self.ci,
                         self.kh, self.kw, self.ho, self.wo,
                         self.hpadding, self.wpadding,
                         self.hstride, self.wstride,
                         self.hdilation, self.wdilation))
    # ----

    def _bwd_max_pool_nhwc(self, dx: np.ndarray, dy: np.ndarray) -> None:
        self.bwd_kernel(self.model.cuda_grid,
                        self.model.cuda_block,
                        (dy, dx, self.idx_max,
                         dy.shape[0], self.hi, self.wi, self.ci,
                         self.kh, self.kw, self.ho, self.wo,
                         self.hpadding, self.wpadding,
                         self.hstride, self.wstride,
                         self.hdilation, self.wdilation))
    # ----
