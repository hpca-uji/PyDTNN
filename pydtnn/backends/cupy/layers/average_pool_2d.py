import logging
logger = logging.getLogger(__name__)

from pydtnn.backends.cupy.layers.layer import LayerCupy
from pydtnn.backends.numpy.layers.average_pool_2d import AveragePool2DNumpy
from pydtnn.libs import numpy as np
from typing import TYPE_CHECKING

from pydtnn.utils.constants import ArrayShape, DTYPE2CTYPE
if TYPE_CHECKING:
    import numpy as np


class AveragePool2DCupy(AveragePool2DNumpy, LayerCupy):

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray | None = None) -> None:
        super()._model_init(prev_shape, x)
        self.defines_replaces = {"\"TYPE\"": DTYPE2CTYPE[self.model.dtype],
                                 "TENSOR_FORMAT": str(self.model.tensor_format)}
        self.fwd_kernel = self._fwd_kernel()
        self.bwd_kernel = self._bwd_kernel()
        #----

    def _fwd_avg_pool_nchw(self, x: np.ndarray, y: np.ndarray) -> None:
        #return super()._fwd_avg_pool_nchw(x, y)
        self.fwd_kernel(self.model.cuda_grid,
                        self.model.cuda_block,
                        (x, y,
                         x.shape[0], self.ci, self.hi, self.wi,
                         self.kh, self.kw, self.ho, self.wo,
                         self.hpadding, self.wpadding,
                         self.hstride, self.wstride,
                         self.hdilation, self.wdilation))
    # ----

    def _fwd_avg_pool_nhwc(self, x: np.ndarray, y: np.ndarray) -> None:
        #return super()._fwd_avg_pool_nhwc(x, y)
        y.fill(0)
        self.fwd_kernel(self.model.cuda_grid,
                        self.model.cuda_block,
                        (x, y,
                         x.shape[0], self.ci, self.hi, self.wi,
                         self.kh, self.kw, self.ho, self.wo,
                         self.hpadding, self.wpadding,
                         self.hstride, self.wstride,
                         self.hdilation, self.wdilation))
    # ----

    def _bwd_avg_pool_nchw(self, dx: np.ndarray, dy: np.ndarray) -> None:
        #return super()._bwd_avg_pool_nchw(dx, dy)
        self.bwd_kernel(self.model.cuda_grid,
                        self.model.cuda_block,
                        (dx, dy,
                         dy.shape[0], self.hi, self.wi, self.ci,
                         self.kh, self.kw, self.ho, self.wo,
                         self.hpadding, self.wpadding,
                         self.hstride, self.wstride,
                         self.hdilation, self.wdilation))
    # ----

    def _bwd_avg_pool_nhwc(self, dx: np.ndarray, dy: np.ndarray) -> None:
        #return super()._bwd_avg_pool_nhwc(dx, dy)
        self.bwd_kernel(self.model.cuda_grid,
                        self.model.cuda_block,
                        (dx, dy,
                         dy.shape[0], self.hi, self.wi, self.ci,
                         self.kh, self.kw, self.ho, self.wo,
                         self.hpadding, self.wpadding,
                         self.hstride, self.wstride,
                         self.hdilation, self.wdilation))
    # ----
