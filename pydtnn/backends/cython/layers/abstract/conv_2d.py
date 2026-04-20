from pydtnn.backends.cython.utils.im2row_nhwc_cython import im2row_nhwc_cython, row2im_nhwc_cython  # , alt_row2im_nhwc_cython
from pydtnn.backends.cython.utils.im2col_nchw_cython import col2im_nchw_cython, im2col_nchw_cython  # , alt_col2im_nchw_cython
from typing import TYPE_CHECKING
from pydtnn.libs import numpy as np
from pydtnn.backends.cython.layers.layer import LayerCython
from pydtnn.backends.numpy.layers.abstract.conv_2d import AbstractConv2DNumpy
import logging
logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import numpy as np


class AbstractConv2DCython(AbstractConv2DNumpy, LayerCython):

    def im2row(self, x: np.ndarray, x_rows: np.ndarray) -> None:
        im2row_nhwc_cython(x, x_rows,
                           self.kh, self.kw, self.ho, self.wo,
                           self.hpadding, self.wpadding,
                           self.hstride, self.wstride, self.hdilation, self.wdilation)
        # -----------------

    def im2col(self, x: np.ndarray, x_cols: np.ndarray) -> None:
        im2col_nchw_cython(x, x_cols,
                           self.kh, self.kw, self.ho, self.wo,
                           self.hpadding, self.wpadding,
                           self.hstride, self.wstride, self.hdilation, self.wdilation)
        # -----------------

    def row2im(self, x_rows: np.ndarray, dx: np.ndarray) -> None:
        row2im_nhwc_cython(x_rows, dx,
                           dx.shape[0], self.hi, self.wi, self.ci,
                           self.kh, self.kw, self.ho, self.wo,
                           self.hpadding, self.wpadding,
                           self.hstride, self.wstride, self.hdilation, self.wdilation)
       # -----------------

    def col2im(self, x_cols: np.ndarray, dx: np.ndarray) -> None:
        col2im_nchw_cython(x_cols, dx,
                           dx.shape[0], self.ci, self.hi, self.wi,
                           self.kh, self.kw, self.ho, self.wo,
                           self.hpadding, self.wpadding,
                           self.hstride, self.wstride, self.hdilation, self.wdilation)
        # -----------------
