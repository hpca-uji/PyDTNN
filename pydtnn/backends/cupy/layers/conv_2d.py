from pydtnn.utils.constants import DTYPE2CTYPE
from pydtnn.utils.constants import ArrayShape
from pydtnn.backends.numpy.layers.conv_2d import Conv2DNumpy
from pydtnn.backends.cupy.layers.layer import LayerCupy
import cupy as np
import logging

from pydtnn.backends.cupy.layers.abstract.conv_2d import AbstractConv2DCupy
logger = logging.getLogger(__name__)


class Conv2DCupy(Conv2DNumpy, AbstractConv2DCupy, LayerCupy):

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray | None = None) -> None:
        super()._model_init(prev_shape, x)

        self.stream_2 = np.cuda.Stream()
        # self.defines_replaces = {"\"TYPE\"": DTYPE2CTYPE[self.model.dtype], "TENSOR_FORMAT": str(self.model.tensor_format)}

        self._im2row = self._get_kernel(func_name="im2row", defines_replaces={"\"TYPE\"": DTYPE2CTYPE[self.model.dtype], "TENSOR_FORMAT": "nhwc"})
        self._im2col = self._get_kernel(func_name="im2col", defines_replaces={"\"TYPE\"": DTYPE2CTYPE[self.model.dtype], "TENSOR_FORMAT": "nchw"})
        self._row2im = self._get_kernel(func_name="row2im", defines_replaces={"\"TYPE\"": DTYPE2CTYPE[self.model.dtype], "TENSOR_FORMAT": "nhwc"})
        self._col2im = self._get_kernel(func_name="col2im", defines_replaces={"\"TYPE\"": DTYPE2CTYPE[self.model.dtype], "TENSOR_FORMAT": "nchw"})
    # ----

    def im2row(self, x: np.ndarray, x_rows: np.ndarray) -> None:
        # return super().im2row(x, x_rows)
        self._im2row(self.model.cuda_grid,
                     self.model.cuda_block,
                     (x, x_rows,
                      x.shape[0], self.ci, self.hi, self.wi,
                      self.kh, self.kw, self.ho, self.wo,
                      self.hpadding, self.wpadding,
                      self.hstride, self.wstride,
                      self.hdilation, self.wdilation))
    # -----

    def row2im(self, x_rows: np.ndarray, dx: np.ndarray) -> None:
        # return super().im2row(x_rows, dx)
        self._row2im(self.model.cuda_grid,
                     self.model.cuda_block,
                     (x_rows, dx,
                      dx.shape[0], self.ci, self.hi, self.wi,
                      self.kh, self.kw, self.ho, self.wo,
                      self.hpadding, self.wpadding,
                      self.hstride, self.wstride,
                      self.hdilation, self.wdilation))
    # -----

    def im2col(self, x: np.ndarray, x_cols: np.ndarray) -> None:
        # return super().im2col(x, x_cols)
        self._im2row(self.model.cuda_grid,
                     self.model.cuda_block,
                     (x, x_cols,
                      x.shape[0], self.ci, self.hi, self.wi,
                      self.kh, self.kw, self.ho, self.wo,
                      self.hpadding, self.wpadding,
                      self.hstride, self.wstride,
                      self.hdilation, self.wdilation))
    # -----

    def col2im(self, x_cols: np.ndarray, dx: np.ndarray) -> None:
        # return super().im2row(x_cols, dx)
        self._col2im(self.model.cuda_grid,
                     self.model.cuda_block,
                     (x_cols, dx,
                      dx.shape[0], self.ci, self.hi, self.wi,
                      self.kh, self.kw, self.ho, self.wo,
                      self.hpadding, self.wpadding,
                      self.hstride, self.wstride,
                      self.hdilation, self.wdilation))
    # -----
