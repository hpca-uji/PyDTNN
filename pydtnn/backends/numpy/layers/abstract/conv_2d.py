from typing import TYPE_CHECKING
from pydtnn.libs import numpy as np
from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.performance_models import im2col_time, matmul_time
from pydtnn.backends.numpy.layers.layer import LayerNumpy
from pydtnn.layers.abstract.conv_2d import AbstractConv2D
import logging
logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import numpy as np


class AbstractConv2DNumpy(AbstractConv2D[np.ndarray], LayerNumpy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # More parameters initialized in initialize()
        self.biases = None  # type: ignore
        self.weights = None  # type: ignore
        self.fwd_time = None  # type: ignore
        self.bwd_time = None  # type: ignore
    # ----

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray | None = None) -> None:
        super()._model_init(prev_shape, x)
        if self.use_bias:
            bias_shape = (self.co,)  # NOTE: Is the same shape in every variant and grouping
            self.biases = np.asarray(self.biases_initializer(bias_shape, self.model.param_dtype), order="C")
            self.memory_used += self.biases.nbytes

        self.weights = np.asarray(self.weights_initializer(self.weights_shape, self.model.param_dtype), order="C")

        self.memory_used += self.weights.nbytes

        if not self.model.evaluate_only:
            if self.use_bias:
                self.db = np.zeros(shape=bias_shape, dtype=self.model.param_dtype, order="C")
                self.memory_used += self.db.nbytes

            self.dw: np.ndarray = np.zeros(self.weights.shape, dtype=self.model.param_dtype, order="C")
            self.memory_used += self.dw.nbytes

        # Performance models
        self.fwd_time = \
            im2col_time(m=(self.ci * self.kh * self.kw), n=(self.model.batch_size * self.ho * self.wo),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype) + \
            matmul_time(m=self.co, n=(self.model.batch_size * self.ho * self.wo), k=(self.ci * self.kh * self.kw),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype)  # type: ignore (It works well.)
        self.bwd_time = \
            matmul_time(m=self.co, n=(self.ci * self.kh * self.kw), k=(self.model.batch_size * self.ho * self.wo),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype)  # type: ignore (It works well.)
        self.bwd_time += matmul_time(m=(self.ci * self.kh * self.kw), n=(self.model.batch_size * self.ho * self.wo),
                                     k=self.co, cpu_speed=self.model.cpu_speed,
                                     memory_bw=self.model.memory_bw, dtype=self.model.dtype)

##########################################################################################################################
##########################################################################################################################
#### TEST ####
##############

    def col2im_alt(self, x: np.ndarray, x_rows: np.ndarray) -> np.ndarray:
        x = np.pad(x, ((0, 0), (0, 0), (self.hpadding, self.hpadding), (self.wpadding, self.wpadding)), mode="constant")
        cols = list[np.ndarray]()

        for kh in range(self.kh):
            for kw in range(self.kw):
                h_start = kh * self.hdilation
                w_start = kw * self.wdilation
                h_end = h_start + self.hstride * self.ho
                w_end = w_start + self.wstride * self.wo

                col = x[:, :, h_start:h_end:self.hstride, w_start:w_end:self.wstride]
                cols.append(col)
        return np.stack(cols, axis=2).reshape(x_rows.shape)

##########################################################################################################################
##########################################################################################################################

    def im2row(self, x: np.ndarray, x_rows: np.ndarray) -> None:
        n, _, _, _ = x.shape
        for nn in range(n):
            for xx in range(self.ho):
                for yy in range(self.wo):
                    row = (nn * self.ho + xx) * self.wo + yy
                    for ii in range(self.kh):
                        x_x = self.hstride * xx + self.hdilation * ii - self.hpadding
                        for jj in range(self.kw):
                            x_y = self.wstride * yy + self.wdilation * jj - self.wpadding
                            for cc in range(self.ci):
                                col = (cc * self.kh + ii) * self.kw + jj
                                if (0 <= x_x < self.hi) and (0 <= x_y < self.wi):
                                    x_rows[row, col] = x[nn, x_x, x_y, cc]
                                else:
                                    x_rows[row, col] = 0.0
    # -----

    def row2im(self, x_rows: np.ndarray, dx: np.ndarray) -> None:
        n, _, _, _ = dx.shape
        for nn in range(n):
            for xx in range(self.ho):
                for yy in range(self.wo):
                    row = (nn * self.ho + xx) * self.wo + yy
                    for cc in range(self.ci):
                        for ii in range(self.kh):
                            x_x = self.hstride * xx + self.hdilation * ii - self.hpadding
                            if 0 <= x_x < self.hi:
                                for jj in range(self.kw):
                                    x_y = self.wstride * yy + self.wdilation * jj - self.wpadding
                                    if 0 <= x_y < self.wi:
                                        col = (cc * self.kh + ii) * self.kw + jj
                                        dx[nn, x_x, x_y, cc] += x_rows[row, col]
    # -----

    def im2col(self, x: np.ndarray, x_cols: np.ndarray) -> None:
        n, _, _, _ = x.shape

        for nn in range(n):
            for cc in range(self.ci):
                for ii in range(self.kh):
                    for jj in range(self.kw):
                        row = (cc * self.kh + ii) * self.kw + jj
                        for xx in range(self.ho):
                            x_x = self.hstride * xx + self.hdilation * ii - self.hpadding
                            for yy in range(self.wo):
                                x_y = self.wstride * yy + self.wdilation * jj - self.wpadding
                                col = (nn * self.ho + xx) * self.wo + yy
                                if (0 <= x_x < self.hi) and (0 <= x_y < self.wi):
                                    x_cols[row, col] = x[nn, cc, x_x, x_y]
                                else:
                                    x_cols[row, col] = 0.0
    # -----

    def col2im(self, x_cols: np.ndarray, dx: np.ndarray) -> None:
        n, _, _, _ = dx.shape
        for cc in range(self.ci):
            for ii in range(self.kh):
                for jj in range(self.kw):
                    row = (cc * self.kh + ii) * self.kw + jj
                    for nn in range(n):
                        for xx in range(self.ho):
                            x_x = self.hstride * xx + self.hdilation * ii - self.hpadding
                            if (0 <= x_x < self.hi):
                                for yy in range(self.wo):
                                    x_y = self.wstride * yy + self.wdilation * jj - self.wpadding
                                    col = (nn * self.ho + xx) * self.wo + yy
                                    if (0 <= x_y < self.wi):
                                        dx[nn, cc, x_x, x_y] = x_cols[row, col]
    # -----

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Use a real forward variant!")

    def backward(self, dy: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Use a real backwards variant!")

    def print_in_convdirect_format(self) -> None:
        if self.wstride != 1 or self.hstride != 1:
            return
        # #l kn wo ho t kh kw ci wi hi"
        ci, hi, wi = self.model.decode_shape(self.prev_shape)
        print(self.id, self.co, self.wo, self.ho, self.model.batch_size, self.kh, self.kw, ci, wi, hi, sep="\t")
