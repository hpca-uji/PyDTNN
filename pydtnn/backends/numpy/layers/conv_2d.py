from pydtnn.libs import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np

from pydtnn.backends.numpy.layers.abstract.conv_2d_standard import AbstractConv2DStandardNumpy

from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum

from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.tensor import TensorFormat, format_transpose


class Conv2DNumpy(AbstractConv2DStandardNumpy):

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray | None = None) -> None:
        super()._model_init(prev_shape, x)

        # dim_n: Dimension where the "n" of NCHW/NHWC is used in the calculations.
        # self.dim_c: Dimension where the "c" of NCHW/NHWC is used in the calculations.
        dim_n = self.model.batch_size * self.ho * self.wo
        self.dim_c = self.ci * self.kh * self.kw

        match self.model.tensor_format:
            case TensorFormat.NCHW:
                self.forward = self._forward_i2c_nchw
                self.backward = self._backward_i2c_nchw
                self._x_cr_shape = (self.dim_c, dim_n)
                _dw_shape = (self.co, self.dim_c)
            case TensorFormat.NHWC:
                self.forward = self._forward_i2c_nhwc
                self.backward = self._backward_i2c_nhwc
                self._x_cr_shape = (dim_n, self.dim_c)
                _dw_shape = (self.dim_c, self.co)
            case _:
                self._x_cr_shape = (None, )
                _dw_shape = (None, )
                raise NotImplementedError(f"\"{self.model.tensor_format}\" format not implemented.")
        # -

        y_shape = (dim_n, self.co)
        self.y_size = np.prod(y_shape)

        # self.y = np.zeros(shape=(self.dim_n, self.co), dtype=self.model.dtype)
        # self.real_memory_size += self.y.nbytes

        if not self.model.evaluate_only:
            self.dx_shape = self.model.encode_shape((self.model.batch_size, self.ci, self.hi, self.wi))
            self._dw_shape = _dw_shape  # This shape is only for a intermediate operation.
        else:
            self.dx_shape = (0,)

        self.dx_shape_size = int(np.prod(self.dx_shape))
        # NOTE: These attributes only store data, their values before the operation doesn't matter; they're initalized due avoid warnings in "LayerAndActivationBase.export".
        self.temp_c_r = np.zeros(shape=self._x_cr_shape, dtype=self.model.dtype)
        # self.temp_c_r_dx: Temporal array where the forward and batckward's cols/rows are stored
        self.memory_used += self.temp_c_r.nbytes

        self.temp_y_dx = np.zeros(shape=(max(self.y_size, self.dx_shape_size), ), dtype=self.model.dtype)
        # self.temp_y_bc_br: Temporal array where the y and backward's cols/rows values are stored.
        self.memory_used += self.temp_y_dx.nbytes

        self.memory_used += self.tmp_memory_used
    # ---

    def get_rows(self, batch_size: int) -> np.ndarray:
        dim_n = batch_size * self.ho * self.wo
        shape = (dim_n, self.dim_c)
        x_rows: np.ndarray = self.temp_c_r[:np.prod(shape)]
        x_rows = x_rows.reshape(shape)
        return x_rows

    def get_cols(self, batch_size: int) -> np.ndarray:
        dim_n = batch_size * self.ho * self.wo
        shape = (self.dim_c, dim_n)
        x_cols: np.ndarray = self.temp_c_r[:np.prod(shape)]
        x_cols = x_cols.reshape(shape)
        return x_cols

    def get_y(self, batch_size: int) -> np.ndarray:
        dim_n = batch_size * self.ho * self.wo
        shape = (dim_n, self.co)
        y: np.ndarray = self.temp_y_dx[:np.prod(shape)]
        y = y.reshape(shape)
        return y

    def get_dx(self, batch_size: int) -> np.ndarray:
        shape = self.model.encode_shape((batch_size, self.ci, self.hi, self.wi))
        dx: np.ndarray = self.temp_y_dx[:np.prod(shape)]
        dx = dx.reshape(shape)
        return dx

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

    def im2row(self, x: np.ndarray, x_rows: np.ndarray):
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

    def im2col(self, x: np.ndarray, x_cols: np.ndarray):
        n, _, _, _ = x.shape

        for cc in range(self.ci):
            for ii in range(self.kh):
                for jj in range(self.kw):
                    row = (cc * self.kh + ii) * self.kw + jj
                    for nn in range(n):
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

    def _forward_i2c_nhwc(self, x: np.ndarray) -> np.ndarray:
        """Version of the forward function that uses im2col and matmul"""

        # x_rows = np.zeros(shape=(dim_n, self.dim_c), dtype=self.model.dtype)
        # x_rows = np.asarray(self._x_rows[:dim_n, :], dtype=self.model.dtype)
        x_rows = self.get_rows(x.shape[0])
        x_rows.fill(0)
        # y = self.y[:shape[-1], :]
        y = self.get_y(x.shape[0])

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_IM2COL)
        self.im2row(x, x_rows)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        self.x_rows = x_rows

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_RESHAPE_W)
        w_cols = self.weights.reshape((-1, self.co), copy=False)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_MATMUL)
        np.matmul(x_rows, w_cols, out=y,
                  dtype=self.model.dtype)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_SUM_BIASES)
            np.add(y, self.biases.reshape((-1, self.co), copy=False), out=y,
                   dtype=self.model.dtype)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_RESHAPE_Y)
        y = y.reshape((-1, self.ho, self.wo, self.co), copy=False)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return np.asarray(y, dtype=self.model.dtype, order="C")

    def _forward_i2c_nchw(self, x: np.ndarray) -> np.ndarray:
        """Version of the forward function that uses im2col and matmul"""

        # x_cols = np.zeros(shape=(self.dim_c, dim_n), dtype=self.model.dtype)
        # x_cols: np.ndarray = np.asarray(self._x_cr[:, :dim_n], dtype=self.model.dtype)
        x_cols = self.get_cols(x.shape[0])
        x_cols.fill(0)
        # y = self.y[:shape[-1], :]
        y = self.get_y(x.shape[0])

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_IM2COL)
        self.im2col(x, x_cols)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.x_cols = x_cols

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_RESHAPE_W)
        w_rows = self.weights.reshape((self.co, -1), copy=False)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_MATMUL)
        np.matmul(w_rows, x_cols, out=y.T,
                  dtype=self.model.dtype)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_SUM_BIASES)
            np.add(y, self.biases.reshape((-1, self.co), copy=False), out=y,
                   dtype=self.model.dtype)

            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_RESHAPE_Y)
        y: np.ndarray = format_transpose(y.reshape((-1, self.ho, self.wo, self.co), copy=False), "NHWC", "NCHW")
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return np.asarray(y, dtype=self.model.dtype, order="C")

    def _backward_i2c_nhwc(self, dy: np.ndarray) -> np.ndarray:
        """Version of the backward function that uses im2col and matmul"""

        # res = np.asarray(self.res_bw[:(dy.shape[0] * self.ho * self.wo), :], dtype=self.model.dtype)
        self.dw = self.dw.reshape(self._dw_shape)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_TRANSPOSE_DY)
        dy_cols: np.ndarray = dy.reshape((-1, self.co), copy=False)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        # Weigths gradient
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DW_MATMUL)
        np.matmul(self.x_rows.T, dy_cols, out=self.dw,
                  dtype=self.model.dtype)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_RESHAPE_DW)
        self.dw = self.dw.reshape(self.weights.shape, copy=False)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        rows: np.ndarray = self.get_rows(dy.shape[0]) # NOTE: rows shares the memory with self.x_rows

        # Biases gradient
        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_SUM_BIASES)
            np.sum(dy, axis=(0, 1, 2), out=self.db)
            # np.sum(dy.reshape((self.co, -1)), axis=1, out=self.db)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        # Data gradient
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_TRANSPOSE_W)
        w_rows = self.weights.reshape((-1, self.co), copy=False).T
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_MATMUL)
        np.matmul(dy_cols, w_rows, out=rows,
                  dtype=self.model.dtype)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        dx: np.ndarray = self.get_dx(dy.shape[0])
        dx.fill(0)  # NOTE: It is necessary that dx is filled with 0s.

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_COL2IM)
        self.row2im(rows, dx)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return np.asarray(dx, dtype=self.model.dtype, order="C")

    def _backward_i2c_nchw(self, dy: np.ndarray) -> np.ndarray:
        """Version of the backward function that uses im2col and matmul"""
        # cols:np.ndarray = np.asarray(self.temp_bw[:, :(dy.shape[0] * self.ho * self.wo)], dtype=self.model.dtype)
        self.dw = self.dw.reshape(self._dw_shape)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_TRANSPOSE_DY)
        dy_rows: np.ndarray = format_transpose(dy, "NCHW", "CNHW").reshape((self.co, -1))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        # Weigths gradient
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DW_MATMUL)
        np.matmul(dy_rows, self.x_cols.T, out=self.dw,
                  dtype=self.model.dtype)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        cols = self.get_cols(dy.shape[0])  # NOTE: cols shares the memory with self.x_cols

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_RESHAPE_DW)
        self.dw = self.dw.reshape(self.weights.shape)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        # Biases gradient
        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_SUM_BIASES)
            np.sum(dy, axis=(0, 2, 3), out=self.db)
            # np.sum(dy.reshape((self.co, -1), copy=False), axis=1, out=self.db)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        # Data gradient
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_TRANSPOSE_W)
        w_cols = self.weights.reshape((self.co, -1), copy=False).T
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_MATMUL)
        np.matmul(w_cols, dy_rows, out=cols,
                  dtype=self.model.dtype, order='C')
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        dx: np.ndarray = self.get_dx(dy.shape[0])
        dx.fill(0)  # NOTE: It is necessary that dx is filled with 0s.

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_COL2IM)
        self.col2im(cols, dx)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return np.asarray(dx, dtype=self.model.dtype, order="C")
