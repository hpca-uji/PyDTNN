import numpy as np
from pydtnn.backends.cpu.layers.abstract.conv_2d_standard import Conv2DStandardCPU
from pydtnn.backends.cpu.utils.im2col_nchw_cython import col2im_nchw_cython, im2col_nchw_cython  # , alt_col2im_nchw_cython
from pydtnn.backends.cpu.utils.im2row_nhwc_cython import im2row_nhwc_cython, row2im_nhwc_cython  # , alt_row2im_nhwc_cython

from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum

from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.tensor import TensorFormat, format_transpose


class Conv2DI2CCPU(Conv2DStandardCPU):

    def initialize(self, prev_shape: ArrayShape, x: np.ndarray | None = None) -> None:
        super().initialize(prev_shape, x)

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

        self.temp_bw_shape = self._x_cr_shape
        self.temp_bw_shape_size = np.prod(self.temp_bw_shape)

        y_shape = (dim_n, self.co)
        self.y_size = np.prod(y_shape)

        # self.y = np.zeros(shape=(self.dim_n, self.co), dtype=self.model.dtype, order="C")
        # self.real_memory_size += self.y.nbytes

        if not self.model.evaluate_only:
            self.dx_shape = self.model.encode_shape((self.model.batch_size, self.ci, self.hi, self.wi))
            self._dw_shape = _dw_shape  # This shape is only for a intermediate operation.
        else:
            self.dx_shape = (0,)

        self.dx_shape_size = int(np.prod(self.dx_shape))
        # NOTE: These attributes only store data, their values before the operation doesn't matter; they're initalized due avoid warnings in "LayerAndActivationBase.export".
        self.temp_c_r_dx = np.zeros(shape=(max(np.prod(self._x_cr_shape), self.dx_shape_size), ), dtype=self.model.dtype, order="C")
        # self.temp_c_r_dx: Temporal array where the cols/rows and dx values are stored.
        self.real_memory_size += self.temp_c_r_dx.nbytes

        self.temp_y_bc_br = np.zeros(shape=(max(self.y_size, self.temp_bw_shape_size), ), dtype=self.model.dtype, order="C")
        # self.temp_y_bc_br: Temporal array where the y and backward's cols/rows values are stored.
        self.real_memory_size += self.temp_y_bc_br.nbytes

        self.real_memory_size += self.temp_memory_size
    # ---

    def get_rows(self, batch_size: int) -> np.ndarray:
        dim_n = batch_size * self.ho * self.wo
        shape = (dim_n, self.dim_c)
        x_rows: np.ndarray = self.temp_c_r_dx[:np.prod(shape)]
        x_rows = x_rows.reshape(shape)
        return x_rows

    def get_cols(self, batch_size: int) -> np.ndarray:
        dim_n = batch_size * self.ho * self.wo
        shape = (self.dim_c, dim_n)
        x_cols: np.ndarray = self.temp_c_r_dx[:np.prod(shape)]
        x_cols = x_cols.reshape(shape)
        return x_cols

    def get_y(self, batch_size: int) -> np.ndarray:
        dim_n = batch_size * self.ho * self.wo
        shape = (dim_n, self.co)
        y: np.ndarray = self.temp_y_bc_br[:np.prod(shape)]
        y = y.reshape(shape)
        return y

    def _forward_i2c_nhwc(self, x: np.ndarray) -> np.ndarray:
        """Version of the forward function that uses im2col and matmul"""

        # x_rows = np.zeros(shape=(dim_n, self.dim_c), dtype=self.model.dtype)
        # x_rows = np.asarray(self._x_rows[:dim_n, :], dtype=self.model.dtype, order="C", copy=None)
        x_rows = self.get_rows(x.shape[0])
        x_rows.fill(0)
        # y = self.y[:shape[-1], :]
        y = self.get_y(x.shape[0])

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_IM2COL)
        im2row_nhwc_cython(x, x_rows,
                           self.kh, self.kw, self.ho, self.wo,
                           self.vpadding, self.hpadding,
                           self.vstride, self.hstride, self.vdilation, self.hdilation)
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

        return np.asarray(y, dtype=self.model.dtype, order='C', copy=None)

    def _forward_i2c_nchw(self, x: np.ndarray) -> np.ndarray:
        """Version of the forward function that uses im2col and matmul"""

        # x_cols = np.zeros(shape=(self.dim_c, dim_n), dtype=self.model.dtype)
        # x_cols: np.ndarray = np.asarray(self._x_cr[:, :dim_n], dtype=self.model.dtype, order="C", copy=None)
        x_cols = self.get_cols(x.shape[0])
        x_cols.fill(0)
        # y = self.y[:shape[-1], :]
        y = self.get_y(x.shape[0])

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_IM2COL)
        im2col_nchw_cython(x, x_cols,
                           self.kh, self.kw, self.ho, self.wo,
                           self.vpadding, self.hpadding,
                           self.vstride, self.hstride, self.vdilation, self.hdilation)
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

        return np.asarray(y, dtype=self.model.dtype, order='C', copy=None)

    def _backward_i2c_nhwc(self, dy: np.ndarray) -> np.ndarray:
        """Version of the backward function that uses im2col and matmul"""

        # res = np.asarray(self.res_bw[:(dy.shape[0] * self.ho * self.wo), :], dtype=self.model.dtype, order="C", copy=None)
        rows: np.ndarray = self.get_rows(dy.shape[0])
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
        self.dw = self.dw.reshape(self.weights.shape, copy=False, order="C")
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        # Biases gradient
        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_SUM_BIASES)
            np.sum(dy, axis=(0, 1, 2), out=self.db)
            # np.sum(dy.reshape((self.co, -1), copy=None), axis=1, out=self.db)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        # Data gradient
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_TRANSPOSE_W)
        w_rows = self.weights.reshape((-1, self.co), copy=False).T
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_MATMUL)
        np.matmul(dy_cols, w_rows, out=rows,
                  dtype=self.model.dtype)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        dx: np.ndarray = self.temp_c_r_dx[:self.dx_shape_size].reshape(self.dx_shape, order="C")
        dx.fill(0)  # NOTE: It is necessary that dx is filled with 0s.

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_COL2IM)
        row2im_nhwc_cython(rows, dx,
                           dy.shape[0], self.hi, self.wi, self.ci,
                           self.kh, self.kw, self.ho, self.wo,
                           self.vpadding, self.hpadding,
                           self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return np.asarray(dx, dtype=self.model.dtype, order='C', copy=None)

    def _backward_i2c_nchw(self, dy: np.ndarray) -> np.ndarray:
        """Version of the backward function that uses im2col and matmul"""
        # cols:np.ndarray = np.asarray(self.temp_bw[:, :(dy.shape[0] * self.ho * self.wo)], dtype=self.model.dtype, order="C", copy=None)
        cols = self.get_cols(dy.shape[0])

        self.dw = self.dw.reshape(self._dw_shape)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_TRANSPOSE_DY)
        dy_rows: np.ndarray = format_transpose(dy, "NCHW", "CNHW").reshape((self.co, -1), copy=None)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        # Weigths gradient
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DW_MATMUL)
        np.matmul(dy_rows, self.x_cols.T, out=self.dw,
                  dtype=self.model.dtype)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_RESHAPE_DW)
        self.dw = self.dw.reshape(self.weights.shape, copy=True, order="C")
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

        dx: np.ndarray = self.temp_c_r_dx[:self.dx_shape_size].reshape(self.dx_shape, order="C")
        dx.fill(0)  # NOTE: It is necessary that dx is filled with 0s.

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_COL2IM)
        col2im_nchw_cython(cols, dx,
                           dy.shape[0], self.ci, self.hi, self.wi,
                           self.kh, self.kw, self.ho, self.wo,
                           self.vpadding, self.hpadding,
                           self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return np.asarray(dx, dtype=self.model.dtype, order='C', copy=None)
