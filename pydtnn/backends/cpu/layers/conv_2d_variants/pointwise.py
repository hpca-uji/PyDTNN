import numpy as np

from pydtnn.backends.cpu.layers.conv_2d import Conv2DCPU
from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_OPS_EVENT_enum, PYDTNN_EVENT_FINISHED
from pydtnn.utils.best_transpose_0231 import best_transpose_0231
from pydtnn.utils.best_transpose_0312 import best_transpose_0312
from pydtnn.model import Model
from pydtnn.utils.constants import ArrayShape, Parameters
from pydtnn.utils.tensor import TensorFormat, format_transpose


class Conv2DPointwiseCPU(Conv2DCPU):

    def _export_prop(self, key: str):
        if key not in {Parameters.WEIGHTS, Parameters.DW}:
            return super()._export_prop(key)
        value = getattr(self, key)

        match self.model.tensor_format:
            case TensorFormat.NHWC:
                # NHWC's src: ci, co
                # NCHW's dst: co, ci
                return np.asarray(format_transpose(value, "IO", "OI"), dtype=np.float64, order="C", copy=True)
        return super()._export_prop(key)
    # -----

    def _import_prop(self, key: str, value) -> None:
        if key not in {Parameters.WEIGHTS, Parameters.DW}:
            return super()._import_prop(key, value)

        match self.model.tensor_format:
            case TensorFormat.NHWC:
                # NCHW's src: co, ci
                # NHWC's dst: ci, co
                ary = getattr(self, key)
                ary[:] = np.asarray(format_transpose(value, "OI", "IO"), dtype=self.model.dtype, order="C", copy=None)
                return
        return super()._import_prop(key, value)
    # ------

    def _initializing_special_parameters(self):
        super()._initializing_special_parameters()
        # Setting other parameters
        self.kh = self.kw = 1
        # Setting weights
        match self.model.tensor_format:
                case TensorFormat.NCHW:
                    self.weights_shape = (self.co, self.ci)
                case TensorFormat.NHWC:
                    self.weights_shape = (self.co, self.ci)
                case _:
                    raise NotImplementedError(f"\"{self.model.tensor_format}\" format not implemented.")
        #--
    # ---

    def initialize(self, prev_shape: ArrayShape, x: np.ndarray | None = None) -> None:
        super().initialize(prev_shape, x)
        match self.model.tensor_format:
            case TensorFormat.NCHW:
                self.forward = self._forward_pointwise_nchw
                self.backward = self._backward_nchw
            case TensorFormat.NHWC:
                self.forward = self._forward_pointwise_nhwc
                self.backward = self._backward_nhwc
        # --
        y_shape = self.model.encode_shape((self.model.batch_size, self.co, self.ho, self.wo))
        # NOTE: These attributes only store data, their values before the operation doesn't matter; they're initalized due avoid warnings in "LayerAndActivationBase.export".
        # self.dw (this one too, but it's initalized in Conv2DCPU)
        self.y = np.zeros(shape=y_shape, dtype=self.model.dtype, order="C")
        self.dx = np.zeros(shape=(self.ci, self.model.batch_size * self.hi * self.wi), dtype=self.model.dtype, order="C")
    # ------

    def _forward_pointwise_nhwc(self, x: np.ndarray) -> np.ndarray:
        if self.model.mode is Model.Mode.TRAIN:
            self.x: np.ndarray = x

        y = self.y[:x.shape[0], :]
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_POINTWISE_CONV)
        np.matmul(x, self.weights, out=y,
                  dtype=self.model.dtype, order="C")
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_SUM_BIASES)
            np.add(y, self.biases.reshape((1, 1, 1, self.co), copy=False), out=y,
                   dtype=self.model.dtype, order="C")
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return np.asarray(y, dtype=self.model.dtype, order='C', copy=None)

    def _forward_pointwise_nchw(self, x: np.ndarray) -> np.ndarray:

        if self.model.mode is Model.Mode.TRAIN:
            self.x: np.ndarray = x

        y = self.y[:x.shape[0], :]

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_TRANSPOSE_Y)
        y = best_transpose_0231(y)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_POINTWISE_CONV)
        np.matmul(best_transpose_0231(x), self.weights.T, out=y,
                  dtype=self.model.dtype, order="C")
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_TRANSPOSE_Y)
        y: np.ndarray = best_transpose_0312(y)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_SUM_BIASES)
            np.add(y, self.biases.reshape((1, self.co, 1, 1), copy=False), out=y,
                   dtype=self.model.dtype, order="C")
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(y, dtype=self.model.dtype, order='C', copy=None)

    def _backward_nhwc(self, dy: np.ndarray) -> np.ndarray:

        _n, _h, _w, _c = dy.shape
        _dim = _n * _h * _w
        x_shape = self.x.shape
        dx = np.asarray(self.dx[:, :_dim], dtype=self.model.dtype, order="C", copy=None)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_TRANSPOSE_DY)
        reshaped_dy = dy.reshape((_dim, _c), copy=False)
        self.x = self.x.reshape((-1, _dim), copy=False)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DW_MATMUL)
        np.matmul(self.x, reshaped_dy, out=self.dw,
                  dtype=self.model.dtype, order="C")
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_SUM_BIASES)
            np.sum(dy, axis=(0, 1, 2), out=self.db)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_TRANSPOSE_W)
        w = self.weights.reshape((self.co, -1), copy=False).T
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        reshaped_dy: np.ndarray = dy.reshape((self.co, -1), copy=False)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_MATMUL)
        np.matmul(w, reshaped_dy, out=dx,
                  dtype=self.model.dtype, order="C")
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return np.asarray(dx.reshape(x_shape, copy=False), dtype=self.model.dtype, order='C', copy=None)

    def _backward_nchw(self, dy: np.ndarray) -> np.ndarray:

        _n, _c, _h, _w = dy.shape
        _dim = _n * _h * _w
        x_shape = self.x.shape
        dx = np.asarray(self.dx[:, :_dim], dtype=self.model.dtype, order="C", copy=None)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_TRANSPOSE_DY)
        reshaped_dy = dy.reshape((_dim, _c), copy=False)
        self.x = self.x.reshape((-1, _dim), copy=False)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DW_MATMUL)
        np.matmul(self.x, reshaped_dy, out=self.dw.T,
                  dtype=self.model.dtype, order="C")
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_SUM_BIASES)
            np.sum(dy, axis=(0, 2, 3), out=self.db)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_TRANSPOSE_W)
        w = self.weights.reshape((self.co, -1), copy=False).T
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        reshaped_dy: np.ndarray = dy.reshape((self.co, -1), copy=False)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_MATMUL)
        np.matmul(w, reshaped_dy, out=dx,
                  dtype=self.model.dtype, order="C")
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return np.asarray(dx.reshape(x_shape, copy=False), dtype=self.model.dtype, order='C', copy=None)
