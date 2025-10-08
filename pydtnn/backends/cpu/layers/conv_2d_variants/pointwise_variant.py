from abc import ABC

import numpy as np

from pydtnn.layers import Conv2D
from pydtnn.tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_OPS_EVENT_enum, PYDTNN_EVENT_FINISHED
from pydtnn.utils.best_transpose_0231 import best_transpose_0231
from pydtnn.utils.best_transpose_0312 import best_transpose_0312
from pydtnn.model import ModelModeEnum


class PointwiseVariant(Conv2D, ABC):

    # NOTE: Attributes defined in conv_2d_cpu.
    y: np.ndarray
    dy: np.ndarray
    dw: np.ndarray
    dx: np.ndarray
    db: np.ndarray
    # ----

    def _forward_pointwise_nhwc(self, x: np.ndarray) -> np.ndarray:
        if self.model.mode is ModelModeEnum.TRAIN:
            self.x: np.ndarray = x

        y = self.y[:x.shape[0], :]
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_POINTWISE_CONV)
        np.matmul(x, self.weights, out=y)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_SUM_BIASES)
            y += self.biases.reshape((1, 1, 1, self.co), copy=False)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        
        return np.asarray(y, dtype=self.model.dtype, order='C', copy=None)
    # --- END _forward_pointwise_nhwc --- #

    def _forward_pointwise_nchw(self, x: np.ndarray) -> np.ndarray:

        if self.model.mode is ModelModeEnum.TRAIN:
            self.x: np.ndarray = x

        y = self.y[:x.shape[0], :]

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_TRANSPOSE_Y)
        y = best_transpose_0231(y)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_POINTWISE_CONV)
        np.matmul(best_transpose_0231(x), self.weights.T, out=y)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_TRANSPOSE_Y)
        y: np.ndarray = best_transpose_0312(y)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_SUM_BIASES)
            y += self.biases.reshape((1, self.co, 1, 1), copy=False)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(y, dtype=self.model.dtype, order='C', copy=None)
    # --- END _forward_pointwise_nchw --- #

    def _backward_pointwise_nhwc(self, dy: np.ndarray) -> np.ndarray:

        _n, _h, _w, _c = dy.shape
        _dim = _n * _h * _w
        x_shape = self.x.shape
        dx = np.asarray(self.dx[:, :_dim], dtype=self.model.dtype, order="C", copy=None)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_TRANSPOSE_DY)
        reshaped_dy = dy.reshape((_dim, _c), copy=False)
        self.x = self.x.reshape((-1, _dim), copy=False)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DW_MATMUL)
        np.matmul(self.x, reshaped_dy, out=self.dw)
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
        np.matmul(w, reshaped_dy, out=dx)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return np.asarray(dx.reshape(x_shape, copy=False) , dtype=self.model.dtype, order='C', copy=None)
    # --- END _backward_pointwise_nhwc --- #

    def _backward_pointwise_nchw(self, dy: np.ndarray) -> np.ndarray:

        _n, _c, _h, _w = dy.shape
        _dim = _n * _h * _w
        x_shape = self.x.shape
        dx = np.asarray(self.dx[:, :_dim], dtype=self.model.dtype, order="C", copy=None)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_TRANSPOSE_DY)
        reshaped_dy = dy.reshape((_dim, _c), copy=False)
        self.x = self.x.reshape((-1, _dim), copy=False)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DW_MATMUL)
        np.matmul(self.x, reshaped_dy, out=self.dw.T)
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
        np.matmul(w, reshaped_dy, out=dx)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return np.asarray(dx.reshape(x_shape, copy=False), dtype=self.model.dtype, order='C', copy=None)
    # --- END _backward_pointwise_nchw --- #
