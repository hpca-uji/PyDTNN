"""
NumPy backend implementation for pointwise 2D convolution layers.
"""

import logging
from typing import TYPE_CHECKING

from pydtnn.backends.numpy.layers.abstract.conv_2d import AbstractConv2DNumpy
from pydtnn.layers.conv_2d_pointwise import Conv2DPointwise
from pydtnn.libs import numpy as np
from pydtnn.tracers.events import (PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT,
                                   PYDTNN_OPS_EVENTS, PYDTNN_OPS_EVENT_enum)
from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.tensor import TensorFormat, format_reshape, format_transpose

__all__ = ("Conv2DPointwiseNumpy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


class Conv2DPointwiseNumpy(Conv2DPointwise, AbstractConv2DNumpy):
    """
    NumPy-based implementation of a pointwise 2D convolution layer.
    """

    def _export_weights_dw(self, key: str):
        """
        Exports weights or gradients to a standard format based on the model's tensor format.
        """
        value = getattr(self, key)

        match self.model.tensor_format:
            case TensorFormat.NHWC:
                # NHWC's src: ci, co
                # NCHW's dst: co, ci
                return np.asarray(
                    format_transpose(value, "IO", "OI"), dtype=np.float64, order="C", copy=True
                )
            case TensorFormat.NCHW:
                return np.asarray(value, dtype=np.float64, order="C", copy=True)
            case tensor_format:
                raise TypeError(f"Unsupported tensor format ({tensor_format})")

    def _import_weights_dw(self, key: str, value) -> None:
        """
        Imports weights or gradients into the layer, adjusting for the model's tensor format.
        """
        ary = getattr(self, key)

        match self.model.tensor_format:
            case TensorFormat.NHWC:
                # NCHW's src: co, ci
                # NHWC's dst: ci, co
                ary[:] = format_transpose(value, "OI", "IO")
                return
            case TensorFormat.NCHW:
                ary[:] = value
                return
            case tensor_format:
                raise TypeError(f"Unsupported tensor format ({tensor_format})")

    def _initializing_special_parameters(self):
        """
        Initializes layer-specific parameters including kernel dimensions and weight shapes.
        """
        super()._initializing_special_parameters()
        # Setting other parameters
        self.kh = self.kw = 1
        # Setting weights
        match self.model.tensor_format:
            case TensorFormat.NCHW:
                self.weights_shape = (self.co, self.ci)
            case TensorFormat.NHWC:
                self.weights_shape = (self.ci, self.co)
            case _:
                raise NotImplementedError(f"{self.model.tensor_format} format not implemented.")

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray | None = None) -> None:
        """
        Initializes model buffers and assigns forward/backward methods based on tensor format.
        """
        super()._model_init(prev_shape, x)
        match self.model.tensor_format:
            case TensorFormat.NCHW:
                self.forward = self._forward_nchw
                self.backward = self._backward_nchw
            case TensorFormat.NHWC:
                self.forward = self._forward_nhwc
                self.backward = self._backward_nhwc

        y_shape = self.model.encode_shape((self.model.batch_size, self.co, self.ho, self.wo))
        # NOTE: These attributes only store data, their values before the operation doesn't matter; they're initalized due avoid warnings in "LayerAndActivationBase.export".
        # self.dw (this one too, but it's initalized in Conv2DNumpy)
        self.y = np.zeros(shape=y_shape, dtype=self.model.dtype)
        self.memory_used += self.y.nbytes

        if not self.model.evaluate_only:
            self.dx = np.zeros(
                shape=(self.ci, self.model.batch_size * self.hi * self.wi), dtype=self.model.dtype
            )
            self.memory_used += self.dx.nbytes

    def _forward_nhwc(self, x: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass for NHWC tensor format.
        """
        self.x: np.ndarray = x

        y = self.y[: x.shape[0], :]
        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT,
            self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_POINTWISE_CONV,
        )
        np.matmul(x, self.weights, out=y, dtype=self.model.dtype)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.model.tracer.emit_event(
                PYDTNN_OPS_EVENT,
                self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_SUM_BIASES,
            )
            np.add(y, self.biases.reshape((1, 1, 1, self.co)), out=y, dtype=self.model.dtype)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return np.asarray(y, dtype=self.model.dtype, order="C")

    def _forward_nchw(self, x: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass for NCHW tensor format.
        """

        self.x: np.ndarray = x

        y: np.ndarray = self.y[: x.shape[0], :]

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT,
            self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_TRANSPOSE_Y,
        )
        y = format_transpose(y, TensorFormat.NCHW, TensorFormat.NHWC)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT,
            self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_POINTWISE_CONV,
        )
        np.matmul(
            format_transpose(x, TensorFormat.NCHW, TensorFormat.NHWC),
            self.weights.T,
            out=y,
            dtype=self.model.dtype,
        )
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT,
            self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_TRANSPOSE_Y,
        )
        y: np.ndarray = format_transpose(y, TensorFormat.NHWC, TensorFormat.NCHW)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.model.tracer.emit_event(
                PYDTNN_OPS_EVENT,
                self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_SUM_BIASES,
            )
            np.add(y, self.biases.reshape((1, self.co, 1, 1)), out=y, dtype=self.model.dtype)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(y, dtype=self.model.dtype, order="C")

    def _backward_nhwc(self, dy: np.ndarray) -> np.ndarray:
        """
        Performs the backward pass for NHWC tensor format.
        """
        _n, _h, _w, _c = dy.shape
        _dim = _n * _h * _w
        x_shape = self.x.shape
        dx: np.ndarray = np.asarray(self.dx[:, :_dim], dtype=self.model.dtype, order="C")

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT,
            self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_TRANSPOSE_DY,
        )
        reshaped_dy: np.ndarray = dy.reshape((_dim, _c))
        self.x: np.ndarray = self.x.reshape((-1, _dim))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DW_MATMUL
        )
        np.matmul(self.x, reshaped_dy, out=self.dw, dtype=self.model.dtype)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.model.tracer.emit_event(
                PYDTNN_OPS_EVENT,
                self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_SUM_BIASES,
            )
            np.sum(dy, axis=(0, 1, 2), out=self.db)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT,
            self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_TRANSPOSE_W,
        )
        w = self.weights.reshape((self.co, -1)).T
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        reshaped_dy: np.ndarray = dy.reshape((self.co, -1))

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_MATMUL
        )
        np.matmul(w, reshaped_dy, out=dx, dtype=self.model.dtype)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        dx = dx.reshape(x_shape)
        return np.asarray(dx, dtype=self.model.dtype, order="C")

    def _backward_nchw(self, dy: np.ndarray) -> np.ndarray:
        """
        Performs the backward pass for NCHW tensor format.
        """
        _n, _c, _h, _w = dy.shape
        _dim = _n * _h * _w
        x_shape = self.x.shape
        dx = np.asarray(self.dx[:, :_dim], dtype=self.model.dtype, order="C")

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT,
            self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_TRANSPOSE_DY,
        )
        reshaped_dy = format_transpose(dy, TensorFormat.NCHW, TensorFormat.NHWC).reshape((_dim, _c))
        self.x = format_transpose(self.x, TensorFormat.NCHW, TensorFormat.NHWC).reshape((-1, _dim))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DW_MATMUL
        )
        np.matmul(self.x, reshaped_dy, out=self.dw.T, dtype=self.model.dtype)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.model.tracer.emit_event(
                PYDTNN_OPS_EVENT,
                self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_SUM_BIASES,
            )
            np.sum(dy, axis=(0, 2, 3), out=self.db)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT,
            self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_TRANSPOSE_W,
        )
        w = format_transpose(self.weights, "IO", "OI").reshape((self.co, -1)).T
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        reshaped_dy: np.ndarray = format_transpose(dy, TensorFormat.NCHW, TensorFormat.NHWC).reshape((self.co, -1))

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_MATMUL
        )
        np.matmul(w, reshaped_dy, out=dx, dtype=self.model.dtype)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        dx = dx.reshape(format_reshape(x_shape, TensorFormat.NCHW, TensorFormat.NHWC))
        return np.asarray(format_transpose(dx, TensorFormat.NHWC, TensorFormat.NCHW), dtype=self.model.dtype, order="C")
