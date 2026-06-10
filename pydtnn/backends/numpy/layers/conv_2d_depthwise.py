"""Numpy backend implementation for 2D depthwise convolution layers."""

import logging
from typing import TYPE_CHECKING

from pydtnn.backends.numpy.layers.abstract.conv_2d import AbstractConv2DNumpy
from pydtnn.layers.conv_2d_depthwise import Conv2DDepthwise
from pydtnn.libs import numpy as np
from pydtnn.tracers.events import (PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT,
                                   PYDTNN_OPS_EVENTS, PYDTNN_OPS_EVENT_enum)
from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.tensor import TensorFormat

__all__ = ("Conv2DDepthwiseNumpy",)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import numpy as np


class Conv2DDepthwiseNumpy(AbstractConv2DNumpy, Conv2DDepthwise):
    """Numpy-based implementation of a 2D depthwise convolution layer."""

    def _initializing_special_parameters(self):
        """Initialize layer-specific parameters for depthwise convolution."""
        super()._initializing_special_parameters()
        # Setting other parameters
        self.co = self.ci
        # Setting weights
        self.weights_shape = (self.ci, *self.filter_shape)

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray | None):
        """Initialize model buffers and select forward/backward implementations."""
        super()._model_init(prev_shape, x)

        match self.model.tensor_format:
            case TensorFormat.NCHW:
                self.forward = self._forward_nchw
                self.backward = self._backward_nchw
            case TensorFormat.NHWC:
                self.forward = self._forward_nhwc
                self.backward = self._backward_nhwc
            case _:
                _y_shape = None
                dx_shape = None
                raise NotImplementedError(
                    f"Format {
                        self.model.tensor_format
                    } is not supported in Conv2DDepthwiseNumpy layer."
                )

        _y_shape = self.model.encode_shape((self.model.batch_size, self.co, self.ho, self.wo))
        dx_shape = self.model.encode_shape((self.model.batch_size, self.ci, self.hi, self.wi))

        self._y = np.zeros(shape=_y_shape, dtype=self.model.dtype)
        self.memory_used += self._y.nbytes

        if not self.model.evaluate_only:
            self.dx = np.zeros(shape=dx_shape, dtype=self.model.dtype)
            self.memory_used += self.dx.nbytes

    def _export_weights_dw(self, key: str):
        """Export weights to a standard numpy array."""
        value = getattr(self, key)
        return np.asarray(value, dtype=np.float64, order="C", copy=True)

    def _import_weights_dw(self, key: str, value) -> None:
        """Import weights from a numpy array into the layer."""
        ary = getattr(self, key)
        ary[:] = value

    def _conv_fwd_nhwc(self, x: np.ndarray, y: np.ndarray) -> None:
        """Perform NHWC depthwise convolution forward pass."""
        for nn in range(x.shape[0]):
            for cc in range(self.ci):
                for ii in range(self.kh):
                    for jj in range(self.kw):
                        for xx in range(self.ho):
                            x_x = self.hstride * xx + self.hdilation * ii - self.hpadding
                            if 0 <= x_x < self.hi:
                                for yy in range(self.wo):
                                    x_y = self.wstride * yy + self.wdilation * jj - self.wpadding
                                    if 0 <= x_y < self.wi:
                                        y[nn, cc, xx, yy] += (
                                            self.weights[cc, ii, jj] * x[nn, cc, x_x, x_y]
                                        )

    def _conv_fwd_nchw(self, x: np.ndarray, y: np.ndarray) -> None:
        """Perform NCHW depthwise convolution forward pass."""
        for nn in range(x.shape[0]):
            for ii in range(self.kh):
                for jj in range(self.kw):
                    for cc in range(self.ci):
                        for xx in range(self.ho):
                            x_x = self.hstride * xx + self.hdilation * ii - self.hpadding
                            if 0 <= x_x < self.hi:
                                for yy in range(self.wo):
                                    x_y = self.wstride * yy + self.wdilation * jj - self.wpadding
                                    if 0 <= x_y < self.wi:
                                        y[nn, xx, yy, cc] += (
                                            self.weights[cc, ii, jj] * x[nn, x_x, x_y, cc]
                                        )

    def _conv_bwd_nhwc(self, dx: np.ndarray, dy: np.ndarray) -> None:
        """Perform NHWC depthwise convolution backward pass."""
        for cc in range(self.ci):
            for ii in range(self.kh):
                for jj in range(self.kw):
                    for nn in range(dy.shape[0]):
                        val_k = self.weights[cc, ii, jj]
                        for xx in range(self.ho):
                            x_x = self.hstride * xx + self.hdilation * ii - self.hpadding
                            if 0 <= x_x < self.hi:
                                for yy in range(self.wo):
                                    x_y = self.wstride * yy + self.wdilation * jj - self.wpadding
                                    val_dy = dy[nn, xx, yy, cc]
                                    if 0 <= x_y < self.wi:
                                        self.dw[cc, ii, jj] = self.x[nn, x_x, x_y, cc] * val_dy
                                        dx[nn, x_x, x_y, cc] += val_k * val_dy

    def _conv_bwd_nchw(self, dx: np.ndarray, dy: np.ndarray) -> None:
        """Perform NCHW depthwise convolution backward pass."""
        for cc in range(self.ci):
            for ii in range(self.kh):
                for jj in range(self.kw):
                    for nn in range(dy.shape[0]):
                        val_k = self.weights[cc, ii, jj]
                        for xx in range(self.ho):
                            x_x = self.hstride * xx + self.hdilation * ii - self.hpadding
                            if 0 <= x_x < self.hi:
                                for yy in range(self.wo):
                                    x_y = self.wstride * yy + self.wdilation * jj - self.wpadding
                                    val_dy = dy[nn, cc, xx, yy]
                                    if 0 <= x_y < self.wi:
                                        self.dw[cc, ii, jj] = self.x[nn, cc, x_x, x_y] * val_dy
                                        dx[nn, cc, x_x, x_y] += val_k * val_dy

    def _forward_nhwc(self, x: np.ndarray) -> np.ndarray:
        """Version of the forward that perform a depthwise convolution"""

        self.x = x
        y: np.ndarray = np.ascontiguousarray(self._y[: x.shape[0],], dtype=self.model.dtype)
        y.fill(0)

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT,
            self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_DEPTHWISE_CONV,
        )
        self._conv_fwd_nhwc(x, y)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.model.tracer.emit_event(
                PYDTNN_OPS_EVENT,
                self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_SUM_BIASES,
            )
            y: np.ndarray = y.reshape((self.co, -1))
            for i in range(self.co):
                np.add(y[i], self.biases[i], out=y[i], dtype=self.model.dtype)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_RESHAPE_Y
        )
        y: np.ndarray = y.reshape((-1, self.ho, self.wo, self.co))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return np.asarray(y, dtype=self.model.dtype, order="C")

    def _forward_nchw(self, x: np.ndarray) -> np.ndarray:
        """Version of the forward that perform a depthwise convolution"""
        self.x = x
        y: np.ndarray = np.ascontiguousarray(self._y[: x.shape[0],], dtype=self.model.dtype)
        y.fill(0)

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT,
            self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_DEPTHWISE_CONV,
        )
        self._conv_fwd_nchw(x, y)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.model.tracer.emit_event(
                PYDTNN_OPS_EVENT,
                self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_SUM_BIASES,
            )
            y: np.ndarray = y.reshape((self.co, -1))
            for i in range(self.co):
                np.add(y[i], self.biases[i], out=y[i], dtype=self.model.dtype)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_RESHAPE_Y
        )
        y: np.ndarray = y.reshape((-1, self.co, self.ho, self.wo))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return np.asarray(y, dtype=self.model.dtype, order="C")

    def _backward_nhwc(self, dy: np.ndarray) -> np.ndarray:
        """Perform NHWC backward pass for depthwise convolution."""

        dx: np.ndarray = np.ascontiguousarray(self.dx[: dy.shape[0],], dtype=self.model.dtype)
        dx.fill(0)

        self._conv_bwd_nhwc(dx, dy)

        if self.use_bias:
            self.model.tracer.emit_event(
                PYDTNN_OPS_EVENT,
                self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_SUM_BIASES,
            )
            np.sum(dy, axis=(0, 1, 2), out=self.db)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return np.asarray(dx, dtype=self.model.dtype, order="C")

    def _backward_nchw(self, dy: np.ndarray) -> np.ndarray:
        """Perform NCHW backward pass for depthwise convolution."""

        dx: np.ndarray = np.ascontiguousarray(self.dx[: dy.shape[0],], dtype=self.model.dtype)
        dx.fill(0)

        self._conv_bwd_nhwc(dx, dy)

        if self.use_bias:
            self.model.tracer.emit_event(
                PYDTNN_OPS_EVENT,
                self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_SUM_BIASES,
            )
            np.sum(dy, axis=(0, 2, 3), out=self.db)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return np.asarray(dx, dtype=self.model.dtype, order="C")
