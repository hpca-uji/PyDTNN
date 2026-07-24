"""Winograd-based 2D convolution layer implementation."""

import logging
from typing import Any

import numpy as np

from pydtnn.backends.cython.utils.im2col_nchw_cython import im2col_nchw_cython
from pydtnn.backends.cython.utils.im2row_nhwc_cython import im2row_nhwc_cython
from pydtnn.backends.numpy.layers.conv_2d import Conv2DNumpy
from pydtnn.backends.winograd.layers.abstract.conv_2d import AbstractConv2DWinograd
from pydtnn.libs.convWinograd import ConvWinograd
from pydtnn.tracers.events import (PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT,
                                   PYDTNN_OPS_EVENTS, OpsEventEnum)
from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.tensor import TensorFormat

__all__ = ("Conv2DWinograd",)

logger = logging.getLogger(__name__)


class Conv2DWinograd(Conv2DNumpy, AbstractConv2DWinograd):
    """2D Convolution layer utilizing Winograd algorithm for optimized computation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the Winograd convolution layer."""
        super().__init__(*args, **kwargs)
        # convWinograd related attributes (will be initialized in initialize())
        self.cw: ConvWinograd = None  # pyright: ignore[reportAttributeAccessIssue]

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray) -> None:
        """Initialize model parameters and select backend implementation based on tensor format."""
        super()._model_init(prev_shape, x)
        # ConvWinograd parameters
        self.cw = ConvWinograd(
            self.kh,
            self.kw,
            self.hstride,
            self.wstride,
            self.hdilation,
            self.wdilation,
            dtype=self.model.dtype,
            tensor_format=self.model.tensor_format,
            debug=self.debug,
            parent_layer=self,
        )

        match self.model.tensor_format:
            case TensorFormat.NCHW:
                self.forward = self._forward_cw_nchw
                self.backward = self._backward_cw_nchw
            case TensorFormat.NHWC:
                self.forward = self._forward_cw_nhwc
                self.backward = self._backward_cw_nhwc
            case _:
                raise NotImplementedError(f"{self.model.tensor_format} format not implemented.")

    def _forward_cw_nhwc(self, x: np.ndarray) -> np.ndarray:
        """Perform forward pass using Winograd algorithm for NHWC format."""

        self.cw_x = x
        w = np.asarray(self.weights, dtype=self.model.dtype)
        biases = np.asarray(self.biases, dtype=self.model.dtype) if self.use_bias else self.biases
        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT,
            self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.FORWARD_CONVWINOGRAD,
        )
        y: np.ndarray = self.cw.conv_winograd_nhwc(
            w,
            x,
            biases=biases,
            vpadding=self.hpadding,
            hpadding=self.wpadding,
            vstride=self.hstride,
            hstride=self.wstride,
            vdilation=self.hdilation,
            hdilation=self.wdilation,
        )
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return y

    def _forward_cw_nchw(self, x: np.ndarray) -> np.ndarray:
        """Perform forward pass using Winograd algorithm for NCHW format."""

        self.cw_x = x

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT,
            self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.FORWARD_CONVWINOGRAD,
        )
        w = np.asarray(self.weights, dtype=self.model.dtype)
        biases = np.asarray(self.biases, dtype=self.model.dtype) if self.use_bias else self.biases
        y: np.ndarray = self.cw.conv_winograd_nchw(
            w,
            x,
            biases=biases,
            vpadding=self.hpadding,
            hpadding=self.wpadding,
            vstride=self.hstride,
            hstride=self.wstride,
            vdilation=self.hdilation,
            hdilation=self.wdilation,
        )
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return y

    def _backward_cw_nhwc(self, dy: np.ndarray) -> np.ndarray:
        """Perform backward pass using im2row transformation for NHWC format."""

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.BACKWARD_IM2COL
        )

        self.x_rows = np.zeros(
            ((dy.shape[0] * self.ho * self.wo), (self.ci * self.kh * self.kw)),
            dtype=self.model.dtype,
        )
        im2row_nhwc_cython(
            self.cw_x,
            self.x_rows,
            self.kh,
            self.kw,
            self.ho,
            self.wo,
            self.hpadding,
            self.wpadding,
            self.hstride,
            self.wstride,
            self.hdilation,
            self.wdilation,
        )
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return self._backward_i2c_nhwc(dy)

    def _backward_cw_nchw(self, dy: np.ndarray) -> np.ndarray:
        """Perform backward pass using im2col transformation for NCHW format."""
        n, c, _, _ = dy.shape
        self.x_cols = np.zeros((c * self.kh * self.kw, n * self.ho * self.wo))
        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.BACKWARD_IM2COL
        )
        im2col_nchw_cython(
            self.cw_x,
            self.x_cols,
            self.kh,
            self.kw,
            self.ho,
            self.wo,
            self.hpadding,
            self.wpadding,
            self.hstride,
            self.wstride,
            self.hdilation,
            self.wdilation,
        )
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return self._backward_i2c_nchw(dy)
