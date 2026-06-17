"""
NumPy backend implementation for pointwise 2D convolution layers.
"""

import logging
from typing import TYPE_CHECKING

from pydtnn.backends.numpy.layers.conv_2d_pointwise import Conv2DPointwiseNumpy
from pydtnn.libs import numpy as np
from pydtnn.tracers.events import (PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT,
                                   PYDTNN_OPS_EVENTS, PYDTNN_OPS_EVENT_enum)
from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.tensor import TensorFormat
from pydtnn.backends.cython.utils.pointwise_conv_nchw_cython import fwd_pointwise_conv_cython_nchw, bwd_pointwise_conv_cython_nchw
from pydtnn.backends.cython.utils.pointwise_conv_nhwc_cython import fwd_pointwise_conv_cython_nhwc, bwd_pointwise_conv_cython_nhwc

__all__ = ("Conv2DPointwiseNumpy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


class Conv2DPointwiseCython(Conv2DPointwiseNumpy):
    """
    Cython-based implementation of a pointwise 2D convolution layer.
    """

    def _forward_nhwc(self, x: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass for NHWC tensor format.
        """
        self.x: np.ndarray = x
        y: np.ndarray = self.y[: x.shape[0], :]

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_POINTWISE_CONV)
        fwd_pointwise_conv_cython_nhwc(x, self.weights, y,
                                       self.hpadding, self.wpadding,
                                       self.hstride, self.wstride,
                                       self.hdilation, self.wdilation)
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

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_TRANSPOSE_Y)
        fwd_pointwise_conv_cython_nchw(x, self.weights, y,
                                       self.hpadding, self.wpadding,
                                       self.hstride, self.wstride,
                                       self.hdilation, self.wdilation)
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
        #dx: np.ndarray = np.asarray(self.dx[:, :_dim], dtype=self.model.dtype, order="C")
        dx = np.asarray(self.dx[: self.x.size].reshape(self.x.shape), dtype=self.model.dtype, order="C")

        bwd_pointwise_conv_cython_nhwc(dy, self.x, self.weights, dx, self.dw,
                                       self.hpadding, self.wpadding,
                                       self.hstride, self.wstride,
                                       self.hdilation, self.wdilation)

        if self.use_bias:
            self.model.tracer.emit_event(
                PYDTNN_OPS_EVENT,
                self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_SUM_BIASES,
            )
            np.sum(dy, axis=(0, 1, 2), out=self.db)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return np.asarray(dx, dtype=self.model.dtype, order="C")

    def _backward_nchw(self, dy: np.ndarray) -> np.ndarray:
        """
        Performs the backward pass for NCHW tensor format.
        """
        dx = np.asarray(self.dx[: self.x.size].reshape(self.x.shape), dtype=self.model.dtype, order="C")


        bwd_pointwise_conv_cython_nchw(dy, self.x, self.weights, dx, self.dw,
                                       self.hpadding, self.wpadding,
                                       self.hstride, self.wstride,
                                       self.hdilation, self.wdilation)
    
        if self.use_bias:
            self.model.tracer.emit_event(
                PYDTNN_OPS_EVENT,
                self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_SUM_BIASES,
            )
            np.sum(dy, axis=(0, 2, 3), out=self.db)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return np.asarray(dx, dtype=self.model.dtype, order="C")
