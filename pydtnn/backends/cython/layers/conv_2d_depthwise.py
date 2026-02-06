from pydtnn.backends.numpy.layers.conv_2d_depthwise import Conv2DDepthwiseNumpy
from pydtnn.backends.cython.utils.depthwise_conv_nchw_cython import depthwise_conv_backward_nchw_cython, depthwise_conv_nchw_cython
from pydtnn.backends.cython.utils.depthwise_conv_nhwc_cython import depthwise_conv_backward_nhwc_cython, depthwise_conv_nhwc_cython
from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum

from pydtnn.libs import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np

class Conv2DDepthwiseCython(Conv2DDepthwiseNumpy):

    def _initializing_special_parameters(self):
        super()._initializing_special_parameters()
        # Setting other parameters
        self.co = self.ci
        # Setting weights
        self.weights_shape = (self.ci, *self.filter_shape)
    # ---

    def _forward_depthwise_nhwc(self, x: np.ndarray) -> np.ndarray:
        """ Version of the forward that perform a depthwise convolution"""

        self.x = x
        y: np.ndarray = self._y[:x.shape[0], ]
        y.fill(0)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_DEPTHWISE_CONV)
        depthwise_conv_nhwc_cython(x, self.weights, y, self.ho, self.wo,
                                   self.hpadding, self.wpadding,
                                   self.hstride, self.wstride, self.hdilation, self.wdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_SUM_BIASES)
            y: np.ndarray = y.reshape((self.co, -1), copy=False)
            for i in range(self.co):
                np.add(y[i], self.biases[i], out=y[i],
                       dtype=self.model.dtype)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_RESHAPE_Y)
        y: np.ndarray = y.reshape((-1, self.ho, self.wo, self.co))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return np.asarray(y, dtype=self.model.dtype, order="C")

    def _forward_depthwise_nchw(self, x: np.ndarray) -> np.ndarray:
        """ Version of the forward that perform a depthwise convolution"""
        self.x = x
        y: np.ndarray = self._y[:x.shape[0], ]
        y.fill(0)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_DEPTHWISE_CONV)
        depthwise_conv_nchw_cython(x, self.weights, y, self.ho, self.wo,
                                   self.hpadding, self.wpadding,
                                   self.hstride, self.wstride, self.hdilation, self.wdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_SUM_BIASES)
            y: np.ndarray = y.reshape((self.co, -1), copy=False)
            for i in range(self.co):
                np.add(y[i], self.biases[i], out=y[i],
                       dtype=self.model.dtype)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_RESHAPE_Y)
        y: np.ndarray = y.reshape((-1, self.co, self.ho, self.wo))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return np.asarray(y, dtype=self.model.dtype, order="C")

    def _backward_nhwc(self, dy: np.ndarray) -> np.ndarray:

        dx: np.ndarray = self.dx[:dy.shape[0], ]
        dx.fill(0)

        depthwise_conv_backward_nhwc_cython(dy, self.x, self.weights,
                                            dx, self.dw,
                                            self.hpadding, self.wpadding,
                                            self.hstride, self.wstride,
                                            self.hdilation, self.wdilation)

        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_SUM_BIASES)
            np.sum(dy, axis=(0, 1, 2), out=self.db)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return np.asarray(dx, dtype=self.model.dtype, order="C")

    def _backward_nchw(self, dy: np.ndarray) -> np.ndarray:

        dx: np.ndarray = self.dx[:dy.shape[0], ]
        dx.fill(0)

        depthwise_conv_backward_nchw_cython(dy, self.x, self.weights,
                                            dx, self.dw,
                                            self.hpadding, self.wpadding,
                                            self.hstride, self.wstride,
                                            self.hdilation, self.wdilation)

        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_SUM_BIASES)
            np.sum(dy, axis=(0, 2, 3), out=self.db)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return np.asarray(dx, dtype=self.model.dtype, order="C")
