from pydtnn.backends.cpu.layers.conv_2d import Conv2DCPU
from pydtnn.backends.cpu.utils.depthwise_conv_nchw_cython import depthwise_conv_backward_nchw_cython, depthwise_conv_nchw_cython
from pydtnn.backends.cpu.utils.depthwise_conv_nhwc_cython import depthwise_conv_backward_nhwc_cython, depthwise_conv_nhwc_cython
from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum

import numpy as np

from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.tensor import TensorFormat


class Conv2DDepthwiseCPU(Conv2DCPU):

    def _initializing_special_parameters(self):
        super()._initializing_special_parameters()
        # Setting other parameters
        self.co = self.ci
        # Setting weights
        self.weights_shape = (self.ci, *self.filter_shape)
    # ---

    def initialize(self, prev_shape: ArrayShape, x: np.ndarray | None):
        super().initialize(prev_shape, x)

        match self.model.tensor_format:
            case TensorFormat.NCHW:
                self.forward = self._forward_depthwise_nchw
                self.backward = self._backward_nchw
                _y_shape = (self.model.batch_size, self.co, self.ho, self.wo)
                dx_shape = (self.model.batch_size, self.hi, self.wi, self.ci)
            case TensorFormat.NHWC:
                self.forward = self._forward_depthwise_nhwc
                self.backward = self._backward_nhwc
                _y_shape = (self.model.batch_size, self.ho, self.wo, self.co)
                dx_shape = (self.model.batch_size, self.hi, self.wi, self.ci)
            case _:
                _y_shape = None
                dx_shape = None
                raise NotImplementedError(f"Format \"{self.model.tensor_format}\" is not supported in \"Conv2DDepthwiseCPU\" layer.")
        # ---

        self.dx = np.zeros(shape=dx_shape, dtype=self.model.dtype, order="C")
        self._y = np.zeros(shape=_y_shape, dtype=self.model.dtype, order="C")
    # ---

    def _forward_depthwise_nhwc(self, x: np.ndarray) -> np.ndarray:
        """ Version of the forward that perform a depthwise convolution"""

        self.x = x
        y = self._y[:x.shape[0], ]
        y.fill(0)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_DEPTHWISE_CONV)
        depthwise_conv_nhwc_cython(x, self.weights, y, self.ho, self.wo,
                                   self.vpadding, self.hpadding,
                                   self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_SUM_BIASES)
            y: np.ndarray = y.reshape((self.co, -1), copy=False)
            for i in range(self.co):
                np.add(y[i], self.biases[i], out=y[i], 
                       dtype=self.model.dtype)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_RESHAPE_Y)
        y: np.ndarray = y.reshape((-1, self.ho, self.wo, self.co), order="C", copy=None)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return np.asarray(y, dtype=self.model.dtype, order='C', copy=None)
    
    def _forward_depthwise_nchw(self, x: np.ndarray) -> np.ndarray:
        """ Version of the forward that perform a depthwise convolution"""
        self.x = x
        y = self._y[:x.shape[0], ]
        y.fill(0)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_DEPTHWISE_CONV)
        depthwise_conv_nchw_cython(x, self.weights, y, self.ho, self.wo,
                                   self.vpadding, self.hpadding,
                                   self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_SUM_BIASES)
            y: np.ndarray = y.reshape((self.co, -1), copy=False)
            for i in range(self.co):
                np.add(y[i], self.biases[i], out=y[i], 
                       dtype=self.model.dtype)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_RESHAPE_Y)
        y: np.ndarray = y.reshape((-1, self.co, self.ho, self.wo), order="C", copy=None)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return np.asarray(y, dtype=self.model.dtype, order='C', copy=None)
    
    def _backward_nhwc(self, dy: np.ndarray) -> np.ndarray:
        
        dx = self.dx[:dy.shape[0], ]
        dx.fill(0)

        depthwise_conv_backward_nhwc_cython(dy, self.x, self.weights,
                                            dx, self.dw,
                                            self.vpadding, self.hpadding,
                                            self.vstride, self.hstride,
                                            self.vdilation, self.hdilation)

        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_SUM_BIASES)
            np.sum(dy, axis=(0, 1, 2), out=self.db)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return np.asarray(dx, dtype=self.model.dtype, order='C', copy=None)
    
    def _backward_nchw(self, dy: np.ndarray) -> np.ndarray:

        dx = self.dx[:dy.shape[0], ]
        dx.fill(0)

        depthwise_conv_backward_nchw_cython(dy, self.x, self.weights,
                                            dx, self.dw,
                                            self.vpadding, self.hpadding,
                                            self.vstride, self.hstride,
                                            self.vdilation, self.hdilation)

        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_SUM_BIASES)
            np.sum(dy, axis=(0, 2, 3), out=self.db)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return np.asarray(dx, dtype=self.model.dtype, order='C', copy=None)
    