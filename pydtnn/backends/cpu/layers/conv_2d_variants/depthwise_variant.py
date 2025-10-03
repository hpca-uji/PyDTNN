from abc import ABC

from pydtnn.cython_modules import depthwise_conv_nchw_cython, depthwise_conv_backward_nchw_cython, \
                                  depthwise_conv_nhwc_cython, depthwise_conv_backward_nhwc_cython
from pydtnn.layers import Conv2D
from pydtnn.tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum


import numpy as np

class DepthwiseVariant(Conv2D, ABC):
    # NOTE: Attributes defined in conv_2d_cpu.
    dw:np.ndarray
    db:np.ndarray
    #---

    def _forward_depthwise_nhwc(self, x:np.ndarray) -> np.ndarray:
        """ Version of the forward that perform a depthwise convolution"""
        
        self.x = x
        y = np.zeros(shape=(x.shape[0], self.ho, self.wo, self.co), dtype=self.model.dtype)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_DEPTHWISE_CONV)
        depthwise_conv_nhwc_cython(x, self.weights, y, self.ho, self.wo,
                                   self.vpadding, self.hpadding,
                                   self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_SUM_BIASES)
            y:np.ndarray = y.reshape((self.co, -1), copy=False)
            for i in range(self.co):
                y[i] += self.biases[i]
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_RESHAPE_Y)
        y:np.ndarray = y.reshape((-1, self.ho, self.wo, self.co), order="C", copy=None)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        
        return y
    # --- END _forward_depthwise_nhwc --- #

    def _forward_depthwise_nchw(self, x:np.ndarray) -> np.ndarray:
        """ Version of the forward that perform a depthwise convolution"""
        self.x = x
        y = np.zeros(shape=(x.shape[0], self.co, self.ho, self.wo), dtype=self.model.dtype)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_DEPTHWISE_CONV)
        depthwise_conv_nchw_cython(x, self.weights, y, self.ho, self.wo,
                                   self.vpadding, self.hpadding,
                                   self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_SUM_BIASES)
            y:np.ndarray = y.reshape((self.co, -1), copy=False)
            for i in range(self.co):
                y[i] += self.biases[i]
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_RESHAPE_Y)
        y:np.ndarray = y.reshape((-1, self.co, self.ho, self.wo), order="C", copy=None)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return y
    # --- END _forward_depthwise_nchw --- #

    def _backward_depthwise_nhwc(self, dy:np.ndarray) -> np.ndarray:
        
        dx = np.zeros(shape=(dy.shape[0], self.hi, self.wi, self.ci), dtype=self.model.dtype)
        

        depthwise_conv_backward_nhwc_cython(dy, self.x, self.weights,
                                            dx, self.dw,
                                            self.vpadding, self.hpadding,
                                            self.vstride, self.hstride,
                                            self.vdilation, self.hdilation)

        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_SUM_BIASES)
            np.sum(dy, axis=(0, 1, 2), out=self.db)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return dx
    # --- END _backward_depthwise_nhwc --- #

    def _backward_depthwise_nchw(self, dy:np.ndarray) -> np.ndarray:
        
        dx = np.zeros(shape=(dy.shape[0], self.ci, self.hi, self.wi), dtype=self.model.dtype)

        depthwise_conv_backward_nchw_cython(dy, self.x, self.weights,
                                            dx, self.dw,
                                            self.vpadding, self.hpadding,
                                            self.vstride, self.hstride,
                                            self.vdilation, self.hdilation)

        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_SUM_BIASES)
            np.sum(dy, axis=(0, 2, 3), out=self.db)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return dx
    # --- END _backward_depthwise_nchw --- #
