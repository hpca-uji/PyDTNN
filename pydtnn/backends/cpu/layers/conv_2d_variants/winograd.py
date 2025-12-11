from pydtnn.backends.cpu.layers.abstract.conv_2d_standard import Conv2DStandardCPU
from pydtnn.backends.cpu.utils.im2col_nchw_cython import im2col_nchw_cython
from pydtnn.backends.cpu.utils.im2row_nhwc_cython import im2row_nhwc_cython
from pydtnn.libs.libconvwinograd import ConvWinograd
from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum

import numpy as np

from pydtnn.utils.tensor import TensorFormat


class Conv2DWinogradCPU(Conv2DStandardCPU):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # convWinograd related attributes (will be initialized in initialize())
        self.cw: ConvWinograd = None  # type: ignore

    def initialize(self, prev_shape, x: np.ndarray | None = None):
        super().initialize(prev_shape, x)
        # ConvWinograd parameters
        self.cw = ConvWinograd(self.kh, self.kw, self.vstride, self.hstride,
                                self.vdilation, self.hdilation,
                                dtype=self.model.dtype, tensor_format=self.model.tensor_format,
                                debug=self.debug, parent_layer=self)
        
        match self.model.tensor_format:
            case TensorFormat.NCHW:
                self.forward = self._forward_cw_nchw
                self.backward = self._backward_cw_nchw
            case TensorFormat.NHWC:
                self.forward = self._forward_cw_nhwc
                self.backward = self._backward_cw_nhwc
            case _:
                raise NotImplementedError(f"\"{self.model.tensor_format}\" format not implemented.")
        # ---

    def _forward_cw_nhwc(self, x: np.ndarray) -> np.ndarray:
        """Version of the forward function that uses the convWinograd library"""

        self.cw_x = x

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CONVWINOGRAD)
        y: np.ndarray = self.cw.conv_winograd_nhwc(self.weights, x, biases=self.biases,
                                                vpadding=self.vpadding, hpadding=self.hpadding,
                                                vstride=self.vstride, hstride=self.hstride,
                                                vdilation=self.vdilation, hdilation=self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return y

    def _forward_cw_nchw(self, x: np.ndarray) -> np.ndarray:
        """Version of the forward function that uses the convWinograd library"""

        self.cw_x = x

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CONVWINOGRAD)
        y: np.ndarray = self.cw.conv_winograd_nchw(self.weights, x, biases=self.biases,
                                                vpadding=self.vpadding, hpadding=self.hpadding,
                                                vstride=self.vstride, hstride=self.hstride,
                                                vdilation=self.vdilation, hdilation=self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return y

    def _backward_cw_nhwc(self, dy: np.ndarray) -> np.ndarray:
        """Version of the backward function that uses the convWinograd library"""

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_IM2COL)

        self.x_rows = np.zeros(((dy.shape[0] * self.ho * self.wo), (self.ci * self.kh * self.kw)), dtype=self.model.dtype)
        im2row_nhwc_cython(self.cw_x, self.x_rows,
                           self.kh, self.kw, self.ho, self.wo,
                           self.vpadding, self.hpadding,
                           self.vstride, self.hstride, self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return self._backward_i2c_nhwc(dy)

    def _backward_cw_nchw(self, dy: np.ndarray) -> np.ndarray:
        """Version of the backward function that uses the convWinograd library"""
        n, c, _, _ = dy.shape
        self.x_cols = np.zeros((c * self.kh * self.kw, n * self.ho * self.wo))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_IM2COL)
        im2col_nchw_cython(self.cw_x,
                           self.x_cols,
                           self.kh, self.kw, 
                           self.ho, self.wo,
                           self.vpadding, self.hpadding,
                           self.vstride, self.hstride, 
                           self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return self._backward_i2c_nchw(dy)
