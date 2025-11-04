from pydtnn.cython.im2col_nchw_cython import im2col_nchw_cython
from pydtnn.cython.im2row_nhwc_cython import im2row_nhwc_cython
from pydtnn.backends.cpu.layers.conv_2d_variants.i2c_variant import I2CVariant
from pydtnn.backends.cpu.libs.conv_winograd import ConvWinograd
from pydtnn.model import Model
from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum
#from pydtnn.utils.types import Array

import numpy as np


class ConvWinogradVariant[T: np.ndarray](I2CVariant[T]):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # convWinograd related attributes (will be initialized in initialize())
        self.cw: ConvWinograd = None  # type: ignore
        self.cw_constraints_fulfilled: bool = None  # type: ignore

    def initialize(self, prev_shape, x: T | None = None):
        super().initialize(prev_shape, x)
        # ConvWinograd parameters
        if self.model.enable_conv_winograd:
            try:
                self.cw = ConvWinograd(self.kh, self.kw, self.vstride, self.hstride,
                                       self.vdilation, self.hdilation,
                                       dtype=self.model.dtype, tensor_format=self.model.tensor_format,
                                       debug=self.debug, parent_layer=self)
            except NotImplementedError:
                self.cw_constraints_fulfilled = False
            else:
                self.cw_constraints_fulfilled = True

    def _forward_cw_nhwc(self, x: np.ndarray) -> np.ndarray:
        """Version of the forward function that uses the convWinograd library"""

        if self.model.mode is Model.Mode.TRAIN:
            self.cw_x = x

        biases_vector = self.biases if self.use_bias else None

        # FIXME: The are some errors with the bias support, we ignore it for now
        biases_vector = None

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CONVWINOGRAD)
        y: np.ndarray = self.cw.conv_winograd_nhwc(self.weights, x, biases=biases_vector,
                                                vpadding=self.vpadding, hpadding=self.hpadding,
                                                vstride=self.vstride, hstride=self.hstride,
                                                vdilation=self.vdilation, hdilation=self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return y

    def _forward_cw_nchw(self, x: np.ndarray) -> np.ndarray:
        """Version of the forward function that uses the convWinograd library"""

        if self.model.mode is Model.Mode.TRAIN:
            self.cw_x = x

        biases_vector = self.biases if self.use_bias else None

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CONVWINOGRAD)
        y: np.ndarray = self.cw.conv_winograd_nchw(self.weights, x, biases=biases_vector,
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
