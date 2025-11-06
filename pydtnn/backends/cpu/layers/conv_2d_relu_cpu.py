from pydtnn.backends.cpu.layers.conv_2d_cpu import Conv2DCPU
from pydtnn.layers.conv_2d_relu import Conv2DRelu
from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum
from pydtnn.utils.types import ArrayShape

import numpy as np

# Next no inspection is because Conv2D _backward_depthwise and _backward_pointwise being considered as abstract methods
# noinspection PyAbstractClass


class Conv2DReluCPU(Conv2DCPU, Conv2DRelu[np.ndarray]):

    def initialize(self, prev_shape: ArrayShape, x: np.ndarray | None = None) -> None:
        super().initialize(prev_shape, x)
        self.forward = {"_forward_cg_nchw": self._forward_nchw_cg,
                        "_forward_cg_nhwc": self._forward_nhwc_cg,
                        "_forward_cw_nchw": self._forward_nchw_cw}[self.forward.__name__]
        self.backward = self._backward

    def _forward_nchw_cg(self, x: np.ndarray) -> np.ndarray:
        """Version of the forward function that uses the convGemm + Relu"""
        biases_vector = self.biases if self.use_bias else None

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CONVGEMM)
        res: np.ndarray = self.cg.conv_gemm_nchw(self.weights, x, biases=None,
                                              vpadding=self.vpadding, hpadding=self.hpadding,
                                              vstride=self.vstride, hstride=self.hstride,
                                              vdilation=self.vdilation, hdilation=self.hdilation,
                                              biases_vector=biases_vector, relu=True)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(res, dtype=self.model.dtype, order='C', copy=None)

    def _forward_nhwc_cg(self, x: np.ndarray) -> np.ndarray:
        """Version of the forward function that uses the convGemm + Relu"""
        biases_vector = self.biases if self.use_bias else None

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CONVGEMM)
        res: np.ndarray = self.cg.conv_gemm_nhwc(self.weights, x, biases=None,
                                              vpadding=self.vpadding, hpadding=self.hpadding,
                                              vstride=self.vstride, hstride=self.hstride,
                                              vdilation=self.vdilation, hdilation=self.hdilation,
                                              biases_vector=biases_vector, relu=True)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(res, dtype=self.model.dtype, order='C', copy=None)

    def _forward_nchw_cw(self, x: np.ndarray) -> np.ndarray:
        """Version of the forward function that uses the convWinograd + Relu"""
        biases_vector = self.biases if self.use_bias else None

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CONVGEMM)
        y: np.ndarray = self.cw.conv_winograd_nchw(self.weights, x, biases_vector,
                                                vpadding=self.vpadding, hpadding=self.hpadding,
                                                vstride=self.vstride, hstride=self.hstride,
                                                vdilation=self.vdilation, hdilation=self.hdilation,
                                                relu=True)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return np.asarray(y, dtype=self.model.dtype, order='C', copy=None)

    def _backward(self, dy: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Use a real backwards variant!")
