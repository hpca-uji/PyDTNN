from pydtnn.backends.cpu.layers.layer_cpu import LayerCPU
from pydtnn.layers.conv_2d_relu import Conv2DRelu
from pydtnn.model import Model
from pydtnn.tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum

from numpy import ndarray, asarray

# Next no inspection is because Conv2D _backward_depthwise and _backward_pointwise being considered as abstract methods
# noinspection PyAbstractClass


class Conv2DReluCPU(LayerCPU, Conv2DRelu):

    def initialize(self, from_parent_dict=None, *args, **kwargs) -> None:
        super().initialize(args, kwargs)
        self.forward = {"_forward_nchw_cg": self._forward_nchw_cg,
                        "_forward_nhwc_cg": self._forward_nhwc_cg,
                        "_forward_nchw_cw": self._forward_nchw_cw}[from_parent_dict["forward"].__name__]
        self.weights = from_parent_dict["weights"]
        self.biases = from_parent_dict["biases"]

    def forward(self, x: ndarray) -> ndarray:
        """This is a fake forward function. It will be masked on initialization by a _forward implementation"""
        pass

    def _forward_nchw_cg(self, x: ndarray) -> ndarray:
        """Version of the forward function that uses the convGemm + Relu"""

        if self.model.mode is Model.Mode.TRAIN:
            raise SystemExit("Sorry, fused layers cannot be used in training mode!")

        biases_vector = self.biases if self.use_bias else None

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CONVGEMM)
        res: ndarray = self.cg.conv_gemm_nchw(self.weights, x, biases=None,
                                              vpadding=self.vpadding, hpadding=self.hpadding,
                                              vstride=self.vstride, hstride=self.hstride,
                                              vdilation=self.vdilation, hdilation=self.hdilation,
                                              biases_vector=biases_vector, relu=True)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return asarray(res, dtype=self.model.dtype, order='C', copy=None)

    def _forward_nhwc_cg(self, x: ndarray) -> ndarray:
        """Version of the forward function that uses the convGemm + Relu"""

        if self.model.mode is Model.Mode.TRAIN:
            raise RuntimeError("Fused layers cannot be used in training mode!")

        biases_vector = self.biases if self.use_bias else None

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CONVGEMM)
        res: ndarray = self.cg.conv_gemm_nhwc(self.weights, x, biases=None,
                                              vpadding=self.vpadding, hpadding=self.hpadding,
                                              vstride=self.vstride, hstride=self.hstride,
                                              vdilation=self.vdilation, hdilation=self.hdilation,
                                              biases_vector=biases_vector, relu=True)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return asarray(res, dtype=self.model.dtype, order='C', copy=None)

    def _forward_nchw_cw(self, x: ndarray) -> ndarray:
        """Version of the forward function that uses the convWinograd + Relu"""

        if self.model.mode is Model.Mode.TRAIN:
            raise SystemExit("Sorry, fused layers cannot be used in training mode!")

        biases_vector = self.biases if self.use_bias else None

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CONVGEMM)
        y: ndarray = self.cw.conv_winograd_nchw(self.weights, x, biases_vector,
                                                vpadding=self.vpadding, hpadding=self.hpadding,
                                                vstride=self.vstride, hstride=self.hstride,
                                                vdilation=self.vdilation, hdilation=self.hdilation,
                                                relu=True)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return asarray(y, dtype=self.model.dtype, order='C', copy=None)

    def backward(self, x: ndarray) -> ndarray:
        raise SystemExit(f"Backward method of {self.__class__.__name__} should not be called")
