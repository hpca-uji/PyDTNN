from pydtnn.backends.cpu.layers import LayerCPU
from pydtnn.layers import Conv2DBatchNormalization
from pydtnn.model import Model
from pydtnn.tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum

from numpy import ndarray, asarray


class Conv2DBatchNormalizationCPU(LayerCPU, Conv2DBatchNormalization):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize(self, from_parent_dict: dict = None, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self.forward = {"_forward_nchw_cw": self._forward_nchw_cw,
                        "_forward_nchw_cg": self._forward_nchw_cg,
                        "_forward_nhwc_cg": self._forward_nhwc_cg}[from_parent_dict["forward"].__name__]
        # self.forward = {"_forward_nchw_cw": self._forward_nchw_cw, \
        #                 "_forward_nchw_best_of": self._forward_nchw_cw}[from_parent_dict["forward"].__name__]
        self.weights = from_parent_dict["weights"]
        self.biases = from_parent_dict["biases"]

    def forward(self, x: ndarray) -> ndarray:
        """This is a fake forward function. It will be masked on initialization by a _forward implementation"""
        pass

    def _forward_nchw_cw(self, x: ndarray) -> ndarray:
        """Version of the forward function that uses the convWinograd + BatchNorm + """

        if self.model.mode is Model.Mode.TRAIN:
            raise SystemExit("Sorry, fused layers cannot be used in training mode!")

        biases_vector = self.biases if self.use_bias else None
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CONVGEMM)
        y: ndarray = self.cw.conv_winograd_nchw(self.weights, x, biases_vector,
                                                vpadding=self.vpadding, hpadding=self.hpadding,
                                                vstride=self.vstride, hstride=self.hstride,
                                                vdilation=self.vdilation, hdilation=self.hdilation,
                                                relu=False, bn=True,
                                                running_mean=self.running_mean,
                                                inv_std=self.inv_std, gamma=self.gamma, beta=self.beta)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return asarray(y, dtype=self.model.dtype, order='C', copy=None)

    def _forward_nchw_cg(self, x: ndarray) -> ndarray:
        """Version of the forward function that uses the convGemm + BatchNorm"""

        if self.model.mode is Model.Mode.TRAIN:
            raise SystemExit("Sorry, fused layers cannot be used in training mode!")

        biases_vector = self.biases if self.use_bias else None
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CONVGEMM)
        res: ndarray = self.cg.conv_gemm_nchw(self.weights, x, biases=None,
                                              vpadding=self.vpadding, hpadding=self.hpadding,
                                              vstride=self.vstride, hstride=self.hstride,
                                              vdilation=self.vdilation, hdilation=self.hdilation,
                                              biases_vector=biases_vector, bn_running_mean=self.running_mean,
                                              bn_inv_std=self.inv_std, bn_gamma=self.gamma, bn_beta=self.beta, relu=False)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return asarray(res, dtype=self.model.dtype, order='C', copy=None)

    def _forward_nhwc_cg(self, x: ndarray) -> ndarray:
        """Version of the forward function that uses the convGemm + BatchNorm"""

        if self.model.mode is Model.Mode.TRAIN:
            raise RuntimeError("Fused layers cannot be used in training mode!")

        biases_vector = self.biases if self.use_bias else None
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CONVGEMM)
        res: ndarray = self.cg.conv_gemm_nhwc(self.weights, x, biases=None,
                                              vpadding=self.vpadding, hpadding=self.hpadding,
                                              vstride=self.vstride, hstride=self.hstride,
                                              vdilation=self.vdilation, hdilation=self.hdilation,
                                              biases_vector=biases_vector, bn_running_mean=self.running_mean,
                                              bn_inv_std=self.inv_std, bn_gamma=self.gamma, bn_beta=self.beta, relu=False)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return asarray(res, dtype=self.model.dtype, order='C', copy=None)

    def backward(self, x: ndarray) -> ndarray:
        raise SystemExit(f"Backward method of {self.__class__.__name__} should not be called")
