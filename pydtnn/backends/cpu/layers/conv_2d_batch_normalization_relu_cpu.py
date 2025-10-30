from pydtnn.backends.cpu.layers.layer_cpu import LayerCPU
from pydtnn.layers.conv_2d_batch_normalization_relu import Conv2DBatchNormalizationRelu
from pydtnn.model import Model
from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum

import numpy as np
import abc

class Conv2DBatchNormalizationReluCPU(LayerCPU, Conv2DBatchNormalizationRelu[np.ndarray]):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # TODO: Check "from_parent_dict" typing
    def initialize(self, from_parent_dict: dict = None, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self.forward = {"_forward_nchw_cw": self._forward_nchw_cw,
                        "_forward_nchw_cg": self._forward_nchw_cg,
                        "_forward_nhwc_cg": self._forward_nhwc_cg}[from_parent_dict["forward"].__name__]
        # self.forward = {"_forward_nchw_cw": self._forward_nchw_cw, \
        #                 "_forward_nchw_best_of": self._forward_nchw_cw}[from_parent_dict["forward"].__name__]
        self.weights = from_parent_dict["weights"]
        self.biases = from_parent_dict["biases"]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """This is a fake forward function. It will be masked on initialization by a _forward implementation"""
        raise NotImplementedError("Use a real forward variant!")

    def _forward_nchw_cw(self, x: np.ndarray) -> np.ndarray:
        """Version of the forward function that uses the convWinograd + BatchNorm + Relu"""

        if self.model.mode is Model.Mode.TRAIN:
            raise SystemExit("Sorry, fused layers cannot be used in training mode!")

        biases_vector = self.biases if self.use_bias else None
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CONVGEMM)
        y: np.ndarray = self.cw.conv_winograd_nchw(self.weights, x, biases_vector,
                                                vpadding=self.vpadding, hpadding=self.hpadding,
                                                vstride=self.vstride, hstride=self.hstride,
                                                vdilation=self.vdilation, hdilation=self.hdilation,
                                                relu=True, bn=True,
                                                running_mean=self.running_mean,
                                                inv_std=self.inv_std, gamma=self.gamma, beta=self.beta)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return np.asarray(y, dtype=self.model.dtype, order='C', copy=None)

    def _forward_nchw_cg(self, x: np.ndarray) -> np.ndarray:
        """Version of the forward function that uses the convGemm + BatchNorm + Relu"""

        if self.model.mode is Model.Mode.TRAIN:
            raise SystemExit("Sorry, fused layers cannot be used in training mode!")

        biases_vector = self.biases if self.use_bias else None
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CONVGEMM)
        res: np.ndarray = self.cg.conv_gemm_nchw(self.weights, x, biases=None,
                                              vpadding=self.vpadding, hpadding=self.hpadding,
                                              vstride=self.vstride, hstride=self.hstride,
                                              vdilation=self.vdilation, hdilation=self.hdilation,
                                              biases_vector=biases_vector, bn_running_mean=self.running_mean,
                                              bn_inv_std=self.inv_std, bn_gamma=self.gamma, bn_beta=self.beta, relu=True)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(res, dtype=self.model.dtype, order='C', copy=None)

    def _forward_nhwc_cg(self, x: np.ndarray) -> np.ndarray:
        """Version of the forward function that uses the convGemm + BatchNorm + Relu"""

        if self.model.mode is Model.Mode.TRAIN:
            raise RuntimeError("Fused layers cannot be used in training mode!")

        biases_vector = self.biases if self.use_bias else None
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CONVGEMM)
        res: np.ndarray = self.cg.conv_gemm_nhwc(self.weights, x, biases=None,
                                              vpadding=self.vpadding, hpadding=self.hpadding,
                                              vstride=self.vstride, hstride=self.hstride,
                                              vdilation=self.vdilation, hdilation=self.hdilation,
                                              biases_vector=biases_vector, bn_running_mean=self.running_mean,
                                              bn_inv_std=self.inv_std, bn_gamma=self.gamma, bn_beta=self.beta, relu=True)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(res, dtype=self.model.dtype, order='C', copy=None)

    def backward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Use a real backwards variant!")
