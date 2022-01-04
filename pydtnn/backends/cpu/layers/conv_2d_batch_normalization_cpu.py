#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021 Universitat Jaume I
#
#  PyDTNN is free software: you can redistribute it and/or modify it under the
#  terms of the GNU General Public License as published by the Free Software
#  Foundation, either version 3 of the License, or (at your option) any later
#  version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
#  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
#  License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with this program. If not, see <https://www.gnu.org/licenses/>.
#

from pydtnn.backends.cpu.layers import LayerCPU
from pydtnn.layers import Conv2DBatchNormalization
from pydtnn.model import TRAIN_MODE
from pydtnn.tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_OPS_FORWARD_CONVGEMM

class Conv2DBatchNormalizationCPU(LayerCPU, Conv2DBatchNormalization):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize(self, prev_shape=None, need_dx=True, from_parent_dict=None):
        self.forward = {"_forward_nchw_cw": self._forward_nchw_cw,
                        "_forward_nchw_cg": self._forward_nchw_cg,
                        "_forward_nhwc_cg": self._forward_nhwc_cg}[from_parent_dict["forward"].__name__]
        # self.forward = {"_forward_nchw_cw": self._forward_nchw_cw, \
        #                 "_forward_nchw_best_of": self._forward_nchw_cw}[from_parent_dict["forward"].__name__]
        self.weights = from_parent_dict["weights"]
        self.biases = from_parent_dict["biases"]

    def forward(self, x):
        """This is a fake forward function. It will be masked on initialization by a _forward implementation"""
        pass

    def _forward_nchw_cw(self, x):
        """Version of the forward function that uses the convWinograd + BatchNorm + """

        if self.model.mode == TRAIN_MODE:
            raise SystemExit("Sorry, fused layers cannot be used in training mode!")

        biases_vector = self.biases if self.use_bias else None
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_CONVGEMM)
        y = self.cw.conv_winograd_nchw(self.weights, x, biases_vector,
                                vpadding=self.vpadding, hpadding=self.hpadding,
                                vstride=self.vstride, hstride=self.hstride,
                                vdilation=self.vdilation, hdilation=self.hdilation, 
                                relu=False, bn=True,
                                running_mean=self.running_mean,
                                inv_std=self.inv_std, gamma=self.gamma, beta=self.beta)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        return y

    def _forward_nchw_cg(self, x):
        """Version of the forward function that uses the convGemm + BatchNorm"""

        if self.model.mode == TRAIN_MODE:
            raise SystemExit("Sorry, fused layers cannot be used in training mode!")

        biases_vector = self.biases if self.use_bias else None
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_CONVGEMM)
        res = self.cg.conv_gemm_nchw(self.weights, x, biases=None,
                                     vpadding=self.vpadding, hpadding=self.hpadding,
                                     vstride=self.vstride, hstride=self.hstride,
                                     vdilation=self.vdilation, hdilation=self.hdilation,
                                     biases_vector=biases_vector, bn_running_mean=self.running_mean,
                                     bn_inv_std=self.inv_std, bn_gamma=self.gamma, bn_beta=self.beta, relu=False)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        return res

    def _forward_nhwc_cg(self, x):
        """Version of the forward function that uses the convGemm + BatchNorm"""

        if self.model.mode == TRAIN_MODE:
            raise RuntimeError("Fused layers cannot be used in training mode!")

        biases_vector = self.biases if self.use_bias else None
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_CONVGEMM)
        res = self.cg.conv_gemm_nhwc(self.weights, x, biases=None,
                                     vpadding=self.vpadding, hpadding=self.hpadding,
                                     vstride=self.vstride, hstride=self.hstride,
                                     vdilation=self.vdilation, hdilation=self.hdilation,
                                     biases_vector=biases_vector, bn_running_mean=self.running_mean,
                                     bn_inv_std=self.inv_std, bn_gamma=self.gamma, bn_beta=self.beta, relu=False)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        return res

    def backward(self, x):
        raise SystemExit(f"Backward method of {self.__class__.__name__} should not be called")
