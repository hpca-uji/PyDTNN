#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021-22 Universitat Jaume I
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

from abc import ABC
from functools import partialmethod

from pydtnn.backends.cpu.libs.conv_direct import ConvDirect
from pydtnn.layers import Conv2D
from pydtnn.tracers import PYDTNN_OPS_EVENTS, PYDTNN_OPS_EVENT, PYDTNN_OPS_FORWARD_CONVDIRECT


class ConvDirectVariant(Conv2D, ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # convDirect related attributes (will be initialized in initialize())
        self.cd = []

    def initialize(self, prev_shape, need_dx=True):
        super().initialize(prev_shape, need_dx)
        # ConvWinograd parameters
        if self.model.enable_conv_direct:
            methods = [self.model.conv_direct_method, ]
            if self.model.enable_best_of:
                if self.model.conv_direct_methods_for_best_of != "":
                    methods = self.model.conv_direct_methods_for_best_of.split(',')
            for n, method in enumerate(methods):
                self.cd.append(ConvDirect(method, dtype=self.model.dtype, tensor_format=self.model.tensor_format,
                                          debug=self.debug, parent_layer=self))
                try:
                    getattr(ConvDirectVariant, f"_forward_cd{n}_nhwc")
                except AttributeError:
                    setattr(ConvDirectVariant, f"_forward_cd{n}_nhwc", partialmethod(ConvDirectVariant._forward_cd, n=n))
                    setattr(ConvDirectVariant, f"_forward_cd{n}_nchw", partialmethod(ConvDirectVariant._forward_cd, n=n))
                    setattr(ConvDirectVariant, f"_backward_cd{n}_nhwc", partialmethod(ConvDirectVariant._backward_cd, n=n))
                    setattr(ConvDirectVariant, f"_backward_cd{n}_nchw", partialmethod(ConvDirectVariant._backward_cd, n=n))

    def _forward_cd(self, x, n=0):
        """Version of the forward function that uses the convDirect library"""

        biases = None
        biases_vector = self.biases if self.use_bias else None

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_CONVDIRECT)
        y = self.cd[n].conv_direct(self.weights, x, biases,
                                   vpadding=self.vpadding, hpadding=self.hpadding,
                                   vstride=self.vstride, hstride=self.hstride,
                                   vdilation=self.vdilation, hdilation=self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        return y

    def _backward_cd(self, y, n=0):
        raise RuntimeError("Backward not implemented yet!")