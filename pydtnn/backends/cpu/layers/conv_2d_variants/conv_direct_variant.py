from abc import ABC
from functools import partialmethod

from pydtnn.backends.cpu.libs.conv_direct import ConvDirect
from pydtnn.layers import Conv2D
from pydtnn.tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum
from numpy import ndarray

class ConvDirectVariant(Conv2D, ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # convDirect related attributes (will be initialized in initialize())
        self.cd = []

    def initialize(self, prev_shape:tuple[int, ...]):
        super().initialize(prev_shape)
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

    def _forward_cd(self, x: ndarray, n=0) -> ndarray:
        """Version of the forward function that uses the convDirect library"""

        biases = None
        biases_vector = self.biases if self.use_bias else None

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CONVDIRECT)
        y = self.cd[n].conv_direct(self.weights, x, biases,
                                   vpadding=self.vpadding, hpadding=self.hpadding,
                                   vstride=self.vstride, hstride=self.hstride,
                                   vdilation=self.vdilation, hdilation=self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return y

    def _backward_cd(self, y: ndarray, n=0) -> ndarray:
        raise RuntimeError("Backward not implemented yet!")