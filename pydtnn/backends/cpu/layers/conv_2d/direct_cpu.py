from functools import partial

from pydtnn.backends.cpu.layers.abstract.conv_2d_standard_cpu import Conv2DStandardCPU
from pydtnn.backends.cpu.libs.conv_direct import ConvDirect
from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum
from pydtnn.utils.constants import ArrayShape

import numpy as np

class Conv2DDirectCPU(Conv2DStandardCPU):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # convDirect related attributes (will be initialized in initialize())
        self.cd = []

    def initialize(self, prev_shape: ArrayShape, x: np.ndarray | None = None):
        super().initialize(prev_shape, x)

        def new(name, func):
            func.__name__ = name
            setattr(self, name, func)

        # ConvDirect parameters
        methods = []
        if self.model.conv_direct_method:
            methods = [self.model.conv_direct_method]
        if self.model.enable_best_of:
            if self.model.conv_direct_methods_for_best_of != "":
                methods = self.model.conv_direct_methods_for_best_of.split(',')
        for n, method in enumerate(methods):
            self.cd.append(ConvDirect(method, dtype=self.model.dtype, tensor_format=self.model.tensor_format,
                                        debug=self.debug, parent_layer=self))
            try:
                getattr(self, f"_forward_cd{n}_nhwc")
            except AttributeError:
                new(f"_forward_cd{n}_nhwc", partial(self._forward_cd, n=n))
                new(f"_forward_cd{n}_nchw", partial(self._forward_cd, n=n))
                new(f"_backward_cd{n}_nhwc", partial(self._backward_cd, n=n))
                new(f"_backward_cd{n}_nchw", partial(self._backward_cd, n=n))

    def _forward_cd(self, x: np.ndarray, n=0) -> np.ndarray:
        """Version of the forward function that uses the convDirect library"""
        biases_vector = self.biases if self.use_bias else None

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CONVDIRECT)
        y = self.cd[n].conv_direct(self.weights, x, biases_vector,
                                   vpadding=self.vpadding, hpadding=self.hpadding,
                                   vstride=self.vstride, hstride=self.hstride,
                                   vdilation=self.vdilation, hdilation=self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return y

    def _backward_cd(self, y: np.ndarray, n=0) -> np.ndarray:
        raise RuntimeError("Backward not implemented yet!")
