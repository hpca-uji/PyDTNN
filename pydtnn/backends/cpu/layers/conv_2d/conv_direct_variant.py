from functools import partial, partialmethod

from pydtnn.backends.cpu.libs.conv_direct import ConvDirect
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum
from pydtnn.utils.constants import ArrayShape, Array

import numpy as np

class ConvDirectVariant[T:Array](Conv2D[T]):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # convDirect related attributes (will be initialized in initialize())
        self.cd = []

    def initialize(self, prev_shape: ArrayShape, x: T | None = None):
        super().initialize(prev_shape, x)

        def new(name, func):
            func.__name__ = name
            setattr(self, name, func)

        # ConvDirect parameters
        if self.model.enable_conv_direct:
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
