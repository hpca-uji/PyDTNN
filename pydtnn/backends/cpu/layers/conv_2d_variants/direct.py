from functools import partial
from warnings import warn

from pydtnn.backends.cpu.layers.abstract.conv_2d_standard import Conv2DStandardCPU
from pydtnn.libs.libconvdirect import ConvDirect
from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum
from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.tensor import encode_shape

import numpy as np

from pydtnn.utils.tensor import TensorFormat

class Conv2DDirectCPU(Conv2DStandardCPU):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # convDirect related attributes (will be initialized in initialize())
        self.cd = []
    
    def _add_forward_backward_methods(self):
        """Add the different forward and backward methods to the class"""

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
    # ----

    def initialize(self, prev_shape: ArrayShape, x: np.ndarray | None = None):
        super().initialize(prev_shape, x)
        self._add_forward_backward_methods()

        match self.model.tensor_format:
            case TensorFormat.NHWC:
                self.forward = self._forward_cd0_nhwc
                self.backward = self._backward_cd0_nhwc
            case TensorFormat.NCHW:
                self.forward = self._forward_cd0_nchw
                self.backward = self._backward_cd0_nchw
            case _:
                raise NotImplementedError(f"\"{self.model.tensor_format}\" format not implemented.")
        # --

        out_shape = encode_shape((self.model.batch_size, self.co, self.ho, self.wo))
        self.out = np.zeros(out_shape, self.weights.dtype, order="C")

        self.out = None

        if self.use_bias:
            warn(f"\"{self.__class__.__name__}\" never uses the biases.", RuntimeWarning)
    # -----

    def _forward_cd(self, x: np.ndarray, n=0) -> np.ndarray:
        """Version of the forward function that uses the convDirect library"""

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CONVDIRECT)
        y = self.cd[n].conv_direct(self.weights, x, self.out,
                                   vpadding=self.vpadding, hpadding=self.hpadding,
                                   vstride=self.vstride, hstride=self.hstride,
                                   vdilation=self.vdilation, hdilation=self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return y

    def _backward_cd(self, y: np.ndarray, n=0) -> np.ndarray:
        raise RuntimeError("Backward not implemented yet!")
