from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pydtnn.activations.activation import Activation
from pydtnn.layers.layer import Layer
from pydtnn.utils.initializers import InitializerFunc, glorot_uniform, zeros
import numpy as np
from pydtnn.utils.constants import Array, ArrayShape, Parameters


class Conv2D[T: Array](Layer[T]):
    def __init__(self, nfilters: int = 1,
                 filter_shape: tuple[int, int] | int = (3, 3),
                 padding: tuple[int, int] | int = 0,
                 stride: tuple[int, int] | int = 1,
                 dilation: tuple[int, int] | int = 1,
                 activation: Optional[type["Activation"]] = None,
                 use_bias=True,
                 weights_initializer: InitializerFunc = glorot_uniform,
                 biases_initializer: InitializerFunc = zeros):

        super().__init__()
        self.co = nfilters
        self.filter_shape = (filter_shape, filter_shape) if isinstance(filter_shape, int) else filter_shape
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.vpadding, self.hpadding = (padding, padding) if isinstance(padding, int) else padding
        self.vstride, self.hstride = (stride, stride) if isinstance(stride, int) else stride
        self.vdilation, self.hdilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.act = activation
        self.use_bias = use_bias
        self.weights_initializer: InitializerFunc = weights_initializer
        self.biases_initializer: InitializerFunc = biases_initializer
        self.grad_vars = {Parameters.WEIGHTS: Parameters.DW}
        if self.use_bias:
            self.grad_vars[Parameters.BIASES] = Parameters.DB
        self.debug = False
        # The following attributes will be initialized later
        self.ci = self.hi = self.wi = self.kh = self.kw = self.ho = self.wo = 0
        self.weights_shape: ArrayShape = None  # type: ignore
        # @warning: do not do this (affects the gpu version) self.forward = self.backward = None
    # ---

    def _initializing_special_parameters(self):
        # NOTE: This method's objective is to define and change the value of some parameters defined before that are needed later in the initialization process,
        #   for example: "self.weights_shape" and, in non-standard cases, "self.co".
        pass
    # ---

    def initialize(self, prev_shape: ArrayShape, x: T | None):
        super().initialize(prev_shape, x)
        self.ci, self.hi, self.wi = self.model.decode_shape(prev_shape)
        self.kh, self.kw = self.filter_shape
        self._initializing_special_parameters()

        self.ho = (self.hi + 2 * self.vpadding - self.vdilation * (self.kh - 1) - 1) // self.vstride + 1
        self.wo = (self.wi + 2 * self.hpadding - self.hdilation * (self.kw - 1) - 1) // self.hstride + 1
        self.shape = self.model.encode_shape((self.co, self.ho, self.wo))

        # NOTE: self.weights_shape must be defined in "self._initializing_special_parameters"
        self.nparams = int(np.prod(self.weights_shape) + (self.co if self.use_bias else 0))
    # --

    def _show_props(self) -> dict:
        props = super()._show_props()

        props["padding"] = (self.vpadding, self.hpadding)
        props["stride"] = (self.vstride, self.hstride)
        props["dilation"] = (self.vdilation, self.hdilation)

        return props
