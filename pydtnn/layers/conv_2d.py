from functools import cached_property
import importlib
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from pydtnn.activations.activation import Activation
from pydtnn.backends import BackendType
from pydtnn.layers.layer import Layer
from pydtnn.utils.initializers import InitializerFunc, glorot_uniform, zeros
import numpy as np
from enum import StrEnum, auto
from pydtnn.utils.constants import Array, ArrayShape, Parameters


class Conv2D[T: Array](Layer[T]):

    class Grouping(StrEnum):
        DEPTHWISE = auto()
        POINTWISE = auto()
        STANDARD = auto()
    # -------

    class Variant(StrEnum):
        BEST_OF = auto()
        I2C = auto()
        #NOTE: The following values are not set by auto due it's necessary that have that value.
        GEMM = "cg"
        WINOGRAD = "cw"
        DIRECT = "cd0"
    # -----

    def _get_backend_cls(self, backend: BackendType) -> None:
        cls = self.__class__
        module_name = cls.__module__.split(".", 1)[1]
        backend_module_name = f"pydtnn.backends.{backend}.{module_name}.{self.grouping.lower()}_variant"
        backend_module = importlib.import_module(backend_module_name)
        cls_name = f"{cls.__name__}{backend.upper()}"
        cls = getattr(backend_module, cls_name)
        return cls

    def __init__(self, nfilters: int = 1,
                 filter_shape: tuple[int, int] | int = (3, 3),
                 grouping: Grouping = Grouping.STANDARD,
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
        self.grouping = Conv2D.Grouping(grouping.lower())
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

    def initialize(self, prev_shape: ArrayShape, x: T | None):
        super().initialize(prev_shape, x)
        self.ci, self.hi, self.wi = self.model.decode_shape(prev_shape)
        self.kh, self.kw = self.filter_shape

    @cached_property
    def ho(self) -> int:
        return (self.hi + 2 * self.vpadding - self.vdilation * (self.kh - 1) - 1) // self.vstride + 1
    
    @cached_property
    def wo(self) -> int:
        return (self.wi + 2 * self.hpadding - self.hdilation * (self.kw - 1) - 1) // self.hstride + 1
    
    @cached_property
    def shape(self) -> ArrayShape:
        return self.model.encode_shape((self.co, self.ho, self.wo))
    
    @cached_property
    def nparams(self) -> int:
        return int(np.prod(self.weights_shape) + (self.co if self.use_bias else 0))

    def show(self, attrs: str = "") -> None:
        self.weights: T
        super().show("|{:^19s}|{:^37s}|".format(str(self.weights.shape),
                                                f"padd=({self.vpadding},{self.hpadding}), "
                                                f"stride=({self.vstride},{self.hstride}), "
                                                f"dilat=({self.vdilation},{self.hdilation}), "
                                                f"grouping=({self.grouping})"
                                                ))
