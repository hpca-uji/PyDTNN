from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Self
if TYPE_CHECKING:
    from pydtnn.activations.activation import Activation
from pydtnn.layers.layer import Layer
from pydtnn.utils.initializers import InitializerFunc, glorot_uniform, zeros
from pydtnn.utils.tensor import TensorFormat
from pydtnn.utils.types import Array
import numpy as np
from enum import StrEnum, auto
from pydtnn.utils.types import ArrayShape


class Conv2D[T: Array](Layer[T]):

    class Grouping(StrEnum):
        DEPTHWISE = auto()
        POINTWISE = auto()
        STANDARD = auto()
    # -------

    class Variant(StrEnum):
        BEST_OF = auto()
        I2C = auto()
        POINTWISE = auto()
        DEPTHWISE = auto()
        # The following values are not set by auto due it's necessary that have that value.
        GEMM = "cg"
        WINOGRAD = "cw"
        DIRECT = "cd0"
    # -----

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
        self.grad_vars = {"weights": "dw"}
        if self.use_bias:
            self.grad_vars["biases"] = "db"
        self.debug = False
        # The following attributes will be initialized later
        self.ci = self.hi = self.wi = self.kh = self.kw = self.ho = self.wo = 0
        self.weights_shape: ArrayShape = None  # type: ignore
        # @warning: do not do this (affects the gpu version) self.forward = self.backward = None

    def initialize(self, prev_shape: ArrayShape, x: T | None = None):
        super().initialize(prev_shape, x)
        self.ci, self.hi, self.wi = self.model.decode_shape(prev_shape)
        self.kh, self.kw = self.filter_shape

        match self.grouping:
            case Conv2D.Grouping.DEPTHWISE:
                self.co = self.ci
                self.weights_shape = (self.ci, *self.filter_shape)
            case Conv2D.Grouping.POINTWISE:
                self.kh = self.kw = 1
                match self.model.tensor_format:
                    case TensorFormat.NCHW:
                        self.weights_shape = (self.co, self.ci)
                    case TensorFormat.NHWC:
                        self.weights_shape = (self.ci, self.co)
                    case tensor_format:
                        raise NotImplementedError(f"\"Conv2D\" is not implemented for \"{tensor_format}\" format.")
            case _:
                match self.model.tensor_format:
                    case TensorFormat.NCHW:
                        self.weights_shape = (self.co, self.ci, *self.filter_shape)
                    case TensorFormat.NHWC:
                        self.weights_shape = (self.ci, *self.filter_shape, self.co)
                    case tensor_format:
                        raise NotImplementedError(f"\"Conv2D\" is not implemented for \"{tensor_format}\" format.")

        self.ho = (self.hi + 2 * self.vpadding - self.vdilation * (self.kh - 1) - 1) // self.vstride + 1
        self.wo = (self.wi + 2 * self.hpadding - self.hdilation * (self.kw - 1) - 1) // self.hstride + 1
        self.shape = self.model.encode_shape((self.co, self.ho, self.wo))
        self.nparams = int(np.prod(self.weights_shape) + (self.co if self.use_bias else 0))

    def copy_from(self, other: Self) -> None:
        super().copy_from(other)

        # Non-objects
        self.ci = other.ci
        self.hi = other.hi
        self.wi = other.wi
        self.kh = other.kh
        self.kw = other.kw
        self.ho = other.ho
        self.wo = other.wo
        self.act = other.act
        self.co = other.co
        self.grouping = other.grouping
        self.stride = other.stride
        self.vpadding = other.vpadding
        self.vstride = other.vstride
        self.vdilation = other.vdilation
        self.use_bias = other.use_bias
        self.weights_initializer = other.weights_initializer # Functions
        self.biases_initializer = other.biases_initializer # Functions

        # "Objects"
        self.weights_shape = deepcopy(other.weights_shape)
        self.filter_shape = deepcopy(other.filter_shape)
        self.padding = deepcopy(other.padding)
        self.dilation = deepcopy(other.dilation)

    def show(self, attrs: str = "") -> None:
        self.weights: T
        super().show("|{:^19s}|{:^37s}|".format(str(self.weights.shape),
                                                f"padd=({self.vpadding},{self.hpadding}), "
                                                f"stride=({self.vstride},{self.hstride}), "
                                                f"dilat=({self.vdilation},{self.hdilation})"
                                                ))
