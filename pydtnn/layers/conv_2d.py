from abc import ABC

from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from pydtnn.activations import Activation
from pydtnn.layers.layer import Layer
from pydtnn.initializers import InitializerFunc, glorot_uniform, zeros
from pydtnn.utils import decode_tensor, encode_tensor, PYDTNN_TENSOR_FORMAT
from pydtnn.utils.types import Array
import numpy as np
from enum import StrEnum, auto


class GroupingEnum(StrEnum):
    DEPTHWISE = auto()
    POINTWISE = auto()
    STANDARD = auto()


class Conv2D(Layer, ABC):

    def __init__(self, nfilters: int = 1,
                 filter_shape: tuple[int, int] | int = (3, 3),
                 grouping: GroupingEnum = GroupingEnum.STANDARD,
                 padding: tuple[int, int] | int = 0,
                 stride: tuple[int, int] | int = 1,
                 dilation: tuple[int, int] | int = 1,
                 activation: Optional["Activation"] = None,
                 use_bias=True,
                 weights_initializer: InitializerFunc = glorot_uniform,
                 biases_initializer: InitializerFunc = zeros):

        super().__init__()
        self.co = nfilters
        self.filter_shape = (filter_shape, filter_shape) if isinstance(filter_shape, int) else filter_shape
        self.grouping = grouping
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
        # The next attributes will be initialized later
        self.ci = self.hi = self.wi = self.kh = self.kw = self.ho = self.wo = 0
        self.weights_shape: tuple[int, ...] = None
        # @warning: do not do this (affects the gpu version) self.forward = self.backward = None

    def initialize(self, prev_shape: tuple[int, ...], x: Array | None = None):
        super().initialize(prev_shape, x)
        self.hi, self.wi, self.ci = decode_tensor(prev_shape, self.model.tensor_format)
        self.kh, self.kw = self.filter_shape

        match self.grouping:
            case GroupingEnum.DEPTHWISE:
                self.co = self.ci
                self.weights_shape = (self.ci, *self.filter_shape)
            case GroupingEnum.POINTWISE:
                self.kh = self.kw = 1
                match self.model.tensor_format:
                    case PYDTNN_TENSOR_FORMAT.NCHW:
                        self.weights_shape = (self.co, self.ci)
                    case PYDTNN_TENSOR_FORMAT.NHWC:
                        self.weights_shape = (self.ci, self.co)
                    case _:
                        raise NotImplementedError(f"\"ConcatenationBlockCPU\" is not implemented for \"{self.model.tensor_format}\" format.")
            case _:
                match self.model.tensor_format:
                    case PYDTNN_TENSOR_FORMAT.NCHW:
                        self.weights_shape = (self.co, self.ci, *self.filter_shape)
                    case PYDTNN_TENSOR_FORMAT.NHWC:
                        self.weights_shape = (self.ci, *self.filter_shape, self.co)
                    case _:
                        raise NotImplementedError(f"\"ConcatenationBlockCPU\" is not implemented for \"{self.model.tensor_format}\" format.")

        self.ho = (self.hi + 2 * self.vpadding - self.vdilation * (self.kh - 1) - 1) // self.vstride + 1
        self.wo = (self.wi + 2 * self.hpadding - self.hdilation * (self.kw - 1) - 1) // self.hstride + 1
        self.shape = encode_tensor((self.ho, self.wo, self.co), self.model.tensor_format)
        self.nparams = np.prod(self.weights_shape) + (self.co if self.use_bias else 0)

    def show(self, attrs: str = "") -> None:
        super().show("|{:^19s}|{:^37s}|".format(str(self.weights.shape),
                                                f"padd=({self.vpadding},{self.hpadding}), "
                                                f"stride=({self.vstride},{self.hstride}), "
                                                f"dilat=({self.vdilation},{self.hdilation})"
                                                ))
