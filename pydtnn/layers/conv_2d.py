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
#  with this program.  If not, see <https://www.gnu.org/licenses/>.
#

from abc import ABC

from .layer import Layer
from pydtnn.activations import Activation 
from ..initializers import InitializerFunc, glorot_uniform, zeros
from pydtnn.utils import decode_tensor, encode_tensor, PYDTNN_TENSOR_FORMAT
import numpy as np
from enum import StrEnum, auto

class GroupingEnum(StrEnum):
    DEPTHWISE = auto()
    POINTWISE = auto()
    STANDARD  = auto()

class Conv2D(Layer, ABC):

    def __init__(self, nfilters:int=1, 
                 filter_shape:tuple[int, int] | int = (3, 3), 
                 grouping:GroupingEnum | None  = None, 
                 padding:tuple[int, int] | int = 0, 
                 stride: tuple[int, int] | int = 1,
                 dilation:tuple[int, int] | int = 1, 
                 activation:Activation | None = None, 
                 use_bias=True, 
                 weights_initializer:InitializerFunc = glorot_uniform,
                 biases_initializer:InitializerFunc = zeros):
        
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
        self.weights_initializer:InitializerFunc = weights_initializer
        self.biases_initializer:InitializerFunc = biases_initializer
        self.grad_vars = {"weights": "dw"}
        if self.use_bias:
            self.grad_vars["biases"] = "db"
        self.debug = False
        # The next attributes will be initialized later
        self.ci = self.hi = self.wi = self.kh = self.kw = self.ho = self.wo = 0
        self.weights_shape:tuple[int, ...] = None
        # @warning: do not do this (affects the gpu version) self.forward = self.backward = None

    def initialize(self, prev_shape:tuple[int, ...], need_dx=True):
        super().initialize(prev_shape, need_dx)
        self.hi, self.wi, self.ci = decode_tensor(prev_shape, self.model.tensor_format)
        self.kh, self.kw = self.filter_shape

        match self.grouping:
            case GroupingEnum.DEPTHWISE:
                self.co = self.ci
                self.weights_shape = (self.ci, *self.filter_shape)
            case GroupingEnum.POINTWISE:
                self.kh = self.kw = 1
                if self.model.tensor_format == PYDTNN_TENSOR_FORMAT.NCHW:
                    self.weights_shape = (self.co, self.ci)
                else:
                    self.weights_shape = (self.ci, self.co)
            case _:
                if self.model.tensor_format == PYDTNN_TENSOR_FORMAT.NCHW:
                    self.weights_shape = (self.co, self.ci, *self.filter_shape)
                else:
                    self.weights_shape = (self.ci, *self.filter_shape, self.co)

        self.ho = (self.hi + 2 * self.vpadding - self.vdilation * (self.kh - 1) - 1) // self.vstride + 1
        self.wo = (self.wi + 2 * self.hpadding - self.hdilation * (self.kw - 1) - 1) // self.hstride + 1
        self.shape = encode_tensor((self.ho, self.wo, self.co), self.model.tensor_format)
        self.nparams = np.prod(self.weights_shape) + (self.co if self.use_bias else 0)

    def show(self, attrs:str="") -> None:
        super().show("|{:^19s}|{:^37s}|".format(str(self.weights.shape),
                                                f"padd=({self.vpadding},{self.hpadding}), "
                                                f"stride=({self.vstride},{self.hstride}), "
                                                f"dilat=({self.vdilation},{self.hdilation})"))
