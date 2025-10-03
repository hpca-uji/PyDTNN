from abc import ABC

import numpy as np

from .layer import Layer
from pydtnn.utils import decode_tensor, encode_tensor


class AbstractPool2DLayer(Layer, ABC):

    def __init__(self, pool_shape:tuple[int, int] | int = (2, 2), padding:tuple[int, int] | int = 0, 
                 stride:tuple[int, int] | int = 1, dilation:tuple[int, int] | int = 1):
        super().__init__()
        self.pool_shape = (pool_shape, pool_shape) if isinstance(pool_shape, int) else pool_shape
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.vpadding, self.hpadding = (padding, padding) if isinstance(padding, int) else padding
        self.vstride, self.hstride = (stride, stride) if isinstance(stride, int) else stride
        self.vdilation, self.hdilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.ci = self.hi = self.wi = self.kh = self.kw = self.ho = self.wo = self.co = self.n = 0

    def initialize(self, prev_shape):
        super().initialize(prev_shape)
        self.hi, self.wi, self.ci = decode_tensor(prev_shape, self.model.tensor_format)
        if self.pool_shape[0] == 0:
            self.pool_shape = (self.hi, self.pool_shape[1])
        if self.pool_shape[1] == 0:
            self.pool_shape = (self.pool_shape[0], self.wi)
        self.kh, self.kw = self.pool_shape
        self.co = self.ci
        self.ho = (self.hi + 2 * self.vpadding - self.vdilation * (self.kh - 1) - 1) // self.vstride + 1
        self.wo = (self.wi + 2 * self.hpadding - self.hdilation * (self.kw - 1) - 1) // self.hstride + 1
        assert self.ho > 0 and self.wo > 0, "Output dimensions must be greater than 0"
        self.shape = encode_tensor((self.ho, self.wo, self.co), self.model.tensor_format)
        self.n = np.prod(self.shape)

    def show(self, attrs=""):
        super().show("|{:^19s}|{:^37s}|".format(str(self.pool_shape),
                                                f"padd=({self.vpadding},{self.hpadding}), "
                                                f"stride=({self.vstride},{self.hstride}), "
                                                f"dilat=({self.vdilation},{self.hdilation})"))
