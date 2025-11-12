from copy import deepcopy
from typing import Self
import numpy as np

from pydtnn.layers.layer import Layer, LayerError
from pydtnn.utils.types import Array


class AbstractPool2DLayer[T: Array](Layer[T]):

    def __init__(self, pool_shape: tuple[int, int] | int = (2, 2), padding: tuple[int, int] | int = 0,
                 stride: tuple[int, int] | int = 1, dilation: tuple[int, int] | int = 1):
        super().__init__()
        self.pool_shape = (pool_shape, pool_shape) if isinstance(pool_shape, int) else pool_shape
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.vpadding, self.hpadding = (padding, padding) if isinstance(padding, int) else padding
        self.vstride, self.hstride = (stride, stride) if isinstance(stride, int) else stride
        self.vdilation, self.hdilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.ci = self.hi = self.wi = self.kh = self.kw = self.ho = self.wo = self.co = self.n = 0

    def initialize(self, prev_shape, x: T | None = None):
        super().initialize(prev_shape, x)
        self.ci, self.hi, self.wi = self.model.decode_shape(prev_shape)
        if self.pool_shape[0] == 0:
            self.pool_shape = (self.hi, self.pool_shape[1])
        if self.pool_shape[1] == 0:
            self.pool_shape = (self.pool_shape[0], self.wi)
        self.kh, self.kw = self.pool_shape
        self.co = self.ci
        self.ho = (self.hi + 2 * self.vpadding - self.vdilation * (self.kh - 1) - 1) // self.vstride + 1
        self.wo = (self.wi + 2 * self.hpadding - self.hdilation * (self.kw - 1) - 1) // self.hstride + 1
        if not (self.ho > 0 and self.wo > 0):
            raise LayerError(f"Output dimensions must be greater than 0. ho: {self.ho}, wo: {self.wo}.")
        self.shape = self.model.encode_shape((self.co, self.ho, self.wo))
        self.n = np.prod(self.shape)

    def copy_from(self, other: Self) -> None:
        super().copy_from(other)

        # Non-objects
        self.hpadding = other.hpadding
        self.vpadding = other.vpadding
        self.hstride = other.hstride
        self.vstride = other.vstride
        self.hdilation = other.hdilation
        self.vdilation = other.vdilation
        self.hi = other.hi
        self.wi = other.wi
        self.ci = other.ci
        self.kh = other.kh
        self.kw = other.kw
        self.ho = other.ho
        self.wo = other.wo
        self.co = other.co
        self.n = other.n

        # "Objects"
        self.pool_shape = deepcopy(other.pool_shape)
        self.padding = deepcopy(other.padding)
        self.stride = deepcopy(other.stride)
        self.dilation = deepcopy(other.dilation)

    def show(self, attrs=""):
        super().show("|{:^19s}|{:^37s}|".format(str(self.pool_shape),
                                                f"padd=({self.vpadding},{self.hpadding}), "
                                                f"stride=({self.vstride},{self.hstride}), "
                                                f"dilat=({self.vdilation},{self.hdilation})"))
