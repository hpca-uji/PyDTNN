#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021 Universitat Jaume I
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
from typing import override

from .layer import Layer

from pydtnn.utils import decode_tensor, encode_tensor
import numpy as np

class AdaptiveAveragePool2D(Layer, ABC):
    
    # This layer will calculate the pool shape and the stride from the output shape (passed as parameter) and the previous layer shape. 
    
    # output_shape:
    #  -> None: if the output shape is equal to the input
    #  -> int: if all the output shape's dimensions share values
    #  -> Tuple[int, int]: if it is necessary or it is preferred to define each output dimension individually
        
    def __init__(self, output_shape: int | tuple[int, int] | None = None):
        super().__init__()
        
        print(" => AdaptativeAveragePool2D <= ")

        self.output_shape = output_shape

        self.padding = 0 
        self.dilation = 1 

        self.vdilation, self.hdilation = (self.dilation, self.dilation)
        self.vpadding, self.hpadding = (self.padding, self.padding)
        
        # The following parameters will be initialized later:
        self.stride = self.pool_shape = (0, 0)
        self.vstride = self.hstride = 0
        self.ci = self.hi = self.wi = self.kh = self.kw = self.ho = self.wo = self.co = self.n = 0
    # ---  END __init__ --- #

    @override
    def initialize(self, prev_shape: tuple[int, int], need_dx: bool = True) -> None:
                
        # We want to override "AbstractPool2DLayer"
        super().initialize(prev_shape, need_dx)
        
        # TODO: Remove
        print(f"prev_shape: {prev_shape}")
        print(f"self.output_shape: {self.output_shape}")
        self.hi, self.wi, self.ci = decode_tensor(prev_shape, self.model.tensor_format)

        if self.output_shape is None:
            self.ho = self.hi
            self.wo = self.wi
        else:
            self.ho, self.wo = (self.output_shape, self.output_shape) if isinstance(self.output_shape, int) else self.output_shape   
        self.co = self.ci
        # https://stackoverflow.com/questions/64284755/what-is-the-upsampling-method-called-area-used-for

        # Unknown values: pool_shape (kh, kw) and stride (vstride, hstride)

        # TODO: Remove 
        print(f"self.ci: {self.ci}")
        print(f"self.hi: {self.hi}")
        print(f"self.wi: {self.wi}")
        print(f"self.co: {self.co}")
        print(f"self.ho: {self.ho}")
        print(f"self.wo: {self.wo}")

        # -> Getting (and setting) the pool_shape (kh, kw):
        self.vstride = self.hi // self.ho
        self.hstride = self.wi // self.wo

        # TODO: Remove
        print(f"self.vstride: {self.vstride}")
        print(f"self.hstride: {self.hstride}")
        print(f"self.vpadding: {self.vpadding}")
        print(f"self.hpadding: {self.hpadding}")
        print(f"self.vdilation: {self.vdilation}")
        print(f"self.hdilation: {self.hdilation}")

        # -> Getting (and setting) the stride (vstride, hstride):
        # Base formula: self.ho = (self.hi + 2 * self.vpadding - self.vdilation * (self.kh - 1) - 1) // self.vstride + 1
        self.kh = (self.vstride * (self.ho - 1) - self.hi -2 * self.vpadding + 1 ) // (-1 * self.vdilation )
        # self.vstride = (self.hi + 2 * self.vpadding - self.vdilation * (self.kh - 1) - 1) // (self.ho - 1)

        # Base formula: self.wo = (self.wi + 2 * self.hpadding - self.hdilation * (self.kw - 1) - 1) // self.hstride + 1
        self.kw = (self.hstride * (self.wo - 1) - self.wi -2 * self.hpadding + 1 ) // (-1 * self.hdilation )
        # self.hstride = (self.wi + 2 * self.hpadding - self.hdilation * (self.kw - 1) - 1) // (self.wo - 1)
    
        # TODO: Remove
        print(f"self.kh: {self.kh}")
        print(f"self.kw: {self.kw}")
        print(f"self.hstride * (self.wo - 1): {self.hstride * (self.wo - 1)}")
        print(f"self.wi -2 * self.hpadding + 1): {self.hdilation * (self.kw - 1) - 1}")
        print(f"(self.hstride * (self.wo - 1) - self.wi -2 * self.hpadding + 1 ): {(self.hstride * (self.wo - 1) - self.wi -2 * self.hpadding + 1 )}")

        self.shape = encode_tensor((self.ho, self.wo, self.co), self.model.tensor_format)        
        self.n = np.prod(self.shape)

        # TODO: Remove
        print(f"self.shape: {self.shape}")
        print(f"self.n: {self.n}")
    # - END initialize - #
    
    
    def show(self, attrs=""):
        super().show("|{:^19s}|{:^37s}|".format(str(self.pool_shape),
                                                f"padd=({self.vpadding},{self.hpadding}), "
                                                f"stride=({self.vstride},{self.hstride}), "
                                                f"dilat=({self.vdilation},{self.hdilation})"))
