#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2025 Universitat Jaume I
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
        self.output_shape = output_shape

        # This value will change in initialize:
        self._forward_pooling_not_needed:bool = None
    # ---  END __init__ --- #

    @override
    def initialize(self, prev_shape: tuple[int, int], need_dx: bool = True) -> None:
        # We want to override "AbstractPool2DLayer"
        super().initialize(prev_shape, need_dx)
        
        self.hi, self.wi, self.ci = decode_tensor(prev_shape, self.model.tensor_format)

        if self.output_shape is None:
            self.ho, self.wo = self.hi, self.wi
        else:
            self.ho, self.wo = (self.output_shape, self.output_shape) if isinstance(self.output_shape, int) else self.output_shape   
        assert (self.ho > 0 and self.wo > 0), f"The output height and width should be grater than 0. height: {self.ho} width: {self.wo}"
        self.co = self.ci
        
        # If the output and the input shapes are the same, there is no need of pooling.
        self.pooling_not_needed = (self.hi == self.ho) and (self.wi == self.wo)

        self.shape = encode_tensor((self.ho, self.wo, self.co), self.model.tensor_format)        
        self.n = np.prod(self.shape)
    # - END initialize - #
    
    def show(self, attrs=""):
        super().show("|{:^19s}|{:^37s}|".format(f"",
                                                f"inp. shape=({self.hi},{self.wi}), "
                                                f"out. shape=({self.ho},{self.wo})"))
