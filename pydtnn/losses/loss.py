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
#  with this program. If not, see <https://www.gnu.org/licenses/>.
#

from abc import ABC, abstractmethod

from ..backends import PromoteToBackendMixin
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pydtnn.model import Model
else: Model = None
from numpy import ndarray
from pydtnn.backends.gpu import TensorGPU
type Array = ndarray | TensorGPU

class Loss(PromoteToBackendMixin, ABC):

    def __init__(self, shape:tuple[int,...], model:Model, eps=1e-8):
        self.shape = shape
        self.model = model
        self.eps = eps

    @abstractmethod
    def __call__(self, y_pred:Array, y_targ:Array, global_batch_size:int) -> tuple[float, Array]:
        pass
