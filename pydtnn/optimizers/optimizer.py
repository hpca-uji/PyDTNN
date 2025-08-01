#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021-25 Universitat Jaume I
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

from abc import ABC, abstractmethod

from pydtnn.backends import PromoteToBackendMixin
import numpy as np

from pydtnn.layers import Layer

class Optimizer(PromoteToBackendMixin, ABC):
    """
    Optimizer abstract base class
    """

    def __init__(self, learning_rate:float = 1e-2, dtype:np.dtype=np.float32):
        super().__init__()
        self.learning_rate:float = learning_rate
        self.dtype:np.dtype = dtype
        self.context:dict = dict()

    @abstractmethod
    def initialize(self, list_layers: list[Layer]) -> None:
        raise NotImplementedError("method \"initialize\" of an Optimizer's child class is not implemented")

    @abstractmethod
    def update(self, layer: Layer) -> None:
        pass
