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
#  with this program. If not, see <https://www.gnu.org/licenses/>.
#

import numpy as np

from pydtnn.backends.cpu.layers import LayerCPU
from pydtnn.layers import Dropout
from pydtnn.model import ModelModeEnum


class DropoutCPU(LayerCPU, Dropout):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask:np.ndarray = None

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)

    def forward(self, x:np.ndarray) -> np.ndarray:

        match self.model.mode:
            case ModelModeEnum.TRAIN:
                # NOTE: Remember, it's necessary a new random mask every training's forward call.
                #self.mask = np.random.binomial(1, (1 - self.rate), size=self.shape).astype(self.model.dtype) / (1 - self.rate)
                self.mask = np.random.binomial(n=1, p=(1 - self.rate), size=self.shape).astype(dtype=self.model.dtype)
                self.mask /= (1 - self.rate)
                return x * self.mask
            case ModelModeEnum.EVALUATE:
                return x
            case _:
                raise RuntimeError(f"Unexpected model mode \'{self.model.mode}\'.")

    def backward(self, dy:np.ndarray) -> np.ndarray:
        dy *= self.mask
        return dy
