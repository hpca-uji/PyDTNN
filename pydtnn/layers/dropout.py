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

from pydtnn.model import EVALUATE_MODE
from .layer import Layer
from ..model import TRAIN_MODE
from ..performance_models import *


class Dropout(Layer):

    def __init__(self, rate=0.5):
        super().__init__()
        self.rate = min(1., max(0., rate))
        # The next attributes will be initialized later
        self.mask = None

    def initialize(self, prev_shape, need_dx=True):
        super().initialize(prev_shape, need_dx)
        self.shape = prev_shape

    def show(self, attrs=""):
        super().show("|{:^19s}|{:^24s}|".format("", "rate=%.2f" % self.rate))

    def forward(self, x):
        if self.model.mode == TRAIN_MODE:
            self.mask = np.random.binomial(1, (1 - self.rate), size=self.shape).astype(self.model.dtype) / (
                    1 - self.rate)
            return x * self.mask
        elif self.model.mode == EVALUATE_MODE:
            return x
        else:
            raise ValueError("Unexpected model mode")

    def backward(self, dy):
        if self.need_dx:
            return dy * self.mask
