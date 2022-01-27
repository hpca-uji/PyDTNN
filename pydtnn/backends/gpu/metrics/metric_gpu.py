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

# noinspection PyUnresolvedReferences
import pycuda.gpuarray as gpuarray

from pydtnn.metrics import Metric


class MetricGPU(Metric, ABC):
    """
    Extends a Metric class with the attributes and methods required by GPU Metrics.
    """

    def __init__(self, shape, model, eps=1e-8):
        super().__init__(shape, model, eps)
        self.cost = gpuarray.empty((self.model.batch_size,), self.model.dtype)
        self.kernel = self.__init_gpu_kernel__()

    @abstractmethod
    def __init_gpu_kernel__(self):
        pass
