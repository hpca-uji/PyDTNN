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

from abc import ABC, abstractmethod

# noinspection PyUnresolvedReferences
import pycuda.gpuarray as gpuarray
# noinspection PyUnresolvedReferences
from pycuda.driver import Function

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pydtnn.model import Model
else:
    Model = object
# noinspection PyUnresolvedReferences
from pydtnn.backends.gpu import TensorGPU
from pydtnn.losses import Loss


class LossGPU(Loss, ABC):
    """
    Extends a Loss class with the attributes and methods required by GPU Losses.
    """

    LIMIT_THREADS_AND_BLOCKS = 1024

    def __init__(self, shape:tuple[int, ...], model:Model, eps=1e-8):
        super().__init__(shape, model, eps)
        self.loss = gpuarray.empty((self.model.batch_size,), self.model.dtype)
        dx_gpu = gpuarray.empty(self.shape, self.model.dtype)
        self.dx = TensorGPU(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)
        self.kernel = self.__init_gpu_kernel__()

    @abstractmethod
    def __init_gpu_kernel__(self) -> Function:
        pass

    def get_threads_and_blocks(self):
        threads = min(self.model.num_real_batches, self.LIMIT_THREADS_AND_BLOCKS)
        blocks = max(self.model.num_real_batches, self.LIMIT_THREADS_AND_BLOCKS) // threads + 1
        return threads, blocks
