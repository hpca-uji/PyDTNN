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

import numpy as np

from pydtnn.activations import Arctanh
from .activation_gpu import ActivationGPU
from ..tensor_gpu import TensorGPU

# noinspection PyUnresolvedReferences
import pycuda.gpuarray as gpuarray
# noinspection PyUnresolvedReferences
from ..libs import libcudnn as cudnn
# noinspection PyUnresolvedReferences
from pycuda.elementwise import ElementwiseKernel


class ArctanhGPU(ActivationGPU, Arctanh):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.atanh = None
        self.datanh = None

    def initialize(self, prev_shape, need_dx, x):
        super().initialize(prev_shape, need_dx, x)

        self.atanh = ElementwiseKernel(
            "T *in, T *out".replace("T", {np.float32: "float", np.float64: "double"}[self.model.dtype]),
            "out[i] = %s(in[i]);" % {np.float32: "atanhf", np.float64: "atanh"}[self.model.dtype],
            "atanh")

        self.datanh = ElementwiseKernel(
            "T *in, T *out".replace("T", {np.float32: "float", np.float64: "double"}[self.model.dtype]),
            "out[i] = 1.0 / (1.0 + %s(in[i], 2));" % {np.float32: "powf", np.float64: "pow"}[self.model.dtype],
            "datanh")

        # Activations y
        y_gpu = gpuarray.empty(x.ary.shape, self.model.dtype)
        self.y = TensorGPU(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        # Derivative dx
        if self.need_dx:
            dx_gpu = gpuarray.empty(x.ary.shape, self.model.dtype)
            self.dx = TensorGPU(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)

    def forward(self, x):
        self.atanh(x.ary, self.y.ary, stream=self.model.stream)
        return self.y

    def backward(self, dy):
        if self.need_dx:
            # Compute dx
            self.datanh(dy.ary, self.dx.ary, stream=self.model.stream)
            return self.dx
