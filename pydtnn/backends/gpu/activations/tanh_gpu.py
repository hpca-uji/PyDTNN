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
#  with this program. If not, see <https://www.gnu.org/licenses/>.
#

from pydtnn.activations import Tanh
from .activation_gpu import ActivationGPU
from ..tensor_gpu import TensorGPU

# noinspection PyUnresolvedReferences
import pycuda.gpuarray as gpuarray
# noinspection PyUnresolvedReferences
from ..libs import libcudnn as cudnn


class TanhGPU(ActivationGPU, Tanh):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.act_desc = None

    def initialize(self, prev_shape, need_dx, x):
        self.shape = prev_shape
        self.need_dx = need_dx
        self.x = x

        self.act_desc = cudnn.cudnnCreateActivationDescriptor()

        mode = cudnn.cudnnActivationMode['CUDNN_ACTIVATION_TANH']
        nan = cudnn.cudnnNanPropagation['CUDNN_NOT_PROPAGATE_NAN']
        cudnn.cudnnSetActivationDescriptor(self.act_desc, mode, nan, 0.0)

        # Activations y
        y_gpu = gpuarray.empty(x.ary.shape, self.model.dtype)
        self.y = TensorGPU(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        # Derivative dx
        if self.need_dx:
            dx_gpu = gpuarray.empty(x.ary.shape, self.model.dtype)
            self.dx = TensorGPU(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)

    def forward(self, x):
        alpha, beta = 1.0, 0.0
        cudnn.cudnnActivationForward(self.model.cudnn_handle, self.act_desc, alpha,
                                     x.desc, x.ptr, beta,
                                     self.y.desc, self.y.ptr)
        return self.y

    def backward(self, dy):
        if self.need_dx:
            alpha, beta = 1.0, 0.0
            cudnn.cudnnActivationBackward(self.model.cudnn_handle, self.act_desc, alpha,
                                          self.y.desc, self.y.ptr,
                                          dy.desc, dy.ptr,
                                          self.x.desc, self.x.ptr, beta,
                                          self.dx.desc, self.dx.ptr)
            return self.dx
