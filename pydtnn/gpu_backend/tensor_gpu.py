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

import ctypes

import numpy as np

try:
    from pydtnn.gpu_backend.libs import libcudnn as cudnn
except OSError:
    pass


class TensorGPU:
    def __init__(self, gpu_arr, tensor_format, cudnn_dtype, tensor_type="tensor", desc=None, gpudirect=False,
                 cublas=False):
        if len(gpu_arr.shape) == 2:
            self.shape = (*gpu_arr.shape, 1, 1)
        else:
            self.shape = gpu_arr.shape
        self.size = gpu_arr.size
        self.ary = gpu_arr
        if gpudirect:
            self.ptr_intp = np.intp(self.ary.base.get_device_pointer())
            self.ptr = ctypes.c_void_p(int(self.ary.base.get_device_pointer()))
        else:
            self.ptr = ctypes.c_void_p(int(gpu_arr.gpudata))
        if desc:
            self.desc = desc
        if tensor_type == "tensor":
            self.desc = cudnn.cudnnCreateTensorDescriptor()
            cudnn.cudnnSetTensor4dDescriptor(self.desc, tensor_format,
                                             cudnn_dtype, *self.shape)
        elif tensor_type == "filter":
            self.desc = cudnn.cudnnCreateFilterDescriptor()
            cudnn.cudnnSetFilter4dDescriptor(self.desc, cudnn_dtype,
                                             tensor_format, *self.shape)
        self.cublas = cublas

    def reshape(self, shape):
        self.ary = self.ary.reshape(shape)
        return self
