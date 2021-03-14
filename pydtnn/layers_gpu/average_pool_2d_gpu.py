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

from .layer_gpu import LayerGPU

from .. import layers
from ..performance_models import *
from ..tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_OPS_FORWARD_CUDNN, PYDTNN_OPS_BACKWARD_CUDNN_DX
from ..utils import TensorGPU

try:
    # noinspection PyUnresolvedReferences
    import libcudnn.libcudnn as cudnn
    # noinspection PyUnresolvedReferences
    import pycuda.gpuarray as gpuarray
except (ImportError, ModuleNotFoundError):
    pass


class AveragePool2DGPU(LayerGPU, layers.AveragePool2D):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool_desc = None
        self.pool_shape = None

    def initialize(self, prev_shape, need_dx, x):
        super().initialize(prev_shape, need_dx, x)
        self.ci, self.hi, self.wi = prev_shape
        if self.pool_shape[0] == 0:
            self.pool_shape = (self.hi, self.pool_shape[1])
        if self.pool_shape[1] == 0:
            self.pool_shape = (self.pool_shape[0], self.wi)
        self.kh, self.kw = self.pool_shape
        self.co = self.ci

        pool_mode = cudnn.cudnnPoolingMode['CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING']
        nan_prop = cudnn.cudnnNanPropagation['CUDNN_NOT_PROPAGATE_NAN']

        self.pool_desc = cudnn.cudnnCreatePoolingDescriptor()
        cudnn.cudnnSetPooling2dDescriptor(self.pool_desc, pool_mode, nan_prop,
                                          self.kh, self.kw, self.vpadding, self.hpadding,
                                          self.vstride, self.hstride)
        # Get output dimensions
        _, _, self.ho, self.wo = cudnn.cudnnGetPooling2dForwardOutputDim(self.pool_desc, x.desc)
        self.shape = (self.co, self.ho, self.wo)

        # Activations y
        y_gpu = gpuarray.empty((self.model.batch_size, *self.shape), self.model.dtype)
        self.y = TensorGPU(y_gpu, self.model.tensor_fmt, self.model.cudnn_dtype)

        if self.need_dx:
            # Derivative dx
            dx_gpu = gpuarray.empty(self.x.ary.shape, self.model.dtype)
            self.dx = TensorGPU(dx_gpu, self.model.tensor_fmt, self.model.cudnn_dtype)

        self.fwd_time = \
            im2col_time(m=(self.kh * self.kw), n=(self.model.batch_size * self.ho * self.wo * self.ci),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype)
        self.bwd_time = \
            col2im_time(m=(self.kh * self.kw), n=(self.model.batch_size * self.ho * self.wo * self.ci),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype) if need_dx else 0

    def forward(self, x):
        alpha, beta = 1.0, 0.0
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_CUDNN)
        cudnn.cudnnPoolingForward(self.model.cudnn_handle, self.pool_desc, alpha,
                                  x.desc, x.ptr, beta,
                                  self.y.desc, self.y.ptr)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        return self.y

    def backward(self, dy):
        if self.need_dx:
            alpha, beta = 1.0, 0.0
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_CUDNN_DX)
            # Compute dx
            cudnn.cudnnPoolingBackward(self.model.cudnn_handle, self.pool_desc, alpha,
                                       self.y.desc, self.y.ptr,
                                       dy.desc, dy.ptr,
                                       self.x.desc, self.x.ptr,
                                       beta, self.dx.desc, self.dx.ptr)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
            return self.dx