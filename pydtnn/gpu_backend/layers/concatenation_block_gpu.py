
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

# noinspection PyUnresolvedReferences
from ..libs import libcudnn as cudnn
# noinspection PyUnresolvedReferences
import pycuda.gpuarray as gpuarray
# noinspection PyUnresolvedReferences
from pycuda.elementwise import ElementwiseKernel

from pydtnn import layers
from pydtnn.performance_models import *
from pydtnn.tracers import PYDTNN_MDL_EVENT, PYDTNN_MDL_EVENTS, PYDTNN_MDL_FORWARD, PYDTNN_MDL_BACKWARD, \
    PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_OPS_FORWARD_CONCAT, \
    PYDTNN_OPS_BACKWARD_ELTW_SUM, PYDTNN_OPS_BACKWARD_SPLIT
from .layer_gpu_mixin import LayerGPUMixin
from ..tensor_gpu import TensorGPU


class ConcatenationBlockGPU(LayerGPUMixin, layers.ConcatenationBlock):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.concat = None
        self.split = None
        self.dy = None

    def initialize(self, prev_shape, need_dx, x):
        # super().initialize(prev_shape, need_dx, x)
        need_dx = True
        self.x = x
        self.prev_shape = prev_shape 
        self.out_shapes = []
        for p in self.paths:
            for i, layer in enumerate(p):
                layer.set_model(self.model)
                layer.initialize(prev_shape, need_dx, x)
                x = layer.y
                prev_shape = layer.shape
                self.fwd_time += layer.fwd_time
                self.bwd_time += layer.bwd_time
                self.nparams += layer.nparams
            self.out_shapes.append(prev_shape)
            prev_shape = self.prev_shape
            x = self.x
        assert all([o[1:] == self.out_shapes[0][1:] for o in self.out_shapes])
        self.out_co = [s[0] for s in self.out_shapes]
        self.idx_co = np.cumsum(self.out_co, axis=0)
        self.shape = (sum(self.out_co), *self.out_shapes[0][1:])
        self.concat = ElementwiseKernel(
            "T *dst, T *src, int N, int C, int H, int W, int first_c, int last_c".replace("T",
                                                                                          {np.float32: "float",
                                                                                           np.float64: "double"}[
                                                                                              self.model.dtype]),
            """int c_ = i / (H*W) % C;
               if (first_c <= c_ && c_ < last_c) {
                   int w_ = i % W;
                   int h_ = i / W % H;
                   int n_ = i / (C*H*W) % N;
                   int i_ = n_ * (last_c-first_c) * H * W + (c_-first_c) * H * W + h_ * W + w_;
                   dst[i] = src[i_];
               }
            """,
            "concat")

        self.split = ElementwiseKernel(
            "T *src, T *dst, int N, int C, int H, int W, int first_c, int last_c".replace("T",
                                                                                          {np.float32: "float",
                                                                                           np.float64: "double"}[
                                                                                              self.model.dtype]),
            """int c_ = i / (H*W) % C;
               if (first_c <= c_ && c_ < last_c) {
                   int w_ = i % W;
                   int h_ = i / W % H;
                   int n_ = i / (C*H*W) % N;
                   int i_ = n_ * (last_c-first_c) * H * W + (c_-first_c) * H * W + h_ * W + w_;
                   dst[i_] = src[i];
               }
            """,
            "split")
        # Activations y
        y_gpu = gpuarray.empty((self.model.batch_size, *self.shape), self.model.dtype)
        self.y = TensorGPU(y_gpu, self.model.tensor_fmt, self.model.cudnn_dtype)
        # Derivative dy
        self.dy = []
        for i, p in enumerate(self.paths):
            dy_gpu = gpuarray.empty((self.model.batch_size, *self.out_shapes[i]), self.model.dtype)
            self.dy.append(TensorGPU(dy_gpu, self.model.tensor_fmt, self.model.cudnn_dtype))

    def forward(self, x):
        for i, p in enumerate(self.paths):
            x_i = x
            for layer in p:
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_FORWARD)
                x_i = layer.forward(x_i)
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, 0)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_CONCAT)
            self.concat(self.y.ary, x_i.ary, self.model.batch_size, *self.shape,
                        0 if i == 0 else self.idx_co[i - 1], self.idx_co[i])
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        return self.y

    def backward(self, dy):
        for i, p in enumerate(self.paths):
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_SPLIT)
            self.split(dy.ary, self.dy[i].ary, self.model.batch_size, *self.shape,
                       0 if i == 0 else self.idx_co[i - 1], self.idx_co[i])
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

            dx_i = self.dy[i]
            for layer in reversed(p):
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_BACKWARD)
                dx_i = layer.backward(dx_i)
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, 0)
            if i == 0:
                dx = dx_i
            else:
                alpha, beta = 1.0, 1.0
                self.model.tracer.emit_event(PYDTNN_OPS_EVENT,
                                             self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_ELTW_SUM)
                cudnn.cudnnAddTensor(self.model.cudnn_handle, alpha, dx_i.desc,
                                     dx_i.ptr, beta, dx.desc, dx.ptr)
                self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        return dx
