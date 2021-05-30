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

# noinspection PyUnresolvedReferences
import pycuda.driver as drv
# noinspection PyUnresolvedReferences
import pycuda.gpuarray as gpuarray

from pydtnn import utils
from pydtnn.layers import FC
from pydtnn.performance_models import *
from pydtnn.tracers import PYDTNN_OPS_FORWARD_CUBLAS_MATMUL, \
    PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_OPS_FORWARD_CUDNN_SUM_BIASES, \
    PYDTNN_OPS_BACKWARD_CUBLAS_MATMUL_DW, PYDTNN_OPS_BACKWARD_CUBLAS_MATVEC_DB, PYDTNN_OPS_BACKWARD_CUBLAS_MATMUL_DX
from .layer_gpu import LayerGPU
from ..libs import libcudnn as cudnn
from ..tensor_gpu import TensorGPU
from ..utils_gpu import matmul_gpu, matvec_gpu


class FCGPU(LayerGPU, FC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.matmul = matmul_gpu
        self.matvec = matvec_gpu

    def initialize(self, prev_shape, need_dx, x):
        super().initialize(prev_shape, need_dx, x)
        self.stream_2 = drv.Stream()

        # Weights
        self.weights_cpu = self.weights_initializer((*prev_shape, *self.shape), self.model.dtype)
        weights_gpu = gpuarray.to_gpu(self.weights_cpu)
        self.weights = TensorGPU(weights_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        if self.use_bias:
            # Biases
            self.biases_cpu = self.biases_initializer((1, *self.shape), self.model.dtype)
            biases_gpu = gpuarray.to_gpu(self.biases_cpu)
            self.biases = TensorGPU(biases_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        y_gpu = gpuarray.empty((self.model.batch_size, self.shape[0]), self.model.dtype)
        self.y = TensorGPU(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        if self.need_dx:
            dx_gpu = gpuarray.empty(x.ary.shape, self.model.dtype)
            self.dx = TensorGPU(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)
            self.dx.reshape((self.model.batch_size, *prev_shape))

        if self.model.gpudirect:
            self.dw_cpu = drv.aligned_zeros(self.weights.ary.shape, self.model.dtype)
            self.dw_cpu = dw_gpu = drv.register_host_memory(self.dw_cpu,
                                                            flags=drv.mem_host_register_flags.DEVICEMAP)
            if self.use_bias:
                self.db_cpu = drv.aligned_zeros(self.biases.ary.shape, self.model.dtype)
                self.db_cpu = db_gpu = drv.register_host_memory(self.db_cpu,
                                                                flags=drv.mem_host_register_flags.DEVICEMAP)
        else:
            self.dw_cpu = np.zeros(self.weights.ary.shape, self.model.dtype)
            dw_gpu = gpuarray.empty(self.dw_cpu.shape, self.model.dtype)
            if self.use_bias:
                self.db_cpu = np.zeros(self.biases.ary.shape, self.model.dtype)
                db_gpu = gpuarray.empty(self.db_cpu.shape, self.model.dtype)

        self.dw = TensorGPU(dw_gpu, self.model.tensor_format, self.model.cudnn_dtype,
                            gpudirect=self.model.gpudirect)
        if self.use_bias:
            # noinspection PyUnboundLocalVariable
            self.db = TensorGPU(db_gpu, self.model.tensor_format, self.model.cudnn_dtype,
                                gpudirect=self.model.gpudirect)

        self.one_vec_gpu = gpuarray.to_gpu(np.ones((self.model.batch_size,), self.model.dtype))
        self.nparams = self.weights.size + (self.biases.size if self.use_bias else 0)

        self.fwd_time = \
            matmul_time(m=self.model.batch_size, n=self.weights_cpu.shape[1], k=self.weights_cpu.shape[0],
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype)
        self.bwd_time = \
            matmul_time(m=self.weights_cpu.shape[0], n=self.weights_cpu.shape[1], k=self.model.batch_size,
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype) + \
            matmul_time(m=self.model.batch_size, n=self.weights_cpu.shape[0], k=self.weights_cpu.shape[1],
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype) if need_dx else 0

    def forward(self, x):
        m = x.ary.shape[0]
        n = ldb = ldc = self.weights.ary.shape[1]
        k = lda = x.ary.shape[1]
        trans_a, trans_b, alpha, beta = 'N', 'N', 1.0, 0.0

        # Compute a' = x @ weights
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT,
                                     self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_CUBLAS_MATMUL)
        self.matmul(self.model.cublas_handle, trans_b, trans_a, n, m, k, alpha,
                    self.weights.ary.gpudata, ldb,
                    x.ary.gpudata, lda, beta,
                    self.y.ary.gpudata, ldc, self.model.dtype)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        if self.use_bias:
            alpha, beta = 1.0, 1.0
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT,
                                         self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_CUDNN_SUM_BIASES)
            # Compute a = a' + biases
            cudnn.cudnnAddTensor(self.model.cudnn_handle, alpha, self.biases.desc,
                                 self.biases.ptr, beta, self.y.desc, self.y.ptr)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        return self.y

    def backward(self, dy):
        # Compute dw
        m = lda = self.x.ary.shape[1]
        n = ldb = ldc = dy.ary.shape[1]
        k = dy.ary.shape[0]
        trans_a, trans_b, alpha, beta = 'T', 'N', 1.0, 0.0

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT,
                                     self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_CUBLAS_MATMUL_DW)
        self.matmul(self.model.cublas_handle, trans_b, trans_a, n, m, k, alpha,
                    dy.ary.gpudata, ldb, self.x.ary.gpudata, lda, beta,
                    self.dw.ptr_intp if self.model.gpudirect else self.dw.ary.gpudata, ldc, self.model.dtype)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        # DtoH dw when data parallelism and no GPU direct/NCCL is used
        if self.model.comm and not self.model.gpudirect and not self.model.enable_nccl:
            self.model.stream.synchronize()
            self.dw.ary.get_async(self.stream_2, self.dw_cpu)

        if self.use_bias:
            # Compute db
            m = dy.ary.shape[0]
            n = lda = dy.ary.shape[1]
            trans_a, alpha, beta, inc_x, inc_y = 'N', 1.0, 0.0, 1, 1

            self.model.tracer.emit_event(PYDTNN_OPS_EVENT,
                                         self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_CUBLAS_MATVEC_DB)
            self.matvec(self.model.cublas_handle, trans_a, n, m, alpha,
                        dy.ary.gpudata, lda, self.one_vec_gpu.gpudata, inc_x, beta,
                        self.db.ptr_intp if self.model.gpudirect else self.db.ary.gpudata,
                        inc_y, self.model.dtype)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

            # DtoH db when data parallelism and no GPU direct/NCCL is used
            if self.model.comm and not self.model.gpudirect and not self.model.enable_nccl:
                self.model.stream.synchronize()
                self.db.ary.get_async(self.stream_2, self.db_cpu)

        if self.need_dx:
            # Compute dx
            m = dy.ary.shape[0]
            n = ldc = self.weights.ary.shape[0]
            k = lda = ldb = dy.ary.shape[1]
            trans_a, trans_b, alpha, beta = 'N', 'T', 1.0, 0.0

            self.model.tracer.emit_event(PYDTNN_OPS_EVENT,
                                         self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_CUBLAS_MATMUL_DX)
            self.matmul(self.model.cublas_handle, trans_b, trans_a, n, m, k, alpha,
                        self.weights.ary.gpudata, ldb,
                        dy.ary.gpudata, lda, beta,
                        self.dx.ary.gpudata, ldc, self.model.dtype)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
            return self.dx
