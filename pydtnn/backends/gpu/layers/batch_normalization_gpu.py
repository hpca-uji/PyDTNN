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

import numpy as np

# noinspection PyUnresolvedReferences
import pycuda.driver as drv
# noinspection PyUnresolvedReferences
import pycuda.gpuarray as gpuarray
# noinspection PyUnresolvedReferences
from pycuda.elementwise import ElementwiseKernel

from pydtnn.layers import BatchNormalization
from pydtnn.model import EVALUATE_MODE, TRAIN_MODE
from pydtnn.tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_OPS_FORWARD_CUDNN, PYDTNN_OPS_BACKWARD_CUDNN_DX
from .layer_gpu import LayerGPU
from ..libs import libcudnn as cudnn
from ..tensor_gpu import TensorGPU
from pydtnn.utils import decode_tensor

class BatchNormalizationGPU(LayerGPU, BatchNormalization):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The next attributes will be initialized later
        self.mode = None
        self.gamma_beta_mean_var_desc = None
        self.gamma_value = None
        self.beta_value = None
        self.dgamma_cpu = None
        self.dbeta_cpu = None
        self.save_mean = None
        self.save_inv_var = None
        self.factor = None

    def initialize(self, prev_shape, need_dx, x):
        super().initialize(prev_shape, need_dx, x)
        self.stream_2 = drv.Stream()

        # Activations y
        y_gpu = gpuarray.empty(x.ary.shape, self.model.dtype)
        self.y = TensorGPU(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        # Derivative dx
        dx_gpu = gpuarray.zeros(x.ary.shape, self.model.dtype)
        self.dx = TensorGPU(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        self.spatial = len(self.shape) > 2
        self.mode = \
            cudnn.cudnnBatchNormMode['CUDNN_BATCHNORM_SPATIAL' if self.spatial else 'CUDNN_BATCHNORM_PER_ACTIVATION']

        self.gamma_beta_mean_var_desc = cudnn.cudnnCreateTensorDescriptor()
        cudnn.cudnnDeriveBNTensorDescriptor(self.gamma_beta_mean_var_desc,
                                            x.desc, self.mode)
        if self.spatial:
            self.hi, self.wi, self.ci = decode_tensor(prev_shape, self.model.tensor_format)
            shape_ = (1, self.ci, 1, 1)  # 1 x C x 1 x 1
        else:
            (self.ci,) = decode_tensor(prev_shape, self.model.tensor_format)
            shape_ = (1, self.ci, 1, 1)  # 1 x C x H x W

        # gamma
        self.gamma_value = np.full(shape_, self.gamma_init_val, self.model.dtype)
        gamma_gpu = gpuarray.to_gpu(self.gamma_value)
        self.gamma = TensorGPU(gamma_gpu, self.model.tensor_format, self.model.cudnn_dtype)
        # beta
        self.beta_value = np.full(shape_, self.beta_init_val, self.model.dtype)
        beta_gpu = gpuarray.to_gpu(self.beta_value)
        self.beta = TensorGPU(beta_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        self.nparams = self.gamma.size + self.beta.size + self.running_mean.size + self.running_var.size

        if self.model.gpudirect:
            self.dgamma_cpu = drv.aligned_zeros(self.gamma.ary.shape, self.model.dtype)
            self.dgamma_cpu = dgamma_gpu = drv.register_host_memory(self.dgamma_cpu,
                                                                    flags=drv.mem_host_register_flags.DEVICEMAP)
            self.dbeta_cpu = drv.aligned_zeros(self.beta.ary.shape, self.model.dtype)
            self.dbeta_cpu = dbeta_gpu = drv.register_host_memory(self.dbeta_cpu,
                                                                  flags=drv.mem_host_register_flags.DEVICEMAP)
        else:
            self.dgamma_cpu = np.zeros(self.gamma.ary.shape, self.model.dtype)
            dgamma_gpu = gpuarray.empty(self.gamma.ary.shape, self.model.dtype)
            self.dbeta_cpu = np.zeros(self.beta.ary.shape, self.model.dtype)
            dbeta_gpu = gpuarray.empty(self.beta.ary.shape, self.model.dtype)

        self.dgamma = TensorGPU(dgamma_gpu, self.model.tensor_format, self.model.cudnn_dtype,
                                gpudirect=self.model.gpudirect)
        self.dbeta = TensorGPU(dbeta_gpu, self.model.tensor_format, self.model.cudnn_dtype,
                               gpudirect=self.model.gpudirect)

        running_mean_gpu = gpuarray.to_gpu(self.moving_mean_initializer(shape_, self.model.dtype))
        self.running_mean = TensorGPU(running_mean_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        running_var_gpu = gpuarray.to_gpu(self.moving_variance_initializer(shape_, self.model.dtype))
        self.running_var = TensorGPU(running_var_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        save_mean_gpu = gpuarray.empty(shape_, self.model.dtype)
        self.save_mean = TensorGPU(save_mean_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        save_inv_var_gpu = gpuarray.empty(shape_, self.model.dtype)
        self.save_inv_var = TensorGPU(save_inv_var_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        self.factor = 1.0 - self.momentum

    def forward(self, x):
        alpha, beta = 1.0, 0.0
        if self.model.mode == TRAIN_MODE:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_CUDNN)
            cudnn.cudnnBatchNormalizationForwardTraining(self.model.cudnn_handle, self.mode,
                                                         alpha, beta, x.desc, x.ptr,
                                                         self.y.desc, self.y.ptr, self.gamma_beta_mean_var_desc,
                                                         self.gamma.ptr,
                                                         self.beta.ptr, self.factor, self.running_mean.ptr,
                                                         self.running_var.ptr,
                                                         self.epsilon, self.save_mean.ptr, self.save_inv_var.ptr)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        elif self.model.mode == EVALUATE_MODE:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_CUDNN)
            cudnn.cudnnBatchNormalizationForwardInference(self.model.cudnn_handle, self.mode,
                                                          alpha, beta, x.desc, x.ptr,
                                                          self.y.desc, self.y.ptr, self.gamma_beta_mean_var_desc,
                                                          self.gamma.ptr,
                                                          self.beta.ptr, self.running_mean.ptr, self.running_var.ptr,
                                                          self.epsilon)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)
        else:
            raise ValueError("Unexpected model mode")
        return self.y

    def backward(self, dy):
        alpha_dx, beta_dx, alpha_dgb, beta_dgb = 1.0, 0.0, 1.0, 0.0
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_BACKWARD_CUDNN_DX)
        # Compute dx, dgamma, dbeta
        cudnn.cudnnBatchNormalizationBackward(self.model.cudnn_handle, self.mode,
                                              alpha_dx, beta_dx, alpha_dgb, beta_dgb,
                                              self.x.desc, self.x.ptr, dy.desc, dy.ptr,
                                              self.dx.desc, self.dx.ptr, self.gamma_beta_mean_var_desc,
                                              self.gamma.ptr, self.dgamma.ptr, self.dbeta.ptr, self.epsilon,
                                              self.save_mean.ptr, self.save_inv_var.ptr)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, 0)

        # DtoH dw when data parallelism and no GPU direct/NCCL is used
        if self.model.comm and not self.model.gpudirect and not self.model.enable_nccl:
            self.model.stream.synchronize()
            self.dgamma.ary.get_async(self.stream_2, self.dgamma_cpu)
            self.dbeta.ary.get_async(self.stream_2, self.dbeta_cpu)
        return self.dx
