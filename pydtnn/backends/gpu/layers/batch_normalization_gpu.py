import numpy as np

# noinspection PyUnresolvedReferences
import pycuda.driver as drv
# noinspection PyUnresolvedReferences
import pycuda.gpuarray as gpuarray
# noinspection PyUnresolvedReferences
from pycuda.elementwise import ElementwiseKernel

from pydtnn.layers import BatchNormalization
from pydtnn.model import Model
from pydtnn.tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum
from pydtnn.backends.gpu.layers.layer_gpu import LayerGPU
from pydtnn.backends.gpu.libs import libcudnn as cudnn
from pydtnn.backends.gpu.tensor_gpu import TensorGPU
from pydtnn.utils.tensor import decode_tensor
from pydtnn.utils.types import shape_t

class BatchNormalizationGPU(LayerGPU, BatchNormalization):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The next attributes will be initialized later
        self.mode: int = None
        self.gamma_beta_mean_var_desc: int | None = None
        self.gamma_cpu: np.ndarray = None
        self.beta_cpu: np.ndarray = None
        self.dgamma_cpu: np.ndarray = None
        self.dbeta_cpu: np.ndarray = None
        self.save_mean: TensorGPU = None
        self.save_inv_var: TensorGPU = None
        self.factor: float = None

    def initialize(self, prev_shape: shape_t, x: TensorGPU) -> TensorGPU:
        super().initialize(prev_shape, x)
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
        self.gamma_cpu = np.full(shape_, self.gamma_init_val, self.model.dtype)
        gamma_gpu = gpuarray.to_gpu(self.gamma_cpu)
        self.gamma = TensorGPU(gamma_gpu, self.model.tensor_format, self.model.cudnn_dtype)
        # beta
        self.beta_cpu = np.full(shape_, self.beta_init_val, self.model.dtype)
        beta_gpu = gpuarray.to_gpu(self.beta_cpu)
        self.beta = TensorGPU(beta_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        self.nparams = self.gamma.size + self.beta.size + self.running_mean.size + self.running_var.size

        if self.model.gpudirect:
            self.dgamma_cpu, self.dgamma = TensorGPU.initialize_gpu_direct(drv, self.gamma.ary.shape, self.model.dtype,
                                                                           tensor_format=self.model.tensor_format,
                                                                           cudnn_dtype=self.model.cudnn_dtype,
                                                                           gpudirect=self.model.gpudirect)

            self.dbeta_cpu, self.dbeta = TensorGPU.initialize_gpu_direct(drv, self.beta.ary.shape, self.model.dtype,
                                                                         tensor_format=self.model.tensor_format,
                                                                         cudnn_dtype=self.model.cudnn_dtype,
                                                                         gpudirect=self.model.gpudirect)
        else:
            self.dgamma_cpu, self.dgamma = TensorGPU.initialize_not_gpu_direct(self.gamma.ary.shape, self.model.dtype,
                                                                               tensor_format=self.model.tensor_format,
                                                                               cudnn_dtype=self.model.cudnn_dtype,
                                                                               gpudirect=self.model.gpudirect)

            self.dbeta_cpu, self.dbeta = TensorGPU.initialize_not_gpu_direct(self.beta.ary.shape, self.model.dtype,
                                                                             tensor_format=self.model.tensor_format,
                                                                             cudnn_dtype=self.model.cudnn_dtype,
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
    # ---

    def forward(self, x):
        alpha, beta = 1.0, 0.0
        match self.model.mode:
            case Model.Mode.TRAIN:
                self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CUDNN)
                cudnn.cudnnBatchNormalizationForwardTraining(self.model.cudnn_handle, self.mode,
                                                             alpha, beta, x.desc, x.ptr,
                                                             self.y.desc, self.y.ptr, self.gamma_beta_mean_var_desc,
                                                             self.gamma.ptr,
                                                             self.beta.ptr, self.factor, self.running_mean.ptr,
                                                             self.running_var.ptr,
                                                             self.epsilon, self.save_mean.ptr, self.save_inv_var.ptr)
                self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
            case Model.Mode.EVALUATE:
                self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CUDNN)
                cudnn.cudnnBatchNormalizationForwardInference(self.model.cudnn_handle, self.mode,
                                                              alpha, beta, x.desc, x.ptr,
                                                              self.y.desc, self.y.ptr, self.gamma_beta_mean_var_desc,
                                                              self.gamma.ptr,
                                                              self.beta.ptr, self.running_mean.ptr, self.running_var.ptr,
                                                              self.epsilon)
                self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
            case _:
                raise RuntimeError(f"Unexpected model mode '{self.model.mode}'.")
        return self.y

    def backward(self, dy):
        alpha_dx, beta_dx, alpha_dgb, beta_dgb = 1.0, 0.0, 1.0, 0.0
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_CUDNN_DX)
        # Compute dx, dgamma, dbeta
        cudnn.cudnnBatchNormalizationBackward(self.model.cudnn_handle, self.mode,
                                              alpha_dx, beta_dx, alpha_dgb, beta_dgb,
                                              self.x.desc, self.x.ptr, dy.desc, dy.ptr,
                                              self.dx.desc, self.dx.ptr, self.gamma_beta_mean_var_desc,
                                              self.gamma.ptr, self.dgamma.ptr, self.dbeta.ptr, self.epsilon,
                                              self.save_mean.ptr, self.save_inv_var.ptr)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        # DtoH dw when data parallelism and no GPU direct/NCCL is used
        if self.model.comm and not self.model.gpudirect and not self.model.enable_nccl:
            self.model.stream.synchronize()
            self.dgamma.ary.get_async(self.stream_2, self.dgamma_cpu)
            self.dbeta.ary.get_async(self.stream_2, self.dbeta_cpu)
        return self.dx
