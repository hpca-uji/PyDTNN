from typing import Any
import numpy as np
import pycuda.driver as drv  # type: ignore
import pycuda.gpuarray as gpuarray  # type: ignore

from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.model import Model
from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum
from pydtnn.backends.gpu.layers.layer import LayerGPU
from pydtnn.libs import libcudnn as cudnn
from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU
from pydtnn.utils.constants import ArrayShape, Parameters


class BatchNormalizationGPU(BatchNormalization[TensorGPU], LayerGPU):

    @property
    def _ary_prop(self) -> set[str]:
        return {Parameters.RUNNING_MEAN, 
                Parameters.RUNNING_VAR, 
                *super()._ary_prop}
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # NOTE: The next attributes will be initialized later
        self.gamma_beta_mean_var_desc: int | None = None
        self.mode: int = None  # type: ignore
        self.gamma_cpu: np.ndarray = None  # type: ignore
        self.beta_cpu: np.ndarray = None  # type: ignore
        self.dgamma_cpu: np.ndarray = None  # type: ignore
        self.dbeta_cpu: np.ndarray = None  # type: ignore
        self.save_mean: TensorGPU = None  # type: ignore
        self.save_inv_var: TensorGPU = None  # type: ignore
        self.factor: float = None  # type: ignore

    def initialize(self, prev_shape: ArrayShape, x: TensorGPU):
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
            self.ci, self.hi, self.wi = self.model.decode_shape(prev_shape)
        else:
            self.ci, = prev_shape

        shape_ = (1, self.ci, 1, 1)

        # gamma
        self.gamma_cpu = np.full(shape_, self.gamma_init_val, self.model.dtype)
        gamma_gpu = gpuarray.to_gpu(self.gamma_cpu)
        self.gamma = TensorGPU(gamma_gpu, self.model.tensor_format, self.model.cudnn_dtype)
        # beta
        self.beta_cpu = np.full(shape_, self.beta_init_val, self.model.dtype)
        beta_gpu = gpuarray.to_gpu(self.beta_cpu)
        self.beta = TensorGPU(beta_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        self.dgamma_cpu, self.dgamma = TensorGPU.initialize(self.gamma.ary.shape, self.model.dtype,
                                                            tensor_format=self.model.tensor_format,
                                                            cudnn_dtype=self.model.cudnn_dtype,
                                                            gpudirect=self.model.gpudirect, 
                                                            drv=(drv if self.model.gpudirect else None))
        self.dbeta_cpu, self.dbeta = TensorGPU.initialize(self.beta.ary.shape, self.model.dtype,
                                                          tensor_format=self.model.tensor_format,
                                                          cudnn_dtype=self.model.cudnn_dtype,
                                                          gpudirect=self.model.gpudirect, 
                                                          drv=(drv if self.model.gpudirect else None))

        running_mean_gpu = gpuarray.to_gpu(self.moving_mean_initializer(shape_, self.model.dtype))
        self.running_mean = TensorGPU(running_mean_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        running_var_gpu = gpuarray.to_gpu(self.moving_variance_initializer(shape_, self.model.dtype))
        self.running_var = TensorGPU(running_var_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        save_mean_gpu = gpuarray.empty(shape_, self.model.dtype)
        self.save_mean = TensorGPU(save_mean_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        save_inv_var_gpu = gpuarray.empty(shape_, self.model.dtype)
        self.save_inv_var = TensorGPU(save_inv_var_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        self.factor = 1.0 - self.momentum

        self.nparams = self.gamma.size + self.beta.size + self.running_mean.size + self.running_var.size
    # ---

    def forward(self, x: TensorGPU) -> TensorGPU:
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

    def backward(self, dy: TensorGPU) -> TensorGPU:
        self.x: TensorGPU
        
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
    # -----

    def _export_gamma_beta(self, key: str) -> Any:
        value = getattr(self, key)
        gpu_ary = value.ary
        cpu_ary = gpu_ary.get()
        return np.asarray(np.squeeze(cpu_ary, axis=(0, 2, 3)), dtype=np.float64, order="C", copy=True)
    # ---

    def _export_prop(self, key: str) -> Any:
        match key:
            case Parameters.GAMMA | Parameters.DGAMMA | Parameters.BETA | Parameters.DBETA :
                return self._export_gamma_beta(key)
            case _:
                return super()._export_prop(key)
    # ----

    def _import_gamma_beta(self, key: str, value: Any) -> None:
        attribute = getattr(self, key)
        cpu_ary = np.asarray(np.expand_dims(value, axis=(0, 2, 3)), dtype=self.model.dtype, order="C", copy=None)
        attribute.ary.set(cpu_ary)
        return
    # ---

    def _import_prop(self, key: str, value) -> None:
        match key:
            case Parameters.GAMMA | Parameters.DGAMMA | Parameters.BETA | Parameters.DBETA :
                return self._import_gamma_beta(key, value)
            case _:
                return super()._import_prop(key, value)
    # -----
