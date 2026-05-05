import logging
from typing import Any

import numpy as np
import pycuda.driver as drv  # type: ignore
from pycuda import gpuarray  # type: ignore

from pydtnn.backends.pycuda.layers.layer import LayerPycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.libs import cudnn as cudnn
from pydtnn.model import Model
from pydtnn.tracers.events import (PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT,
                                   PYDTNN_OPS_EVENTS, PYDTNN_OPS_EVENT_enum)
from pydtnn.utils.constants import ArrayShape, Parameters

__all__=(
    "BatchNormalizationPycuda",
)

logger = logging.getLogger(__name__)


class BatchNormalizationPycuda(BatchNormalization[TensorArray], LayerPycuda):

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
        self.save_mean: TensorArray = None  # type: ignore
        self.save_inv_var: TensorArray = None  # type: ignore
        self.factor: float = None  # type: ignore

    def _model_init(self, prev_shape: ArrayShape, x: TensorArray):
        super()._model_init(prev_shape, x)
        self.stream_2 = drv.Stream()

        # Activations y
        y_gpu = gpuarray.zeros(x.shape, self.model.dtype)
        self.y = TensorArray(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)
        self.memory_used += self.y.nbytes

        # Derivative dx
        dx_gpu = gpuarray.zeros(x.shape, self.model.dtype)
        self.dx = TensorArray(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)
        self.memory_used += self.dx.nbytes

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
        self.gamma = TensorArray(gamma_gpu, self.model.tensor_format, self.model.cudnn_dtype)
        self.memory_used += self.gamma.nbytes

        # beta
        self.beta_cpu = np.full(shape_, self.beta_init_val, self.model.dtype)
        beta_gpu = gpuarray.to_gpu(self.beta_cpu)
        self.beta = TensorArray(beta_gpu, self.model.tensor_format, self.model.cudnn_dtype)
        self.memory_used += self.beta.nbytes

        self.dgamma_cpu, self.dgamma = TensorArray.new(self.gamma.shape, self.model.dtype,
                                                       tensor_format=self.model.tensor_format,
                                                       cudnn_dtype=self.model.cudnn_dtype,
                                                       gpudirect=self.model.gpudirect,
                                                       drv=(drv if self.model.gpudirect else None))
        self.memory_used += self.dgamma.nbytes

        self.dbeta_cpu, self.dbeta = TensorArray.new(self.beta.shape, self.model.dtype,
                                                     tensor_format=self.model.tensor_format,
                                                     cudnn_dtype=self.model.cudnn_dtype,
                                                     gpudirect=self.model.gpudirect,
                                                     drv=(drv if self.model.gpudirect else None))
        self.memory_used += self.dbeta.nbytes

        running_mean_gpu = gpuarray.to_gpu(self.moving_mean_initializer(shape_, self.model.dtype))
        self.running_mean = TensorArray(running_mean_gpu, self.model.tensor_format, self.model.cudnn_dtype)
        self.memory_used += self.running_mean.nbytes

        running_var_gpu = gpuarray.to_gpu(self.moving_variance_initializer(shape_, self.model.dtype))
        self.running_var = TensorArray(running_var_gpu, self.model.tensor_format, self.model.cudnn_dtype)
        self.memory_used += self.running_var.nbytes

        save_mean_gpu = gpuarray.zeros(shape_, self.model.dtype)
        self.save_mean = TensorArray(save_mean_gpu, self.model.tensor_format, self.model.cudnn_dtype)
        self.memory_used += self.save_mean.nbytes

        save_inv_var_gpu = gpuarray.zeros(shape_, self.model.dtype)
        self.save_inv_var = TensorArray(save_inv_var_gpu, self.model.tensor_format, self.model.cudnn_dtype)
        self.memory_used += self.save_inv_var.nbytes

        self.factor = 1.0 - self.momentum

        self.nparams = self.gamma.size + self.beta.size + self.running_mean.size + self.running_var.size

        self.memory_used += self.gamma.nbytes

    def forward(self, x: TensorArray) -> TensorArray:
        alpha, beta = 1.0, 0.0
        match self.model.mode:
            case Model.Mode.TRAIN:
                self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CUDNN)
                cudnn.cudnnBatchNormalizationForwardTraining(self.model.cudnn_handle, self.mode,
                                                             alpha, beta, x.desc, x.ptr_voidp,
                                                             self.y.desc, self.y.ptr_voidp, self.gamma_beta_mean_var_desc,
                                                             self.gamma.ptr_voidp,
                                                             self.beta.ptr_voidp, self.factor, self.running_mean.ptr_voidp,
                                                             self.running_var.ptr_voidp,
                                                             self.epsilon, self.save_mean.ptr_voidp, self.save_inv_var.ptr_voidp)
                self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
            case Model.Mode.EVALUATE:
                self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CUDNN)
                cudnn.cudnnBatchNormalizationForwardInference(self.model.cudnn_handle, self.mode,
                                                              alpha, beta, x.desc, x.ptr_voidp,
                                                              self.y.desc, self.y.ptr_voidp, self.gamma_beta_mean_var_desc,
                                                              self.gamma.ptr_voidp,
                                                              self.beta.ptr_voidp, self.running_mean.ptr_voidp, self.running_var.ptr_voidp,
                                                              self.epsilon)
                self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
            case _:
                raise RuntimeError(f"Unexpected model mode '{self.model.mode}'.")
        return self.y

    def backward(self, dy: TensorArray) -> TensorArray:
        self.x: TensorArray

        alpha_dx, beta_dx, alpha_dgb, beta_dgb = 1.0, 0.0, 1.0, 0.0
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_CUDNN_DX)
        # Compute dx, dgamma, dbeta
        cudnn.cudnnBatchNormalizationBackward(self.model.cudnn_handle, self.mode,
                                              alpha_dx, beta_dx, alpha_dgb, beta_dgb,
                                              self.x.desc, self.x.ptr_voidp, dy.desc, dy.ptr_voidp,
                                              self.dx.desc, self.dx.ptr_voidp, self.gamma_beta_mean_var_desc,
                                              self.gamma.ptr_voidp, self.dgamma.ptr_voidp, self.dbeta.ptr_voidp, self.epsilon,
                                              self.save_mean.ptr_voidp, self.save_inv_var.ptr_voidp)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        # DtoH dw when data parallelism and no GPU direct/NCCL is used
        if self.model.comm and not self.model.gpudirect and not self.model.enable_nccl:
            # self.model.stream.synchronize()
            self.dgamma.get_async(self.stream_2, self.dgamma_cpu)
            self.dbeta.get_async(self.stream_2, self.dbeta_cpu)
        return self.dx

    def _export_gamma_beta(self, key: str) -> Any:
        value = getattr(self, key)
        gpu_ary = value
        cpu_ary = gpu_ary.get()
        return np.asarray(cpu_ary, dtype=np.float64, order="C", copy=True)

    def _export_prop(self, key: str) -> Any:
        match key:
            case Parameters.GAMMA | Parameters.DGAMMA | Parameters.BETA | Parameters.DBETA:
                return self._export_gamma_beta(key)
            case _:
                return super()._export_prop(key)

    def _import_gamma_beta(self, key: str, value: Any) -> None:
        attribute = getattr(self, key)
        attribute.set(value)
        return

    def _import_prop(self, key: str, value) -> None:
        match key:
            case Parameters.GAMMA | Parameters.DGAMMA | Parameters.BETA | Parameters.DBETA:
                return self._import_gamma_beta(key, value)
            case _:
                return super()._import_prop(key, value)
