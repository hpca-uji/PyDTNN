"""PyCUDA implementation of the Batch Normalization layer."""

import logging
from typing import Any

import numpy as np
import pycuda.driver as drv
from pycuda import gpuarray  # pyright: ignore[reportAttributeAccessIssue]

from pydtnn.backends.pycuda.layers.abstract.layer import LayerPycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.libs import cudnn as cudnn
from pydtnn.model import Model
from pydtnn.tracers.events import (PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT,
                                   PYDTNN_OPS_EVENTS, OpsEventEnum)
from pydtnn.utils.constants import ArrayShape, Parameters

__all__ = ("BatchNormalizationPycuda",)

logger = logging.getLogger(__name__)


class BatchNormalizationPycuda(BatchNormalization[TensorArray], LayerPycuda):
    """PyCUDA-accelerated Batch Normalization layer using cuDNN."""

    @property
    def _ary_prop(self) -> set[str]:
        """Returns the set of parameter names that are stored as TensorArrays."""
        return {Parameters.RUNNING_MEAN, Parameters.RUNNING_VAR, *super()._ary_prop}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the BatchNormalizationPycuda layer."""
        super().__init__(*args, **kwargs)
        # NOTE: The next attributes will be initialized later
        self.weights_biases_mean_var_desc: int = None  # pyright: ignore[reportAttributeAccessIssue]
        self.mode: int = None  # pyright: ignore[reportAttributeAccessIssue]
        self.weights_cpu: np.ndarray = None  # pyright: ignore[reportAttributeAccessIssue]
        self.biases_cpu: np.ndarray = None  # pyright: ignore[reportAttributeAccessIssue]
        self.dw_cpu: np.ndarray = None  # pyright: ignore[reportAttributeAccessIssue]
        self.db_cpu: np.ndarray = None  # pyright: ignore[reportAttributeAccessIssue]
        self.save_mean: TensorArray = None  # pyright: ignore[reportAttributeAccessIssue]
        self.save_inv_var: TensorArray = None  # pyright: ignore[reportAttributeAccessIssue]
        self.factor: float = None  # pyright: ignore[reportAttributeAccessIssue]

    def _model_init(self, prev_shape: ArrayShape, x: TensorArray) -> None:
        """Initializes GPU memory and cuDNN descriptors for the layer."""
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
        self.mode = cudnn.cudnnBatchNormMode[
            "CUDNN_BATCHNORM_SPATIAL" if self.spatial else "CUDNN_BATCHNORM_PER_ACTIVATION"
        ]

        self.weights_biases_mean_var_desc = cudnn.cudnnCreateTensorDescriptor()
        cudnn.cudnnDeriveBNTensorDescriptor(self.weights_biases_mean_var_desc, x.desc, self.mode)
        if self.spatial:
            self.ci, self.hi, self.wi = self.model.decode_shape(prev_shape)
        else:
            (self.ci,) = prev_shape

        shape_ = (1, self.ci, 1, 1)

        # weights
        self.weights_cpu = self.weights_initializer(shape_, self.model.dtype)
        weights_gpu = gpuarray.to_gpu(self.weights_cpu)
        self.weights = TensorArray(weights_gpu, self.model.tensor_format, self.model.cudnn_dtype)
        self.memory_used += self.weights.nbytes

        # biases
        self.biases_cpu = self.biases_initializer(shape_, self.model.dtype)
        biases_gpu = gpuarray.to_gpu(self.biases_cpu)
        self.biases = TensorArray(biases_gpu, self.model.tensor_format, self.model.cudnn_dtype)
        self.memory_used += self.biases.nbytes

        self.dw_cpu, self.dw = TensorArray.new(
            self.weights.shape,
            self.model.dtype,
            tensor_format=self.model.tensor_format,
            cudnn_dtype=self.model.cudnn_dtype,
            gpudirect=self.model.use_gpudirect,
            drv=(drv if self.model.use_gpudirect else None),
        )
        self.memory_used += self.dw.nbytes

        self.db_cpu, self.db = TensorArray.new(
            self.biases.shape,
            self.model.dtype,
            tensor_format=self.model.tensor_format,
            cudnn_dtype=self.model.cudnn_dtype,
            gpudirect=self.model.use_gpudirect,
            drv=(drv if self.model.use_gpudirect else None),
        )
        self.memory_used += self.db.nbytes

        running_mean_gpu = gpuarray.to_gpu(self.running_mean_initializer(shape_, self.model.dtype))
        self.running_mean = TensorArray(
            running_mean_gpu, self.model.tensor_format, self.model.cudnn_dtype
        )
        self.memory_used += self.running_mean.nbytes

        running_var_gpu = gpuarray.to_gpu(self.running_var_initializer(shape_, self.model.dtype))
        self.running_var = TensorArray(
            running_var_gpu, self.model.tensor_format, self.model.cudnn_dtype
        )
        self.memory_used += self.running_var.nbytes

        save_mean_gpu = gpuarray.zeros(shape_, self.model.dtype)
        self.save_mean = TensorArray(
            save_mean_gpu, self.model.tensor_format, self.model.cudnn_dtype
        )
        self.memory_used += self.save_mean.nbytes

        save_inv_var_gpu = gpuarray.zeros(shape_, self.model.dtype)
        self.save_inv_var = TensorArray(
            save_inv_var_gpu, self.model.tensor_format, self.model.cudnn_dtype
        )
        self.memory_used += self.save_inv_var.nbytes

        self.factor = 1.0 - self.momentum

        self.nparams = (
            self.weights.size + self.biases.size + self.running_mean.size + self.running_var.size
        )

        self.memory_used += self.weights.nbytes

    def forward(self, x: TensorArray) -> TensorArray:
        """Performs the forward pass using cuDNN."""
        alpha, beta = 1.0, 0.0
        match self.model.mode:
            case Model.Mode.TRAIN:
                self.model.tracer.emit_event(
                    PYDTNN_OPS_EVENT,
                    self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.FORWARD_CUDNN,
                )
                cudnn.cudnnBatchNormalizationForwardTraining(
                    self.model.cudnn_handle,
                    self.mode,
                    alpha,
                    beta,
                    x.desc,
                    x.ptr_voidp,
                    self.y.desc,
                    self.y.ptr_voidp,
                    self.weights_biases_mean_var_desc,
                    self.weights.ptr_voidp,
                    self.biases.ptr_voidp,
                    self.factor,
                    self.running_mean.ptr_voidp,
                    self.running_var.ptr_voidp,
                    self.epsilon,
                    self.save_mean.ptr_voidp,
                    self.save_inv_var.ptr_voidp,
                )
                self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
            case Model.Mode.EVALUATE:
                self.model.tracer.emit_event(
                    PYDTNN_OPS_EVENT,
                    self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.FORWARD_CUDNN,
                )
                cudnn.cudnnBatchNormalizationForwardInference(
                    self.model.cudnn_handle,
                    self.mode,
                    alpha,
                    beta,
                    x.desc,
                    x.ptr_voidp,
                    self.y.desc,
                    self.y.ptr_voidp,
                    self.weights_biases_mean_var_desc,
                    self.weights.ptr_voidp,
                    self.biases.ptr_voidp,
                    self.running_mean.ptr_voidp,
                    self.running_var.ptr_voidp,
                    self.epsilon,
                )
                self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
            case _:
                raise RuntimeError(f"Unexpected model mode '{self.model.mode}'.")
        return self.y

    def backward(self, dy: TensorArray) -> TensorArray:
        """Performs the backward pass using cuDNN."""
        self.x: TensorArray

        alpha_dx, beta_dx, alpha_dgb, beta_dgb = 1.0, 0.0, 1.0, 0.0
        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.BACKWARD_CUDNN_DX
        )
        # Compute dx, dgamma, dbeta
        cudnn.cudnnBatchNormalizationBackward(
            self.model.cudnn_handle,
            self.mode,
            alpha_dx,
            beta_dx,
            alpha_dgb,
            beta_dgb,
            self.x.desc,
            self.x.ptr_voidp,
            dy.desc,
            dy.ptr_voidp,
            self.dx.desc,
            self.dx.ptr_voidp,
            self.weights_biases_mean_var_desc,
            self.weights.ptr_voidp,
            self.dw.ptr_voidp,
            self.db.ptr_voidp,
            self.epsilon,
            self.save_mean.ptr_voidp,
            self.save_inv_var.ptr_voidp,
        )
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        # DtoH dw when data parallelism and no GPU direct/NCCL is used
        if self.model.comm and not self.model.use_gpudirect and not self.model.use_nccl:
            # self.model.stream.synchronize()
            self.dw.get_async(self.stream_2, self.dw_cpu)
            self.db.get_async(self.stream_2, self.db_cpu)
        return self.dx

    def _export_gamma_beta(self, key: str) -> np.ndarray:
        """Exports gamma or beta parameters to CPU."""
        value = getattr(self, key)
        gpu_ary = value
        cpu_ary = gpu_ary.get()
        return np.asarray(cpu_ary, dtype=np.float64, order="C", copy=True)

    def _export_prop(self, key: str) -> Any:
        """Exports layer properties."""
        match key:
            case Parameters.WEIGHTS | Parameters.DW | Parameters.BIASES | Parameters.DB:
                return self._export_gamma_beta(key)
            case _:
                return super()._export_prop(key)

    def _import_gamma_beta(self, key: str, value: np.ndarray) -> None:
        """Imports gamma or beta parameters from CPU."""
        attribute = getattr(self, key)
        attribute.set(value)
        return

    def _import_prop(self, key: str, value: Any) -> None:
        """Imports layer properties."""
        match key:
            case Parameters.WEIGHTS | Parameters.DW | Parameters.BIASES | Parameters.DB:
                return self._import_gamma_beta(key, value)
            case _:
                return super()._import_prop(key, value)
