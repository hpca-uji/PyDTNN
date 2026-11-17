"""PyCUDA implementation of the Fully Connected (FC) layer."""

import logging
from typing import Any

import numpy as np

from pydtnn.backends.pycuda.layers.abstract.layer import LayerPycuda
from pydtnn.backends.pycuda.utils import matmul_gpu, matvec_gpu
from pydtnn.layers.fc import FC
from pydtnn.libs import cudnn as cudnn
from pydtnn.tracers.events import (PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT,
                                   PYDTNN_OPS_EVENTS, OpsEventEnum)
from pydtnn.utils.constants import ArrayShape, Parameters
from pydtnn.utils.performance_models import matmul_time
from pydtnn.utils.tensor_array import DescriptorType, MemoryType, TensorArray

__all__ = ("FCPycuda",)

logger = logging.getLogger(__name__)


class FCPycuda(FC[TensorArray], LayerPycuda):
    """Fully connected layer implementation for PyCUDA backend."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the FCPycuda layer."""
        super().__init__(*args, **kwargs)
        self.matmul = matmul_gpu
        self.matvec = matvec_gpu

    def _import_biases_db(self, key: str, value: Any) -> None:
        """Import bias or gradient data from CPU to GPU."""
        attribute = getattr(self, key)

        cpu_ary = value
        attribute.set(cpu_ary)

    def _import_prop(self, key: str, value: Any) -> None:
        """Import layer property from CPU to GPU."""
        match key:
            case Parameters.BIASES | Parameters.DB:
                return self._import_biases_db(key, value)

            case _:
                return super()._import_prop(key, value)

    def _export_biases_db(self, key: str) -> Any:
        """Export bias or gradient data from GPU to CPU."""
        value = getattr(self, key)
        gpu_ary = value
        cpu_ary = gpu_ary.get()

        return np.asarray(cpu_ary, dtype=np.float64, order="C", copy=True)

    def _export_prop(self, key: str) -> Any:
        """Export layer property from GPU to CPU."""
        match key:
            case Parameters.BIASES | Parameters.DB:
                return self._export_biases_db(key)
            case _:
                return super()._export_prop(key)

    def _model_init(self, prev_shape: ArrayShape, x: TensorArray) -> None:
        """Initialize model parameters and GPU buffers."""
        super()._model_init(prev_shape, x)

        # Weights
        self.weights = TensorArray.asarray(
            self.weights_initializer(self.weights_shape, self.model.dtype, self.model.random),
            tensor_format=self.model.tensor_format,
            mem_type=MemoryType.GLOBAL if self.model.use_gpudirect else MemoryType.DEVICE,
            desc_type=DescriptorType.TENSOR
        )
        self.memory_used += self.weights.nbytes

        if self.use_bias:
            # Biases
            self.biases = TensorArray.asarray(
                self.biases_initializer((1, *self.shape), self.model.dtype, self.model.random),
                tensor_format=self.model.tensor_format,
                mem_type=MemoryType.GLOBAL if self.model.use_gpudirect else MemoryType.DEVICE,
                desc_type=DescriptorType.TENSOR
            )
            self.memory_used += self.biases.nbytes

        self.y = TensorArray.zeros(
            (self.model.batch_size, self.shape[0]),
            self.model.dtype,
            tensor_format=self.model.tensor_format,
            mem_type=MemoryType.GLOBAL if self.model.use_gpudirect else MemoryType.DEVICE,
            desc_type=DescriptorType.TENSOR
        )
        self.memory_used += self.y.nbytes

        self.dx = TensorArray.zeros(
            (self.model.batch_size, *prev_shape),
            self.model.dtype,
            tensor_format=self.model.tensor_format,
            mem_type=MemoryType.GLOBAL if self.model.use_gpudirect else MemoryType.DEVICE,
            desc_type=DescriptorType.TENSOR
        )
        self.memory_used += self.dx.nbytes

        self.dw = TensorArray.zeros(
            self.weights.host_shape,
            self.model.dtype,
            tensor_format=self.model.tensor_format,
            mem_type=MemoryType.GLOBAL if self.model.use_gpudirect else MemoryType.DEVICE,
            desc_type=DescriptorType.TENSOR
        )
        self.memory_used += self.dw.nbytes

        if self.use_bias:
            self.db = TensorArray.zeros(
                self.biases.host_shape,
                self.model.dtype,
                tensor_format=self.model.tensor_format,
                mem_type=MemoryType.GLOBAL if self.model.use_gpudirect else MemoryType.DEVICE,
                desc_type=DescriptorType.TENSOR
            )
            self.memory_used += self.db.nbytes

        self.one_vec_gpu = TensorArray.asarray(
            np.ones((self.model.batch_size,), self.model.dtype),
            mem_type=MemoryType.DEVICE,
            desc_type=DescriptorType.NONE
        )
        self.memory_used += self.one_vec_gpu.nbytes

        self.nparams = self.weights.nbytes + (self.biases.nbytes if self.use_bias else 0)

        self.fwd_time = self.bwd_time = np.zeros((4,), dtype=np.float32)
        self.fwd_time += matmul_time(
            m=self.model.batch_size,
            n=self.weights.host_shape[1],
            k=self.weights.host_shape[0],
            cpu_speed=self.model.cpu_speed,
            memory_bw=self.model.memory_bw,
            dtype=self.model.dtype,
        )
        self.bwd_time += matmul_time(
            m=self.weights.host_shape[0],
            n=self.weights.host_shape[1],
            k=self.model.batch_size,
            cpu_speed=self.model.cpu_speed,
            memory_bw=self.model.memory_bw,
            dtype=self.model.dtype,
        )
        self.bwd_time += matmul_time(
            m=self.model.batch_size,
            n=self.weights.host_shape[0],
            k=self.weights.host_shape[1],
            cpu_speed=self.model.cpu_speed,
            memory_bw=self.model.memory_bw,
            dtype=self.model.dtype,
        )

    def forward(self, x: TensorArray) -> TensorArray:
        """Perform forward pass computation."""
        m = x.shape[0]
        n = ldb = ldc = self.weights.shape[1]
        k = lda = x.shape[1]
        trans_a, trans_b, alpha, beta = "N", "N", 1.0, 0.0

        # Compute a' = x @ weights
        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT,
            self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.FORWARD_CUBLAS_MATMUL,
        )
        self.matmul(
            self.model.cublas_handle,
            trans_b,
            trans_a,
            n,
            m,
            k,
            alpha,
            self.weights.ptr,
            ldb,
            x.ptr,
            lda,
            beta,
            self.y.ptr,
            ldc,
            self.model.dtype,
        )
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.biases: TensorArray
            alpha, beta = 1.0, 1.0
            self.model.tracer.emit_event(
                PYDTNN_OPS_EVENT,
                self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.FORWARD_CUDNN_SUM_BIASES,
            )
            # Compute a = a' + biases
            cudnn.cudnnAddTensor(
                self.model.cudnn_handle,
                alpha,
                self.biases.desc,
                self.biases.ptr,
                beta,
                self.y.desc,
                self.y.ptr,
            )
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return self.y

    def backward(self, dy: TensorArray) -> TensorArray:
        """Perform backward pass computation."""
        # Compute dw
        m = lda = self.x.shape[1]
        n = ldb = ldc = dy.shape[1]
        k = dy.shape[0]
        trans_a, trans_b, alpha, beta = "T", "N", 1.0, 0.0

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT,
            self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.BACKWARD_CUBLAS_MATMUL_DW,
        )
        self.matmul(
            self.model.cublas_handle,
            trans_b,
            trans_a,
            n,
            m,
            k,
            alpha,
            dy.ptr,
            ldb,
            self.x.ptr,
            lda,
            beta,
            self.dw.ptr,
            ldc,
            self.model.dtype,
        )
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.biases: TensorArray
            # Compute db
            m = dy.shape[0]
            n = lda = dy.shape[1]
            trans_a, alpha, beta, inc_x, inc_y = "N", 1.0, 0.0, 1, 1

            self.model.tracer.emit_event(
                PYDTNN_OPS_EVENT,
                self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.BACKWARD_CUBLAS_MATVEC_DB,
            )
            self.matvec(
                self.model.cublas_handle,
                trans_a,
                n,
                m,
                alpha,
                dy.ptr,
                lda,
                self.one_vec_gpu.ptr,
                inc_x,
                beta,
                self.db.ptr,
                inc_y,
                self.model.dtype,
            )
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        # Compute dx
        m = dy.shape[0]
        n = ldc = self.weights.shape[0]
        k = lda = ldb = dy.shape[1]
        trans_a, trans_b, alpha, beta = "N", "T", 1.0, 0.0

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT,
            self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.BACKWARD_CUBLAS_MATMUL_DX,
        )
        self.matmul(
            self.model.cublas_handle,
            trans_b,
            trans_a,
            n,
            m,
            k,
            alpha,
            self.weights.ptr,
            ldb,
            dy.ptr,
            lda,
            beta,
            self.dx.ptr,
            ldc,
            self.model.dtype,
        )
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return self.dx
