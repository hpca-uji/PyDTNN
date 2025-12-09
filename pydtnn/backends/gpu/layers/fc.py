from typing import Any
import numpy as np

import pycuda.driver as drv  # type: ignore
import pycuda.gpuarray as gpuarray  # type: ignore

from pydtnn.layers.fc import FC
from pydtnn.utils.performance_models import matmul_time
from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum
from pydtnn.backends.gpu.layers.layer import LayerGPU
from pydtnn.libs import libcudnn as cudnn
from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU
from pydtnn.backends.gpu.utils import matmul_gpu, matvec_gpu
from pydtnn.utils.constants import ArrayShape, Parameters

class FCGPU(FC[TensorGPU], LayerGPU):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.matmul = matmul_gpu
        self.matvec = matvec_gpu

    def _import_biases_db(self, key: str, value: Any) -> None:
        attribute = getattr(self, key)
        
        cpu_ary = np.asarray(np.expand_dims(value, axis=0), dtype=self.model.dtype, order="C", copy=None)
        attribute.ary.set(cpu_ary)
        return
    # ---

    def _import_prop(self, key: str, value) -> None:
        match key:
            case Parameters.BIASES | Parameters.DB:
                return self._import_biases_db(key, value)
            # 
            case _:
                return super()._import_prop(key, value)
    # -----

    def _export_biases_db(self, key: str) -> Any:
        value = getattr(self, key)
        gpu_ary = value.ary
        cpu_ary = gpu_ary.get()

        return np.asarray(np.squeeze(cpu_ary, axis=0), dtype=np.float64, order="C", copy=True)
    # ---

    def _export_prop(self, key: str) -> Any:
        match key:
            case Parameters.BIASES | Parameters.DB:
                return self._export_biases_db(key)
            case _:
                return super()._export_prop(key)

    def initialize(self, prev_shape: ArrayShape, x: TensorGPU) -> None:
        super().initialize(prev_shape, x)
        self.stream_2 = drv.Stream()

        # Weights
        self.weights_cpu = self.weights_initializer(self.weights_shape, self.model.dtype)
        weights_gpu = gpuarray.to_gpu(self.weights_cpu)
        self.weights = TensorGPU(weights_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        if self.use_bias:
            # Biases
            self.biases_cpu = self.biases_initializer((1, *self.shape), self.model.dtype)
            biases_gpu = gpuarray.to_gpu(self.biases_cpu)
            self.biases = TensorGPU(biases_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        y_gpu = gpuarray.empty((self.model.batch_size, self.shape[0]), self.model.dtype)
        self.y = TensorGPU(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        dx_gpu = gpuarray.empty(x.ary.shape, self.model.dtype)
        self.dx = TensorGPU(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)
        self.dx.reshape((self.model.batch_size, *prev_shape))

        self.dw_cpu, self.dw = TensorGPU.initialize(self.weights.ary.shape, self.model.dtype,
                                                    tensor_format=self.model.tensor_format,
                                                    cudnn_dtype=self.model.cudnn_dtype,
                                                    gpudirect=self.model.gpudirect, 
                                                    drv=(drv if self.model.gpudirect else None))
        if self.use_bias:
            self.biases: TensorGPU
            self.db_cpu, self.db = TensorGPU.initialize(self.biases.ary.shape, self.model.dtype,
                                                        tensor_format=self.model.tensor_format,
                                                        cudnn_dtype=self.model.cudnn_dtype,
                                                        gpudirect=self.model.gpudirect, 
                                                        drv=(drv if self.model.gpudirect else None))

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
                        dtype=self.model.dtype)  # type: ignore (This is correct)

    def forward(self, x: TensorGPU) -> TensorGPU:
        m = x.ary.shape[0]
        n = ldb = ldc = self.weights.ary.shape[1]
        k = lda = x.ary.shape[1]
        trans_a, trans_b, alpha, beta = 'N', 'N', 1.0, 0.0

        # Compute a' = x @ weights
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT,
                                     self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CUBLAS_MATMUL)
        self.matmul(self.model.cublas_handle, trans_b, trans_a, n, m, k, alpha,
                    self.weights.ary.gpudata, ldb,
                    x.ary.gpudata, lda, beta,
                    self.y.ary.gpudata, ldc, self.model.dtype)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.biases: TensorGPU
            alpha, beta = 1.0, 1.0
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT,
                                         self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CUDNN_SUM_BIASES)
            # Compute a = a' + biases
            cudnn.cudnnAddTensor(self.model.cudnn_handle, alpha, self.biases.desc,
                                 self.biases.ptr, beta, self.y.desc, self.y.ptr)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return self.y

    def backward(self, dy: TensorGPU) -> TensorGPU:
        # Compute dw
        m = lda = self.x.ary.shape[1]
        n = ldb = ldc = dy.ary.shape[1]
        k = dy.ary.shape[0]
        trans_a, trans_b, alpha, beta = 'T', 'N', 1.0, 0.0

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT,
                                     self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_CUBLAS_MATMUL_DW)
        self.matmul(self.model.cublas_handle, trans_b, trans_a, n, m, k, alpha,
                    dy.ary.gpudata, ldb, self.x.ary.gpudata, lda, beta,
                    self.dw.ptr_intp if self.model.gpudirect else self.dw.ary.gpudata, ldc, self.model.dtype)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        # DtoH dw when data parallelism and no GPU direct/NCCL is used
        if self.model.comm and not self.model.gpudirect and not self.model.enable_nccl:
            self.model.stream.synchronize()
            self.dw.ary.get_async(self.stream_2, self.dw_cpu)

        if self.use_bias:
            self.biases: TensorGPU
            # Compute db
            m = dy.ary.shape[0]
            n = lda = dy.ary.shape[1]
            trans_a, alpha, beta, inc_x, inc_y = 'N', 1.0, 0.0, 1, 1

            self.model.tracer.emit_event(PYDTNN_OPS_EVENT,
                                         self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_CUBLAS_MATVEC_DB)
            self.matvec(self.model.cublas_handle, trans_a, n, m, alpha,
                        dy.ary.gpudata, lda, self.one_vec_gpu.gpudata, inc_x, beta,
                        self.db.ptr_intp if self.model.gpudirect else self.db.ary.gpudata,
                        inc_y, self.model.dtype)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

            # DtoH db when data parallelism and no GPU direct/NCCL is used
            if self.model.comm and not self.model.gpudirect and not self.model.enable_nccl:
                self.model.stream.synchronize()
                self.db.ary.get_async(self.stream_2, self.db_cpu)

        # Compute dx
        m = dy.ary.shape[0]
        n = ldc = self.weights.ary.shape[0]
        k = lda = ldb = dy.ary.shape[1]
        trans_a, trans_b, alpha, beta = 'N', 'T', 1.0, 0.0

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT,
                                     self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_CUBLAS_MATMUL_DX)
        self.matmul(self.model.cublas_handle, trans_b, trans_a, n, m, k, alpha,
                    self.weights.ary.gpudata, ldb,
                    dy.ary.gpudata, lda, beta,
                    self.dx.ary.gpudata, ldc, self.model.dtype)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return self.dx
