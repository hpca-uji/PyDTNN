import numpy as np
# noinspection PyUnresolvedReferences
import pycuda.gpuarray as gpuarray
# noinspection PyUnresolvedReferences
from pycuda.compiler import SourceModule
# noinspection PyUnresolvedReferences
from pycuda.driver import Function

from pydtnn.metrics import CategoricalAccuracy
from pydtnn.backends.gpu.metrics.metric_gpu import MetricGPU
from pydtnn.backends.gpu.tensor_gpu import TensorGPU


class CategoricalAccuracyGPU(MetricGPU, CategoricalAccuracy[TensorGPU]):

    def __init_gpu_kernel__(self) -> Function:
        module = SourceModule("""
        __global__ void categorical_accuracy(T *y_targ, T *y_pred, T *res, int b, int n)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < b){
                int i = 0, max = 0;
                T max_value = y_pred[idx * n];
                for ( i = 1; i < n; i++ ) {
                    if ( y_pred[idx * n + i] > max_value ){
                        max = i;
                        max_value = y_pred[idx * n + i];
                    }
                }
                res[idx] = y_targ[idx * n + max];
            }
            return;
        }
        """.replace("T", {np.float32: "float", np.float64: "double"}[self.model.dtype]))
        return module.get_function("categorical_accuracy")

    def __call__(self, y_pred: TensorGPU, y_targ: TensorGPU) -> float:
        threads = min(self.model.batch_size, 1024)
        blocks = max(self.model.batch_size, 1024) // threads + 1
        self.kernel(y_targ, y_pred, self.cost,
                    np.int32(self.model.batch_size), np.int32(self.shape[1]),
                    grid=(blocks, 1, 1), block=(threads, 1, 1),
                    stream=self.model.stream)
        return gpuarray.sum(self.cost).get() * 100 / self.model.batch_size
