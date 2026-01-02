import numpy as np
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU
from pydtnn.metrics.kl_divergence_metric import KLDivergenceMetric
from pydtnn.backends.gpu.metrics.metric import MetricGPU


class KLDivergenceMetricGPU(KLDivergenceMetric[TensorGPU], MetricGPU):

    def __init_gpu_kernel__(self):
        module = SourceModule("""
        __global__ void kl_divergence_metric(T *y_targ, T *y_pred, T *res, int b, int n, float eps)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < b) {
                int i = 0;
                res[idx * n] = y_targ[idx * n];
                for ( i = 1; i < n; i++ ) {
                    res[idx * n + i] = fabs(y_pred[idx * n + i] * logf(fabs(y_pred[idx * n + i] / (y_targ[idx * n + i] + eps)) + eps));
                }
            }
            return;
        }
        """.replace("T", {np.float32: "float", np.float64: "double"}[self.model.dtype]))
        return module.get_function("kl_divergence_metric")

    def compute(self, y_pred, y_targ):
        threads = min(self.model.batch_size, 1024)
        blocks = max(self.model.batch_size, 1024) // threads + 1
        self.kernel(y_targ, y_pred, self.cost,
                    np.int32(self.model.batch_size), np.int32(self.shape[1]),
                    np.float32(self.eps),
                    grid=(blocks, 1, 1), block=(threads, 1, 1),
                    stream=self.model.stream)
        return gpuarray.sum(self.cost).get() / self.model.batch_size
