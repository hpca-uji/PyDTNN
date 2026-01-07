import numpy as np
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU
from pydtnn.losses.kl_divergence import KLDivergence
from pydtnn.backends.gpu.losses.loss import LossGPU


class KLDivergenceGPU(KLDivergence[TensorGPU], LossGPU):

    def __init_gpu_kernel__(self):
        module = SourceModule("""
        __global__ void kl_divergence(T *y_targ, T *y_pred, T *res,
                                      T *dx, int b, int bs, int n, T eps)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < b) {
                int i = 0;
                double partial = 0;
                double loss = 0;
                for ( i = 0; i < n; i++ ) {
                    partial = logf(fabs(y_targ[idx * n + i] / (y_pred[idx * n + i] + eps)) + 1.0) / bs;
                    loss += fabs(y_targ[idx * n + i] * partial);
                    dx[idx * n + i] = (T) partial;
                }
                res[idx] = (T) loss;
            }
            return;
        }
        """.replace("T", {np.float32: "float", np.float64: "double"}[self.model.dtype]))
        return module.get_function("kl_divergence")

    def compute(self, y_pred, y_targ, batch_size):
        # loss = SUM(|pred * log(|pred / (targ + eps)| + eps) / N
        # dx = log(|pred / targ + eps| + eps) + 1 / N
        threads = min(self.model.batch_size, 1024)
        blocks = max(self.model.batch_size, 1024) // threads + 1
        self.kernel(y_targ.ary, y_pred.ary, self.loss, self.dx.ary,
                    np.int32(self.model.batch_size), np.int32(batch_size),
                    np.int32(np.prod(self.shape[1:])), np.float32(self.eps),
                    grid=(blocks, 1, 1), block=(threads, 1, 1),
                    stream=self.model.stream)
        # loss = gpuarray.sum(self.loss).get()
        loss = gpuarray.sum(self.dx.ary).get()
        return loss, self.dx
