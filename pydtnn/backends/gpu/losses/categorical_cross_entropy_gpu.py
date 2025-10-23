import numpy as np
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from pycuda.driver import Function

from pydtnn.losses import CategoricalCrossEntropy
from pydtnn.backends.gpu.losses.loss_gpu import LossGPU
from pydtnn.backends.gpu import TensorGPU
from pydtnn.utils.types import DTYPE2CTYPE

class CategoricalCrossEntropyGPU(LossGPU, CategoricalCrossEntropy):

    def __init_gpu_kernel__(self) -> Function:
        module = SourceModule("""
        __global__ void categorical_cross_entropy(T *y_targ, T *y_pred, T *res,
                                                  T *dx, int b, int n, float eps)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < b) {
                int i = 0, max = 0;
                T max_value = y_targ[idx * n];
                dx[idx * n] = y_targ[idx * n];
                for ( i = 1; i < n; i++ ) {
                    dx[idx * n + i] = y_targ[idx * n + i];
                    if ( y_targ[idx * n + i] > max_value ) {
                        max = i;
                        max_value = y_targ[idx * n + i];
                    }
                }
                T pred = y_pred[idx * n + max];
                if ( pred < eps )          pred = eps;
                else if ( pred > (1-eps) ) pred = (1-eps);
                res[idx] = logf(pred);
                dx[idx * n + max] /= -(pred * b);
            }
            return;
        }
        """.replace("T", DTYPE2CTYPE[self.model.dtype]))
        return module.get_function("categorical_cross_entropy")

    def compute(self, y_pred: TensorGPU, y_targ: TensorGPU, batch_size: int) -> tuple[float, TensorGPU]:
        threads, blocks = self.get_threads_and_blocks()
        self.kernel(y_targ.ary, y_pred.ary, self.loss, self.dx.ary,
                    np.int32(batch_size), np.int32(self.shape[1]), np.float32(self.eps),
                    grid=(blocks, 1, 1), block=(threads, 1, 1),
                    stream=self.model.stream)
        loss: float = -gpuarray.sum(self.loss[:batch_size]).get() / batch_size
        return loss, self.dx
