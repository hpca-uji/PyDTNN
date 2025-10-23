import numpy as np
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from pycuda.driver import Function

from pydtnn.losses import BinaryCrossEntropy
from pydtnn.backends.gpu.losses.loss_gpu import LossGPU
from pydtnn.backends.gpu import TensorGPU
from pydtnn.utils.types import DTYPE2CTYPE

class BinaryCrossEntropyGPU(LossGPU, BinaryCrossEntropy):

    def __init_gpu_kernel__(self) -> Function:
        module = SourceModule("""
        __global__ void binary_cross_entropy(T *y_targ, T *y_pred, T *res,
                                             T *dx, int b, int n, T eps)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < b) {
                int i = 0, max = 0;
                T pred;
                res[idx] = 0;
                for ( i = 0; i < n; i++ ) {
                    res[idx]+= logf(fmaxf((1 - y_targ[idx * n + i] ) -
                                               y_pred[idx * n + i], eps));
                    pred = y_pred[idx * n + max];
                    if ( pred < eps )          pred = eps;
                    else if ( pred > (1-eps) ) pred = (1-eps);
                    dx[idx * n + i] = (-(y_targ[idx * n + i]  / pred) +
                                   ((1 - y_targ[idx * n + i]) / pred) ) / b;
                }
            }
            return;
        }
        """.replace("T", DTYPE2CTYPE[self.model.dtype]))
        return module.get_function("binary_cross_entropy")

    def compute(self, y_pred: TensorGPU, y_targ: TensorGPU, batch_size: int) -> tuple[float, TensorGPU]:
        assert len(y_targ.shape) == 2
        threads, blocks = self.get_threads_and_blocks()
        self.kernel(y_targ, y_pred, self.loss, self.dx.ary,
                    batch_size, self.shape[1], self.eps,
                    grid=(blocks, 1, 1), block=(threads, 1, 1),
                    stream=self.model.stream)
        loss: float = -gpuarray.sum(self.loss[:batch_size]) / batch_size
        return loss, self.dx
