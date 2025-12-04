import pycuda.gpuarray as gpuarray  # type: ignore
from pycuda.compiler import SourceModule  # type: ignore
from pycuda.driver import Function  # type: ignore

from pydtnn.losses.binary_cross_entropy import BinaryCrossEntropy
from pydtnn.backends.gpu.losses.loss import LossGPU
from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU
from pydtnn.utils.constants import DTYPE2CTYPE


class BinaryCrossEntropyGPU(LossGPU, BinaryCrossEntropy[TensorGPU]):

    def __init_gpu_kernel__(self) -> Function:
        _name = "binary_cross_entropy"
        code = """
        __global__ void {name}({T} *y_targ, {T} *y_pred, {T} *res,
                               {T} *dx, int b, int n, {T} eps)
        {{
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < b) {
                int i = 0, max = 0;
                {T} pred;
                res[idx] = 0;
                for ( i = 0; i < n; i++ ) 
                {{
                    res[idx]+= logf(fmaxf((1 - y_targ[idx * n + i] ) -
                                               y_pred[idx * n + i], eps));
                    pred = y_pred[idx * n + max];
                    if ( pred < eps )          pred = eps;
                    else if ( pred > (1-eps) ) pred = (1-eps);
                    dx[idx * n + i] = (-(y_targ[idx * n + i]  / pred) +
                                   ((1 - y_targ[idx * n + i]) / pred) ) / b;
                }}
            }}
            return;
        }}
        """.format(
            T = DTYPE2CTYPE[self.model.dtype],
            name = _name
            )
        
        module = SourceModule(code).get_function(_name)
        return module

    def compute(self, y_pred: TensorGPU, y_targ: TensorGPU, batch_size: int) -> tuple[float, TensorGPU]:
        assert len(y_targ.shape) == 2
        self.kernel(y_targ, y_pred, self.loss, self.dx.ary,
                    batch_size, self.shape[1], self.eps,
                    grid=self.grid, block=self.block,
                    stream=self.model.stream)
        loss: float = -gpuarray.sum(self.loss[:batch_size]) / batch_size
        return loss, self.dx
