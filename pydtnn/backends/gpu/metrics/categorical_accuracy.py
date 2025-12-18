import numpy as np
import pycuda.gpuarray as gpuarray  # type: ignore
from pycuda.compiler import SourceModule  # type: ignore
from pycuda.driver import Function  # type: ignore

from pydtnn.metrics.categorical_accuracy import CategoricalAccuracy
from pydtnn.backends.gpu.metrics.metric import MetricGPU
from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU
from pydtnn.utils.constants import DTYPE2CTYPE


class CategoricalAccuracyGPU(CategoricalAccuracy[TensorGPU], MetricGPU):

    def initialize(self) -> None:
        super().initialize()
        self.cost = gpuarray.empty((self.model.batch_size,), self.model.dtype)

    def __init_gpu_kernel__(self) -> Function:
        _name = "categorical_accuracy"
        code = """
        __global__ void {name} ({T} *y_targ, {T} *y_pred, {T} *res, int b, int n)
        {{
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            const int workers = blockDim.x * gridDim.x;

            for(; idx < b; idx += workers)
            {{
                int i = 0, max = 0;
                {T} max_value = y_pred[idx * n];
                for ( i = 1; i < n; i++ ) 
                {{
                    if ( y_pred[idx * n + i] > max_value )
                    {{
                        max = i;
                        max_value = y_pred[idx * n + i];
                    }}
                }}
                res[idx] = y_targ[idx * n + max];
            }}
            return;
        }}
        """.format(T=DTYPE2CTYPE[self.model.dtype],
                   name=_name)
        
        module = SourceModule(code).get_function(_name)
        return module

    def compute(self, y_pred: TensorGPU, y_targ: TensorGPU) -> float:
        self.kernel(y_targ.ary, y_pred.ary, self.cost,
                    np.int32(self.model.batch_size), np.int32(self.shape[1]),
                    grid=self.grid, block=self.block,
                    stream=self.model.stream)
        return gpuarray.sum(self.cost).get() * 100 / self.model.batch_size
