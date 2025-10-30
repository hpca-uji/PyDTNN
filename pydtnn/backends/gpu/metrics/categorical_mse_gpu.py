import numpy as np

from pydtnn.metrics.categorical_mse import CategoricalMSE
from pydtnn.backends.gpu.metrics.metric_gpu import MetricGPU
from pydtnn.backends.gpu.tensor_gpu import TensorGPU
import pycuda.gpuarray as gpuarray  # type: ignore
from pycuda.compiler import SourceModule  # type: ignore
from pycuda.driver import Function  # type: ignore

from pydtnn.utils.types import DTYPE2CTYPE


class CategoricalMSEGPU(MetricGPU, CategoricalMSE[TensorGPU]):

    def __init_gpu_kernel__(self) -> Function:
        _name = "categorical_accuracy"
        code = """
        __global__ void {name} ({T} *y_targ, {T} *y_pred, {T} *res, int b, int n)
        {{
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < b)
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

    def compute(self, y_pred: TensorGPU, y_targ: TensorGPU) -> TensorGPU:
        b = y_targ.shape[0]
        # return np.square(1 - y_pred[np.arange(b), np.argmax(y_targ, axis=1)]).mean()

        y = y_pred[np.arange(b), np.argmax(y_targ, axis=1)]
        np.multiply(y, -1, out=y, dtype=self.model.dtype)
        np.add(y, 1, out=y, dtype=self.model.dtype)
        np.square(y, out=y, dtype=self.model.dtype, casting="unsafe")
        return y.mean(dtype=self.model.dtype)
