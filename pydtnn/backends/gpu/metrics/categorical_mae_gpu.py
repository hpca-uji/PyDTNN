import numpy as np

from pydtnn.metrics.categorical_mae import CategoricalMAE

from pydtnn.backends.gpu.metrics.metric_gpu import MetricGPU
from pydtnn.backends.gpu.tensor_gpu import TensorGPU
import pycuda.gpuarray as gpuarray  # type: ignore
from pycuda.compiler import SourceModule  # type: ignore
from pycuda.driver import Function  # type: ignore

from pydtnn.utils.types import DTYPE2CTYPE

class CategoricalMAEGPU(MetricGPU, CategoricalMAE[TensorGPU]):

    def __init_gpu_kernel__(self) -> Function:
        _name = "categorical_accuracy"
        code = """
        #define SHIFT_2D_AR(p, i, j, dim_i) (p + ((i * dim_i) + j))

        __global__ void {name} ({T} *y_targ, {T} *y_pred, {T} *res, {T} *local_res, int n, int labels)
        {{
            int i, idx;
            {T} val_targ, val_pred, error;
        
            int base_idx = blockIdx.x * blockDim.x + threadIdx.x;
            int workers = blockDim.x * gridDim.x;

            for(idx = base_idx; idx < n; idx += workers)
            {{
                for(i = 0, sum = ({T}) 0.0, max = ({T}) 0.0; i < labels; i++)
                {{
                    // val_targ = y_targ[idx][i];
                    val_targ = (*SHIFT_2D_AR(y_targ, idx, i, n));

                    // val_pred = y_pred[idx][i];
                    val_pred = (*SHIFT_2D_AR(y_pred, idx, i, n));
                    
                    error = ({T}) (val_targ - val_pred);
                    (error * error)
                    if ( (i == 0) || (max < neg))
                        max = neg;
                }}
                max = ({T}) ((max - pos) + 1);
                *(local_res + idx) = ({T}) (max > 0 ? max : 0);
            }}
            
            
            if(base_idx == 0)
            {{
                for(idx = 1; idx < n; idx++)
                    *(res) += *(local_res + idx);

                *(res) = ({T}) (*(res) / n);
            }}
        }}
        """..format(T=DTYPE2CTYPE[self.model.dtype],
                   name=_name)
        
        module = SourceModule(code).get_function(_name)
        return module

    def compute(self, y_pred: TensorGPU, y_targ: TensorGPU) -> TensorGPU:
        b = y_targ.shape[0]
        # return np.sum(np.absolute(1 - y_pred[np.arange(b), np.argmax(y_targ, axis=1)]))

        # Obtenemos la matriz con los valores predichos donde deberían haber 1s.
        y = y_pred[np.arange(b), np.argmax(y_targ, axis=1)]

        # Lo invertimos (0 * -1 + 1 = 1; (1*-1 +1 = 0)
        np.multiply(y, -1, out=y, dtype=self.model.dtype)
        np.add(y, 1, out=y, dtype=self.model.dtype)

        # Acamos el valor absoluto
        np.absolute(y, out=y, dtype=self.model.dtype, casting="unsafe")

        # Sumamos lso valores
        return np.sum(y)
