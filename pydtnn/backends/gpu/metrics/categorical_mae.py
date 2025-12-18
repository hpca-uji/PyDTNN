import numpy as np

from pydtnn.metrics.categorical_mae import CategoricalMAE

from pydtnn.backends.gpu.metrics.metric import MetricGPU
from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU
from pycuda.compiler import SourceModule  # type: ignore
from pycuda.driver import Function  # type: ignore

from pydtnn.utils.constants import DTYPE2CTYPE

class CategoricalMAEGPU(CategoricalMAE[TensorGPU], MetricGPU):

    def initialize(self) -> None:
        super().initialize()
        self.res = TensorGPU.create_zeros_tensor(shape=(1, ), dtype=np.dtype(self.model.dtype),
                                            tensor_format=self.model.tensor_format, cudnn_dtype=self.model.cudnn_dtype)
        self.local_res = TensorGPU.create_zeros_tensor(shape=(self.model.batch_size, ), dtype=np.dtype(self.model.dtype),
                                                  tensor_format=self.model.tensor_format, cudnn_dtype=self.model.cudnn_dtype)

    def __init_gpu_kernel__(self) -> Function:
        _name = "categorical_mae"
        code = """
        #define SHIFT_2D_AR(p, i, j, dim_j) (p + ((i * dim_j) + j))

        __global__ void {name} ({T} *y_targ, {T} *y_pred, {T} *res, {T} *local_res, int n, int labels)
        {{
            int i, idx;
            {T} val_targ, val_pred, error;
        
            int base_idx = blockIdx.x * blockDim.x + threadIdx.x;
            int workers = blockDim.x * gridDim.x;

            for(idx = base_idx; idx < n; idx += workers)
            {{
                for(i = 0; i < labels; i++)
                {{
                    // val_targ = y_targ[idx][i];
                    val_targ = (*SHIFT_2D_AR(y_targ, idx, i, labels));

                    // val_pred = y_pred[idx][i];
                    val_pred = (*SHIFT_2D_AR(y_pred, idx, i, labels));
                    
                    error = ({T}) (val_targ - val_pred);
                    error = error > 0 ? error : (-1) * error; // absolute error
                    *(local_res + idx) += error;
                }}
            }}

            // Getting the mean and accumulating it on the output's buffer.
            if(base_idx == 0)
            {{
                for(idx = 1; idx < n; idx++)
                    *(local_res) += *(local_res + idx);

                *(res) = ({T}) (*(local_res) / (n * labels));
            }}
        }}
        """.format(T=DTYPE2CTYPE[self.model.dtype],
                   name=_name)
        
        module = SourceModule(code).get_function(_name)
        return module

    def compute(self, y_pred: TensorGPU, y_targ: TensorGPU) -> float:
        n = y_pred.shape[0]

        self.res.fill(0)
        self.local_res.fill(0)

        n = np.int32(n)
        num_classes = np.int32(y_pred.shape[1])

        self.kernel(y_targ.ary, y_pred.ary, 
                    self.res.ary, self.local_res.ary,
                    n, num_classes,
                    grid=self.grid, block=self.block,
                    stream=self.model.stream)
        return self.res.ary.get()[0]
