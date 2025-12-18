from pydtnn.backends.gpu.metrics.metric import MetricGPU
from pydtnn.metrics.multiclass_confusion_matrix import MulticlassConfusionMatrix

from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU
import numpy as np
from pycuda.compiler import SourceModule  # type: ignore
from pycuda.driver import Function  # type: ignore
from pydtnn.utils.constants import DTYPE2CTYPE

class MulticlassConfusionMatrixGPU(MulticlassConfusionMatrix[TensorGPU], MetricGPU):

    def initialize(self) -> None:
        super().initialize()
        n = self.model.batch_size
        target_classes = self.model.output_shape[0]
        
        self.conf_matrix = TensorGPU.create_zeros_tensor(shape=(1, 1, target_classes, target_classes), dtype=np.dtype(np.int32), 
                                                        tensor_format=self.model.tensor_format, cudnn_dtype=self.model.cudnn_dtype)
        self.local_cm = TensorGPU.create_zeros_tensor(shape=(1, n, target_classes, target_classes), dtype=np.dtype(np.int32), 
                                                      tensor_format=self.model.tensor_format, cudnn_dtype=self.model.cudnn_dtype)
    # ----

    def __init_gpu_kernel__(self) -> Function:
        
        _name = "multiclass_confusion_matrix"
        code = """
        
        #define INDEX_FIRST_ONE_ON(y, var_class) for(i = 0; (i < num_classes) && !(y[i]); i++); var_class = i;
        #define SHIFT_POINTER_CM(p, i, j, num_classes) p + (i * num_classes + j)
        #define SHIFT_POINTER_LOCAL_CM(p, idx, i, j, num_i, num_j) p + ((idx * num_i + i) * num_j + j)
        
        __global__ void {name}({T} *y_targ, {T} *y_pred, int *cm, int *local_cm, const int num_classes, const int n)
        {{
            int idx, idx_i, i, j, target_class, predicted_class;

            const int base_idx = blockIdx.x * blockDim.x + threadIdx.x;
            const int workers = blockDim.x * gridDim.x;

            for(idx = base_idx; idx < n; idx += workers)
            {{
                INDEX_FIRST_ONE_ON(y_targ, target_class)
                INDEX_FIRST_ONE_ON(y_pred, predicted_class)

                (*(SHIFT_POINTER_LOCAL_CM(local_cm, idx, target_class, predicted_class, num_classes, num_classes))) += 1;
            }}
            
            // Accumulating the local values
            if (base_idx == 0)
            {{   
                for(idx_i = blockDim.x/2; idx_i > 0; idx_i >>= 1)
                {{
                    if(base_idx < idx_i)
                    {{
                        for(i = 0; i < num_classes; i++) for(j = 0; j < num_classes; j++)
                        {{
                            (*(SHIFT_POINTER_LOCAL_CM(local_cm, base_idx, i, j, num_classes, num_classes))) += (*(SHIFT_POINTER_LOCAL_CM(local_cm, base_idx + idx_i, i, j, num_classes, num_classes)));
                        }}
                    }}
                    __syncthreads();
                }}
            }}
            
            // Accumulating the local values into the output's tensor.
            if (base_idx == 0)
            {{
                for(i = 0; i < num_classes; i++) for(j = 0; j < num_classes; j++)
                {{
                    (*(SHIFT_POINTER_CM(cm, i, j, num_classes))) = (*(SHIFT_POINTER_LOCAL_CM(local_cm, base_idx, i, j, num_classes, num_classes)));
                }}
            }}
        }}
        """
        
        code = code.format(
            T = DTYPE2CTYPE[self.model.dtype],
            name = _name
        )
        module = SourceModule(code).get_function(_name)

        return module 
    #---


    def compute(self, y_pred: TensorGPU, y_targ: TensorGPU) -> np.ndarray:
        """
        The output will be a confusion matrix like this:
                |Predicted     |
        ________| 0  | 1  | 2  |
        Target|0| T0 | F1 | F2 |
              |1| F0 | T1 | F2 |
              |2| F0 | F1 | T2 |
        """

        n = y_pred.shape[0]
        target_classes = self.model.output_shape[0]

        self.conf_matrix.fill(0)
        self.local_cm.fill(0)

        n = np.int32(n)
        num_classes = np.int32(target_classes)

        self.kernel(y_targ.ary, y_pred.ary, 
                    self.conf_matrix.ary, self.local_cm.ary,
                    num_classes, n,
                    grid=self.grid, block=self.block,
                    stream=self.model.stream)
        return self.conf_matrix.ary.get()
