from pydtnn.backends.gpu.metrics.metric_gpu import MetricGPU
from pydtnn.metrics.confusion_matrix import ConfusionMatrix

from pydtnn.backends.gpu.tensor_gpu import TensorGPU
import numpy as np
from pycuda.compiler import SourceModule  # type: ignore
from pycuda.driver import Function  # type: ignore
from pydtnn.utils.types import DTYPE2CTYPE

class MulticlassConfusionMatrixGPU(MetricGPU, ConfusionMatrix[TensorGPU]):

    def __init_gpu_kernel__(self) -> Function:
        _name = "multiclass_confusion_matrix"
        module = SourceModule("""
        
        #define INDEX_FIRST_ONE_ON(y, var_class) for(i = 0; (i < num_classes) && !(y[i]); i++); var_class = i;
        
        __global__ void {name}({T} *y_targ, {T} *y_pred, int *cm, int num_classes, int workers, int n)
        {{
            int idx, idx_i, i, j, target_class, predicted_class;

            int base_idx = blockIdx.x * blockDim.x + threadIdx.x;
            int local_cm[n][num_classes][num_classes];

            for(idx = base_idx; idx < n; idx += workers)
            {{
                // Initializing the "thread"'s local confusion matrix.
                for(i = 0; i < num_classes; i++) for(j = 0; j < num_classes; j++, local_cm[idx][i][j] = 0);

                INDEX_FIRST_ONE_ON(y_targ, target_class)
                INDEX_FIRST_ONE_ON(y_pred, predicted_class)
            
                local_cm[idx][label][target_class][predicted_class] += 1;
            }}
            
            // Accumulating the local values
            if (base_idx == 0)
            {{   
                for(idx_i = blockDim.x/2; s > 0; s >>= 1)
                {{
                    if(base_idx < idx_i)
                        for(label = 0; label < num_classes; label++)
                            for(i = 0; i < 2; i++) for(j = 0; j < 2; j++)
                                local_cm[base_idx][label][i][j] += local_cm[base_idx + idx_i][label][i][j];

                    __syncthreads();
                }}
            }}
            
            // Accumulating the local values into the output's tensor.
            if (base_idx == 0)
            {{
                for(label = 0; label < num_classes; label++)
                    for(i = 0; i < 2; i++) for(j = 0; j < 2; j++)
                        cm[label][i][j] = local_cm[base_idx][label][i][j];
            }}
        }}
        """.format(
            T = DTYPE2CTYPE[self.model.dtype]),
            name = _name
        )
        return module.get_function(_name)
    #---


    def compute(self, y_pred: TensorGPU, y_targ: TensorGPU) -> TensorGPU:
        """
        The output will be a confusion matrix like this:
                |Predicted     |
        ________| 0  | 1  | 2  |
        Target|0| T0 | F1 | F2 |
              |1| F0 | T1 | F2 |
              |2| F0 | F1 | T2 |
        """
        threads = min(self.model.batch_size, 1024)
        blocks = max(self.model.batch_size, 1024) // threads + 1
        
        grid = (blocks, 1, 1)
        block = (threads, 1, 1)

        target_classes = self.model.output_shape[0]
        conf_matrix = TensorGPU.create_zeros_tensor(shape=(target_classes, target_classes), dtype=np.dtype(np.int32), 
                                                    tensor_format=self.model.tensor_format, cudnn_dtype=self.model.cudnn_dtype)
        
        num_classes = np.int32(target_classes)
        num_workers = np.prod(grid, dtype=np.int32) * np.prod(block, dtype=np.int32)
        n = np.int32(y_pred.size)

        self.kernel(y_targ.ary, y_pred.ary, conf_matrix,
                    num_classes, num_workers, n,
                    grid=grid, block=block,
                    stream=self.model.stream)
        return conf_matrix