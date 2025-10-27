from pydtnn.backends.gpu.metrics.metric_gpu import MetricGPU
from pydtnn.metrics.confusion_matrix import ConfusionMatrix

from pydtnn.backends.gpu.tensor_gpu import TensorGPU
import numpy as np
import pycuda.gpuarray as gpuarray  # type: ignore
from pycuda.compiler import SourceModule  # type: ignore
from pycuda.driver import Function  # type: ignore
from pydtnn.utils.types import DTYPE2CTYPE

class BinaryConfusionMatrixGPU(MetricGPU, ConfusionMatrix[TensorGPU]):

    def __init_gpu_kernel__(self) -> Function:
        _name = "binary_confusion_matrix"
        module = SourceModule("""
        #define TRUE_POSITIVE  {{0,0}}
        #define TRUE_NEGATIVE  {{1,1}}
        #define FALSE_NEGATIVE {{0,1}}
        #define FALSE_POSITIVE {{1,0}}

        __shared__ const short indexes[2][2][2] = {{
            {{TRUE_POSITIVE, TRUE_NEGATIVE}}, 
            {{FALSE_NEGATIVE, FALSE_POSITIVE}}
        }};
        
        
        __global__ void {name}({T} *y_targ, {T} *y_pred, int *cm, int num_classes, int workers, int n)
        {{
            int label, i, j, idx_i, is_pred_correct, idx;

            int base_idx = blockIdx.x * blockDim.x + threadIdx.x;
            int local_cm[n][num_classes][2][2];

            for(idx = base_idx; idx < n; idx += workers)
            {{
                for(label = 0; label < num_classes; label++)
                {{
                    for(i = 0; i < 2; i++) for(j = 0; j < 2; j++)
                    {{
                        local_cm[idx][label][i][j] = 0;
                    }}

                    // NOTE: y_pred[idx][label]' only possible values are 0 or 1.
                    is_pred_correct = (y_targ[idx][label] == y_pred[idx][label]);
                    local_cm[idx][label][is_pred_correct][(y_pred[idx][label])] += 1;
                }}
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
        For every label in target class, there is one confusion matrix like this:
                |Predicted|
        ________| T  | F  |
        Target|T| TP | FN |
              |F| FP | TN |
        """
        threads = min(self.model.batch_size, 1024)
        blocks = max(self.model.batch_size, 1024) // threads + 1
        
        grid = (blocks, 1, 1)
        block = (threads, 1, 1)

        target_classes = self.model.output_shape[0]
        conf_matrix = TensorGPU.create_zeros_tensor(shape=(target_classes, 2, 2), dtype=np.dtype(np.int32), 
                                                    tensor_format=self.model.tensor_format, cudnn_dtype=self.model.cudnn_dtype)
        
        num_classes = np.int32(target_classes)
        num_workers = np.prod(grid, dtype=np.int32) * np.prod(block, dtype=np.int32)
        n = np.int32(y_pred.size)

        self.kernel(y_targ.ary, y_pred.ary, conf_matrix,
                    num_classes, num_workers, n,
                    grid=grid, block=block,
                    stream=self.model.stream)
        return conf_matrix