from pydtnn.backends.gpu.metrics.metric import MetricGPU
from pydtnn.metrics.binary_confusion_matrix import BinaryConfusionMatrix

from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU
import numpy as np
from pycuda.compiler import SourceModule  # type: ignore
from pycuda.driver import Function  # type: ignore
from pydtnn.utils.constants import DTYPE2CTYPE

class BinaryConfusionMatrixGPU(MetricGPU, BinaryConfusionMatrix[TensorGPU]):

    def __init_gpu_kernel__(self) -> Function:
        _name = "binary_confusion_matrix"
        code = """
        #define TRUE_POSITIVE  {{0,0}}
        #define TRUE_NEGATIVE  {{1,1}}
        #define FALSE_NEGATIVE {{0,1}}
        #define FALSE_POSITIVE {{1,0}}

        #define SHIFT_POINTER_CM(p, label, i, j, n_clss) p + (label * n_clss + i) * 2 + j
        #define SHIFT_POINTER_LOCAL_CM(p, idx, label, i, j, num_n, n_clss) p + (((idx * num_n + label) * n_clss + i) * 2 + j)
        #define SHIFT_POINTER_Y(p, i, j, n) p + (i * n + j)

        __constant__ const short indexes[2][2][2] = {{
            {{TRUE_POSITIVE, TRUE_NEGATIVE}}, 
            {{FALSE_NEGATIVE, FALSE_POSITIVE}}
        }};
        
        
        __global__ void {name}({T} *y_targ, {T} *y_pred, int *cm, int *local_cm, const int num_classes, const int n)
        {{
            int label, i, j, idx_i, is_pred_correct, idx;
            short index_0, index_1;
            {T} value_targ, value_pred;

            int base_idx = blockIdx.x * blockDim.x + threadIdx.x;
            const int workers = blockDim.x * gridDim.x;

            for(idx = base_idx; idx < n; idx += workers)
            {{
                for(label = 0; label < num_classes; label++)
                {{
                    value_targ = (*(SHIFT_POINTER_Y(y_targ, idx, label, n)));
                    value_pred = (*(SHIFT_POINTER_Y(y_pred, idx, label, n)));
                
                    // NOTE: y_pred[idx][label]' only possible values are 0 or 1.
                    is_pred_correct = (value_targ == value_pred);
                    index_0 = indexes[is_pred_correct][((int) value_pred)][0];
                    index_1 = indexes[is_pred_correct][((int) value_pred)][1];
                    (*(SHIFT_POINTER_LOCAL_CM(local_cm, idx, label, index_0, index_1, n, num_classes))) += 1;
                }}
            }}
            
            // Accumulating the local values
            if (base_idx == 0)
            {{   
                for(idx_i = blockDim.x/2; idx_i > 0; idx_i >>= 1)
                {{
                    if(base_idx < idx_i)
                        for(label = 0; label < num_classes; label++)
                            for(i = 0; i < 2; i++) for(j = 0; j < 2; j++)
                                (*(SHIFT_POINTER_LOCAL_CM(local_cm, base_idx, label, i, j, n, num_classes))) += (*(SHIFT_POINTER_LOCAL_CM(local_cm, base_idx + idx_i, label, i, j, n, num_classes)));

                    __syncthreads();
                }}
            }}
            
            // Accumulating the local values into the output's tensor.
            if (base_idx == 0)
            {{
                for(label = 0; label < num_classes; label++)
                    for(i = 0; i < 2; i++) for(j = 0; j < 2; j++)
                        (*(SHIFT_POINTER_CM(cm, label, i, j, num_classes))) = (*(SHIFT_POINTER_LOCAL_CM(local_cm, base_idx, label, i, j, n, num_classes)));
            }}
        }}
        """.format(
            T = DTYPE2CTYPE[self.model.dtype],
            name = _name
        )
        module = SourceModule(code).get_function(_name)

        return module
    #---


    def compute(self, y_pred: TensorGPU, y_targ: TensorGPU) -> TensorGPU:
        """
        For every label in target class, there is one confusion matrix like this:
                |Predicted|
        ________| T  | F  |
        Target|T| TP | FN |
              |F| FP | TN |
        """

        target_classes = self.model.output_shape[0]
        conf_matrix = TensorGPU.create_zeros_tensor(shape=(1, target_classes, 2, 2), dtype=np.dtype(np.int32), 
                                                    tensor_format=self.model.tensor_format, cudnn_dtype=self.model.cudnn_dtype)
        
        num_classes = np.int32(target_classes)
        n = np.int32(y_pred.size)
        local_cm = TensorGPU.create_zeros_tensor(shape=(y_pred.shape[0], target_classes, 2, 2), dtype=np.dtype(np.int32), 
                                                 tensor_format=self.model.tensor_format, cudnn_dtype=self.model.cudnn_dtype)

        self.kernel(y_targ.ary, y_pred.ary, 
                    conf_matrix.ary, local_cm.ary,
                    num_classes, n,
                    grid=self.grid, block=self.block,
                    stream=self.model.stream)
        self.conf_matrix = conf_matrix
        
        return conf_matrix.ary.get()