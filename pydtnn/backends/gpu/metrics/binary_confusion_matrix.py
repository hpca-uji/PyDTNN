from pydtnn.backends.gpu.metrics.metric import MetricGPU
from pydtnn.metrics.binary_confusion_matrix import BinaryConfusionMatrix

from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU
import numpy as np
from pycuda.compiler import SourceModule  # type: ignore
from pycuda.driver import Function  # type: ignore
from pydtnn.utils.constants import DTYPE2CTYPE

class BinaryConfusionMatrixGPU(BinaryConfusionMatrix[TensorGPU], MetricGPU):

    def initialize(self) -> None:
        super().initialize()
        n = self.model.batch_size
        target_classes = self.model.output_shape[0]

        self.conf_matrix = TensorGPU.create_zeros_tensor(shape=(1, target_classes, 2, 2), dtype=np.dtype(np.int32),
                                                         tensor_format=self.model.tensor_format, cudnn_dtype=self.model.cudnn_dtype)
        self.local_cm = TensorGPU.create_zeros_tensor(shape=(n, target_classes, 2, 2), dtype=np.dtype(np.int32),
                                                      tensor_format=self.model.tensor_format, cudnn_dtype=self.model.cudnn_dtype)
    # ----

    def __init_gpu_kernel__(self) -> Function:
        _name = "binary_confusion_matrix"
        code = """
        #define TRUE_POSITIVE  {{0,0}}
        #define TRUE_NEGATIVE  {{1,1}}
        #define FALSE_NEGATIVE {{0,1}}
        #define FALSE_POSITIVE {{1,0}}

        #define SHIFT_POINTER_CM(label, i, j, num_rows, num_columns) (((label * num_rows + i) * num_columns) + j)
        #define SHIFT_POINTER_LOCAL_CM(idx, label, i, j, num_labels, num_rows, num_columns) ((((idx * num_labels + label) * num_rows + i) * num_columns) + j)
        #define SHIFT_POINTER_Y(i, j, dim_j) (i * dim_j + j)

        __constant__ const short indexes[2][2][2] = {{
            {{TRUE_POSITIVE, TRUE_NEGATIVE}}, 
            {{FALSE_NEGATIVE, FALSE_POSITIVE}}
        }};
        
        
        __global__ void {name}({T} *y_targ, {T} *y_pred, int *cm, int *local_cm, const int num_classes, const int n)
        {{
            int label, i, j, is_pred_correct, idx;
            short index_0, index_1;
            {T} value_targ, value_pred;

            int base_idx = blockIdx.x * blockDim.x + threadIdx.x;
            const int workers = blockDim.x * gridDim.x;

            for(idx = base_idx; idx < n; idx += workers)
            {{
                for(label = 0; label < num_classes; label++)
                {{
                    value_targ = *(y_targ + SHIFT_POINTER_Y(idx, label, num_classes));
                    value_pred = *(y_pred + SHIFT_POINTER_Y(idx, label, num_classes));

                    // NOTE: y_pred[idx][label]' only possible values are 0 or 1.
                    is_pred_correct = (value_targ == value_pred);
                    index_0 = indexes[is_pred_correct][((int) value_pred)][0];
                    index_1 = indexes[is_pred_correct][((int) value_pred)][1];

                    *(local_cm + SHIFT_POINTER_LOCAL_CM(idx, label, index_0, index_1, num_classes, 2, 2)) += 1;
                }}
            }}
            // Accumulating the local values
            if (base_idx == 0)
            {{   
                for(idx = blockDim.x/2; idx > 0; idx >>= 1)
                {{
                    if(base_idx < idx)
                        for(label = 0; label < num_classes; label++)
                            for(i = 0; i < 2; i++) for(j = 0; j < 2; j++)
                                *(local_cm + SHIFT_POINTER_LOCAL_CM(base_idx, label, i, j, num_classes, 2, 2)) += *(local_cm + SHIFT_POINTER_LOCAL_CM(base_idx + idx, label, i, j, num_classes, 2, 2));
                }}
            }}
            __syncthreads();
            
            // Accumulating the local values into the output's tensor.
            if (base_idx == 0)
            {{
                for(label = 0; label < num_classes; label++)
                    for(i = 0; i < 2; i++) for(j = 0; j < 2; j++)
                        *(cm + SHIFT_POINTER_CM(label, i, j, 2, 2)) = *(local_cm + SHIFT_POINTER_LOCAL_CM(base_idx, label, i, j, num_classes, 2, 2));
            }}
        }}
        """.format(
            T = DTYPE2CTYPE[self.model.dtype],
            name = _name
        )
        module = SourceModule(code).get_function(_name)

        return module
    #---

    def compute(self, y_pred: TensorGPU, y_targ: TensorGPU) -> np.ndarray:
        """
        For every label in target class, there is one confusion matrix like this:
                |Predicted|
        ________| T  | F  |
        Target|T| TP | FN |
              |F| FP | TN |
        """

        n = self.model.batch_size
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
