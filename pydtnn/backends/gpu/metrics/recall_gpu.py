import numpy as np

from pydtnn.backends.gpu.metrics.metric import MetricGPU
from pydtnn.backends.gpu.metrics.binary_confusion_matrix import BinaryConfusionMatrixGPU
from pydtnn.metrics.recall import Recall
from pydtnn.utils.constants import DTYPE2CTYPE
from pycuda.compiler import SourceModule  # type: ignore
from pycuda.driver import Function  # type: ignore

from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU


class RecallGPU(MetricGPU, Recall[TensorGPU]):

    conf_matrix_metric: BinaryConfusionMatrixGPU

    def __init_gpu_kernel__(self) -> Function:
        _name = "binary_confusion_matrix"
        code = """
        //#define TRUE_POSITIVE  {{0,0}}
        #define TRUE_POSITIVE_0  0
        #define TRUE_POSITIVE_1  0
        
        //#define FALSE_NEGATIVE {{0,1}}
        #define FALSE_NEGATIVE_0 0
        #define FALSE_NEGATIVE_1 1

        #define SHIFT_POINTER_CM(p, label, i, j, n_clss) p + (label * n_clss + i) * 2 + j
        
        __global__ void {name}({T} *recall, int *cm, {T} *local_recall, const int num_classes)
        {{
            int label, idx, true_positive, false_negative, div;

            int base_idx = blockIdx.x * blockDim.x + threadIdx.x;
            const int workers = blockDim.x * gridDim.x;

            for(idx = base_idx, local_recall[idx] = 0; idx < num_classes; idx += workers)
            {{
                true_positive = (*(SHIFT_POINTER_CM(cm, label, (TRUE_POSITIVE_0), (TRUE_POSITIVE_1), num_classes)));
                false_negative = (*(SHIFT_POINTER_CM(cm, label, (FALSE_NEGATIVE_0), (FALSE_NEGATIVE_1), num_classes)));
                div = true_positive + false_negative;

                (*(local_recall + idx)) += ({T}) (div == 0 ? 0 : (true_positive / div));
            }}

            // Accumulating the local values into the output's tensor.
            if (base_idx == 0)
            {{
                for(idx = 0; label < num_classes; label++)
                    (*local_recall) += local_recall[idx];

                (*recall) = ({T}) ((*local_recall) / num_classes);
            }}
        }}
        """.format(
            T = DTYPE2CTYPE[self.model.dtype],
            name = _name
        )
        module = SourceModule(code).get_function(_name)

        return module
    #---

    def compute(self, y_pred: TensorGPU, y_targ: TensorGPU) -> float:

        target_classes = self.model.output_shape[0]

        num_classes = np.int32(target_classes)
        recall = TensorGPU.create_zeros_tensor(shape=(1, ), dtype=np.dtype(np.int32), 
                                                 tensor_format=self.model.tensor_format, cudnn_dtype=self.model.cudnn_dtype)
        local_recall = TensorGPU.create_zeros_tensor(shape=(int(num_classes), ), dtype=np.dtype(np.int32), 
                                                 tensor_format=self.model.tensor_format, cudnn_dtype=self.model.cudnn_dtype)

        self.kernel(recall.ary, self.conf_matrix_metric.conf_matrix.ary, 
                    local_recall.ary, num_classes,
                    grid=self.grid, block=self.block,
                    stream=self.model.stream)
        
        return recall.ary.get()[0]
