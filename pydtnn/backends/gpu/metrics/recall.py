import numpy as np

from pydtnn.backends.gpu.metrics.metric import MetricGPU
from pydtnn.metrics.recall import Recall
from pydtnn.utils.constants import DTYPE2CTYPE
from pycuda.compiler import SourceModule  # type: ignore
from pycuda.driver import Function  # type: ignore

from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU


class RecallGPU(Recall[TensorGPU], MetricGPU):

    def initialize(self) -> None:
        super().initialize()
        target_classes = self.model.output_shape[0]
        self.recall = TensorGPU.create_zeros_tensor(shape=(1, ), dtype=np.dtype(np.int32), 
                                                    tensor_format=self.model.tensor_format, cudnn_dtype=self.model.cudnn_dtype)
        self.local_recall = TensorGPU.create_zeros_tensor(shape=(target_classes, ), dtype=np.dtype(np.int32), 
                                                          tensor_format=self.model.tensor_format, cudnn_dtype=self.model.cudnn_dtype)

    def __init_gpu_kernel__(self) -> Function:
        _name = "binary_confusion_matrix"
        code = """
        //#define TRUE_POSITIVE  {{0,0}}
        #define TRUE_POSITIVE_0  0
        #define TRUE_POSITIVE_1  0
        
        //#define FALSE_NEGATIVE {{0,1}}
        #define FALSE_NEGATIVE_0 0
        #define FALSE_NEGATIVE_1 1

        #define SHIFT_POINTER_CM(p, label, i, j, num_i, num_j) p + ((label * num_i + i) * num_j + j)
        
        __global__ void {name}({T} *recall, int *cm, {T} *local_recall, const int num_classes)
        {{
            int label, idx, true_positive, false_negative, div;

            int base_idx = blockIdx.x * blockDim.x + threadIdx.x;
            const int workers = blockDim.x * gridDim.x;

            for(idx = base_idx; idx < num_classes; idx += workers)
            {{
                *(local_recall + idx) = 0;
                true_positive = (*(SHIFT_POINTER_CM(cm, label, (TRUE_POSITIVE_0), (TRUE_POSITIVE_1), 2, 2)));
                false_negative = (*(SHIFT_POINTER_CM(cm, label, (FALSE_NEGATIVE_0), (FALSE_NEGATIVE_1), 2, 2)));
                div = true_positive + false_negative;

                (*(local_recall + idx)) += ({T}) (div == 0 ? 0 : (true_positive / div));
            }}

            // Accumulating the local values into the output's tensor.
            if (base_idx == 0)
            {{
                for(idx = 0; label < num_classes; label++)
                    (*local_recall) += *(local_recall + idx);

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

        self.recall.fill(0)
        self.local_recall.fill(0)

        target_classes = np.int32(target_classes)
        self.kernel(self.recall.ary, self.conf_matrix_metric.conf_matrix.ary, 
                    self.local_recall.ary, target_classes,
                    grid=self.grid, block=self.block,
                    stream=self.model.stream)
        
        return self.recall.ary.get()[0]
