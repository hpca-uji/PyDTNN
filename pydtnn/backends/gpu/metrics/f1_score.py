import numpy as np

from pydtnn.backends.gpu.metrics.metric import MetricGPU
from pydtnn.metrics.f1_score import F1Score
from pydtnn.utils.constants import DTYPE2CTYPE
from pycuda.compiler import SourceModule  # type: ignore
from pycuda.driver import Function  # type: ignore

from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU


class F1ScoreGPU(F1Score[TensorGPU], MetricGPU):

    def initialize(self) -> None:
        super().initialize()
        target_classes = self.model.output_shape[0]
        self.f1 = TensorGPU.create_zeros_tensor(shape=(1, ), dtype=np.dtype(np.int32),
                                                tensor_format=self.model.tensor_format, cudnn_dtype=self.model.cudnn_dtype)
        self.local_f1 = TensorGPU.create_zeros_tensor(shape=(target_classes, ), dtype=np.dtype(np.int32),
                                                      tensor_format=self.model.tensor_format, cudnn_dtype=self.model.cudnn_dtype)
    # ----

    def __init_gpu_kernel__(self) -> Function:
        _name = "binary_confusion_matrix"
        code = """
        //#define TRUE_POSITIVE  {{0,0}}
        #define TRUE_POSITIVE_0  0
        #define TRUE_POSITIVE_1  0

        //#define FALSE_NEGATIVE {{0,1}}
        #define FALSE_NEGATIVE_0 0
        #define FALSE_NEGATIVE_1 1
        
        //#define FALSE_POSITIVE {{1,0}}
        #define FALSE_POSITIVE_0 1
        #define FALSE_POSITIVE_1 0

        #define SHIFT_POINTER_CM(p, label, i, j, num_i, num_j) p + (label * num_i + i) * num_j + j
        
        __global__ void {name}({T} *f1, int *cm, {T} *local_f1, const int num_classes)
        {{
            int label, idx, true_positive, false_negative, false_positive, div;

            int base_idx = blockIdx.x * blockDim.x + threadIdx.x;
            const int workers = blockDim.x * gridDim.x;

            for(idx = base_idx; idx < num_classes; idx += workers)
            {{
                *(local_f1 + idx) = 0;
                true_positive = (*(SHIFT_POINTER_CM(cm, label, TRUE_POSITIVE_0, TRUE_POSITIVE_1, 2, 2)));
                false_negative = (*(SHIFT_POINTER_CM(cm, label, FALSE_NEGATIVE_0, FALSE_NEGATIVE_1, 2, 2)));
                false_positive = (*(SHIFT_POINTER_CM(cm, label, FALSE_POSITIVE_0, FALSE_POSITIVE_1, 2, 2)));
                div = 2 * true_positive + false_positive + false_negative;

                (*(local_f1 + idx)) += ({T}) (div == 0 ? 0 : (2 * true_positive / div));
            }}
            
            // Accumulating the local values into the output's tensor.
            if (base_idx == 0)
            {{
                for(idx = 0; label < num_classes; label++)
                    (*local_f1) += *(local_f1 + idx);

                (*f1) = ({T}) ((*local_f1) / num_classes);
            }}
        }}
        """.format(
            T=DTYPE2CTYPE[self.model.dtype],
            name=_name
        )
        module = SourceModule(code).get_function(_name)

        return module
    # ---

    def compute(self, y_pred: TensorGPU, y_targ: TensorGPU) -> float:

        target_classes = self.model.output_shape[0]

        self.f1.fill(0)
        self.local_f1.fill(0)

        target_classes = np.int32(target_classes)

        self.kernel(self.f1.ary, self.conf_matrix_metric.conf_matrix.ary,
                    self.local_f1.ary, target_classes,
                    grid=self.grid, block=self.block,
                    stream=self.model.stream)

        return self.f1.ary.get()[0]
