import numpy as np

from pydtnn.backends.gpu.metrics.metric_gpu import MetricGPU
from pydtnn.backends.gpu.metrics.binary_confusion_matrix_gpu import BinaryConfusionMatrixGPU
from pydtnn.metrics.precision import Precision
from pydtnn.utils.types import DTYPE2CTYPE
from pycuda.compiler import SourceModule  # type: ignore
from pycuda.driver import Function  # type: ignore

from pydtnn.backends.gpu.tensor_gpu import TensorGPU


class PrecisionGPU(MetricGPU, Precision[TensorGPU]):

    conf_matrix_metric: BinaryConfusionMatrixGPU

    def __init_gpu_kernel__(self) -> Function:
        _name = "binary_confusion_matrix"
        code = """
        #define TRUE_POSITIVE  {{0,0}}
        #define FALSE_POSITIVE {{1,0}}

        #define SHIFT_POINTER_CM(p, label, i, j, n_clss) p + (label * n_clss + i) * 2 + j
        
        __global__ void {name}({T} *precision, int *cm, {T} *local_precision, const int num_classes)
        {{
            int label, idx, true_positive, false_negative, false_positive;

            int base_idx = blockIdx.x * blockDim.x + threadIdx.x;
            const int workers = blockDim.x * gridDim.x;

            for(idx = base_idx, local_precision[idx] = 0; idx < num_classes; idx += workers)
            {{
                true_positive = (*(SHIFT_POINTER_CM(cm, label, TRUE_POSITIVE[0], TRUE_POSITIVE[1], num_classes)));
                false_positive = (*(SHIFT_POINTER_CM(cm, label, FALSE_POSITIVE[0], FALSE_POSITIVE[1], num_classes)));

                (*(local_precision + idx)) += ({T}) (true_positive / (true_positive + false_negative));
            }}
            
            // Accumulating the local values into the output's tensor.
            if (base_idx == 0)
            {{
                for(idx = 0; label < num_classes; label++)
                    (*local_precision) += local_precision[idx];

                (*precision) = ({T}) ((*local_precision) / num_classes);
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

        target_classes = self.model.output_shape[0]

        num_classes = np.int32(target_classes)
        precision = TensorGPU.create_zeros_tensor(shape=(1, ), dtype=np.dtype(np.int32), 
                                                 tensor_format=self.model.tensor_format, cudnn_dtype=self.model.cudnn_dtype)
        local_precision = TensorGPU.create_zeros_tensor(shape=(int(num_classes), ), dtype=np.dtype(np.int32), 
                                                 tensor_format=self.model.tensor_format, cudnn_dtype=self.model.cudnn_dtype)

        self.kernel(precision.ary, self.conf_matrix_metric.conf_matrix.ary, 
                    local_precision.ary, num_classes,
                    grid=self.grid, block=self.block,
                    stream=self.model.stream)
        
        return precision
