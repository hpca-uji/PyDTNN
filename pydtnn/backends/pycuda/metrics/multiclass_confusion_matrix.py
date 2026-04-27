import logging

import numpy as np

from pydtnn.backends.pycuda.metrics.metric import MetricPycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.metrics.multiclass_confusion_matrix import \
    MulticlassConfusionMatrix

logger = logging.getLogger(__name__)


class MulticlassConfusionMatrixPycuda(MulticlassConfusionMatrix[TensorArray], MetricPycuda):

    def _model_init(self) -> None:
        super()._model_init()
        n = self.model.batch_size
        target_classes = self.model.output_shape[0]

        self.conf_matrix = TensorArray.new_zeros(shape=(1, 1, target_classes, target_classes), dtype=np.dtype(np.int32),
                                                 tensor_format=self.model.tensor_format, cudnn_dtype=self.model.cudnn_dtype)
        self.local_cm = TensorArray.new_zeros(shape=(1, n, target_classes, target_classes), dtype=np.dtype(np.int32),
                                              tensor_format=self.model.tensor_format, cudnn_dtype=self.model.cudnn_dtype)
    # ----

    def compute(self, y_pred: TensorArray, y_targ: TensorArray) -> np.ndarray:
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
        return np.asarray(self.conf_matrix)
